"""Neural low policy adapter for the merged H-MOD pipeline.

Wraps the trained R-PADPP / DMORL model (the low policy `w -> action`) so the
H-MOD dynamic controller can drive it turn by turn. The H-MOD meta-controller
produces the dynamic 3-D weight `w_t`; this adapter feeds `(state, w_t)` to the
DMORL `predict()` and renders the chosen `(strategy, bin)` into a buyer
utterance plus its committed price.

The DMORL stack is heavy (torch, RoBERTa backbone, generation LLM), so all of it
is constructed lazily here and kept out of `hmod.policy` to avoid import cycles.
"""

import re
from typing import Any, Dict, List, Optional

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs

from utils.utils import (
    get_datasets_by_names, get_model_by_names, get_scenario_by_name,
    load_config_from_yaml_file, get_text_generation_model_by_name,
)
from config.config import DatasetConfigForRecommendation
from config.constants import RECOMMENDATION, NEGOTIATION, BART_GENERATION


# Strategies that commit to a buyer price (mirrors hmod.policy.PRICE_ACTIONS and
# the generation-side price-bearing set).
_PRICE_STRATEGIES = {"propose", "counter", "final_offer", "agree"}
_BIN_STRATEGIES = {"propose", "counter", "final_offer"}


def _first_price(text: str) -> Optional[float]:
    nums = re.findall(r"[-+]?\d*\.?\d+", (text or "").replace(",", ""))
    return float(nums[0]) if nums else None


class NeuralLowPolicy:
    """Builds the DMORL trainer from a checkpoint and exposes `.act`."""

    def __init__(
        self,
        checkpoint: str,
        scenario: str = "negotiation",
        datasets: str = "craigslist_bargain",
        models: str = "dmorl",
        gen_models: str = "fpt",
        model_type: str = "fpt",
        device: Optional[str] = None,
        mask_redundant_actions: bool = True,
    ):
        # A real Accelerator is needed because the trainer reads
        # self.device = accelerator.device and uses it inside predict().
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(device_placement=True, kwargs_handlers=[ddp_kwargs])
        self.device = str(self.accelerator.device)

        # ── Scenario / game ───────────────────────────────────────────────
        game_config_file, game_config_class, game_class, _ = get_scenario_by_name(scenario)
        game_config = game_config_class(load_config_from_yaml_file(game_config_file))
        game_config.set_params({"model_type": model_type})
        self.game_config = game_config

        # ── Dataset (only needed for the action mapping) ──────────────────
        data_cfgs = get_datasets_by_names(scenario, datasets)
        data_config_path, dataset_class, dataset_scenario_config_class = data_cfgs[0]
        dataset_config = dataset_scenario_config_class(
            load_config_from_yaml_file(data_config_path))
        if isinstance(dataset_config, DatasetConfigForRecommendation):
            dataset_config.set_params({"domain": "all"})
        dataset = dataset_class(dataset_config)

        # ── Model config ──────────────────────────────────────────────────
        model_cfgs = get_model_by_names(scenario, models)
        config_file, config_class, model_class, _, trainer_class = model_cfgs[0]
        model_config = config_class(load_config_from_yaml_file(config_file))
        model_config.set_params({"model_type": game_config.model_type})
        scen_params = {
            "n_goals": dataset.n_goals,
            "n_objectives": game_config.n_objectives,
            "device": self.device,
            "scenario_name": scenario,
        }
        if scenario == RECOMMENDATION:
            scen_params["n_topics"] = dataset.n_topics
            scen_params["domain"] = "all"
        model_config.set_params(scen_params)
        model_config.set_params({"run_sft": False, "run_rlt": False, "test_phase": True})
        model_config.set_params({"mask_redundant_actions": mask_redundant_actions})
        self.model_config = model_config

        game = game_class(game_config=game_config, dataset_config=dataset_config)

        # ── Generation method (LLM that renders utterances) ───────────────
        gen_packages = get_text_generation_model_by_name(scenario, gen_models)
        gen_name = gen_models.split(",")[0].strip()
        if gen_name == BART_GENERATION:
            raise NotImplementedError("NeuralLowPolicy needs an LLM gen model (fpt/llama3/chatgpt).")
        (gen_config_path, gen_prompt, gen_config_class, gen_class) = gen_packages[0]
        gen_config = gen_config_class(load_config_from_yaml_file(gen_config_path))
        gen_config.set_params({
            "prompt": gen_prompt,
            "scenario_name": game_config.name,
            "dataset": dataset_config.dataset_name,
        })
        generation_method = gen_class(gen_config, None, None)
        # deterministic inference where supported
        try:
            generation_method.generation_config.set_params({"temperature": 0.0})
        except Exception:
            pass
        self.generation_method = generation_method

        # ── Trainer + checkpoint ──────────────────────────────────────────
        model = model_class(model_config)
        trainer = trainer_class(
            accelerator=self.accelerator,
            game_config=game_config,
            model_config=model_config,
            game=game,
            model=model,
            generation_method=generation_method,
            offline_evaluator=None,
            online_evaluator=None,
            loggers=[],
        )
        trainer.load_model(checkpoint)
        trainer.model.to(self.device)
        trainer.model.eval()
        self.trainer = trainer

        self.action_mapping = dataset.construct_action_mapping(
            combine=model_config.combined_action)
        self.n_bins = getattr(model_config, "n_topics", 5)

        # The data processor always computes a (goal, bin) -> id label, even at
        # inference where it is unused. It reads instance['response'] and
        # instance['goal']; a fresh H-MOD state has neither. Pick any real goal
        # so action_to_id[(goal, bin)] never KeyErrors (value is irrelevant).
        _mapping = (self.action_mapping[0]
                    if isinstance(self.action_mapping, tuple)
                    else self.action_mapping)
        _first_key = next(iter(_mapping))
        self._default_goal = _first_key[0] if isinstance(_first_key, tuple) else _first_key

    # ─────────────────────────────────────────────────────────────────────
    def _price_from_action(self, strategy: str, bin_idx: int, dmorl_state: Dict[str, Any]) -> Optional[float]:
        tb = dmorl_state["task_background"]
        buyer_p = float(tb["buyer_price"])
        seller_p = float(tb["seller_price"])
        if strategy in _BIN_STRATEGIES:
            bin_width = (seller_p - buyer_p) / max(self.n_bins, 1)
            return float(int(buyer_p + bin_width * bin_idx))
        if strategy == "agree":
            # accept the seller's most recent offer
            for turn in reversed(dmorl_state.get("dialogue_context", [])):
                if turn.get("role") == "user":
                    p = _first_price(turn.get("content", ""))
                    if p is not None:
                        return p
            return None
        return None

    def act(self, dmorl_state: Dict[str, Any], weight: List[float]) -> Dict[str, Any]:
        """Map (state, w) -> {strategy, price, utterance} via the low policy."""
        dmorl_state = dict(dmorl_state)
        dmorl_state["w"] = list(weight)
        # Dummy fields the data processor reads to build its (unused) SFT label.
        dmorl_state.setdefault("response", "")
        dmorl_state.setdefault("goal", self._default_goal)
        w_tensor = torch.FloatTensor(weight).to(self.device)

        with torch.no_grad():
            action, _, _ = self.trainer.predict(
                dmorl_state, w_tensor, self.action_mapping,
                is_test=True, is_computing_reward=False, use_gpi=False,
            )

        # action is (strategy, bin)
        if isinstance(action, tuple):
            strategy, bin_idx = action[0], int(action[1])
        else:
            strategy, bin_idx = action, 0

        dmorl_state["pred_goal"] = action
        try:
            utterance = self.generation_method.generate_response(dmorl_state)
        except Exception as exc:  # generation failure should not crash eval
            utterance = f"(generation_error: {exc})"

        # strip any leftover price tag the generator might emit
        utterance = re.sub(r"\[\[\s*PRICE\s*:.*?\]\]", "", utterance or "").strip()

        price = self._price_from_action(strategy, bin_idx, dmorl_state)
        return {"strategy": strategy, "price": price, "utterance": utterance}
