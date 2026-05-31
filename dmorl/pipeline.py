"""
DMORL Pipeline
Extends PADPP pipelines with the 3-phase DMORL training execution:
  Phase 1a: Basic-skill curriculum (fixed-weight RLT per skill)
  Phase 1b: Advanced-skill training (skill-biased RLT)
  Phase 2:  Full PADPP RLT (random weight sampling, generalisation)
  Phase 3:  Post-dialogue refinement at inference time

One concrete pipeline class per scenario (Recommendation, Negotiation,
EmotionalSupport) mirrors the PADPP structure.
"""

import os
import random
import copy
import torch

from loguru import logger
from sklearn.model_selection import train_test_split

from padpp.pipeline import (
    PADPPPipeline,
    PADPPPipelineForRecommendation,
    PADPPPipelineForNegotiation,
    PADPPPipelineForEmotionalSupport,
)
from dmorl.llm_controller import DMORLController
from dmorl.trainer import DMORLTrainer
from hmod.training import HMODController, HMOD_OBJECTIVE_ORDER
from utils.game import create_target_set, create_cases
from text_gen.bart_generation import BARTGeneration

# Objective name maps per scenario (must match config.constants metric names)
SCENARIO_OBJECTIVE_NAMES = {
    "recommendation":   ["user_reward", "item_freq"],
    "negotiation":      ["sl_ratio", "fairness", "deal_rate", "avg_turn"],
    "emotional_support": ["user_reward", "toxicity", "avg_turn"],
}

# Short human-readable semantics for each objective, used to ground the LLM
# during skill discovery (so it understands e.g. that avg_turn is a per-turn
# PENALTY and a high weight on it means "prefer closing fast").
OBJECTIVE_DESCRIPTIONS = {
    # Negotiation (agent acts as the buyer)
    "sl_ratio":    "Reward in [0, 1]: 1.0 = agent (buyer) secured a very low price; 0.5 = midpoint; 0.0 = paid the seller's full ask. Higher weight => agent is more price-aggressive.",
    "fairness":    "Reward in [-0.5, 0.5], max 0.5 when the proposed price equals the midpoint between buyer's target and seller's ask. Higher weight => agent prefers equitable mid-range outcomes.",
    "deal_rate":   "Binary terminal reward, 1.0 if a deal is closed, 0.0 if the conversation times out. Higher weight => agent strongly prefers reaching ANY agreement.",
    "avg_turn":    "Constant per-turn PENALTY of -0.1 (negative); the longer the dialogue the larger the cumulative penalty. Higher weight => agent strongly prefers closing in fewer turns (urgency).",
    # Recommendation
    "user_reward": "User satisfaction with the recommended item (positive = liked).",
    "item_freq":   "Coverage of the long-tail item space (positive = recommending diverse items).",
    # Emotional support
    "toxicity":    "Negative toxicity score; higher weight => agent avoids harmful language.",
}


class DMORLPipeline(PADPPPipeline):
    """
    Base DMORL pipeline. Injects the DMORLController into the trainer,
    runs skill discovery, then executes the 3-phase training.
    """

    def _init_dmorl_controller(self):
        """Build and attach a DMORLController to the trainer."""
        scenario = self.game_config.name
        if getattr(self.model_config, "hmod_enabled", False):
            objective_names = HMOD_OBJECTIVE_ORDER[: self.model_config.n_objectives]
        else:
            objective_names = SCENARIO_OBJECTIVE_NAMES.get(
                scenario, [f"obj_{i}" for i in range(self.model_config.n_objectives)]
            )
        objective_descriptions = {
            name: OBJECTIVE_DESCRIPTIONS[name]
            for name in objective_names if name in OBJECTIVE_DESCRIPTIONS
        }
        saved_dir = getattr(self.model_config, "saved_dir", "checkpoints")
        skill_log_file = os.path.join(saved_dir, "skill_discovery.txt")
        if getattr(self.model_config, "hmod_enabled", False):
            controller = HMODController(
                n_objectives=self.model_config.n_objectives,
                objective_names=objective_names,
                scenario=scenario,
                objective_file=self.model_config.hmod_objective_file,
                objective_id=self.model_config.hmod_objective_id,
                n_basic_skills=self.model_config.n_basic_skills,
                n_advanced_skills=self.model_config.n_advanced_skills,
                dynamic_weight_horizon=self.model_config.dynamic_weight_horizon,
                skills_file=self.model_config.skills_file,
                hints_file=self.model_config.hints_file,
                skill_log_file=skill_log_file,
                controller_mode=getattr(self.model_config, "hmod_controller_mode", "rule_scaffold"),
                macro_goal=getattr(self.model_config, "hmod_macro_goal", None),
                llm_model=getattr(self.model_config, "hmod_llm_model", None),
                llm_api_key=getattr(self.model_config, "hmod_llm_api_key", None),
                llm_api_key_env=getattr(self.model_config, "hmod_llm_api_key_env", "HMOD_LLM_API_KEY"),
                llm_base_url=getattr(self.model_config, "hmod_llm_base_url", None),
                llm_temperature=getattr(self.model_config, "hmod_llm_temperature", 0.0),
                llm_max_tokens=getattr(self.model_config, "hmod_llm_max_tokens", 500),
                llm_fallback_to_rule=getattr(self.model_config, "hmod_llm_fallback_to_rule", False),
            )
        else:
            controller = DMORLController(
                n_objectives=self.model_config.n_objectives,
                objective_names=objective_names,
                objective_descriptions=objective_descriptions,
                scenario=scenario,
                n_basic_skills=self.model_config.n_basic_skills,
                n_advanced_skills=self.model_config.n_advanced_skills,
                dynamic_weight_horizon=self.model_config.dynamic_weight_horizon,
                skills_file=self.model_config.skills_file,
                hints_file=self.model_config.hints_file,
                skill_log_file=skill_log_file,
            )
        # Phase 0: Discover skills (loads from file if already done)
        controller.initialize_skills(
            force_rediscover=self.model_config.force_rediscover_skills
        )
        # Attach skill library to model (enables GPI over skills)
        self.trainer.model.set_skill_library(controller.skill_library)
        # Attach controller to trainer (enables dynamic weight + hints)
        self.trainer.dmorl_controller = controller
        if getattr(self.model_config, "hmod_enabled", False):
            logger.info("[H-MOD] Controller initialised and attached to trainer.")
        else:
            logger.info("[DMORL] Controller initialised and attached to trainer.")
        return controller

    def execute(self):
        """
        Full DMORL pipeline:
          SFT → offline eval → Phase 1a → Phase 1b → Phase 2 (PADPP RLT) → online eval
        """
        offline_eval_results, online_eval_results = None, None

        # ── Testing-only mode ────────────────────────────────────────────────
        if self.model_config.test_phase:
            if self.model_config.run_online_eval:
                self._init_dmorl_controller()
                if self.model_config.ablation not in (None, ''):
                    self.load_pretrained_model(is_rl=False)
                else:
                    self.load_pretrained_model(is_rl=True)

                if not isinstance(self.trainer.generation_method, BARTGeneration):
                    self.trainer.generation_method.generation_config.set_params(
                        {'temperature': 0.0}
                    )
                online_eval_results = self.run_online_test()
            return offline_eval_results, online_eval_results

        # ── Full training + evaluation ───────────────────────────────────────
        if self.model_config.run_sft:
            logger.info("[DMORL] Running SFT ...")
            self.run_sft()
            self.trainer.global_step = 0

        if self.model_config.run_offline_eval:
            self.load_pretrained_model(is_rl=False)
            logger.info("[DMORL] Running offline evaluation ...")
            offline_eval_results = self.run_offline_test()

        if self.model_config.run_rlt:
            self.trainer.global_step = 0
            self.load_pretrained_model(is_rl=False)

            # Phase 0: initialise skill library
            self._init_dmorl_controller()

            if self.model_config.run_curriculum:
                logger.info("[DMORL] Phase 1a: Basic skill curriculum ...")
                self.run_basic_skills()

                if getattr(self.model_config, "phase1a_only", False):
                    logger.info("[DMORL] phase1a_only=True — stopping after Phase 1a.")
                    return offline_eval_results, None

                logger.info("[DMORL] Phase 1b: Advanced skill training ...")
                self.run_advanced_skills()

            if getattr(self.model_config, "hmod_enabled", False) and getattr(
                self.model_config, "hmod_phase2_dynamic_training", False
            ):
                logger.info("[H-MOD] Phase 2: Dynamic objective-conditioned RLT ...")
                self.run_hmod_phase2()
            else:
                logger.info("[DMORL] Phase 2: Full PADPP RLT (generalisation) ...")
                self.run_rlt()

        if self.model_config.run_online_eval:
            self.load_pretrained_model(is_rl=True)
            logger.info("[DMORL] Running online evaluation ...")
            if not isinstance(self.trainer.generation_method, BARTGeneration):
                self.trainer.generation_method.generation_config.set_params(
                    {'temperature': 0.0}
                )
            online_eval_results = self.run_online_test()

        return offline_eval_results, online_eval_results

    # ── Subclasses implement run_basic_skills / run_advanced_skills / run_rlt
    # ── and run_online_test (scenario-specific case/simulator setup)

    def run_basic_skills(self):
        raise NotImplementedError

    def run_advanced_skills(self):
        raise NotImplementedError

    def run_hmod_phase2(self):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation
# ─────────────────────────────────────────────────────────────────────────────

class DMORLPipelineForRecommendation(DMORLPipeline, PADPPPipelineForRecommendation):
    """DMORL pipeline for the recommendation scenario."""

    def load_pretrained_model(self, is_rl=False, is_last=False):
        if not is_rl:
            path = os.path.join(
                self.model_config.saved_dir, f"model_{self.model_config.domain}.pth")
        else:
            path = os.path.join(
                self.model_config.saved_dir, f"rl_model_{self.model_config.domain}.pth")
        if not os.path.exists(path):
            raise Exception(f"No pretrained model at {path}")
        self.model = self.trainer.load_model(path)

    def _get_rlt_splits(self, dev_ratio=0.1):
        dev_target_items = create_target_set(
            self.dataset.train_convs,
            test_instances=self.dataset.dev_instances,
            num_items=self.dataset_config.num_dev_items,
            domain=self.dataset_config.domain,
        )
        action_mapping = self.dataset.construct_action_mapping(
            combine=self.model_config.combined_action)
        assert self.dev_simulators is not None
        train_items, dev_items = train_test_split(
            dev_target_items, test_size=dev_ratio, random_state=self.game_config.seed)
        train_sims, dev_sims = train_test_split(
            self.dev_simulators, test_size=dev_ratio, random_state=self.game_config.seed)
        if len(dev_sims) > len(dev_items):
            dev_sims = random.sample(dev_sims, len(dev_items))
        return train_items, dev_items, train_sims, dev_sims, action_mapping

    def run_basic_skills(self):
        train_items, _, train_sims, _, action_mapping = self._get_rlt_splits()
        assert isinstance(self.trainer, DMORLTrainer)
        self.trainer.train_basic_skills(
            cases=train_items, device=self.device,
            simulators=train_sims, action_mapping=action_mapping)

    def run_advanced_skills(self):
        train_items, _, train_sims, _, action_mapping = self._get_rlt_splits()
        assert isinstance(self.trainer, DMORLTrainer)
        self.trainer.train_advanced_skills(
            cases=train_items, device=self.device,
            simulators=train_sims, action_mapping=action_mapping)

    def run_hmod_phase2(self):
        train_items, dev_items, train_sims, dev_sims, action_mapping = self._get_rlt_splits()
        assert isinstance(self.trainer, DMORLTrainer)
        return self.trainer.train_hmod_phase2(
            cases=train_items, dev_cases=dev_items,
            device=self.device, simulators=train_sims,
            dev_simulators=dev_sims, action_mapping=action_mapping)

    def run_rlt(self, dev_ratio=0.1):
        train_items, dev_items, train_sims, dev_sims, action_mapping = \
            self._get_rlt_splits(dev_ratio)
        return self.trainer.train_rlt(
            cases=train_items, dev_cases=dev_items,
            device=self.device, simulators=train_sims,
            dev_simulators=dev_sims, action_mapping=action_mapping)

    def run_online_test(self, target_items=None, simulators=None):
        if target_items is None:
            target_items = create_target_set(
                self.dataset.train_convs,
                test_instances=self.dataset.test_instances,
                num_items=self.dataset_config.num_test_items,
                domain=self.dataset_config.domain,
            )
            simulators = self.test_simulators
            if self.model_config.num_test_cases:
                random.seed(self.game_config.seed)
                target_items = random.sample(target_items, self.model_config.num_test_cases)

        action_mapping = self.dataset.construct_action_mapping(
            combine=self.model_config.combined_action)
        if len(simulators) > len(target_items):
            simulators = random.sample(simulators, len(target_items))

        assert self.test_simulators is not None
        with torch.no_grad():
            # Use DMORL dynamic online test (Phase 2 + Phase 3)
            results = self.trainer.online_test_dmorl(
                target_items, device=self.device,
                simulators=simulators, action_mapping=action_mapping,
                stage='test')
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Negotiation
# ─────────────────────────────────────────────────────────────────────────────

class DMORLPipelineForNegotiation(DMORLPipeline, PADPPPipelineForNegotiation):
    """DMORL pipeline for the negotiation scenario."""

    def load_pretrained_model(self, is_rl=False, is_last=False):
        if not is_rl:
            path = os.path.join(self.model_config.saved_dir, "model.pth")
        else:
            path = os.path.join(self.model_config.saved_dir, "rl_model.pth")
        if not os.path.exists(path):
            raise Exception(f"No pretrained model at {path}")
        self.model = self.trainer.load_model(path)

    def _get_rlt_splits(self, dev_ratio=0.1):
        dev_cases = create_cases(
            test_instances=self.dataset.dev_instances,
            num_cases=self.dataset_config.num_dev_cases)
        action_mapping = self.dataset.construct_action_mapping(
            combine=self.model_config.combined_action)
        assert self.dev_simulators is not None
        train_cases, dev_cases_split = train_test_split(
            dev_cases, test_size=dev_ratio, random_state=self.game_config.seed)
        train_sims, dev_sims = train_test_split(
            self.dev_simulators, test_size=dev_ratio, random_state=self.game_config.seed)
        if len(dev_sims) > len(dev_cases_split):
            dev_sims = random.sample(dev_sims, len(dev_cases_split))
        return train_cases, dev_cases_split, train_sims, dev_sims, action_mapping

    def run_basic_skills(self):
        train_cases, _, train_sims, _, action_mapping = self._get_rlt_splits()
        self.trainer.train_basic_skills(
            cases=train_cases, device=self.device,
            simulators=train_sims, action_mapping=action_mapping)

    def run_advanced_skills(self):
        train_cases, _, train_sims, _, action_mapping = self._get_rlt_splits()
        self.trainer.train_advanced_skills(
            cases=train_cases, device=self.device,
            simulators=train_sims, action_mapping=action_mapping)

    def run_hmod_phase2(self):
        train_cases, dev_cases, train_sims, dev_sims, action_mapping = \
            self._get_rlt_splits()
        return self.trainer.train_hmod_phase2(
            cases=train_cases, dev_cases=dev_cases,
            device=self.device, simulators=train_sims,
            dev_simulators=dev_sims, action_mapping=action_mapping)

    def run_rlt(self, dev_ratio=0.1):
        train_cases, dev_cases, train_sims, dev_sims, action_mapping = \
            self._get_rlt_splits(dev_ratio)
        return self.trainer.train_rlt(
            cases=train_cases, dev_cases=dev_cases,
            device=self.device, simulators=train_sims,
            dev_simulators=dev_sims, action_mapping=action_mapping)

    def run_online_test(self, cases=None, simulators=None):
        if cases is None:
            cases = create_cases(
                test_instances=self.dataset.test_instances,
                num_cases=self.dataset_config.num_test_cases)
            simulators = self.test_simulators
        action_mapping = self.dataset.construct_action_mapping(
            combine=self.model_config.combined_action)
        if len(simulators) > len(cases):
            simulators = random.sample(simulators, len(cases))

        assert self.test_simulators is not None
        with torch.no_grad():
            results = self.trainer.online_test_dmorl(
                cases, device=self.device,
                simulators=simulators, action_mapping=action_mapping,
                stage='test')
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Emotional Support
# ─────────────────────────────────────────────────────────────────────────────

class DMORLPipelineForEmotionalSupport(DMORLPipeline, PADPPPipelineForEmotionalSupport):
    """DMORL pipeline for the emotional support scenario."""

    def load_pretrained_model(self, is_rl=False, is_last=False):
        if not is_rl:
            path = os.path.join(self.model_config.saved_dir, "model.pth")
        else:
            path = os.path.join(self.model_config.saved_dir, "rl_model.pth")
        if not os.path.exists(path):
            raise Exception(f"No pretrained model at {path}")
        self.model = self.trainer.load_model(path)

    def _get_rlt_splits(self, dev_ratio=0.1):
        dev_cases = create_cases(
            test_instances=self.dataset.dev_instances,
            num_cases=self.dataset_config.num_dev_cases)
        action_mapping = self.dataset.construct_action_mapping(
            combine=self.model_config.combined_action)
        assert self.dev_simulators is not None
        train_cases, dev_cases_split = train_test_split(
            dev_cases, test_size=dev_ratio, random_state=self.game_config.seed)
        train_sims, dev_sims = train_test_split(
            self.dev_simulators, test_size=dev_ratio, random_state=self.game_config.seed)
        if len(dev_sims) > len(dev_cases_split):
            dev_sims = random.sample(dev_sims, len(dev_cases_split))
        return train_cases, dev_cases_split, train_sims, dev_sims, action_mapping

    def run_basic_skills(self):
        train_cases, _, train_sims, _, action_mapping = self._get_rlt_splits()
        self.trainer.train_basic_skills(
            cases=train_cases, device=self.device,
            simulators=train_sims, action_mapping=action_mapping)

    def run_advanced_skills(self):
        train_cases, _, train_sims, _, action_mapping = self._get_rlt_splits()
        self.trainer.train_advanced_skills(
            cases=train_cases, device=self.device,
            simulators=train_sims, action_mapping=action_mapping)

    def run_hmod_phase2(self):
        train_cases, dev_cases, train_sims, dev_sims, action_mapping = \
            self._get_rlt_splits()
        return self.trainer.train_hmod_phase2(
            cases=train_cases, dev_cases=dev_cases,
            device=self.device, simulators=train_sims,
            dev_simulators=dev_sims, action_mapping=action_mapping)

    def run_rlt(self, dev_ratio=0.1):
        train_cases, dev_cases, train_sims, dev_sims, action_mapping = \
            self._get_rlt_splits(dev_ratio)
        return self.trainer.train_rlt(
            cases=train_cases, dev_cases=dev_cases,
            device=self.device, simulators=train_sims,
            dev_simulators=dev_sims, action_mapping=action_mapping)

    def run_online_test(self, cases=None, simulators=None):
        if cases is None:
            cases = create_cases(
                test_instances=self.dataset.test_instances,
                num_cases=self.dataset_config.num_test_cases)
            simulators = self.test_simulators
        action_mapping = self.dataset.construct_action_mapping(
            combine=self.model_config.combined_action)
        if len(simulators) > len(cases):
            simulators = random.sample(simulators, len(cases))

        assert self.test_simulators is not None
        with torch.no_grad():
            results = self.trainer.online_test_dmorl(
                cases, device=self.device,
                simulators=simulators, action_mapping=action_mapping,
                stage='test')
        return results
