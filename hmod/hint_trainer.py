"""H-MOD hint training loop.

Self-play the LLM meta-controller against the drift simulator with the trained
R-PADPP low policy, read back the full metric feedback each epoch, and let an
LLM distiller turn that feedback into a growing playbook of general hints. The
hints are injected into the controller's reflection prompt on the next epoch,
so the controller bootstraps better w_t choices over epochs. The final playbook
is saved to a JSON file for eval-time loading.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from hmod.hint_distiller import LLMHintDistiller, build_episode_digest
from hmod.hints import HintStore
from hmod.runner import HMODEvaluator
from hmod.scenario import HMODScenario


def _fmt_metrics(m: Dict[str, Any]) -> str:
    def g(k):
        v = m.get(k)
        return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
    return (f"GSR={g('gsr')} llm_sr={g('llm_sr')} T2DA={g('t2da')} "
            f"CVR={g('cvr')} (n={m.get('num_dialogues')})")


class HMODHintTrainer:
    def __init__(
        self,
        scenarios: List[HMODScenario],
        buyer_policy: Any,
        hint_store: HintStore,
        distiller: LLMHintDistiller,
        *,
        mode: str = "hmod_dynamic",
        judge_model: str = "rule",
        reflection_horizon: int = 3,
        turn_limit_mult: float = 1.0,
        verbose: bool = False,
        use_llm_simulator: bool = False,
        audit_sample_size: int = 0,
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_api_key_env: str = "DEEPINFRA_API_KEY",
        llm_base_url: Optional[str] = None,
        llm_temperature: float = 0.0,
        llm_max_tokens: int = 500,
        llm_fallback_to_rule: bool = True,
    ):
        self.scenarios = scenarios
        self.buyer_policy = buyer_policy
        self.hint_store = hint_store
        self.distiller = distiller
        self.cfg = dict(
            mode=mode,
            judge_model=judge_model,
            reflection_horizon=reflection_horizon,
            turn_limit_mult=turn_limit_mult,
            verbose=verbose,
            use_llm_simulator=use_llm_simulator,
            audit_sample_size=audit_sample_size,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_api_key_env=llm_api_key_env,
            llm_base_url=llm_base_url,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            llm_fallback_to_rule=llm_fallback_to_rule,
        )

    def _make_evaluator(self) -> HMODEvaluator:
        c = self.cfg
        return HMODEvaluator(
            mode=c["mode"],
            judge_model=c["judge_model"],
            use_llm_simulator=c["use_llm_simulator"],
            audit_sample_size=c["audit_sample_size"],
            reflection_horizon=c["reflection_horizon"],
            controller_mode="llm_reflection",
            llm_model=c["llm_model"],
            llm_api_key=c["llm_api_key"],
            llm_api_key_env=c["llm_api_key_env"],
            llm_base_url=c["llm_base_url"],
            llm_temperature=c["llm_temperature"],
            llm_max_tokens=c["llm_max_tokens"],
            llm_fallback_to_rule=c["llm_fallback_to_rule"],
            buyer_policy=self.buyer_policy,
            verbose=c["verbose"],
            turn_limit_mult=c["turn_limit_mult"],
            hint_provider=self.hint_store.provider(),
        )

    def train(self, epochs: int) -> HintStore:
        logger.info(
            f"===== H-MOD hint training: {epochs} epochs over "
            f"{len(self.scenarios)} scenarios | starting hints={len(self.hint_store.hints)} ====="
        )
        for epoch in range(1, epochs + 1):
            evaluator = self._make_evaluator()
            result = evaluator.run(self.scenarios)
            metrics = result["metrics"]
            logger.info(f"[epoch {epoch}/{epochs}] self-play metrics: {_fmt_metrics(metrics)}")

            digest = build_episode_digest(result["dialogues"])
            try:
                new_hints = self.distiller.distill(
                    self.hint_store.hints, digest, metrics, epoch
                )
            except Exception as exc:
                logger.warning(f"[epoch {epoch}] hint distillation failed ({exc}); keeping current hints")
                continue

            self.hint_store.update(new_hints, metrics)
            logger.info(f"[epoch {epoch}] updated playbook -> {len(self.hint_store.hints)} hints:")
            for i, h in enumerate(self.hint_store.hints, 1):
                logger.info(f"    {i}. {h}")

        logger.info(
            f"===== Hint training done. {len(self.hint_store.hints)} hints saved to "
            f"{self.hint_store.path} ====="
        )
        return self.hint_store
