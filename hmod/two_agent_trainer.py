"""Two-agent H-MOD training loop.

Two agents are trained independently, each distilling its own hint playbook:

  * High-Policy agent — driven by the GOLD seller intent each epoch, produces
    w_local; scored by dialogue quality (GSR / T2DA / CVR) and reviews its
    weight-setting hints.
  * Intent-Drift Detector — predicts drift/intent every turn; scored against the
    gold intent and reviews its detection hints.

Each epoch ends with a REVIEW step per agent: hints proposed for removal on two
consecutive epochs are dropped, and new hints are added.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from hmod.hint_distiller import (
    ReviewHintDistiller,
    build_detector_digest,
    build_episode_digest,
)
from hmod.hints import HintStore
from hmod.high_policy import LLMHighPolicy
from hmod.intent_detector import (
    SELLER_INTENT_TYPES,
    LLMIntentDetector,
    build_intent_fewshot,
)
from hmod.llm_reflection import LLMWeightReflector
from hmod.policy import RuleMetaController
from hmod.runner import HMODEvaluator
from hmod.scenario import HMODScenario
from hmod.two_agent_controller import TwoAgentMetaController


def _fmt(m: Dict[str, Any]) -> str:
    def g(k):
        v = m.get(k)
        return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
    return f"GSR={g('gsr')} llm_sr={g('llm_sr')} T2DA={g('t2da')} CVR={g('cvr')}"


class TwoAgentHintTrainer:
    def __init__(
        self,
        scenarios: List[HMODScenario],
        buyer_policy: Any,
        reflector: LLMWeightReflector,
        policy_hints: HintStore,
        detector_hints: HintStore,
        *,
        fewshot_scenario_file: str,
        judge_model: str = "rule",
        turn_limit_mult: float = 1.0,
        verbose: bool = False,
        use_llm_simulator: bool = False,
        llm_fallback_to_rule: bool = True,
    ):
        self.scenarios = scenarios
        self.buyer_policy = buyer_policy
        self.policy_hints = policy_hints
        self.detector_hints = detector_hints
        self.judge_model = judge_model
        self.turn_limit_mult = turn_limit_mult
        self.verbose = verbose
        self.use_llm_simulator = use_llm_simulator
        self.llm_fallback_to_rule = llm_fallback_to_rule

        # Build the two agents once; their hint providers read the (mutating)
        # stores so each epoch automatically sees the updated playbooks.
        fewshot = build_intent_fewshot(fewshot_scenario_file)
        self.detector = LLMIntentDetector(
            reflector=reflector, fewshot=fewshot,
            hint_provider=detector_hints.provider(),
        )
        self.high_policy = LLMHighPolicy(
            reflector=reflector, hint_provider=policy_hints.provider(),
        )
        self.policy_distiller = ReviewHintDistiller(reflector, kind="high_policy")
        self.detector_distiller = ReviewHintDistiller(reflector, kind="intent_detection")
        self._rule = RuleMetaController()

    def _epoch_controller(self) -> TwoAgentMetaController:
        return TwoAgentMetaController(
            detector=self.detector,
            high_policy=self.high_policy,
            fallback_controller=self._rule,
            fallback_to_rule=self.llm_fallback_to_rule,
            use_gold_intent=True,   # training: gold intent drives w_local
        )

    def train(self, epochs: int) -> Dict[str, HintStore]:
        logger.info(
            f"===== Two-agent H-MOD training: {epochs} epochs over "
            f"{len(self.scenarios)} scenarios | policy_hints={len(self.policy_hints.hints)} "
            f"detector_hints={len(self.detector_hints.hints)} ====="
        )
        for epoch in range(1, epochs + 1):
            controller = self._epoch_controller()
            evaluator = HMODEvaluator(
                mode="hmod_dynamic",
                judge_model=self.judge_model,
                use_llm_simulator=self.use_llm_simulator,
                audit_sample_size=0,
                buyer_policy=self.buyer_policy,
                meta_controller=controller,
                verbose=self.verbose,
                turn_limit_mult=self.turn_limit_mult,
            )
            result = evaluator.run(self.scenarios)
            metrics = result["metrics"]

            det_digest = build_detector_digest(controller.detector_records)
            logger.info(
                f"[epoch {epoch}/{epochs}] policy: {_fmt(metrics)} | "
                f"detector: intent_acc={det_digest.get('intent_accuracy')} "
                f"drift_acc={det_digest.get('drift_accuracy')} (turns={det_digest.get('n_turns')})"
            )

            # ── review high-policy (w_local) hints ──────────────────────
            try:
                pol = self.policy_distiller.review(
                    self.policy_hints.hints,
                    {"aggregate_metrics": metrics,
                     "episode_digest": build_episode_digest(result["dialogues"])},
                    epoch,
                )
                rep = self.policy_hints.review_update(pol["remove"], pol["add"], metrics)
                logger.info(f"[epoch {epoch}] high-policy hints -> {rep['n_hints']} "
                            f"(+{len(rep['added'])} / -{len(rep['dropped'])})")
            except Exception as exc:
                logger.warning(f"[epoch {epoch}] high-policy review failed ({exc})")

            # ── review intent-detection hints ───────────────────────────
            try:
                det = self.detector_distiller.review(
                    self.detector_hints.hints,
                    {"detector_digest": det_digest, "intent_definitions": SELLER_INTENT_TYPES},
                    epoch,
                )
                rep = self.detector_hints.review_update(
                    det["remove"], det["add"],
                    {"intent_accuracy": det_digest.get("intent_accuracy"),
                     "drift_accuracy": det_digest.get("drift_accuracy")},
                )
                logger.info(f"[epoch {epoch}] detector hints -> {rep['n_hints']} "
                            f"(+{len(rep['added'])} / -{len(rep['dropped'])})")
            except Exception as exc:
                logger.warning(f"[epoch {epoch}] detector review failed ({exc})")

        logger.info("===== Two-agent training done =====")
        for i, h in enumerate(self.policy_hints.hints, 1):
            logger.info(f"  [policy {i}] {h}")
        for i, h in enumerate(self.detector_hints.hints, 1):
            logger.info(f"  [detector {i}] {h}")
        return {"policy_hints": self.policy_hints, "detector_hints": self.detector_hints}
