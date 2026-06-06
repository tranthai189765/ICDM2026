"""Two-agent H-MOD meta-controller.

Combines the Intent-Drift Detector (agent 1) and the High-Policy weight
controller (agent 2) behind the same `select_local_weight` interface the runner
already uses, so it drops into HMODEvaluator unchanged.

Pipeline per dialogue:
  turn 0          : no intent, no dialogue, goal only -> high policy -> w_local.
  later turns     : detector watches the dialogue. While no drift, the low policy
                    keeps acting under the carried w_local. On detected drift, the
                    high policy regenerates w_local for the new intent.

Training vs inference:
  use_gold_intent=True  (training)  -> the GOLD seller intent (from the simulator
                                       trace) drives w_local regeneration, while
                                       the detector still predicts and is scored.
  use_gold_intent=False (inference) -> the detector's prediction drives w_local.
"""

from typing import Any, Dict, List, Optional

from hmod.high_policy import LLMHighPolicy
from hmod.intent_detector import LLMIntentDetector
from hmod.objectives import normalize_weight
from hmod.scenario import HMODScenario


class TwoAgentMetaController:
    name = "two_agent"

    def __init__(
        self,
        detector: LLMIntentDetector,
        high_policy: LLMHighPolicy,
        fallback_controller=None,
        fallback_to_rule: bool = False,
        use_gold_intent: bool = False,
        max_carry: Optional[int] = None,
    ):
        self.detector = detector
        self.high_policy = high_policy
        self.fallback_controller = fallback_controller
        self.fallback_to_rule = fallback_to_rule
        self.use_gold_intent = use_gold_intent
        self.max_carry = max_carry  # optional forced refresh after N carried turns

        # per-dialogue state
        self._scenario_id: Optional[str] = None
        self._w: Optional[List[float]] = None
        self._intent: str = "unknown"
        self._gold_prev: str = "neutral"
        self._carried: int = 0
        # cross-episode detector scoring (read + cleared by the trainer)
        self.detector_records: List[Dict[str, Any]] = []

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _gold_intent_now(simulator_trace: Dict[str, Any], scenario: HMODScenario) -> str:
        seq = simulator_trace.get("intent_state_by_turn", []) if simulator_trace else []
        if seq:
            return seq[-1].get("intent_state", scenario.initial_intent)
        return scenario.initial_intent

    def _reset_for(self, scenario: HMODScenario) -> None:
        self._scenario_id = scenario.id
        self._w = None
        self._intent = "unknown"
        self._gold_prev = scenario.initial_intent
        self._carried = 0

    def _high_policy_w(
        self,
        scenario: HMODScenario,
        dialogue_history: Optional[List[Dict[str, str]]],
        turn: int,
    ) -> Dict[str, Any]:
        from hmod.simulator import first_price

        last_offer = None
        for row in reversed(dialogue_history or []):
            if row.get("role") == "user":
                p = first_price(row.get("content", ""))
                if p is not None:
                    last_offer = p
                    break
        return self.high_policy.generate(
            macro_goal=scenario.macro_goal,
            dialogue_history=dialogue_history or [],
            current_intent=self._intent,
            previous_weight=self._w,
            turn=turn,
            buyer_constraints={
                "max_acceptable_price": scenario.max_acceptable_price(),
                "target_price": scenario.target_price(),
                "turn_limit": scenario.turn_limit,
            },
            item_context=scenario.case,
            last_seller_offer=last_offer,
        )

    def _emit(self, reflection_step: bool, allocation: str = "", reason: str = "",
              detector_out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "weight_vector": list(self._w),
            "w_t": list(self._w),
            "intent_state": self._intent,
            "decision_summary": (
                f"[two-agent] intent={self._intent} "
                + ("REGENERATED w_local: " + allocation if reflection_step else "carry w_local")
            ),
            "selected_objective_id": None,
            "objective_cluster": None,
            "objective_source": "two_agent_intent_then_weight",
            "objective_match_reason": reason,
            "base_weight": list(self._w),
            "reflection_step": reflection_step,
            "reflection_horizon": None,
            "controller_mode": "two_agent",
            "llm_reflection": {
                "allocation": allocation,
                "reason": reason,
                "detector": detector_out,
                "intent": self._intent,
            },
            "llm_error": None,
            "mode": "hmod_dynamic",
        }

    # ── main entry (runner interface) ────────────────────────────────────
    def select_local_weight(
        self,
        scenario: HMODScenario,
        simulator_trace: Dict[str, Any],
        previous_weight: Optional[List[float]],
        mode: str,
        turn: int = 0,
        dialogue_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        if turn == 0 or self._scenario_id != scenario.id:
            self._reset_for(scenario)

        try:
            # ── turn 0: goal only, no intent, no dialogue ────────────────
            if turn == 0:
                self._intent = "unknown"
                hp = self._high_policy_w(scenario, [], 0)
                self._w = hp["weight_vector"]
                self._carried = 0
                return self._emit(True, hp["allocation_text"], hp["reason"])

            gold_now = self._gold_intent_now(simulator_trace, scenario)

            # detector always predicts (drives inference; scored in training)
            det = self.detector.detect(dialogue_history or [], turn, self._intent)

            if self.use_gold_intent:
                # training: gold drives w_local; record detector correctness
                drift = (gold_now != self._gold_prev)
                self.detector_records.append({
                    "scenario_id": scenario.id, "turn": turn,
                    "gold_intent": gold_now, "gold_drift": drift,
                    "pred_intent": det["current_intent"], "pred_drift": det["drift_detected"],
                    "intent_correct": det["current_intent"] == gold_now,
                    "drift_correct": det["drift_detected"] == drift,
                })
                new_intent = gold_now
            else:
                # inference: detector drives w_local
                drift = det["drift_detected"]
                new_intent = det["current_intent"]

            self._gold_prev = gold_now
            self._intent = new_intent  # display + drive (gold or detector)

            force = self.max_carry is not None and self._carried >= self.max_carry
            if drift or force:
                hp = self._high_policy_w(scenario, dialogue_history, turn)
                self._w = hp["weight_vector"]
                self._carried = 0
                return self._emit(True, hp["allocation_text"], hp["reason"], det)

            # no drift -> keep acting under the carried w_local
            self._carried += 1
            return self._emit(False, "", det.get("reason", ""), det)

        except Exception as exc:
            return self._fallback(scenario, simulator_trace, previous_weight, mode,
                                  turn, dialogue_history, exc)

    # ── robustness ───────────────────────────────────────────────────────
    def _fallback(self, scenario, simulator_trace, previous_weight, mode, turn,
                  dialogue_history, error) -> Dict[str, Any]:
        if self.fallback_to_rule and self.fallback_controller is not None:
            selected = self.fallback_controller.select_local_weight(
                scenario=scenario, simulator_trace=simulator_trace,
                previous_weight=previous_weight, mode=mode, turn=turn,
                dialogue_history=dialogue_history,
            )
            selected["controller_mode"] = "two_agent_fallback_rule"
            selected["llm_error"] = str(error)
            self._w = list(selected["weight_vector"])
            return selected
        # last-resort: carry or uniform
        if self._w is None:
            self._w = normalize_weight([1.0, 1.0, 1.0])
        out = self._emit(False, "", f"error={error}")
        out["controller_mode"] = "two_agent_error_carry"
        out["llm_error"] = str(error)
        return out
