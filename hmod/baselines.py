"""Baseline controllers for H-MOD ablations.

StaticHighPolicyController: the high-policy LLM (with its loaded hint playbook)
produces w_local ONCE at turn 0 from the macro goal alone, then the low policy
runs under that fixed w_local for the whole episode. No intent detection, no
drift-triggered re-generation. This isolates the value of dynamic adaptation:
compared with the full two-agent controller (same high policy + hints), any GSR
/ T2DA gap is attributable to adapting w_local on drift rather than holding it.
"""

from typing import Any, Dict, List, Optional

from hmod.high_policy import LLMHighPolicy
from hmod.objectives import normalize_weight
from hmod.scenario import HMODScenario


class StaticHighPolicyController:
    name = "static_high_policy"

    def __init__(
        self,
        high_policy: LLMHighPolicy,
        fallback_controller=None,
        fallback_to_rule: bool = False,
    ):
        self.high_policy = high_policy
        self.fallback_controller = fallback_controller
        self.fallback_to_rule = fallback_to_rule
        self._scenario_id: Optional[str] = None
        self._w: Optional[List[float]] = None
        self._alloc: str = ""
        self._reason: str = ""

    def _emit(self, reflection_step: bool) -> Dict[str, Any]:
        return {
            "weight_vector": list(self._w),
            "w_t": list(self._w),
            "intent_state": "static",
            "decision_summary": (
                "[static-high-policy] " + (
                    "w_local set once at turn 0: " + self._alloc
                    if reflection_step else "carry initial w_local")
            ),
            "selected_objective_id": None,
            "objective_cluster": None,
            "objective_source": "static_high_policy_once",
            "objective_match_reason": self._reason,
            "base_weight": list(self._w),
            "reflection_step": reflection_step,
            "reflection_horizon": None,
            "controller_mode": "static_high_policy",
            "llm_reflection": {"allocation": self._alloc, "reason": self._reason},
            "llm_error": None,
            "mode": "hmod_dynamic",
        }

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
            self._scenario_id = scenario.id
            self._w = None

        if self._w is None:
            try:
                hp = self.high_policy.generate(
                    macro_goal=scenario.macro_goal,
                    dialogue_history=[],          # goal only, no dialogue yet
                    current_intent="unknown",
                    previous_weight=None,
                    turn=0,
                    buyer_constraints={
                        "max_acceptable_price": scenario.max_acceptable_price(),
                        "target_price": scenario.target_price(),
                        "turn_limit": scenario.turn_limit,
                    },
                    item_context=scenario.case,
                    last_seller_offer=None,
                )
                self._w = hp["weight_vector"]
                self._alloc = hp["allocation_text"]
                self._reason = hp["reason"]
            except Exception as exc:
                if self.fallback_to_rule and self.fallback_controller is not None:
                    selected = self.fallback_controller.select_local_weight(
                        scenario=scenario, simulator_trace=simulator_trace,
                        previous_weight=previous_weight, mode=mode, turn=turn,
                        dialogue_history=dialogue_history,
                    )
                    self._w = list(selected["weight_vector"])
                    selected["controller_mode"] = "static_high_policy_fallback_rule"
                    selected["llm_error"] = str(exc)
                    return selected
                self._w = normalize_weight([1.0, 1.0, 1.0])
                self._alloc = ""
                self._reason = f"error={exc}"
            return self._emit(reflection_step=True)

        # all later turns: carry the turn-0 weight, no re-generation
        return self._emit(reflection_step=False)
