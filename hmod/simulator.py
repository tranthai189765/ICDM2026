"""Seller-mode negotiation simulator with deterministic intent drift."""

import copy
import re
from typing import Any, Dict, List, Optional

from base.simulator import Simulator
from config.constants import LLAMA3
from hmod.scenario import HMODScenario
from utils.prompt import call_llm


PRICE_PATTERN = re.compile(r"\$?\s*([-+]?\d[\d,]*\.?\d*)")


def extract_prices(text: str) -> List[float]:
    prices = []
    for match in PRICE_PATTERN.findall(text or ""):
        try:
            prices.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return prices


def first_price(text: str) -> Optional[float]:
    prices = extract_prices(text)
    return prices[0] if prices else None


class DynamicSellerNegotiationSimulator(Simulator):
    """A seller simulator that follows a persona and scripted drift rules.

    H-MOD trains/evaluates the assistant as Buyer. User messages come from this
    Seller simulator, matching the legacy PADPP/DMORL negotiation role setup.
    """

    def __init__(
        self,
        scenario: HMODScenario,
        use_llm_response: bool = False,
        model_type: str = LLAMA3,
    ):
        self.scenario = scenario
        self.use_llm_response = use_llm_response
        self.model_type = model_type
        self.use_persona = True
        self.user_profile_description = scenario.seller_persona.description
        self.reset_trace()

    def reset_trace(self) -> None:
        self.turn_count = 0
        self.round_without_deal = 0
        self.frustration_level = 0
        self.low_offer_streak = 0
        self.rejection_count = 0
        self.intent_state = self.scenario.initial_intent
        self.t_drift = None
        self.drift_trigger_reason = None
        self.turns_since_drift = 0
        self.trace = {
            "scenario_id": self.scenario.id,
            "persona_type": self.scenario.seller_persona.type,
            "drift_mode": self.scenario.drift_mode,
            "intent_state_by_turn": [],
            "seller_offer_by_turn": [],
            "buyer_offer_by_turn": [],
            "frustration_by_turn": [],
            "round_without_deal_by_turn": [],
            "t_drift": None,
            "drift_trigger_reason": None,
        }

    def get_trace(self) -> Dict[str, Any]:
        trace = copy.deepcopy(self.trace)
        trace["t_drift"] = self.t_drift
        trace["drift_trigger_reason"] = self.drift_trigger_reason
        return trace

    def _price_from_ratio(self, ratio: float) -> float:
        buyer_price = float(self.scenario.case["buyer_price"])
        seller_price = float(self.scenario.case["seller_price"])
        return round(buyer_price + ratio * (seller_price - buyer_price), 2)

    def _buyer_offer_from_state(self, state: Dict[str, Any]) -> Optional[float]:
        if state.get("last_buyer_price") is not None:
            return float(state["last_buyer_price"])
        pred_goal = state.get("pred_goal")
        if isinstance(pred_goal, dict) and pred_goal.get("price") is not None:
            return float(pred_goal["price"])
        for turn in reversed(state.get("dialogue_context", [])):
            if turn.get("role") == "assistant":
                price = first_price(turn.get("content", ""))
                if price is not None:
                    return price
        return None

    def _current_accept_price(self) -> float:
        persona = self.scenario.seller_persona
        if self.intent_state == "final_offer":
            return self._price_from_ratio(persona.final_ask_ratio)
        if self.intent_state in {"firm", "walkaway_risk", "frustrated"}:
            return self._price_from_ratio(persona.post_drift_ask_ratio)
        return self._price_from_ratio(persona.neutral_accept_ratio)

    def _current_counter_offer(self) -> float:
        persona = self.scenario.seller_persona
        if self.intent_state == "final_offer":
            return self._price_from_ratio(persona.final_ask_ratio)
        if self.intent_state in {"firm", "walkaway_risk", "frustrated"}:
            return self._price_from_ratio(persona.post_drift_ask_ratio)
        concession = min(self.round_without_deal, 3) * 0.07
        ratio = max(persona.neutral_accept_ratio, persona.initial_ask_ratio - concession)
        return self._price_from_ratio(ratio)

    def _set_drift(self, turn_id: int, new_state: str, reason: str) -> None:
        if self.t_drift is not None:
            return
        self.t_drift = turn_id
        self.intent_state = new_state
        self.drift_trigger_reason = reason
        self.trace["t_drift"] = turn_id
        self.trace["drift_trigger_reason"] = reason

    def _update_internal_state(self, state: Dict[str, Any], buyer_offer: Optional[float]) -> None:
        turn_id = int(state.get("turn_id", self.turn_count))
        trigger = self.scenario.drift_trigger
        fair_target = self._price_from_ratio(self.scenario.seller_persona.neutral_accept_ratio)
        strategy = state.get("pred_goal", {})
        if isinstance(strategy, dict):
            strategy = strategy.get("strategy", "")
        elif isinstance(strategy, tuple):
            strategy = strategy[0]

        self.round_without_deal += 1
        if buyer_offer is not None and buyer_offer < fair_target * trigger.low_offer_multiplier:
            self.low_offer_streak += 1
            self.frustration_level += 1
        else:
            self.low_offer_streak = 0

        if strategy in {"reject", "counter", "final_offer", "walk_away"}:
            self.rejection_count += 1
            if buyer_offer is None or buyer_offer < fair_target:
                self.frustration_level += 1

        mode = self.scenario.drift_mode
        if mode == "static_no_drift":
            return
        if mode == "gradual_firming":
            if self.round_without_deal >= trigger.round_without_deal:
                self._set_drift(turn_id, "firm", "round_without_deal threshold reached")
            elif self.low_offer_streak >= trigger.low_offer_streak:
                self._set_drift(turn_id, "firm", "buyer kept offers below seller tolerance")
        elif mode == "abrupt_final_offer":
            if turn_id >= trigger.turn_id or self.rejection_count >= 2:
                self._set_drift(turn_id, "final_offer", "configured final-offer trigger")
        elif mode == "frustrated_walkaway":
            if self.frustration_level >= trigger.frustration_threshold:
                self._set_drift(turn_id, "walkaway_risk", "frustration threshold reached")

        if self.t_drift is not None:
            self.turns_since_drift = max(0, turn_id - self.t_drift)

    def _deterministic_response(self, buyer_offer: Optional[float]) -> str:
        accept_price = self._current_accept_price()
        counter = self._current_counter_offer()
        mode = self.scenario.drift_mode
        trigger = self.scenario.drift_trigger

        if buyer_offer is not None and buyer_offer >= accept_price:
            self.round_without_deal = 0
            self.trace["seller_offer_by_turn"].append(buyer_offer)
            return f"Deal, I can sell it for ${buyer_offer:.0f}."

        if (
            mode == "frustrated_walkaway"
            and self.intent_state == "walkaway_risk"
            and self.turns_since_drift >= trigger.walkaway_after_turns
        ):
            self.trace["seller_offer_by_turn"].append(None)
            return "I do not think this is going anywhere. I will sell to someone else."

        self.trace["seller_offer_by_turn"].append(counter)
        if self.intent_state == "firm":
            return (
                f"We have gone back and forth for too long. "
                f"I can only consider offers close to ${counter:.0f}."
            )
        if self.intent_state == "final_offer":
            return f"My final price is ${counter:.0f}. If you cannot do that, I will pass."
        if self.intent_state == "walkaway_risk":
            return f"I am getting frustrated. I can do ${counter:.0f}, otherwise I will sell elsewhere."
        return f"I can come down a bit, but I would need about ${counter:.0f}."

    def _llm_response(self, buyer_offer: Optional[float], deterministic_response: str) -> str:
        prompt = (
            "You are the Seller in a price bargaining game.\n"
            f"Seller persona: {self.user_profile_description}\n"
            f"Current seller intent: {self.intent_state}\n"
            f"Buyer latest offer: {buyer_offer}\n"
            f"Private instruction: express this intent in one short sentence: {deterministic_response}\n"
            "Return only the seller utterance."
        )
        try:
            response = call_llm(
                [{"role": "system", "content": prompt}],
                n=1,
                temperature=0.2,
                max_token=self.max_gen_token,
                model_type=self.model_type,
            )
            return response[0]
        except Exception:
            return deterministic_response

    def respond(self, state: Dict[str, Any]) -> str:
        buyer_offer = self._buyer_offer_from_state(state)
        self.trace["buyer_offer_by_turn"].append(buyer_offer)
        self._update_internal_state(state, buyer_offer)

        deterministic_response = self._deterministic_response(buyer_offer)
        response = (
            self._llm_response(buyer_offer, deterministic_response)
            if self.use_llm_response
            else deterministic_response
        )

        turn_id = int(state.get("turn_id", self.turn_count))
        self.trace["intent_state_by_turn"].append(
            {"turn": turn_id, "intent_state": self.intent_state}
        )
        self.trace["frustration_by_turn"].append(
            {"turn": turn_id, "frustration_level": self.frustration_level}
        )
        self.trace["round_without_deal_by_turn"].append(
            {"turn": turn_id, "round_without_deal": self.round_without_deal}
        )
        self.turn_count += 1
        return response


# Backward-compatible alias for older imports.
DynamicBuyerNegotiationSimulator = DynamicSellerNegotiationSimulator
