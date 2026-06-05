"""High-Policy LLM (agent 2 of the two-agent H-MOD controller).

Given the macro goal, the visible dialogue and the CURRENT seller intent (from
the detector at inference, or the gold label during training), this agent emits
the local weight w_local for the low policy. It is forced to reason in natural
language first — an explicit percentage allocation across the three objectives —
and only then commit to the numeric vector, which is parsed back out.

Output format the LLM must follow:
    "In the short term, focus X% on sl_ratio (price saving), Y% on fairness,
     Z% on deal_rate"  ->  w_t = [X/100, Y/100, Z/100]
"""

import json
import re
from typing import Any, Dict, List, Optional

from hmod.llm_reflection import LLMWeightReflector, parse_reflection_json
from hmod.objectives import normalize_weight
from hmod.scenario import OBJECTIVE_ORDER

OBJECTIVE_MEANING = {
    "sl_ratio": "price saving — bargain harder for a lower price (buyer surplus)",
    "fairness": "fairness / relationship — keep offers reasonable, avoid lowballing",
    "deal_rate": "closing — prioritise sealing the deal when the price is acceptable",
}

# Recommended direction to move w_local for each seller intent. This is general
# domain knowledge (intent semantics), NOT the per-scenario gold label, so it can
# legitimately ground the LLM at inference. It tells the high policy HOW w_local
# should shift when the seller's intent changes; T2DA then checks it actually did.
INTENT_ADAPTATION_GUIDE = {
    "unknown": "No seller signal yet (turn 0). Start balanced, leaning on sl_ratio "
               "for price saving while there is room to bargain.",
    "neutral": "Seller is flexible. Emphasise sl_ratio (price saving) to push the "
               "price down; keep deal_rate modest.",
    "firm": "Seller has hardened. LOWER sl_ratio and RAISE deal_rate; keep fairness "
            "moderate to de-escalate. Stop aggressive lowballing.",
    "final_offer": "Take-it-or-leave-it. If the offer is within the ceiling, sharply "
                   "RAISE deal_rate and LOWER sl_ratio to close now; if above the "
                   "ceiling, hold deal_rate low and prepare to walk.",
    "walkaway_risk": "Seller may leave. RAISE deal_rate (and fairness) and LOWER "
                     "sl_ratio to secure the deal before they walk.",
}


def parse_allocation_to_weight(parsed: Dict[str, Any]) -> List[float]:
    """Extract w_t from the LLM JSON: prefer the explicit w_t dict/list, else
    fall back to parsing percentages out of the natural-language allocation."""
    raw = parsed.get("w_t")
    if isinstance(raw, dict):
        weight = [float(raw.get(name, 0.0)) for name in OBJECTIVE_ORDER]
        return normalize_weight(weight)
    if isinstance(raw, list) and raw:
        return normalize_weight([float(x) for x in raw])

    # Fallback: read "X% on sl_ratio" style tokens from the allocation text.
    text = str(parsed.get("allocation", ""))
    weight = [0.0, 0.0, 0.0]
    found = False
    for i, name in enumerate(OBJECTIVE_ORDER):
        # bind each objective to its NEAREST preceding percentage (do not cross
        # another '%', otherwise every name would match the first percentage).
        m = re.search(rf"(\d+(?:\.\d+)?)\s*%[^%]*?{re.escape(name)}", text, re.IGNORECASE)
        if m:
            weight[i] = float(m.group(1)) / 100.0
            found = True
    if not found:
        raise ValueError("high policy returned neither w_t nor a parseable allocation")
    return normalize_weight(weight)


class LLMHighPolicy:
    """Agent 2: produce w_local from (goal, dialogue, current intent)."""

    def __init__(self, reflector: LLMWeightReflector, hint_provider=None):
        self.reflector = reflector
        self.hint_provider = hint_provider

    def generate(
        self,
        macro_goal: str,
        dialogue_history: List[Dict[str, str]],
        current_intent: Optional[str],
        previous_weight: Optional[List[float]],
        turn: int,
        buyer_constraints: Dict[str, Any],
        item_context: Optional[Dict[str, Any]] = None,
        last_seller_offer: Optional[float] = None,
    ) -> Dict[str, Any]:
        hints = None
        if self.hint_provider is not None:
            try:
                hints = self.hint_provider("high_policy", None)
            except Exception:
                hints = None
        messages = self._build_messages(
            macro_goal, dialogue_history, current_intent, previous_weight,
            turn, buyer_constraints, item_context or {}, last_seller_offer, hints,
        )
        content = self.reflector._complete(messages)
        parsed = parse_reflection_json(content)
        weight = parse_allocation_to_weight(parsed)
        return {
            "weight_vector": weight,
            "allocation_text": str(parsed.get("allocation", "")),
            "reason": str(parsed.get("reason", "")),
            "current_intent": current_intent,
            "raw_response": content,
        }

    def _build_messages(
        self,
        macro_goal: str,
        dialogue_history: List[Dict[str, str]],
        current_intent: Optional[str],
        previous_weight: Optional[List[float]],
        turn: int,
        buyer_constraints: Dict[str, Any],
        item_context: Dict[str, Any],
        last_seller_offer: Optional[float],
        hints: Optional[str],
    ) -> List[Dict[str, str]]:
        dialogue_text = "\n".join(
            f"{'Buyer' if r.get('role') == 'assistant' else 'Seller'}: {r.get('content', '')}"
            for r in dialogue_history[-12:]
        ) or "(no dialogue yet — this is turn 0)"
        previous = (
            dict(zip(OBJECTIVE_ORDER, [round(float(x), 2) for x in previous_weight]))
            if previous_weight is not None else None
        )
        system = (
            "You are the High-Policy weight controller for a BUYER agent. Given the "
            "buyer's macro goal, the visible dialogue and the CURRENT seller intent, set "
            "the short-term weight w_local over three objectives for the low policy. "
            "Follow the weight_adaptation_guideline for the current seller intent to "
            "decide WHICH WAY to shift w_local (e.g. when the seller becomes firm or "
            "issues a final offer, lower sl_ratio and raise deal_rate). FIRST reason in "
            "natural language as an explicit percentage allocation, THEN give the matching "
            "numeric vector. The buyer must never plan to exceed the price ceiling."
        )
        user = {
            "macro_goal": macro_goal,
            "objectives_in_order": OBJECTIVE_ORDER,
            "objective_meaning": OBJECTIVE_MEANING,
            "current_seller_intent": current_intent or "unknown",
            "weight_adaptation_guideline": INTENT_ADAPTATION_GUIDE.get(
                current_intent or "unknown", INTENT_ADAPTATION_GUIDE["neutral"]),
            "previous_w_local": previous,
            "learned_weight_hints": hints or "none yet",
            "buyer_constraints": buyer_constraints,
            "last_seller_offer": last_seller_offer,
            "item_context": item_context,
            "current_turn": turn,
            "visible_dialogue": dialogue_text,
            "required_json_schema": {
                "allocation": (
                    "In the short term, focus X% on sl_ratio (price saving), Y% on "
                    "fairness, Z% on deal_rate"
                ),
                "w_t": {"sl_ratio": "X/100", "fairness": "Y/100", "deal_rate": "Z/100"},
                "reason": "brief justification tied to the current seller intent",
            },
            "output_instruction": (
                "Return ONLY valid JSON. 'allocation' MUST use the 'focus X% on ...' "
                "phrasing and the three percentages MUST sum to 100; 'w_t' MUST be the "
                "same numbers as fractions summing to 1."
            ),
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]
