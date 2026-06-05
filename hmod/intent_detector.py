"""Intent-drift Detection LLM (agent 1 of the two-agent H-MOD controller).

This agent watches the dialogue and decides, each step, whether the seller's
intent has drifted and what the current intent is. It is given the catalog of
possible seller intents plus a few few-shot examples per intent grounded in the
training scenarios, and it accumulates its own playbook of detection hints.

Input  : visible dialogue + current turn + previously-believed intent.
Output : {drift_detected, current_intent, reason}.
"""

import json
from typing import Any, Dict, List, Optional

from hmod.llm_reflection import LLMWeightReflector, parse_reflection_json
from hmod.scenario import load_scenarios


# ── Seller intent catalog (matches DynamicSellerNegotiationSimulator) ────────
SELLER_INTENT_TYPES = {
    "neutral": "Bargaining normally; willing to come down gradually toward a fair midpoint.",
    "firm": "Hardened after long back-and-forth; will only consider offers near its own counter.",
    "final_offer": "Issued a take-it-or-leave-it final price; will pass if the buyer cannot meet it.",
    "walkaway_risk": "Frustrated and threatening to sell to someone else if pushed further.",
}

# Canonical seller utterances the deterministic simulator emits per intent.
CANONICAL_UTTERANCES = {
    "neutral": "I can come down a bit, but I would need about $X.",
    "firm": "We have gone back and forth for too long. I can only consider offers close to $X.",
    "final_offer": "My final price is $X. If you cannot do that, I will pass.",
    "walkaway_risk": "I am getting frustrated. I can do $X, otherwise I will sell elsewhere.",
}

# Which intent each drift_mode eventually drifts the seller into.
DRIFT_TARGET_INTENT = {
    "static_no_drift": "neutral",
    "gradual_firming": "firm",
    "abrupt_final_offer": "final_offer",
    "frustrated_walkaway": "walkaway_risk",
}


def build_intent_fewshot(scenario_file: str, max_per_intent: int = 2) -> List[Dict[str, Any]]:
    """Build few-shot detection examples per intent, grounded in train data.

    Each example pairs a real seller persona (from the training scenarios) and
    the canonical utterance for the intent that persona drifts into, with the
    gold detection label.
    """
    try:
        scenarios = load_scenarios(scenario_file)
    except Exception:
        scenarios = []

    examples: List[Dict[str, Any]] = []
    # one no-drift example (stay neutral)
    examples.append({
        "previous_intent": "neutral",
        "seller_says": CANONICAL_UTTERANCES["neutral"],
        "persona": "a seller still bargaining normally",
        "label": {"drift_detected": False, "current_intent": "neutral"},
    })

    per_intent: Dict[str, int] = {k: 0 for k in SELLER_INTENT_TYPES}
    for sc in scenarios:
        target = DRIFT_TARGET_INTENT.get(sc.drift_mode)
        if not target or target == "neutral":
            continue
        if per_intent.get(target, 0) >= max_per_intent:
            continue
        per_intent[target] += 1
        examples.append({
            "previous_intent": "neutral",
            "seller_says": CANONICAL_UTTERANCES[target],
            "persona": sc.seller_persona.description,
            "label": {"drift_detected": True, "current_intent": target},
        })
        if all(per_intent[k] >= max_per_intent for k in CANONICAL_UTTERANCES if k != "neutral"):
            break
    return examples


class IntentDetectorError(RuntimeError):
    pass


class LLMIntentDetector:
    """Agent 1: detect seller intent drift from the visible dialogue."""

    def __init__(
        self,
        reflector: LLMWeightReflector,
        fewshot: Optional[List[Dict[str, Any]]] = None,
        hint_provider=None,
    ):
        # Reuse the shared LLM backend (FPT/DeepInfra routing via _complete).
        self.reflector = reflector
        self.fewshot = fewshot or []
        self.hint_provider = hint_provider

    def detect(
        self,
        dialogue_history: List[Dict[str, str]],
        turn: int,
        previous_intent: Optional[str],
    ) -> Dict[str, Any]:
        hints = None
        if self.hint_provider is not None:
            try:
                hints = self.hint_provider("intent_detection", None)
            except Exception:
                hints = None
        messages = self._build_messages(dialogue_history, turn, previous_intent, hints)
        content = self.reflector._complete(messages)
        parsed = parse_reflection_json(content)
        current = str(parsed.get("current_intent", previous_intent or "neutral")).strip()
        if current not in SELLER_INTENT_TYPES:
            current = previous_intent or "neutral"
        drift = bool(parsed.get("drift_detected", current != (previous_intent or "neutral")))
        return {
            "drift_detected": drift,
            "current_intent": current,
            "reason": str(parsed.get("reason", "")),
            "raw_response": content,
        }

    def _build_messages(
        self,
        dialogue_history: List[Dict[str, str]],
        turn: int,
        previous_intent: Optional[str],
        hints: Optional[str],
    ) -> List[Dict[str, str]]:
        dialogue_text = "\n".join(
            f"{'Buyer' if r.get('role') == 'assistant' else 'Seller'}: {r.get('content', '')}"
            for r in dialogue_history[-12:]
        )
        system = (
            "You are the Intent-Drift Detector for a negotiation. The opponent is the "
            "SELLER. From the visible dialogue, the current turn and the previously "
            "believed seller intent, decide whether the seller's intent has DRIFTED and "
            "what the seller's CURRENT intent is. Judge only from visible evidence; do "
            "not assume hidden labels."
        )
        user = {
            "possible_seller_intents": SELLER_INTENT_TYPES,
            "few_shot_examples": self.fewshot,
            "learned_detection_hints": hints or "none yet",
            "current_turn": turn,
            "previously_believed_intent": previous_intent or "unknown",
            "visible_dialogue": dialogue_text,
            "required_json_schema": {
                "drift_detected": "true if the seller intent changed from the previous one",
                "current_intent": "one of: neutral | firm | final_offer | walkaway_risk",
                "reason": "brief evidence from the dialogue",
            },
            "output_instruction": "Return ONLY valid JSON with those three keys.",
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]
