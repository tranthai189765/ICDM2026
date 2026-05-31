"""LLM self-reflection for H-MOD dynamic buyer weights.

The runtime paper path should expose one natural-language buyer objective to
the controller. The controller then asks an LLM to infer the current local
objective and produce W_t from visible dialogue context.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from hmod.objectives import normalize_weight
from hmod.scenario import OBJECTIVE_ORDER

load_dotenv()


OBJECTIVE_DESCRIPTIONS = {
    "sl_ratio": (
        "buyer price gain. Higher means bargain harder for a lower price and "
        "protect buyer surplus."
    ),
    "fairness": (
        "fairness and relationship preservation. Higher means avoid antagonistic "
        "lowballing and keep the offer reasonable."
    ),
    "deal_rate": (
        "probability of closing a deal. Higher means prioritize agreement when "
        "the price is acceptable."
    ),
    "avg_turn": (
        "time efficiency and urgency. Higher means reduce negotiation length and "
        "move toward closure."
    ),
}


class LLMReflectionError(RuntimeError):
    pass


class LLMWeightReflector:
    """OpenAI-compatible LLM client for H-MOD W_t reflection."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: str = "DEEPINFRA_API_KEY",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ):
        self.model = (
            model
            or os.getenv("HMOD_LLM_MODEL")
            or os.getenv("DEEPINFRA_MODEL")
            or "Qwen/Qwen3-32B"
        )
        self.api_key_env = api_key_env
        self.api_key = api_key if api_key is not None else (
            os.getenv(api_key_env)
            or os.getenv("HMOD_LLM_API_KEY")
            or os.getenv("DEEPINFRA_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("API_KEY")
        )
        self.base_url = (
            base_url
            or os.getenv("HMOD_LLM_BASE_URL")
            or os.getenv("DEEPINFRA_BASE_URL")
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _client(self):
        if not self.api_key:
            raise LLMReflectionError(
                "Missing LLM API key. Set DEEPINFRA_API_KEY in .env."
            )
        try:
            import openai
        except ImportError as exc:
            raise LLMReflectionError("The openai package is required for llm_reflection mode") from exc
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return openai.OpenAI(**kwargs)

    def reflect(
        self,
        macro_goal: str,
        dialogue_history: List[Dict[str, str]],
        previous_weight: Optional[List[float]],
        turn: int,
        buyer_constraints: Dict[str, Any],
        item_context: Optional[Dict[str, Any]] = None,
        last_seller_offer: Optional[float] = None,
    ) -> Dict[str, Any]:
        messages = self._build_messages(
            macro_goal=macro_goal,
            dialogue_history=dialogue_history,
            previous_weight=previous_weight,
            turn=turn,
            buyer_constraints=buyer_constraints,
            item_context=item_context or {},
            last_seller_offer=last_seller_offer,
        )
        client = self._client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content
        parsed = parse_reflection_json(content)
        weight = coerce_reflection_weight(parsed.get("w_t"))
        return {
            "weight_vector": weight,
            "detected_seller_intent": str(parsed.get("detected_seller_intent", "unknown")),
            "local_objective": str(parsed.get("local_objective", "")),
            "adaptation_signal": str(parsed.get("adaptation_signal", "")),
            "reason": str(parsed.get("reason", "")),
            "raw_response": content,
            "model": self.model,
        }

    def _build_messages(
        self,
        macro_goal: str,
        dialogue_history: List[Dict[str, str]],
        previous_weight: Optional[List[float]],
        turn: int,
        buyer_constraints: Dict[str, Any],
        item_context: Dict[str, Any],
        last_seller_offer: Optional[float],
    ) -> List[Dict[str, str]]:
        objective_text = "\n".join(
            f"- {name}: {OBJECTIVE_DESCRIPTIONS[name]}" for name in OBJECTIVE_ORDER
        )
        dialogue_text = "\n".join(
            f"{'Buyer' if row.get('role') == 'assistant' else 'Seller'}: {row.get('content', '')}"
            for row in dialogue_history[-12:]
        )
        previous = (
            dict(zip(OBJECTIVE_ORDER, previous_weight))
            if previous_weight is not None
            else None
        )
        system = (
            "You are the H-MOD meta-controller for a negotiation buyer agent. "
            "The user gives one ambiguous natural-language buyer objective. "
            "At each reflection step, infer the current local objective from visible dialogue only, "
            "then output the MORL weight vector W_t. Never assume hidden drift labels."
        )
        user = {
            "macro_goal": macro_goal,
            "objective_order": OBJECTIVE_ORDER,
            "objective_meaning": OBJECTIVE_DESCRIPTIONS,
            "hard_rule": (
                "The buyer must not intentionally exceed max_acceptable_price when it is provided. "
                "If seller pressure conflicts with the ceiling, preserve the ceiling."
            ),
            "current_turn": turn,
            "previous_w": previous,
            "buyer_constraints": buyer_constraints,
            "last_seller_offer": last_seller_offer,
            "item_context": item_context,
            "visible_dialogue": dialogue_text,
            "required_json_schema": {
                "detected_seller_intent": (
                    "neutral | firm | final_offer | walkaway_risk | scarcity_pressure | "
                    "over_budget | cooperative | unknown"
                ),
                "local_objective": "short natural-language local objective for the next few turns",
                "adaptation_signal": "what changed compared with previous_w, if anything",
                "w_t": {
                    "sl_ratio": "non-negative number",
                    "fairness": "non-negative number",
                    "deal_rate": "non-negative number",
                    "avg_turn": "non-negative number"
                },
                "reason": "brief explanation",
            },
            "output_instruction": (
                "Return only valid JSON. W_t must contain all four keys and should sum to 1. "
                "Use higher sl_ratio for price saving, higher deal_rate/avg_turn for closing urgency, "
                "and higher fairness for relationship/fairness recovery."
            ),
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]


def parse_reflection_json(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise LLMReflectionError(f"LLM did not return JSON: {text!r}")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise LLMReflectionError("LLM reflection response must be a JSON object")
    return parsed


def coerce_reflection_weight(raw_weight: Any) -> List[float]:
    if isinstance(raw_weight, dict):
        weight = [float(raw_weight.get(name, 0.0)) for name in OBJECTIVE_ORDER]
    elif isinstance(raw_weight, list):
        weight = [float(x) for x in raw_weight]
    else:
        raise LLMReflectionError("LLM reflection JSON missing w_t")
    if len(weight) != len(OBJECTIVE_ORDER):
        raise LLMReflectionError(f"w_t must have {len(OBJECTIVE_ORDER)} values")
    return normalize_weight(weight)
