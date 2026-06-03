"""Deal judging helpers for H-MOD buyer-agent evaluation."""

import json
import re
from typing import Any, Dict, List, Optional

from config.constants import LLAMA3
from hmod.simulator import extract_prices, first_price
from utils.prompt import call_llm


ACCEPT_PATTERNS = (
    "deal",
    "i accept",
    "accept it",
    "sounds good",
    "i can buy",
    "i'll buy",
    "i will buy",
    "let's do",
)

REJECT_PATTERNS = (
    "walk away",
    "will pass",
    "not buy",
    "buy from someone else",
    "not going anywhere",
)


def _last_price_before(dialogue: List[Dict[str, str]], idx: int) -> Optional[float]:
    for turn in reversed(dialogue[: idx + 1]):
        price = first_price(turn.get("content", ""))
        if price is not None:
            return price
    return None


def rule_judge_deal(dialogue: List[Dict[str, str]]) -> Dict[str, Any]:
    """Deterministic judge used for tests and offline smoke runs."""
    deal_price = None
    evidence = ""
    for idx, turn in enumerate(dialogue):
        content = turn.get("content", "") or ""
        lower = content.lower()
        prices = extract_prices(content)
        if any(pattern in lower for pattern in ACCEPT_PATTERNS):
            deal_price = prices[0] if prices else _last_price_before(dialogue, idx)
            evidence = content
    if deal_price is not None:
        return {
            "deal": True,
            "deal_price": float(deal_price),
            "success": True,
            "evidence": evidence,
            "judge_model": "rule",
        }
    rejected = any(
        any(pattern in (turn.get("content", "") or "").lower() for pattern in REJECT_PATTERNS)
        for turn in dialogue
    )
    return {
        "deal": False,
        "deal_price": None,
        "success": False,
        "evidence": "walkaway/rejection detected" if rejected else "no acceptance detected",
        "judge_model": "rule",
    }


def _dialogue_to_text(dialogue: List[Dict[str, str]]) -> str:
    lines = []
    for turn in dialogue:
        role = "Buyer" if turn.get("role") == "assistant" else "Seller"
        lines.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(lines)


def _parse_llm_json(text: str) -> Dict[str, Any]:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


def llm_judge_deal(
    dialogue: List[Dict[str, str]],
    model_type: str = LLAMA3,
    fallback_to_rule: bool = True,
) -> Dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an evaluator for a buyer-agent negotiation. "
                "Assistant messages are from the Buyer. User messages are from the Seller. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Determine whether the Buyer and Seller reached a deal, and extract the final deal price.\n"
                f"Dialogue:\n{_dialogue_to_text(dialogue)}\n\n"
                'Return JSON: {"deal": true|false, "deal_price": number|null, '
                '"success": true|false, "evidence": "short quote or reason"}'
            ),
        },
    ]
    try:
        response = call_llm(messages, n=1, temperature=0.0, max_token=160, model_type=model_type)[0]
        parsed = _parse_llm_json(response)
        parsed["deal"] = bool(parsed.get("deal", False))
        parsed["success"] = bool(parsed.get("success", parsed["deal"]))
        parsed["deal_price"] = (
            float(parsed["deal_price"]) if parsed.get("deal_price") is not None else None
        )
        parsed["judge_model"] = model_type
        return parsed
    except Exception as exc:
        if not fallback_to_rule:
            raise
        judged = rule_judge_deal(dialogue)
        judged["evidence"] = f"{judged['evidence']} (LLM judge fallback: {exc})"
        return judged


def judge_deal(dialogue: List[Dict[str, str]], model_type: str = "rule") -> Dict[str, Any]:
    if model_type == "rule":
        return rule_judge_deal(dialogue)
    return llm_judge_deal(dialogue, model_type=model_type)
