"""Deal judging helpers for H-MOD buyer-agent evaluation."""

import json
import re
from typing import Any, Dict, List, Optional

from config.constants import LLAMA3
from hmod.simulator import extract_prices, first_price
from utils.prompt import call_llm


# Phrases that should NEVER count as an accept even though they contain a
# matching substring (e.g. "good deal", "no deal"). Stripped from the
# seller utterance before matching ACCEPT_REGEXES.
ACCEPT_NEGATORS = (
    "good deal",
    "great deal",
    "best deal",
    "fair deal",
    "better deal",
    "no deal",
    "not a deal",
    "deal-breaker",
    "deal breaker",
)

# Year-like tokens that should not be treated as prices.
_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

# Verb / monetary contexts that indicate a number is a real committed price.
_PRICE_CONTEXT_REGEX = re.compile(
    r"(?:offer(?:s|ing|ed)?|pay(?:s|ing|ed)?|paid|accept(?:s|ing|ed)?|settle(?:s|d)?\s+(?:on|at|for)|"
    r"meet(?:\s+(?:you|me))?\s+(?:at|in\s+the\s+middle\s+at)|do|come\s+down\s+to|"
    r"go\s+(?:as\s+low\s+as|down\s+to|to)|sell(?:\s+(?:it|them))?\s+(?:for|at|to\s+you\s+for)|"
    r"sold\s+(?:for|at)|buy(?:\s+(?:it|them))?\s+for|bought\s+for|deal\s+at|"
    r"price\s+(?:is|of)|propose(?:s|d)?|counter(?:offer|s|ed)?\s+(?:of|at)?|"
    r"yours\s+for|for\s+a\s+total\s+of)"
    r"\s*\$?\s*([-+]?\d[\d,]*\.?\d*)",
    flags=re.IGNORECASE,
)
_DOLLAR_PRICE_REGEX = re.compile(r"\$\s*([-+]?\d[\d,]*\.?\d*)")
_SUFFIX_PRICE_REGEX = re.compile(
    r"([-+]?\d[\d,]*\.?\d*)\s*(?:dollars?|usd|bucks)\b",
    flags=re.IGNORECASE,
)

# Whole-word, seller-side accept patterns.
ACCEPT_REGEXES = (
    r"\bwe(?:'ve| have) got a deal\b",
    r"\byou(?:'ve| have) got a deal\b",
    r"\bwe have a deal\b",
    r"\bit(?:'s| is) a deal\b",
    r"\bdeal[,.!]",
    r"^\s*deal\b",
    r"\bdeal\s*$",
    r"\bi (?:can )?accept\b",
    r"\bi(?:'ll| will) accept\b",
    r"\baccepted\b",
    r"\bsold(?:\s+to\s+you)?\b",
    r"\b(?:it(?:'s| is)|the\s+\w+\s+is)\s+yours\b",
    r"\byou(?:'ve| have) bought\b",
    r"\bi(?:'ll| will) sell (?:it|them) (?:to you )?for\b",
)

# Seller-side reject patterns. If the LAST seller utterance contains any of
# these, the dialogue is treated as no-deal regardless of earlier accepts.
REJECT_REGEXES = (
    r"\bwalk away\b",
    r"\bwill pass\b",
    r"\bnot buy\b",
    r"\bbuy from someone else\b",
    r"\bnot going anywhere\b",
    r"\btoo low\b",
    r"\bwon(?:'t| not)\b",
    r"\bcan(?:'t| not) (?:accept|do|go|sell)\b",
    r"\bnot acceptable\b",
    r"\bunable to accept\b",
    r"\b(?:i'll|i will|have to|i have to) (?:decline|pass)\b",
    r"\bdecline (?:your )?offer\b",
    r"\bthe lowest i(?:'?ll| can| will) (?:go|do|accept)\b",
    r"\bonly come down to\b",
    r"\bmy (?:absolute )?minimum\b",
    r"\bnot able to (?:do|accept|go|sell)\b",
    r"\bsorry,? i(?:'m| am) not (?:willing|able)\b",
)


def _normalise_role(turn: Dict[str, str]) -> str:
    role = str(turn.get("role", "")).lower()
    # H-MOD: assistant = Buyer, user/seller = Seller.
    if role in {"assistant", "buyer", "system"}:
        return "buyer"
    return "seller"


def _seller_turns(dialogue: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [t for t in dialogue if _normalise_role(t) == "seller"]


def _has_accept(seller_text: str) -> bool:
    lower = seller_text.lower()
    stripped = lower
    for neg in ACCEPT_NEGATORS:
        stripped = stripped.replace(neg, "")
    return any(re.search(pat, stripped) for pat in ACCEPT_REGEXES)


def _has_reject(seller_text: str) -> bool:
    lower = seller_text.lower()
    return any(re.search(pat, lower) for pat in REJECT_REGEXES)


def _extract_committed_prices(text: str) -> List[float]:
    """Strict price extraction used by the rule judge.

    Only returns numbers that are either preceded by ``$``, followed by a
    money suffix (``dollars``, ``USD``, ``bucks``), or appear in a verb
    context that commits a price. Year-like tokens are stripped first.
    """
    if not text:
        return []
    cleaned = _YEAR_PATTERN.sub(" ", text)
    candidates: List[str] = []
    for match in _DOLLAR_PRICE_REGEX.finditer(cleaned):
        candidates.append(match.group(1))
    for match in _PRICE_CONTEXT_REGEX.finditer(cleaned):
        candidates.append(match.group(1))
    for match in _SUFFIX_PRICE_REGEX.finditer(cleaned):
        candidates.append(match.group(1))
    prices: List[float] = []
    seen: set = set()
    for token in candidates:
        norm = token.replace(",", "")
        try:
            value = float(norm)
        except ValueError:
            continue
        if (norm, value) in seen:
            continue
        seen.add((norm, value))
        prices.append(value)
    return prices


def _last_price_before(dialogue: List[Dict[str, str]], idx: int) -> Optional[float]:
    """Return the most recent committed price prior to (and including) idx."""
    for turn in reversed(dialogue[: idx + 1]):
        text = turn.get("content", "") or ""
        committed = _extract_committed_prices(text)
        if committed:
            return committed[-1]
        # Last-resort: a $-marked number, if any.
        match = _DOLLAR_PRICE_REGEX.search(text)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                pass
    # Final fallback: legacy loose first-price scan.
    for turn in reversed(dialogue[: idx + 1]):
        price = first_price(turn.get("content", ""))
        if price is not None:
            return price
    return None


def rule_judge_deal(dialogue: List[Dict[str, str]]) -> Dict[str, Any]:
    """Deterministic seller-side rule judge.

    The previous implementation matched substrings like ``"deal"`` anywhere
    in the dialogue, so phrases such as ``"that would be a great deal"`` or
    ``"good deal"`` triggered false positives. This version requires (1) an
    explicit seller-side accept on its own turn, (2) word-boundary matching
    with negator stripping, and (3) a veto when the LAST seller utterance
    contains a reject pattern.
    """
    sellers = _seller_turns(dialogue)
    if not sellers:
        return {
            "deal": False,
            "deal_price": None,
            "success": False,
            "evidence": "no seller turn",
            "judge_model": "rule",
        }

    last_seller_text = sellers[-1].get("content", "") or ""
    if _has_reject(last_seller_text):
        return {
            "deal": False,
            "deal_price": None,
            "success": False,
            "evidence": f"seller rejected: {last_seller_text.strip()[:120]}",
            "judge_model": "rule",
        }

    accept_turn_idx = None
    for idx in range(len(dialogue) - 1, -1, -1):
        if _normalise_role(dialogue[idx]) != "seller":
            continue
        text = dialogue[idx].get("content", "") or ""
        if _has_accept(text) and not _has_reject(text):
            accept_turn_idx = idx
            break

    if accept_turn_idx is None:
        return {
            "deal": False,
            "deal_price": None,
            "success": False,
            "evidence": "no explicit seller acceptance",
            "judge_model": "rule",
        }

    accept_text = dialogue[accept_turn_idx].get("content", "") or ""
    # Strict price extraction: prefer $-marked / verb-context numbers; only
    # fall back to the previous price if the accept turn has no committed
    # price (e.g. a bare "Deal." turn).
    prices = _extract_committed_prices(accept_text)
    deal_price = prices[-1] if prices else _last_price_before(dialogue, accept_turn_idx)

    return {
        "deal": True,
        "deal_price": float(deal_price) if deal_price is not None else None,
        "success": True,
        "evidence": accept_text.strip()[:160],
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
                "You are a strict evaluator for a buyer-agent negotiation. "
                "Assistant messages are from the Buyer. User messages are from the Seller. "
                "Return strict JSON only. Be conservative: only call it a deal "
                "when the SELLER explicitly accepts a concrete price."
            ),
        },
        {
            "role": "user",
            "content": (
                "Decide whether the Buyer and Seller closed a deal.\n"
                "Strict rules:\n"
                "  1. ONLY the Seller can close the deal. Buyer agreeing alone is not enough.\n"
                "  2. The Seller's acceptance must be explicit (e.g. 'Deal.', 'Sold.', "
                "'I accept your offer of $X', 'It is yours for $X'). Descriptive phrases "
                "like 'good deal', 'great deal', 'fair deal', 'best deal' do NOT count.\n"
                "  3. If the LAST seller turn contains a refusal ('too low', 'cannot', "
                "'will not', 'have to decline', 'the lowest I can', 'my minimum', "
                "'not acceptable'), the result is no-deal even if earlier turns hinted at agreement.\n"
                "  4. deal_price must be the concrete price the Seller accepted on that turn. "
                "If you cannot identify it, set deal_price to null.\n"
                "  5. Ignore numbers that are clearly model numbers, years, or product specs "
                "(e.g. 'Nishiki Altron 7000', '1970 model').\n"
                f"Dialogue:\n{_dialogue_to_text(dialogue)}\n\n"
                'Return JSON: {"deal": true|false, "deal_price": number|null, '
                '"success": true|false, "evidence": "short quote of the seller turn"}'
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
        # Post-check: if the LLM said deal=True but the LAST seller turn is a
        # clear rejection, override to no-deal. This catches the common LLM
        # failure mode of fixating on an earlier "deal" phrase.
        if parsed["deal"]:
            sellers = _seller_turns(dialogue)
            if sellers and _has_reject(sellers[-1].get("content", "") or ""):
                parsed["deal"] = False
                parsed["success"] = False
                parsed["deal_price"] = None
                parsed["evidence"] = (
                    "overridden: last seller turn rejects; "
                    + str(parsed.get("evidence", ""))
                )[:240]
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
