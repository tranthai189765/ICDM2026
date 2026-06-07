import csv
import json
import os
import re
import sys
from collections import defaultdict


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from hmod.judge import rule_judge_deal as _hmod_rule_judge_deal
except Exception:
    _hmod_rule_judge_deal = None


OBJECTIVE_WEIGHTS = {
    "uniform": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    "sl_ratio": [1.0, 0.0, 0.0],
    "fairness": [0.0, 1.0, 0.0],
    "deal_rate": [0.0, 0.0, 1.0],
}

OBJECTIVE_ORDER = ["sl_ratio", "fairness", "deal_rate"]
# Actions that genuinely commit a price. `confirm` is a question/clarification
# and `inquire`/`inform` are descriptive, so they are excluded.
PRICE_COMMITTING_ACTIONS = {"propose", "counter", "agree"}

# Numbers that look like years should never be treated as prices. Used by
# `extract_committed_prices` to filter out tokens like "1970", "2025".
_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

# Words / phrases that, when they appear immediately before a number,
# indicate the number is a real committed price.
_PRICE_VERB_CONTEXTS = (
    r"offer(?:s|ing|ed)?",
    r"pay(?:s|ing|ed)?",
    r"paid",
    r"accept(?:s|ing|ed)?",
    r"settle(?:s|d)?\s+(?:on|at|for)",
    r"meet(?:\s+(?:you|me))?\s+(?:at|in\s+the\s+middle\s+at)",
    r"do",
    r"come\s+down\s+to",
    r"go\s+(?:as\s+low\s+as|down\s+to|to)",
    r"sell(?:\s+(?:it|them))?\s+(?:for|at|to\s+you\s+for)",
    r"sold\s+(?:for|at)",
    r"buy(?:\s+(?:it|them))?\s+for",
    r"bought\s+for",
    r"deal\s+at",
    r"price\s+(?:is|of)",
    r"propose(?:s|d)?",
    r"counter(?:offer|s|ed)?\s+(?:of|at)?",
)
_PRICE_VERB_REGEX = re.compile(
    r"(?:" + "|".join(_PRICE_VERB_CONTEXTS) + r")\s*\$?\s*([-+]?\d[\d,]*\.?\d*)",
    flags=re.IGNORECASE,
)
# Standalone dollar-marked numbers (e.g. "$1,250", "$45.50").
_DOLLAR_PRICE_REGEX = re.compile(r"\$\s*([-+]?\d[\d,]*\.?\d*)")


def objective_weight(name):
    if name not in OBJECTIVE_WEIGHTS:
        raise ValueError(f"Unknown objective {name!r}. Expected one of {sorted(OBJECTIVE_WEIGHTS)}")
    return OBJECTIVE_WEIGHTS[name]


def scalarize(vector, objective):
    weight = objective_weight(objective)
    return sum(float(w) * float(v) for w, v in zip(weight, vector))


def extract_prices(text):
    """Loose numeric scanner kept for backward compatibility (used by judge).

    Returns every numeric token in the text. Prefer `extract_committed_prices`
    when deciding whether the *system* committed a price.
    """
    nums = re.findall(r"[-+]?\d*\.?\d+", (text or "").replace(",", ""))
    prices = []
    for n in nums:
        try:
            prices.append(float(n))
        except ValueError:
            continue
    return prices


def extract_committed_prices(text):
    """Strict price extractor that requires monetary/verb context.

    A number is treated as a committed price only when it is preceded by an
    explicit monetary marker (``$``, ``USD``, ``dollars``) or a verb that
    commits a price (``offer``, ``pay``, ``accept``, ``come down to``, etc).
    Years (1900-2099) are excluded outright.
    """
    raw = text or ""
    if not raw.strip():
        return []
    cleaned = _YEAR_PATTERN.sub(" ", raw)
    candidates = []
    for match in _DOLLAR_PRICE_REGEX.finditer(cleaned):
        candidates.append(match.group(1))
    for match in _PRICE_VERB_REGEX.finditer(cleaned):
        candidates.append(match.group(1))
    # Also accept numbers followed by an explicit dollar/USD suffix.
    for match in re.finditer(
        r"([-+]?\d[\d,]*\.?\d*)\s*(?:dollars?|usd|bucks)\b",
        cleaned,
        flags=re.IGNORECASE,
    ):
        candidates.append(match.group(1))

    prices = []
    seen = set()
    for token in candidates:
        norm = token.replace(",", "")
        try:
            value = float(norm)
        except ValueError:
            continue
        # Defend against tiny tokens that snuck in as bare decimals.
        if (norm, value) in seen:
            continue
        seen.add((norm, value))
        prices.append(value)
    return prices


def bounded_price_from_text(text, case, strict=True):
    low = min(float(case["buyer_price"]), float(case["seller_price"]))
    high = max(float(case["buyer_price"]), float(case["seller_price"]))
    extractor = extract_committed_prices if strict else extract_prices
    prices = [p for p in extractor(text) if low <= p <= high]
    if not prices:
        return None
    return prices[-1]


def compute_price_objectives(case, system_response, action=None):
    strategy = action[0] if isinstance(action, tuple) else action
    price = bounded_price_from_text(system_response, case)
    if price is None or (strategy and strategy not in PRICE_COMMITTING_ACTIONS):
        return 0.0, 0.0, price

    buyer_price = float(case["buyer_price"])
    seller_price = float(case["seller_price"])
    denom = buyer_price - seller_price
    span = seller_price - buyer_price
    if denom == 0 or span == 0:
        return 0.0, 0.0, price

    sl_ratio = (price - seller_price) / denom
    sl_ratio = max(-1.0, min(1.0, sl_ratio))

    mid_price = (buyer_price + seller_price) / 2.0
    fairness = 0.5 - abs(price - mid_price) / span
    fairness = max(-0.5, min(0.5, fairness))
    return sl_ratio, fairness, price


def parse_deal_outputs(outputs):
    deal_votes = []
    deal_prices = []
    for output in outputs or []:
        lower = output.lower()
        if "have not" in lower or "not reached" in lower:
            deal_votes.append(False)
        elif "have reached" in lower or "reached a deal" in lower:
            deal_votes.append(True)
        prices = extract_prices(output)
        if prices:
            deal_prices.append(prices[0])

    if not deal_votes:
        return False, None

    reached = sum(1 for vote in deal_votes if vote) >= (len(deal_votes) / 2.0)
    deal_price = None
    if reached and deal_prices:
        deal_price = max(set(deal_prices), key=deal_prices.count)
    return reached, deal_price


def _conversation_to_hmod_dialogue(conversation):
    """Translate PPDPP role names to the hmod judge's assistant/user contract."""
    rows = []
    for turn in conversation:
        role = str(turn.get("role", "")).lower()
        if role in {"buyer", "assistant", "system"}:
            mapped = "assistant"
        else:
            mapped = "user"
        rows.append({"role": mapped, "content": turn.get("content", "")})
    return rows


def rule_judge_dialogue(conversation):
    """Delegate to hmod.judge.rule_judge_deal so PPDPP and HMOD share the judge."""
    if _hmod_rule_judge_deal is not None:
        result = _hmod_rule_judge_deal(_conversation_to_hmod_dialogue(conversation))
        return bool(result.get("deal")), result.get("deal_price")

    # Fallback (kept for environments where hmod is unavailable).
    accept_patterns = (
        "deal", "i accept", "accept it", "sounds good",
        "i can buy", "i'll buy", "i will buy", "let's do",
    )
    deal_price = None
    for idx, turn in enumerate(conversation):
        content = str(turn.get("content", "") or "")
        lower = content.lower()
        if any(p in lower for p in accept_patterns):
            prices = extract_prices(content)
            if prices:
                deal_price = prices[0]
            else:
                for prev in reversed(conversation[: idx + 1]):
                    p = extract_prices(str(prev.get("content", "") or ""))
                    if p:
                        deal_price = p[0]
                        break
    if deal_price is None:
        return False, None
    return True, float(deal_price)


def compute_reward_info(case, conversation, system_response, action, objective,
                        judge_outputs=None, judge_result=None):
    sl_ratio, fairness, system_price = compute_price_objectives(case, system_response, action)

    if judge_result is not None:
        deal_success, deal_price = judge_result
    else:
        deal_success, deal_price = parse_deal_outputs(judge_outputs)

    deal_reward = 1.0 if deal_success else -0.1
    vector = [sl_ratio, fairness, deal_reward]
    scalar = scalarize(vector, objective)

    max_price = case.get("max_acceptable_price")
    price_violation = bool(
        max_price is not None and system_price is not None and system_price > float(max_price)
    )

    return {
        "scalar_reward": scalar,
        "reward_vector": {
            "sl_ratio": sl_ratio,
            "fairness": fairness,
            "deal_rate": deal_reward,
        },
        "system_price": system_price,
        "deal_success": bool(deal_success),
        "deal_price": deal_price,
        "price_violation": price_violation,
        "blocked_violation": False,
        "actual_violation": price_violation,
        "objective": objective,
        "objective_weight": objective_weight(objective),
    }


def _mean(rows, key, default=0.0):
    vals = [float(row.get(key, 0.0) or 0.0) for row in rows]
    if not vals:
        return default
    return sum(vals) / len(vals)


def aggregate_episode_records(records, objective):
    if not records:
        return {
            "num_dialogues": 0,
            "objective": objective,
            "objective_weight": objective_weight(objective),
            "sr": 0.0,
            "llm_sr": 0.0,
            "deal_rate": 0.0,
            "avg_turn": 0.0,
            "gsr": 0.0,
            "blocked_cvr": 0.0,
            "actual_cvr": 0.0,
            "cvr": 0.0,
            "t2da": None,
            "t2da_status": "not_applicable",
        }

    summary_rows = []
    for rec in records:
        rewards = rec.get("cumulative_reward_vector", {})
        static_w = rec.get("static_w")
        static_w_score = None
        if static_w:
            static_w_score = sum(
                float(w) * float(rewards.get(obj, 0.0))
                for w, obj in zip(static_w, OBJECTIVE_ORDER)
            )
        summary_rows.append({
            **rec,
            "cum_sl_ratio": rewards.get("sl_ratio", 0.0),
            "cum_fairness": rewards.get("fairness", 0.0),
            "cum_deal_rate": rewards.get("deal_rate", 0.0),
            "static_w_score": static_w_score,
        })

    gsr_values = [float(row.get("gsr", 0.0)) for row in summary_rows]
    deal_rate = sum(float(row.get("success", False)) for row in summary_rows) / len(summary_rows)

    def _avg(field):
        vals = [row.get(field) for row in summary_rows if row.get(field) is not None]
        return (sum(float(v) for v in vals) / len(vals)) if vals else 0.0

    blocked_total = sum(int(row.get("blocked_violation_count", 0) or 0) for row in summary_rows)
    actual_total = sum(int(row.get("actual_violation_count", 0) or 0) for row in summary_rows)
    attempts_total = sum(int(row.get("violation_attempt_count", 0) or 0) for row in summary_rows)

    return {
        "num_dialogues": len(summary_rows),
        "objective": objective,
        "objective_weight": objective_weight(objective),
        "sr": deal_rate,
        "llm_sr": deal_rate,
        "deal_rate": deal_rate,
        "avg_turn": _mean(summary_rows, "turns"),
        "sl_ratio": _mean(summary_rows, "cum_sl_ratio"),
        "fairness": _mean(summary_rows, "cum_fairness"),
        "weighted_return": _mean(summary_rows, "weighted_return"),
        "static_w_return": _mean(
            [row for row in summary_rows if row.get("static_w_score") is not None],
            "static_w_score",
            default=None,
        ),
        "gsr": sum(gsr_values) / len(gsr_values),
        "blocked_cvr": _avg("blocked_cvr"),
        "actual_cvr": _avg("actual_cvr"),
        "cvr": _avg("actual_cvr"),
        "blocked_violation_count": blocked_total,
        "actual_violation_count": actual_total,
        "violation_attempt_count": attempts_total,
        "t2da": None,
        "t2da_status": "not_applicable",
    }


def subgroup_metrics(records, objective, key):
    groups = defaultdict(list)
    for rec in records:
        value = rec.get(key)
        if value is None:
            value = "unknown"
        groups[str(value)].append(rec)
    return {name: aggregate_episode_records(rows, objective) for name, rows in groups.items()}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path, rows):
    fieldnames = [
        "scenario_id", "source_dataset", "recommendation_domain", "drift_mode",
        "seller_persona_type", "buyer_intent_id", "objective", "success",
        "gsr", "turns", "weighted_return", "cum_sl_ratio", "cum_fairness",
        "cum_deal_rate", "static_w_score", "deal_price", "max_acceptable_price",
        "price_violation", "price_attempt_count",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            rewards = row.get("cumulative_reward_vector", {})
            out = dict(row)
            out["cum_sl_ratio"] = rewards.get("sl_ratio", 0.0)
            out["cum_fairness"] = rewards.get("fairness", 0.0)
            out["cum_deal_rate"] = rewards.get("deal_rate", 0.0)
            static_w = row.get("static_w")
            out["static_w_score"] = (
                sum(float(w) * float(rewards.get(obj, 0.0))
                    for w, obj in zip(static_w, OBJECTIVE_ORDER))
                if static_w else None
            )
            writer.writerow({k: out.get(k) for k in fieldnames})


def write_evaluation_outputs(output_dir, metrics_payload, episode_records, dialogue_records):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2, ensure_ascii=False)
    write_jsonl(os.path.join(output_dir, "dialogues.jsonl"), dialogue_records)
    write_summary_csv(os.path.join(output_dir, "summary.csv"), episode_records)
