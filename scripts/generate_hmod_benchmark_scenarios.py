"""Generate H-MOD buyer-agent train/test scenarios from local benchmark data.

The generated files keep H-MOD's current role setup:
assistant = Buyer agent, simulator = Seller.

Sources:
- Craigslist Bargain raw JSONL for bargain scenarios.
- DuReCDial English raw JSONL for recommendation-derived scenarios.

DuReCDial does not contain prices, so recommendation scenarios use deterministic
synthetic price bands while preserving the recommendation topic, goal, user
profile, and context as the ambiguous buyer objective context.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
OBJECTIVE_FILE = REPO_ROOT / "config" / "scenario" / "hmod_buyer_objectives.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "config" / "scenario" / "generated"

BARGAIN_TRAIN_SOURCE = REPO_ROOT / "data" / "neg_data" / "craigslist" / "cb-train.txt"
BARGAIN_TEST_SOURCE = REPO_ROOT / "data" / "neg_data" / "craigslist" / "cb-test.txt"
REC_TRAIN_SOURCE = REPO_ROOT / "data" / "rec_data" / "durecdial" / "data" / "en_train.txt"
REC_TEST_SOURCE = REPO_ROOT / "data" / "rec_data" / "durecdial" / "data" / "en_test.txt"

DRIFT_MODES = [
    "static_no_drift",
    "gradual_firming",
    "abrupt_final_offer",
    "frustrated_walkaway",
]

PRIMARY_OBJECTIVE_IDS = [
    "AGGRESSIVE_SAVINGS_THEN_RECOVERY",
    "URGENT_GIFT_WITH_HARD_CEILING",
    "QUALITY_RISK_THEN_PRICE_PUSH",
    "BACKUP_OPTION_PRESSURE_CONTROL",
    "SCARCITY_ADAPTIVE_BUYER",
    "FAIR_RELATIONSHIP_REPEAT_BUYER",
    "BUDGET_LOCKED_FLEXIBLE_TIMING",
    "LONG_HAGGLE_FATIGUE_CONTROL",
]

OBJECTIVE_CONSTRAINTS = {
    "AGGRESSIVE_SAVINGS_THEN_RECOVERY": (0.62, 0.32, 9),
    "URGENT_GIFT_WITH_HARD_CEILING": (0.66, 0.48, 7),
    "QUALITY_RISK_THEN_PRICE_PUSH": (0.60, 0.38, 9),
    "BACKUP_OPTION_PRESSURE_CONTROL": (0.61, 0.34, 9),
    "SCARCITY_ADAPTIVE_BUYER": (0.65, 0.45, 8),
    "FAIR_RELATIONSHIP_REPEAT_BUYER": (0.62, 0.44, 8),
    "BUDGET_LOCKED_FLEXIBLE_TIMING": (0.54, 0.34, 8),
    "LONG_HAGGLE_FATIGUE_CONTROL": (0.63, 0.36, 10),
}

DRIFT_TRIGGERS = {
    "static_no_drift": {},
    "gradual_firming": {
        "round_without_deal": 3,
        "low_offer_streak": 2,
        "low_offer_multiplier": 0.88,
    },
    "abrupt_final_offer": {
        "turn_id": 4,
    },
    "frustrated_walkaway": {
        "frustration_threshold": 3,
        "walkaway_after_turns": 2,
        "low_offer_multiplier": 0.88,
    },
}

DRIFT_EXPECTED_SHIFT = {
    "static_no_drift": {},
    "gradual_firming": {
        "sl_ratio": "down",
        "fairness": "up",
        "deal_rate": "up",
    },
    "abrupt_final_offer": {
        "sl_ratio": "down",
        "deal_rate": "up",
        "avg_turn": "up",
    },
    "frustrated_walkaway": {
        "sl_ratio": "down",
        "fairness": "up",
        "deal_rate": "up",
    },
}

DRIFT_GOAL_CLAUSES = {
    "static_no_drift": (
        "The seller is expected to bargain consistently, so the buyer should keep the original objective "
        "unless the price itself crosses the private ceiling."
    ),
    "gradual_firming": (
        "If several rounds pass without agreement or repeated low offers irritate the seller, the seller may "
        "become firm; the buyer must notice this and shift from pure savings toward a bounded close."
    ),
    "abrupt_final_offer": (
        "The seller may suddenly issue a final take-it-or-leave-it offer; the buyer should adapt quickly "
        "without paying above the ceiling."
    ),
    "frustrated_walkaway": (
        "If the buyer keeps pushing too hard, the seller may become frustrated and threaten to sell elsewhere; "
        "the buyer must recover the deal only when the price is still safe."
    ),
}

SELLER_PERSONAS_BY_DRIFT = {
    "static_no_drift": [
        {
            "type": "cooperative_midpoint_seller",
            "description": "A polite seller who starts high but accepts a fair midpoint if the buyer is respectful.",
            "initial_ask_ratio": 0.94,
            "neutral_accept_ratio": 0.54,
            "post_drift_ask_ratio": 0.60,
            "final_ask_ratio": 0.57,
        },
        {
            "type": "steady_value_seller",
            "description": "A stable seller who negotiates predictably and cares about a reasonable value exchange.",
            "initial_ask_ratio": 0.96,
            "neutral_accept_ratio": 0.57,
            "post_drift_ask_ratio": 0.61,
            "final_ask_ratio": 0.58,
        },
    ],
    "gradual_firming": [
        {
            "type": "firm_after_lowballs_seller",
            "description": "A seller who negotiates at first but becomes firm after repeated low offers.",
            "initial_ask_ratio": 0.98,
            "neutral_accept_ratio": 0.58,
            "post_drift_ask_ratio": 0.62,
            "final_ask_ratio": 0.60,
        },
        {
            "type": "guarded_condition_seller",
            "description": "A seller who answers questions but resists when the buyer uses uncertainty to push too low.",
            "initial_ask_ratio": 0.95,
            "neutral_accept_ratio": 0.56,
            "post_drift_ask_ratio": 0.61,
            "final_ask_ratio": 0.59,
        },
    ],
    "abrupt_final_offer": [
        {
            "type": "scarcity_pressure_seller",
            "description": "A seller who claims another buyer is ready and may suddenly give a final price.",
            "initial_ask_ratio": 0.98,
            "neutral_accept_ratio": 0.61,
            "post_drift_ask_ratio": 0.69,
            "final_ask_ratio": 0.64,
        },
        {
            "type": "deadline_final_offer_seller",
            "description": "A seller with limited patience who switches to a firm final offer mid-negotiation.",
            "initial_ask_ratio": 0.96,
            "neutral_accept_ratio": 0.60,
            "post_drift_ask_ratio": 0.68,
            "final_ask_ratio": 0.63,
        },
    ],
    "frustrated_walkaway": [
        {
            "type": "impatient_walkaway_seller",
            "description": "An impatient seller who dislikes repeated pressure and threatens to sell elsewhere.",
            "initial_ask_ratio": 0.97,
            "neutral_accept_ratio": 0.58,
            "post_drift_ask_ratio": 0.62,
            "final_ask_ratio": 0.60,
        },
        {
            "type": "lowball_sensitive_seller",
            "description": "A seller who becomes frustrated when the buyer keeps making offers far below tolerance.",
            "initial_ask_ratio": 0.99,
            "neutral_accept_ratio": 0.59,
            "post_drift_ask_ratio": 0.63,
            "final_ask_ratio": 0.61,
        },
    ],
}

REC_DOMAIN_PRICE_BANDS = {
    "movie": (18, 55),
    "music": (15, 50),
    "food": (25, 90),
    "poi": (30, 120),
    "celebrity": (20, 70),
    "news": (10, 35),
    "general": (20, 80),
}

REC_DOMAIN_ITEMS = {
    "movie": "movie-night recommendation bundle",
    "music": "music discovery package",
    "food": "restaurant meal voucher",
    "poi": "local experience booking",
    "celebrity": "fan content package",
    "news": "curated information service",
    "general": "personalized recommendation package",
}

REC_ALLOWED_DOMAINS = ["movie", "music", "poi"]


class NoAliasSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def sanitize_text(value: Any, max_chars: int = 360) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        value = " ".join(sanitize_text(item, max_chars=max_chars) for item in value)
    elif isinstance(value, Mapping):
        parts = []
        for key, item in value.items():
            if isinstance(item, (list, tuple, set)):
                item = ", ".join(sanitize_text(x, max_chars=80) for x in item[:6])
            parts.append(f"{key}: {item}")
        value = "; ".join(parts)
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def stable_int(text: str, modulo: int = 10_000_000) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def round_money(value: float) -> float:
    if value >= 100:
        return float(round(value / 5.0) * 5)
    return float(round(value))


def normalize_weight(values: Iterable[float]) -> List[float]:
    clean = [max(0.0, float(x)) for x in values]
    total = sum(clean)
    if total <= 0:
        return [0.25, 0.25, 0.25, 0.25]
    return [round(x / total, 4) for x in clean]


def load_objectives(path: Path) -> Dict[str, Dict[str, Any]]:
    namespace: Dict[str, Any] = {}
    exec(path.read_text(encoding="utf-8"), namespace)
    raw = namespace.get("BUYER_STRATEGY_INTENTS")
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must define BUYER_STRATEGY_INTENTS")
    return raw


def objective_static_w(objectives: Mapping[str, Dict[str, Any]], objective_id: str) -> List[float]:
    raw = objectives[objective_id]
    stage_weights = raw.get("stage_weights", {})
    if isinstance(stage_weights, dict) and "initial" in stage_weights:
        return normalize_weight(stage_weights["initial"])
    return [0.25, 0.25, 0.25, 0.25]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


def load_bargain_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for idx, row in enumerate(read_jsonl(path)):
        try:
            buyer_price = float(row["buyer_price"])
            seller_price = float(row["seller_price"])
        except (KeyError, TypeError, ValueError):
            continue
        if buyer_price <= 0 or seller_price <= buyer_price:
            continue
        item_name = sanitize_text(row.get("item_name", "used item"), max_chars=80) or "used item"
        rows.append(
            {
                "source_index": idx,
                "source_dataset": "craigslist_bargain",
                "item_name": item_name,
                "buyer_price": round_money(buyer_price),
                "seller_price": round_money(seller_price),
                "seller_item_description": sanitize_text(
                    row.get("seller_item_description", ""), max_chars=520
                )
                or f"Seller listing for {item_name}.",
                "buyer_item_description": sanitize_text(
                    row.get("buyer_item_description", ""), max_chars=420
                )
                or f"Buyer is interested in {item_name} and wants a bounded deal.",
                "source_dialog_turns": len(row.get("dialog", [])) if isinstance(row.get("dialog"), list) else 0,
            }
        )
    return rows


def flatten_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        return [str(value)]
    if isinstance(value, Mapping):
        values: List[str] = []
        for item in value.values():
            values.extend(flatten_values(item))
        return values
    if isinstance(value, Iterable):
        values = []
        for item in value:
            values.extend(flatten_values(item))
        return values
    return [str(value)]


def infer_rec_domain(row: Mapping[str, Any]) -> str:
    goal_text = sanitize_text(str(row.get("goal", "")), max_chars=1000).lower()
    goal_matches = re.findall(r"(movie|music|food|poi|celebrity|news)\s+recommendation", goal_text)
    if goal_matches:
        return goal_matches[-1]

    text = sanitize_text(" ".join(flatten_values(row.get("goal_type_list"))), max_chars=1000).lower()
    type_matches = re.findall(r"(movie|music|food|poi|celebrity|news)\s+recommendation", text)
    if type_matches:
        return type_matches[-1]
    for key in ["movie", "music", "food", "poi", "celebrity", "news"]:
        if key in text:
            return key
    return "general"


def infer_rec_topic(row: Mapping[str, Any], domain: str) -> str:
    topics = [
        sanitize_text(x, max_chars=80)
        for x in flatten_values(row.get("goal_topic_list"))
        if sanitize_text(x, max_chars=80)
    ]
    filtered = [
        topic
        for topic in topics
        if topic.lower() not in {"say goodbye", "goodbye", "weather", "q&a"}
    ]
    if filtered:
        return filtered[-1]

    goal = str(row.get("goal", ""))
    matches = re.findall(r"\(([^()]+)\)", goal)
    cleaned = [sanitize_text(match, max_chars=80) for match in matches]
    cleaned = [match for match in cleaned if match and match.lower() not in {"say goodbye"}]
    if cleaned:
        return cleaned[-1]
    return REC_DOMAIN_ITEMS.get(domain, "personalized recommendation")


def profile_summary(profile: Mapping[str, Any]) -> str:
    if not isinstance(profile, Mapping):
        return "User profile is sparse; infer preferences from the conversation context."
    preferred_keys = [
        "Age Range",
        "Gender",
        "Occupation",
        "Residence",
        "Accepted movies",
        "Accepted movie",
        "Rejected movies",
        "Accepted Music",
        "Accepted music",
        "Rejected music",
        "Accepted food",
        "Accepted POI",
        "Reject",
    ]
    selected: MutableMapping[str, Any] = {}
    for key in preferred_keys:
        if key in profile and profile[key]:
            selected[key] = profile[key]
    return sanitize_text(selected or profile, max_chars=520)


def synthetic_rec_case(row: Mapping[str, Any], idx: int) -> Dict[str, Any]:
    domain = infer_rec_domain(row)
    topic = infer_rec_topic(row, domain)
    seed = stable_int(f"{idx}:{row.get('goal', '')}:{topic}:{domain}")
    rng = random.Random(seed)
    low, high = REC_DOMAIN_PRICE_BANDS.get(domain, REC_DOMAIN_PRICE_BANDS["general"])
    buyer_price = round_money(rng.uniform(low, high))
    seller_multiplier = rng.uniform(1.45, 2.15)
    seller_price = round_money(max(buyer_price + 8.0, buyer_price * seller_multiplier))
    if seller_price <= buyer_price:
        seller_price = buyer_price + 10.0

    conversation = row.get("conversation", [])
    context_snippet = sanitize_text(conversation[:6] if isinstance(conversation, list) else "", max_chars=480)
    knowledge = sanitize_text(row.get("knowledge", [])[:5] if isinstance(row.get("knowledge"), list) else "", max_chars=420)
    situation = sanitize_text(row.get("situation", ""), max_chars=180)
    item_label = REC_DOMAIN_ITEMS.get(domain, REC_DOMAIN_ITEMS["general"])
    item_name = sanitize_text(f"{item_label}: {topic}", max_chars=90)

    return {
        "source_index": idx,
        "source_dataset": "durecdial_recommendation",
        "recommendation_domain": domain,
        "item_name": item_name,
        "buyer_price": buyer_price,
        "seller_price": seller_price,
        "seller_item_description": sanitize_text(
            (
                f"Seller offers a {item_label} around '{topic}'. "
                f"Original recommendation goal: {row.get('goal', '')}. "
                f"Situation: {situation}. Knowledge hints: {knowledge}."
            ),
            max_chars=620,
        ),
        "buyer_item_description": sanitize_text(
            (
                f"Recommendation-derived buyer preferences: {profile_summary(row.get('user_profile', {}))}. "
                f"Conversation seed: {context_snippet}"
            ),
            max_chars=620,
        ),
        "source_goal": sanitize_text(row.get("goal", ""), max_chars=260),
    }


def load_rec_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for idx, row in enumerate(read_jsonl(path)):
        if not isinstance(row, dict) or not row.get("goal"):
            continue
        rows.append(synthetic_rec_case(row, idx))
    return rows


def cycle_sample(rows: Sequence[Dict[str, Any]], size: int, rng: random.Random) -> List[Dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot sample from an empty source dataset")
    pool = list(rows)
    rng.shuffle(pool)
    selected: List[Dict[str, Any]] = []
    cursor = 0
    while len(selected) < size:
        if cursor >= len(pool):
            rng.shuffle(pool)
            cursor = 0
        selected.append(dict(pool[cursor]))
        cursor += 1
    return selected


def target_counts(total: int, labels: Sequence[str]) -> Dict[str, int]:
    base = total // len(labels)
    remainder = total % len(labels)
    return {
        label: base + (1 if idx < remainder else 0)
        for idx, label in enumerate(labels)
    }


def balanced_domain_sample(
    rows: Sequence[Dict[str, Any]],
    size: int,
    domains: Sequence[str],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {domain: [] for domain in domains}
    for row in rows:
        domain = row.get("recommendation_domain")
        if domain in buckets:
            buckets[domain].append(row)
    missing = [domain for domain, values in buckets.items() if not values]
    if missing:
        raise ValueError(f"Missing recommendation domains in source data: {missing}")

    quotas = target_counts(size, domains)
    sampled_by_domain = {
        domain: cycle_sample(values, quotas[domain], rng)
        for domain, values in buckets.items()
    }
    cursors = {domain: 0 for domain in domains}
    selected: List[Dict[str, Any]] = []
    while len(selected) < size:
        round_domains = list(domains)
        rng.shuffle(round_domains)
        for domain in round_domains:
            if cursors[domain] >= len(sampled_by_domain[domain]):
                continue
            selected.append(dict(sampled_by_domain[domain][cursors[domain]]))
            cursors[domain] += 1
            if len(selected) >= size:
                break
    return selected


def jitter_constraints(
    objective_id: str,
    drift_mode: str,
    rng: random.Random,
) -> Dict[str, Any]:
    base_max, base_target, base_turn_limit = OBJECTIVE_CONSTRAINTS[objective_id]
    max_ratio = base_max + rng.choice([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02])
    target_ratio = base_target + rng.choice([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02])

    if drift_mode == "abrupt_final_offer":
        base_turn_limit = min(base_turn_limit, 8)
        max_ratio += 0.01
    elif drift_mode == "frustrated_walkaway":
        base_turn_limit = max(base_turn_limit, 9)
    elif drift_mode == "gradual_firming":
        base_turn_limit = max(base_turn_limit, 9)

    max_ratio = min(0.72, max(0.50, max_ratio))
    target_ratio = min(max_ratio - 0.08, max(0.24, target_ratio))
    return {
        "max_acceptable_price_ratio": round(max_ratio, 2),
        "target_price_ratio": round(target_ratio, 2),
        "turn_limit": int(base_turn_limit),
    }


def drift_trigger_for(mode: str, idx: int) -> Dict[str, Any]:
    trigger = dict(DRIFT_TRIGGERS[mode])
    if mode == "abrupt_final_offer":
        trigger["turn_id"] = 3 + (idx % 3)
    elif mode == "gradual_firming":
        trigger["round_without_deal"] = 3 + (idx % 2)
        trigger["low_offer_multiplier"] = round(0.84 + (idx % 5) * 0.015, 2)
    elif mode == "frustrated_walkaway":
        trigger["frustration_threshold"] = 2 + (idx % 2)
        trigger["walkaway_after_turns"] = 2
    return trigger


def seller_persona_for(mode: str, idx: int, rng: random.Random) -> Dict[str, Any]:
    pool = SELLER_PERSONAS_BY_DRIFT[mode]
    persona = dict(pool[(idx // len(DRIFT_MODES)) % len(pool)])
    # Small deterministic jitter keeps personas diverse while preserving behavior.
    for key in ["initial_ask_ratio", "neutral_accept_ratio", "post_drift_ask_ratio", "final_ask_ratio"]:
        persona[key] = round(persona[key] + rng.choice([-0.015, 0.0, 0.015]), 3)
    persona["initial_ask_ratio"] = min(0.995, max(0.90, persona["initial_ask_ratio"]))
    persona["neutral_accept_ratio"] = min(0.66, max(0.50, persona["neutral_accept_ratio"]))
    persona["post_drift_ask_ratio"] = min(0.72, max(persona["neutral_accept_ratio"], persona["post_drift_ask_ratio"]))
    persona["final_ask_ratio"] = min(0.68, max(persona["neutral_accept_ratio"], persona["final_ask_ratio"]))
    return persona


def build_macro_goal(
    objectives: Mapping[str, Dict[str, Any]],
    objective_id: str,
    drift_mode: str,
    case: Mapping[str, Any],
    domain_label: str,
) -> str:
    objective = objectives[objective_id]
    item = sanitize_text(case.get("item_name", "the item"), max_chars=90)
    natural_intent = sanitize_text(objective.get("natural_language_intent", ""), max_chars=420)
    steps = objective.get("typical_steps", [])
    steps_text = sanitize_text("; ".join(str(step) for step in steps[:4]), max_chars=360)
    drift_clause = DRIFT_GOAL_CLAUSES[drift_mode]
    source_context = "real Craigslist bargain listing" if domain_label == "bargain" else "recommendation-derived user preference case"
    return sanitize_text(
        (
            f"You are the buyer agent for {item}. Ambiguous business objective: {natural_intent} "
            f"Operational hints, not fixed weights: {steps_text}. "
            f"Context comes from a {source_context}. {drift_clause} "
            "Infer the local objective weights by reflection during the dialogue, and revise them when seller intent changes."
        ),
        max_chars=900,
    )


def build_scenario(
    *,
    prefix: str,
    idx: int,
    case: Mapping[str, Any],
    objectives: Mapping[str, Dict[str, Any]],
    rng: random.Random,
    domain_label: str,
) -> Dict[str, Any]:
    drift_mode = DRIFT_MODES[idx % len(DRIFT_MODES)]
    objective_id = PRIMARY_OBJECTIVE_IDS[(idx // len(DRIFT_MODES)) % len(PRIMARY_OBJECTIVE_IDS)]
    constraints = jitter_constraints(objective_id, drift_mode, rng)
    scenario = {
        "id": f"{prefix}_{idx + 1:04d}",
        "macro_goal": build_macro_goal(objectives, objective_id, drift_mode, case, domain_label),
        "buyer_intent_id": objective_id,
        "static_w": objective_static_w(objectives, objective_id),
        "buyer_constraints": constraints,
        "seller_persona": seller_persona_for(drift_mode, idx, rng),
        "initial_intent": "neutral",
        "drift_mode": drift_mode,
        "drift_trigger": drift_trigger_for(drift_mode, idx),
        "turn_limit": constraints["turn_limit"],
        "expected_weight_shift": dict(DRIFT_EXPECTED_SHIFT[drift_mode]),
        "case": dict(case),
    }
    return scenario


def scenario_distribution(scenarios: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    counters = {
        "drift_mode": Counter(),
        "buyer_intent_id": Counter(),
        "seller_persona": Counter(),
        "source_dataset": Counter(),
        "recommendation_domain": Counter(),
    }
    for scenario in scenarios:
        counters["drift_mode"][str(scenario.get("drift_mode"))] += 1
        counters["buyer_intent_id"][str(scenario.get("buyer_intent_id"))] += 1
        persona = scenario.get("seller_persona", {})
        if isinstance(persona, Mapping):
            counters["seller_persona"][str(persona.get("type"))] += 1
        case = scenario.get("case", {})
        if isinstance(case, Mapping):
            counters["source_dataset"][str(case.get("source_dataset"))] += 1
            if case.get("recommendation_domain"):
                counters["recommendation_domain"][str(case.get("recommendation_domain"))] += 1
    return {key: dict(counter) for key, counter in counters.items() if counter}


def write_scenarios(path: Path, scenarios: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> None:
    payload = {
        "metadata": dict(metadata),
        "scenarios": list(scenarios),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.dump(payload, fh, Dumper=NoAliasSafeDumper, sort_keys=False, allow_unicode=False, width=120)


def validate_scenarios(paths: Sequence[Path]) -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from hmod.scenario import load_scenarios

    for path in paths:
        scenarios = load_scenarios(str(path))
        if not scenarios:
            raise ValueError(f"Generated no scenarios in {path}")


def generate_split(
    *,
    prefix: str,
    source_rows: Sequence[Dict[str, Any]],
    size: int,
    objectives: Mapping[str, Dict[str, Any]],
    seed: int,
    domain_label: str,
    balance_rec_domains: bool = False,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    if balance_rec_domains:
        rows = balanced_domain_sample(source_rows, size, REC_ALLOWED_DOMAINS, rng)
    else:
        rows = cycle_sample(source_rows, size, rng)
    return [
        build_scenario(
            prefix=prefix,
            idx=idx,
            case=row,
            objectives=objectives,
            rng=rng,
            domain_label=domain_label,
        )
        for idx, row in enumerate(rows)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bargain_train_size", type=int, default=1000)
    parser.add_argument("--bargain_test_size", type=int, default=250)
    parser.add_argument("--rec_train_size", type=int, default=1000)
    parser.add_argument("--rec_test_size", type=int, default=250)
    parser.add_argument("--no_validate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    objectives = load_objectives(OBJECTIVE_FILE)

    bargain_train_rows = load_bargain_rows(BARGAIN_TRAIN_SOURCE)
    bargain_test_rows = load_bargain_rows(BARGAIN_TEST_SOURCE)
    rec_train_rows = load_rec_rows(REC_TRAIN_SOURCE)
    rec_test_rows = load_rec_rows(REC_TEST_SOURCE)

    splits = {
        "hmod_bargain_train_scenarios.yaml": (
            generate_split(
                prefix="bargain_train",
                source_rows=bargain_train_rows,
                size=args.bargain_train_size,
                objectives=objectives,
                seed=args.seed + 11,
                domain_label="bargain",
            ),
            "craigslist_bargain_train",
            str(BARGAIN_TRAIN_SOURCE.relative_to(REPO_ROOT)),
        ),
        "hmod_bargain_test_scenarios.yaml": (
            generate_split(
                prefix="bargain_test",
                source_rows=bargain_test_rows,
                size=args.bargain_test_size,
                objectives=objectives,
                seed=args.seed + 13,
                domain_label="bargain",
            ),
            "craigslist_bargain_test",
            str(BARGAIN_TEST_SOURCE.relative_to(REPO_ROOT)),
        ),
        "hmod_recommendation_train_scenarios.yaml": (
            generate_split(
                prefix="rec_train",
                source_rows=rec_train_rows,
                size=args.rec_train_size,
                objectives=objectives,
                seed=args.seed + 17,
                domain_label="recommendation",
                balance_rec_domains=True,
            ),
            "durecdial_recommendation_train",
            str(REC_TRAIN_SOURCE.relative_to(REPO_ROOT)),
        ),
        "hmod_recommendation_test_scenarios.yaml": (
            generate_split(
                prefix="rec_test",
                source_rows=rec_test_rows,
                size=args.rec_test_size,
                objectives=objectives,
                seed=args.seed + 19,
                domain_label="recommendation",
                balance_rec_domains=True,
            ),
            "durecdial_recommendation_test",
            str(REC_TEST_SOURCE.relative_to(REPO_ROOT)),
        ),
    }

    written_paths: List[Path] = []
    manifest: Dict[str, Any] = {
        "seed": args.seed,
        "objective_order": ["sl_ratio", "fairness", "deal_rate", "avg_turn"],
        "drift_modes": DRIFT_MODES,
        "primary_objective_ids": PRIMARY_OBJECTIVE_IDS,
        "recommendation_allowed_domains": REC_ALLOWED_DOMAINS,
        "source_files": {
            "bargain_train": str(BARGAIN_TRAIN_SOURCE.relative_to(REPO_ROOT)),
            "bargain_test": str(BARGAIN_TEST_SOURCE.relative_to(REPO_ROOT)),
            "recommendation_train": str(REC_TRAIN_SOURCE.relative_to(REPO_ROOT)),
            "recommendation_test": str(REC_TEST_SOURCE.relative_to(REPO_ROOT)),
        },
        "source_row_counts": {
            "bargain_train": len(bargain_train_rows),
            "bargain_test": len(bargain_test_rows),
            "recommendation_train": len(rec_train_rows),
            "recommendation_test": len(rec_test_rows),
        },
        "source_recommendation_domain_counts": {
            "train": dict(Counter(row["recommendation_domain"] for row in rec_train_rows)),
            "test": dict(Counter(row["recommendation_domain"] for row in rec_test_rows)),
        },
        "splits": {},
        "note": (
            "Recommendation scenarios are derived from DuReCDial preferences and goals, filtered to Movie/Music/POI. "
            "Prices are deterministic synthetic bands because DuReCDial has no transaction prices."
        ),
    }

    for filename, (scenarios, split_name, source_path) in splits.items():
        output_path = output_dir / filename
        metadata = {
            "split": split_name,
            "source_path": source_path,
            "count": len(scenarios),
            "seed": args.seed,
            "generator": str(Path(__file__).relative_to(REPO_ROOT)),
            "distribution": scenario_distribution(scenarios),
        }
        write_scenarios(output_path, scenarios, metadata)
        written_paths.append(output_path)
        manifest["splits"][filename] = metadata

    manifest_path = output_dir / "hmod_benchmark_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    if not args.no_validate:
        validate_scenarios(written_paths)

    print(f"Wrote {len(written_paths)} scenario files to {output_dir}")
    for path in written_paths:
        split_meta = manifest["splits"][path.name]
        print(f"- {path}: {split_meta['count']} scenarios")
    print(f"- {manifest_path}: manifest")


if __name__ == "__main__":
    main()
