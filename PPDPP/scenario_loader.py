import json
from pathlib import Path

import yaml


OBJECTIVE_ORDER = ["sl_ratio", "fairness", "deal_rate"]


def coerce_static_w(values):
    vals = [float(v) for v in values]
    n = len(OBJECTIVE_ORDER)
    if len(vals) > n:
        vals = vals[:n]
    elif len(vals) < n:
        vals = vals + [0.0] * (n - len(vals))
    total = sum(vals)
    if total <= 0:
        return [1.0 / n] * n
    return [v / total for v in vals]


def _load_raw(path):
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data.get("scenarios", data) if isinstance(data, dict) else data

    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _price_span(case):
    return float(case["seller_price"]) - float(case["buyer_price"])


def convert_scenario(raw):
    case = dict(raw.get("case", raw))
    scenario_id = str(raw.get("id", case.get("scenario_id", "")))
    if not scenario_id:
        scenario_id = f"{case.get('source_dataset', 'scenario')}_{case.get('source_index', '')}"

    case.setdefault("item_name", "used item")
    case.setdefault("buyer_item_description", case.get("seller_item_description", ""))
    case.setdefault("seller_item_description", case.get("buyer_item_description", ""))
    case["buyer_price"] = float(case["buyer_price"])
    case["seller_price"] = float(case["seller_price"])

    static_w = coerce_static_w(raw.get("static_w", [1.0, 1.0, 1.0]))
    span = _price_span(case)
    constraints = raw.get("buyer_constraints", raw.get("seller_constraints", {})) or {}
    max_ratio = constraints.get("max_acceptable_price_ratio")
    if max_ratio is None:
        max_ratio = constraints.get("min_acceptable_price_ratio")
    target_ratio = constraints.get("target_price_ratio")

    if max_ratio is not None:
        max_acceptable_price = case["buyer_price"] + float(max_ratio) * span
    else:
        max_acceptable_price = None

    if target_ratio is not None:
        target_price = case["buyer_price"] + float(target_ratio) * span
    else:
        target_price = None

    seller_persona = raw.get("seller_persona", raw.get("buyer_persona", {})) or {}
    case.update({
        "scenario_id": scenario_id,
        "macro_goal": raw.get("macro_goal"),
        "buyer_intent_id": raw.get("buyer_intent_id", raw.get("seller_intent_id")),
        "drift_mode": raw.get("drift_mode"),
        "expected_weight_shift": raw.get("expected_weight_shift", {}),
        "seller_persona": seller_persona,
        "seller_persona_type": seller_persona.get("type"),
        "recommendation_domain": case.get("recommendation_domain"),
        "source_dataset": case.get("source_dataset"),
        "static_w": static_w,
        "turn_limit": int(raw.get("turn_limit", constraints.get("turn_limit", 0)) or 0),
        "max_acceptable_price": max_acceptable_price,
        "target_price": target_price,
    })
    return case


def load_scenario_cases(path, limit=None):
    rows = _load_raw(path)
    cases = [convert_scenario(row) for row in rows]
    if limit is not None:
        cases = cases[: int(limit)]
    return cases


def load_custom_dataset(train_path=None, test_path=None, valid_path=None,
                        train_limit=None, test_limit=None, valid_limit=None):
    if not train_path and not test_path and not valid_path:
        raise ValueError("At least one custom scenario path is required.")

    train_cases = load_scenario_cases(train_path, train_limit) if train_path else []
    test_cases = load_scenario_cases(test_path, test_limit) if test_path else []
    valid_cases = load_scenario_cases(valid_path, valid_limit) if valid_path else []

    if not test_cases:
        test_cases = list(valid_cases or train_cases)
    if not valid_cases:
        valid_cases = list(test_cases or train_cases)
    if not train_cases:
        train_cases = list(valid_cases or test_cases)

    return {"train": train_cases, "valid": valid_cases, "test": test_cases}
