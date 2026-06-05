"""Metric computation for H-MOD buyer-agent evaluation."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from hmod.scenario import HMODScenario, OBJECTIVE_ORDER


def compute_gsr(
    judge_result: Dict[str, Any],
    min_acceptable_price: float,
    turn_count: int,
    turn_limit: int,
    price_direction: str = "at_most",
) -> Tuple[int, Dict[str, Any]]:
    deal = bool(judge_result.get("deal", False))
    deal_price = judge_result.get("deal_price")
    if price_direction == "at_least":
        price_ok = deal_price is not None and float(deal_price) >= float(min_acceptable_price)
    else:
        price_ok = deal_price is not None and float(deal_price) <= float(min_acceptable_price)
    turn_ok = int(turn_count) <= int(turn_limit)
    gsr = int(deal and price_ok and turn_ok)
    return gsr, {
        "deal": deal,
        "deal_price": deal_price,
        "constraint_price": min_acceptable_price,
        "price_direction": price_direction,
        "price_ok": bool(price_ok),
        "turn_count": turn_count,
        "turn_limit": turn_limit,
        "turn_ok": bool(turn_ok),
    }


def _direction_ok(
    before: List[float],
    after: List[float],
    expected_weight_shift: Dict[str, str],
) -> bool:
    """Adaptation direction check (relaxed: no-violation + at-least-one-correct).

    The shift counts as a correct adaptation if NO expected objective moves the
    OPPOSITE way and AT LEAST ONE expected objective moves the correct way. A
    flat objective (delta == 0) is allowed — this avoids penalising a sensible
    adaptation that, e.g., raises deal_rate and lowers sl_ratio but keeps
    fairness moderate, when the gold shift also lists fairness 'up'.
    """
    if not expected_weight_shift:
        return True
    index = {name: i for i, name in enumerate(OBJECTIVE_ORDER)}
    checked = False
    satisfied = 0
    for objective, direction in expected_weight_shift.items():
        if objective not in index:
            continue
        checked = True
        delta = after[index[objective]] - before[index[objective]]
        if direction == "up":
            if delta < 0:          # moved opposite -> violation
                return False
            if delta > 0:
                satisfied += 1
        elif direction == "down":
            if delta > 0:          # moved opposite -> violation
                return False
            if delta < 0:
                satisfied += 1
    return checked and satisfied >= 1


def compute_t2da(
    weight_trace: List[Dict[str, Any]],
    t_drift: Optional[int],
    turn_limit: int,
    expected_weight_shift: Optional[Dict[str, str]] = None,
    threshold: float = 0.25,
) -> Dict[str, Any]:
    if t_drift is None:
        return {"t_drift": None, "t_adapt": None, "t2da": None, "adapted": None}
    if not weight_trace:
        penalty = int(turn_limit) - int(t_drift) + 1
        return {"t_drift": t_drift, "t_adapt": None, "t2da": penalty, "adapted": False}

    before_candidates = [
        row for row in weight_trace if int(row["turn"]) < int(t_drift)
    ]
    before = (
        before_candidates[-1]["weight_vector"]
        if before_candidates
        else weight_trace[0]["weight_vector"]
    )
    expected_weight_shift = expected_weight_shift or {}

    for row in weight_trace:
        turn = int(row["turn"])
        if turn < int(t_drift):
            continue
        current = row["weight_vector"]
        l1 = sum(abs(float(a) - float(b)) for a, b in zip(current, before))
        if l1 >= threshold and _direction_ok(before, current, expected_weight_shift):
            return {
                "t_drift": int(t_drift),
                "t_adapt": turn,
                "t2da": max(0, turn - int(t_drift)),
                "adapted": True,
            }

    penalty = int(turn_limit) - int(t_drift) + 1
    return {"t_drift": int(t_drift), "t_adapt": None, "t2da": penalty, "adapted": False}


def compute_cvr(violation_trace: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    rows = list(violation_trace)
    denom = max(len(rows), 1)
    blocked = sum(1 for row in rows if row.get("blocked_violation"))
    actual = sum(1 for row in rows if row.get("actual_violation"))
    return {
        "blocked_cvr": blocked / denom,
        "actual_cvr": actual / denom,
        "cvr": actual / denom,
        "blocked_violation_count": blocked,
        "actual_violation_count": actual,
        "violation_attempt_count": len(rows),
    }


def aggregate_dialogue_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "num_dialogues": 0,
            "llm_sr": 0.0,
            "gsr": 0.0,
            "t2da": None,
            "blocked_cvr": 0.0,
            "actual_cvr": 0.0,
            "cvr": 0.0,
        }
    t2da_values = [
        rec["t2da"]["t2da"]
        for rec in records
        if rec.get("t2da", {}).get("t2da") is not None
    ]
    # Deal-reached rate. Use the judge's `deal` verdict, not `success`: the LLM
    # judge tends to set `success=True` (it reads it as "buyer negotiated well")
    # even when no deal closed, which inflates the metric. `deal` is the actual
    # agreement signal and matches GSR's deal component.
    def _deal(rec):
        jr = rec["judge_result"]
        return bool(jr.get("deal", jr.get("success", False)))

    return {
        "num_dialogues": len(records),
        "llm_sr": sum(float(_deal(rec)) for rec in records) / len(records),
        "gsr": sum(float(rec.get("gsr", 0)) for rec in records) / len(records),
        "t2da": (sum(t2da_values) / len(t2da_values)) if t2da_values else None,
        "t2da_count": len(t2da_values),
        "blocked_cvr": sum(float(rec["cvr"]["blocked_cvr"]) for rec in records) / len(records),
        "actual_cvr": sum(float(rec["cvr"]["actual_cvr"]) for rec in records) / len(records),
        "cvr": sum(float(rec["cvr"]["cvr"]) for rec in records) / len(records),
    }


def subgroup_metrics(records: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault(str(rec.get(key, "unknown")), []).append(rec)
    return {name: aggregate_dialogue_metrics(group) for name, group in groups.items()}


def min_acceptable_price(scenario: HMODScenario) -> float:
    return scenario.min_acceptable_price()


def max_acceptable_price(scenario: HMODScenario) -> float:
    return scenario.max_acceptable_price()
