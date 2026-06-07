import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PPDPP_ROOT = os.path.join(ROOT, "PPDPP")
if PPDPP_ROOT not in sys.path:
    sys.path.insert(0, PPDPP_ROOT)

from ppdpp_rewards import (  # noqa: E402
    aggregate_episode_records,
    compute_reward_info,
    write_evaluation_outputs,
)
from scenario_loader import load_scenario_cases  # noqa: E402


def test_ppdpp_loads_generated_bargain_and_recommendation_scenarios():
    paths = [
        "config/scenario/generated/hmod_bargain_test_scenarios.yaml",
        "config/scenario/generated/hmod_recommendation_test_scenarios.yaml",
    ]
    for path in paths:
        cases = load_scenario_cases(path, limit=2)
        assert len(cases) == 2
        assert all(case["item_name"] for case in cases)
        assert all(case["buyer_price"] < case["seller_price"] for case in cases)
        assert all(len(case["static_w"]) == 3 for case in cases)
        assert all(abs(sum(case["static_w"]) - 1.0) < 1e-8 for case in cases)


def test_ppdpp_reward_vector_scalarization_and_static_w_score(tmp_path):
    case = {
        "buyer_price": 100.0,
        "seller_price": 200.0,
        "max_acceptable_price": 160.0,
        "static_w": [0.5, 0.25, 0.25],
    }
    dialogue = [
        {"role": "Buyer", "content": "I can do $140, deal?"},
        {"role": "Seller", "content": "Deal, $140 works."},
    ]
    info = compute_reward_info(
        case=case,
        conversation=dialogue,
        system_response="I can do $140, deal?",
        action="propose",
        objective="uniform",
        judge_result=(True, 140.0),
    )
    assert info["reward_vector"] == {
        "sl_ratio": 0.6,
        "fairness": 0.4,
        "deal_rate": 1.0,
    }
    assert round(info["scalar_reward"], 6) == round((0.6 + 0.4 + 1.0) / 3.0, 6)

    record = {
        "scenario_id": "case_1",
        "objective": "uniform",
        "success": True,
        "gsr": 1,
        "turns": 1,
        "weighted_return": info["scalar_reward"],
        "cumulative_reward_vector": info["reward_vector"],
        "static_w": case["static_w"],
        "deal_price": 140.0,
        "max_acceptable_price": 160.0,
        "price_violation": False,
        "price_attempt_count": 1,
    }
    metrics = aggregate_episode_records([record], "uniform")
    assert metrics["sr"] == 1.0
    assert metrics["gsr"] == 1.0
    assert metrics["cvr"] == 0.0
    assert metrics["t2da_status"] == "not_applicable"
    assert round(metrics["static_w_return"], 6) == 0.65

    write_evaluation_outputs(
        str(tmp_path),
        {"metrics": metrics},
        [record],
        [{"scenario_id": "case_1", "dialogue": dialogue}],
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "dialogues.jsonl",
        "metrics.json",
        "summary.csv",
    ]
    with (tmp_path / "metrics.json").open(encoding="utf-8") as fh:
        assert json.load(fh)["metrics"]["num_dialogues"] == 1
