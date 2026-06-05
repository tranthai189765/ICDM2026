import json

from hmod.llm_reflection import LLMWeightReflector, coerce_reflection_weight, parse_reflection_json
from hmod.metrics import compute_cvr, compute_gsr, compute_t2da
from hmod.objectives import BuyerObjectiveLibrary
from hmod.policy import RuleMetaController
from hmod.runner import run_and_write
from hmod.scenario import load_scenarios
from hmod.simulator import DynamicSellerNegotiationSimulator
from hmod.training import HMODController
from scripts.score_hmod_human_audit import score_rows


SCENARIO_FILE = "config/scenario/hmod_buyer_drift_scenarios.yaml"


def _write_objective_file(tmp_path):
    objective_file = tmp_path / "buyer_objectives.py"
    objective_file.write_text(
        """
BUYER_STRATEGY_INTENTS = {
    "PRICE_GAIN_BUT_DEAL_AWARE": {
        "description": "Maximize buyer price gain but close if seller walkaway risk becomes high.",
        "natural_language_intent": "Push for a low price early, then adapt when the seller becomes firm.",
        "typical_steps": ["Offer low", "Observe seller firmness", "Concede if needed"],
        "stage_weights": {
            "initial": [0.70, 0.10, 0.15, 0.05],
            "firm_response": [0.35, 0.25, 0.32, 0.08],
            "walkaway_response": [0.24, 0.28, 0.38, 0.10]
        },
        "adaptation_rules": [
            {
                "when": {"seller_intent": "firm"},
                "target_stage": "firm_response",
                "summary": "test firm seller rule"
            },
            {
                "when": {"seller_intent": "walkaway_risk"},
                "target_stage": "walkaway_response",
                "summary": "test walkaway seller rule"
            }
        ]
    },
    "URGENT_PURCHASE_WITH_BUDGET_CEILING": {
        "description": "Prioritize buying quickly, but only below a maximum acceptable budget ceiling.",
        "natural_language_intent": "Move closer to the seller price when the item is needed urgently, but never above the ceiling.",
        "typical_steps": ["Signal serious purchase intent", "Move closer", "Stop at ceiling"],
        "stage_weights": {
            "initial": [0.24, 0.16, 0.48, 0.12],
            "above_ceiling_defense": [0.58, 0.14, 0.20, 0.08]
        },
        "adaptation_rules": [
            {
                "when": {"seller_offer_above_ceiling": True},
                "target_stage": "above_ceiling_defense",
                "summary": "test budget ceiling rule"
            }
        ]
    }
}

BUYER_STRATEGY_MACRO_CLUSTERS = {
    "PRICE_GAIN": ["PRICE_GAIN_BUT_DEAL_AWARE"],
    "FAST_PURCHASE": ["URGENT_PURCHASE_WITH_BUDGET_CEILING"]
}
""",
        encoding="utf-8",
    )
    return objective_file


def test_load_hmod_scenarios():
    scenarios = load_scenarios(SCENARIO_FILE)
    assert len(scenarios) >= 8
    assert all(len(s.static_w) == 3 for s in scenarios)
    assert all(s.buyer_intent_id for s in scenarios)
    assert {s.buyer_intent_id for s in scenarios} >= {
        "AGGRESSIVE_SAVINGS_THEN_RECOVERY",
        "URGENT_GIFT_WITH_HARD_CEILING",
        "QUALITY_RISK_THEN_PRICE_PUSH",
        "BACKUP_OPTION_PRESSURE_CONTROL",
        "SCARCITY_ADAPTIVE_BUYER",
        "BUDGET_LOCKED_FLEXIBLE_TIMING",
        "LONG_HAGGLE_FATIGUE_CONTROL",
    }
    assert {s.drift_mode for s in scenarios} >= {
        "static_no_drift",
        "gradual_firming",
        "abrupt_final_offer",
        "frustrated_walkaway",
    }


def test_gradual_firming_sets_drift():
    scenario = [s for s in load_scenarios(SCENARIO_FILE) if s.drift_mode == "gradual_firming"][0]
    simulator = DynamicSellerNegotiationSimulator(scenario)
    state = {
        "dialogue_context": [{"role": "assistant", "content": "I can do $130."}],
        "pred_goal": {"strategy": "counter", "price": 130},
        "last_buyer_price": 130,
        "turn_id": 0,
    }
    for turn in range(3):
        state["turn_id"] = turn
        simulator.respond(state)
    trace = simulator.get_trace()
    assert trace["t_drift"] is not None
    assert trace["drift_trigger_reason"]


def test_metrics_gsr_t2da_cvr():
    gsr, parts = compute_gsr(
        {"deal": True, "deal_price": 150},
        min_acceptable_price=160,
        turn_count=5,
        turn_limit=8,
        price_direction="at_most",
    )
    assert gsr == 1
    assert parts["price_ok"] is True

    no_gsr, _ = compute_gsr(
        {"deal": True, "deal_price": 170},
        min_acceptable_price=160,
        turn_count=5,
        turn_limit=8,
        price_direction="at_most",
    )
    assert no_gsr == 0

    t2da = compute_t2da(
        weight_trace=[
            {"turn": 0, "weight_vector": [0.6, 0.1, 0.2, 0.1]},
            {"turn": 2, "weight_vector": [0.3, 0.2, 0.4, 0.1]},
        ],
        t_drift=2,
        turn_limit=8,
        expected_weight_shift={"sl_ratio": "down", "deal_rate": "up"},
    )
    assert t2da["adapted"] is True
    assert t2da["t2da"] == 0

    cvr = compute_cvr([
        {"blocked_violation": True, "actual_violation": False},
        {"blocked_violation": False, "actual_violation": True},
    ])
    assert cvr["blocked_cvr"] == 0.5
    assert cvr["actual_cvr"] == 0.5


def test_buyer_objective_library_maps_ambiguous_intent(tmp_path):
    objective_file = _write_objective_file(tmp_path)
    library = BuyerObjectiveLibrary.from_file(str(objective_file))

    mapping = library.map_objective(
        macro_goal="Buy fast but never above budget ceiling.",
        scenario_objective_id="URGENT_PURCHASE_WITH_BUDGET_CEILING",
        fallback_weight=[0.5, 0.2, 0.2, 0.1],
    )

    assert mapping.objective_id == "URGENT_PURCHASE_WITH_BUDGET_CEILING"
    assert mapping.cluster == "FAST_PURCHASE"
    assert mapping.source == "buyer_strategy_objective"
    assert round(sum(mapping.weight_vector), 6) == 1
    assert mapping.weight_vector[2] > mapping.weight_vector[0]
    # stage_weights are collapsed to the 3-D objective space on load
    # ([0.35, 0.25, 0.32, 0.08] drops avg_turn and renormalises).
    assert [
        round(x, 2)
        for x in library.get("PRICE_GAIN_BUT_DEAL_AWARE").stage_weights["firm_response"]
    ] == [0.38, 0.27, 0.35]


def test_llm_reflection_json_parsing_and_weight_normalization():
    parsed = parse_reflection_json(
        """```json
        {
          "detected_seller_intent": "firm",
          "local_objective": "Recover deal without overpaying",
          "w_t": {"sl_ratio": 2, "fairness": 1, "deal_rate": 1, "avg_turn": 0}
        }
        ```"""
    )
    weight = coerce_reflection_weight(parsed["w_t"])

    assert parsed["detected_seller_intent"] == "firm"
    assert [round(x, 2) for x in weight] == [0.5, 0.25, 0.25]


def test_llm_reflector_defaults_to_deepinfra_env(monkeypatch):
    monkeypatch.delenv("HMOD_LLM_API_KEY", raising=False)
    monkeypatch.delenv("HMOD_LLM_MODEL", raising=False)
    monkeypatch.delenv("HMOD_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    monkeypatch.setenv("DEEPINFRA_MODEL", "test/model")
    monkeypatch.setenv("DEEPINFRA_BASE_URL", "https://deepinfra.test/v1/openai")

    reflector = LLMWeightReflector()

    assert reflector.api_key == "test-key"
    assert reflector.model == "test/model"
    assert reflector.base_url == "https://deepinfra.test/v1/openai"


def test_meta_controller_reflects_every_t_steps(tmp_path):
    objective_file = _write_objective_file(tmp_path)
    library = BuyerObjectiveLibrary.from_file(str(objective_file))
    scenario = [s for s in load_scenarios(SCENARIO_FILE) if s.drift_mode == "gradual_firming"][0]
    controller = RuleMetaController(
        objective_library=library,
        objective_id="PRICE_GAIN_BUT_DEAL_AWARE",
        reflection_horizon=2,
    )

    first = controller.select_local_weight(
        scenario=scenario,
        simulator_trace={},
        previous_weight=None,
        mode="hmod_dynamic",
        turn=0,
    )
    second = controller.select_local_weight(
        scenario=scenario,
        simulator_trace={"intent_state_by_turn": [{"turn": 0, "intent_state": "neutral"}]},
        previous_weight=first["weight_vector"],
        mode="hmod_dynamic",
        turn=1,
    )
    third = controller.select_local_weight(
        scenario=scenario,
        simulator_trace={
            "intent_state_by_turn": [{"turn": 1, "intent_state": "firm"}],
            "seller_offer_by_turn": [{"turn": 1, "price": 240}],
        },
        previous_weight=second["weight_vector"],
        mode="hmod_dynamic",
        turn=2,
    )

    assert first["reflection_step"] is True
    assert second["reflection_step"] is False
    assert third["reflection_step"] is True
    assert third["selected_objective_id"] == "PRICE_GAIN_BUT_DEAL_AWARE"
    assert third["weight_vector"] != second["weight_vector"]
    assert [round(x, 2) for x in third["weight_vector"]] == [0.38, 0.27, 0.35]


def test_eval_hmod_runner_writes_outputs(tmp_path):
    result = run_and_write(
        scenario_file=SCENARIO_FILE,
        mode="hmod_dynamic",
        num_cases=2,
        output_dir=str(tmp_path),
        audit_sample_size=2,
        judge_model="rule",
    )
    run_dir = tmp_path / result["run_dir"].split("/")[-1]
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "dialogues.jsonl").exists()
    assert (run_dir / "weight_trace.jsonl").exists()
    assert (run_dir / "violation_trace.jsonl").exists()
    assert (run_dir / "human_audit.jsonl").exists()

    with open(run_dir / "metrics.json", "r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    assert metrics["metrics"]["num_dialogues"] == 2
    assert "gsr" in metrics["metrics"]
    assert "blocked_cvr" in metrics["metrics"]
    assert metrics["human_audit_samples"] == 2


def test_eval_hmod_runner_accepts_objective_file(tmp_path):
    objective_file = _write_objective_file(tmp_path)
    result = run_and_write(
        scenario_file=SCENARIO_FILE,
        mode="hmod_dynamic",
        num_cases=1,
        output_dir=str(tmp_path),
        audit_sample_size=1,
        judge_model="rule",
        objective_file=str(objective_file),
        objective_id="PRICE_GAIN_BUT_DEAL_AWARE",
        reflection_horizon=2,
    )
    run_dir = tmp_path / result["run_dir"].split("/")[-1]
    with open(run_dir / "weight_trace.jsonl", "r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]

    assert rows
    assert rows[0]["selected_objective_id"] == "PRICE_GAIN_BUT_DEAL_AWARE"
    assert rows[0]["objective_source"] == "buyer_strategy_objective"
    assert rows[0]["reflection_horizon"] == 2
    assert result["metrics_by_objective"]


def test_eval_hmod_runner_llm_reflection_fallback_mode(tmp_path):
    result = run_and_write(
        scenario_file=SCENARIO_FILE,
        mode="hmod_dynamic",
        num_cases=1,
        output_dir=str(tmp_path),
        audit_sample_size=1,
        judge_model="rule",
        objective_file=None,
        controller_mode="llm_reflection",
        llm_api_key="",
        llm_fallback_to_rule=True,
    )
    run_dir = tmp_path / result["run_dir"].split("/")[-1]
    with open(run_dir / "weight_trace.jsonl", "r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]

    assert result["controller_mode"] == "llm_reflection"
    assert rows
    assert rows[0]["controller_mode"] == "llm_reflection_fallback_rule"
    assert rows[0]["llm_error"]


def test_hmod_training_controller_builds_dmorl_skill_library(tmp_path):
    objective_file = _write_objective_file(tmp_path)
    controller = HMODController(
        n_objectives=3,
        objective_names=["sl_ratio", "fairness", "deal_rate"],
        scenario="negotiation",
        objective_file=str(objective_file),
        objective_id="PRICE_GAIN_BUT_DEAL_AWARE",
        n_basic_skills=2,
        n_advanced_skills=2,
        skills_file=str(tmp_path / "skills.json"),
        hints_file=str(tmp_path / "hints.json"),
    )

    controller.initialize_skills(force_rediscover=True)

    assert len(controller.skill_library.basic_skills) == 2
    assert controller.skill_library.basic_skills[0]["name"] == "PRICE_GAIN_BUT_DEAL_AWARE"
    assert len(controller.skill_library.advanced_skills) == 2
    assert all(len(skill["weight_vector"]) == 3 for skill in controller.skill_library.basic_skills)
    assert (tmp_path / "skills.json").exists()


def test_hmod_training_controller_reflects_dynamic_weight(tmp_path):
    objective_file = _write_objective_file(tmp_path)
    controller = HMODController(
        n_objectives=3,
        objective_names=["sl_ratio", "fairness", "deal_rate"],
        scenario="negotiation",
        objective_file=str(objective_file),
        objective_id="PRICE_GAIN_BUT_DEAL_AWARE",
        skills_file=str(tmp_path / "skills.json"),
        hints_file=str(tmp_path / "hints.json"),
    )

    neutral = controller.get_dynamic_weight([
        {"role": "user", "content": "I am interested."},
    ])
    pressure = controller.get_dynamic_weight([
        {"role": "user", "content": "That offer is too low and I cannot go lower."},
    ])

    assert round(sum(pressure), 6) == 1
    assert pressure[0] < neutral[0]
    assert pressure[2] > neutral[2]


def test_score_human_audit_rows():
    score = score_rows([
        {
            "llm_judge": {"deal": True, "success": True, "deal_price": 100},
            "human_deal": True,
            "human_success": True,
            "human_deal_price": 110,
        },
        {
            "llm_judge": {"deal": False, "success": False, "deal_price": None},
            "human_deal": True,
            "human_success": False,
            "human_deal_price": None,
        },
    ])
    assert score["total_samples"] == 2
    assert score["labeled_samples"] == 2
    assert score["deal_agreement"] == 0.5
    assert score["success_agreement"] == 1.0
    assert score["avg_abs_price_error"] == 10
