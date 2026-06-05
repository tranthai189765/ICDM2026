"""Tests for the two-agent H-MOD controller (intent detector + high policy)."""

import json
from unittest.mock import MagicMock

from hmod.high_policy import LLMHighPolicy, parse_allocation_to_weight
from hmod.hints import HintStore
from hmod.intent_detector import LLMIntentDetector, build_intent_fewshot
from hmod.metrics import compute_t2da
from hmod.scenario import load_scenarios
from hmod.two_agent_controller import TwoAgentMetaController

SCENARIO_FILE = "config/scenario/hmod_buyer_drift_scenarios.yaml"


# ── high-policy NL allocation parsing ────────────────────────────────────────
def test_parse_allocation_from_w_t_dict():
    w = parse_allocation_to_weight({"w_t": {"sl_ratio": 0.2, "fairness": 0.3, "deal_rate": 0.5}})
    assert len(w) == 3 and abs(sum(w) - 1.0) < 1e-6


def test_parse_allocation_nl_fallback():
    parsed = {"allocation": "In the short term, focus 20% on sl_ratio, 30% on fairness, 50% on deal_rate"}
    w = parse_allocation_to_weight(parsed)
    assert abs(sum(w) - 1.0) < 1e-6 and w[2] > w[0] > 0


def test_high_policy_generate_returns_normalised_vector():
    refl = MagicMock()
    refl._complete.return_value = json.dumps({
        "allocation": "focus 10% on sl_ratio, 20% on fairness, 70% on deal_rate",
        "w_t": {"sl_ratio": 0.1, "fairness": 0.2, "deal_rate": 0.7},
        "reason": "final offer within budget",
    })
    hp = LLMHighPolicy(refl)
    out = hp.generate("goal", [], "final_offer", None, 4,
                      {"max_acceptable_price": 100, "target_price": 60, "turn_limit": 8}, {})
    assert abs(sum(out["weight_vector"]) - 1.0) < 1e-6 and out["weight_vector"][2] > 0.5


def test_high_policy_injects_intent_adaptation_guideline():
    refl = MagicMock()
    refl._complete.return_value = json.dumps({"w_t": {"sl_ratio": 0.1, "fairness": 0.2, "deal_rate": 0.7}})
    hp = LLMHighPolicy(refl)
    hp.generate("goal", [], "final_offer", None, 4,
                {"max_acceptable_price": 100, "target_price": 60, "turn_limit": 8}, {})
    user_content = refl._complete.call_args[0][0][1]["content"]
    assert "weight_adaptation_guideline" in user_content
    assert "deal_rate" in user_content  # guideline tells it to raise deal_rate on final_offer


# ── intent detector ──────────────────────────────────────────────────────────
def test_detector_parses_drift_and_intent():
    refl = MagicMock()
    refl._complete.return_value = json.dumps({
        "drift_detected": True, "current_intent": "final_offer", "reason": "final price stated"})
    det = LLMIntentDetector(refl, fewshot=[])
    out = det.detect([{"role": "user", "content": "My final price is $284."}], 3, "neutral")
    assert out["drift_detected"] and out["current_intent"] == "final_offer"


def test_detector_clamps_unknown_intent_to_previous():
    refl = MagicMock()
    refl._complete.return_value = json.dumps({"drift_detected": True, "current_intent": "bogus"})
    det = LLMIntentDetector(refl)
    out = det.detect([], 2, "firm")
    assert out["current_intent"] == "firm"


def test_build_fewshot_covers_drift_intents():
    fs = build_intent_fewshot(SCENARIO_FILE)
    intents = {e["label"]["current_intent"] for e in fs}
    assert intents & {"firm", "final_offer", "walkaway_risk"}


# ── two-agent controller (gold-intent training path) ─────────────────────────
def test_controller_turn0_then_gold_drift_triggers_regen():
    sc = load_scenarios(SCENARIO_FILE)[0]
    det = MagicMock()
    det.detect.return_value = {"drift_detected": False, "current_intent": "neutral", "reason": ""}
    hp = MagicMock()
    hp.generate.return_value = {"weight_vector": [0.5, 0.3, 0.2], "allocation_text": "a",
                                "reason": "r", "current_intent": "x", "raw_response": ""}
    ctrl = TwoAgentMetaController(det, hp, use_gold_intent=True)

    out0 = ctrl.select_local_weight(sc, {"intent_state_by_turn": []}, None,
                                    "hmod_dynamic", turn=0, dialogue_history=[])
    assert out0["reflection_step"] is True and hp.generate.call_count == 1

    out1 = ctrl.select_local_weight(
        sc, {"intent_state_by_turn": [{"turn": 0, "intent_state": "neutral"}]},
        out0["weight_vector"], "hmod_dynamic", turn=1,
        dialogue_history=[{"role": "user", "content": "hi"}])
    assert out1["reflection_step"] is False and hp.generate.call_count == 1  # carried

    out2 = ctrl.select_local_weight(
        sc, {"intent_state_by_turn": [{"turn": 1, "intent_state": "final_offer"}]},
        out1["weight_vector"], "hmod_dynamic", turn=2,
        dialogue_history=[{"role": "user", "content": "My final price is $284."}])
    assert out2["reflection_step"] is True and hp.generate.call_count == 2
    assert out2["intent_state"] == "final_offer"
    # detector scored on each non-zero turn
    assert len(ctrl.detector_records) == 2
    assert ctrl.detector_records[-1]["gold_intent"] == "final_offer"


def test_controller_inference_uses_detector():
    sc = load_scenarios(SCENARIO_FILE)[0]
    det = MagicMock()
    det.detect.return_value = {"drift_detected": True, "current_intent": "firm", "reason": ""}
    hp = MagicMock()
    hp.generate.return_value = {"weight_vector": [0.3, 0.3, 0.4], "allocation_text": "",
                                "reason": "", "current_intent": "", "raw_response": ""}
    ctrl = TwoAgentMetaController(det, hp, use_gold_intent=False)
    ctrl.select_local_weight(sc, {"intent_state_by_turn": []}, None, "hmod_dynamic", 0, [])
    out1 = ctrl.select_local_weight(sc, {"intent_state_by_turn": []}, [0.3, 0.3, 0.4],
                                    "hmod_dynamic", 1, [{"role": "user", "content": "firm"}])
    # detector said drift -> regen, intent from detector
    assert out1["reflection_step"] is True and out1["intent_state"] == "firm"


# ── review hint store ────────────────────────────────────────────────────────
def test_review_drops_hint_after_two_consecutive_proposals(tmp_path):
    store = HintStore(path=str(tmp_path / "h.json"), max_hints=10)
    store.review_update([], ["A", "B"], {})
    assert set(store.hints) == {"A", "B"}
    store.review_update(["A"], [], {})          # streak 1 -> kept
    assert "A" in store.hints
    store.review_update(["A"], [], {})          # streak 2 -> dropped
    assert "A" not in store.hints and "B" in store.hints


def test_review_streak_resets_when_not_proposed(tmp_path):
    store = HintStore(path=str(tmp_path / "h.json"))
    store.review_update([], ["A"], {})
    store.review_update(["A"], [], {})          # streak 1
    store.review_update([], [], {})             # reset
    store.review_update(["A"], [], {})          # streak 1 again, never 2-in-a-row
    assert "A" in store.hints


# ── T2DA = magnitude >= 0.25 AND relaxed direction ───────────────────────────
def test_t2da_relaxed_allows_flat_objective():
    wt = [{"turn": 0, "weight_vector": [0.6, 0.2, 0.2]},
          {"turn": 1, "weight_vector": [0.3, 0.2, 0.5]}]   # sl down, deal up, fairness flat
    out = compute_t2da(wt, t_drift=1, turn_limit=8,
                       expected_weight_shift={"sl_ratio": "down", "fairness": "up", "deal_rate": "up"})
    assert out["adapted"] is True and out["t2da"] == 0


def test_t2da_penalises_opposite_move():
    wt = [{"turn": 0, "weight_vector": [0.6, 0.2, 0.2]},
          {"turn": 1, "weight_vector": [0.8, 0.1, 0.1]}]   # sl_ratio UP = opposite of expected down
    out = compute_t2da(wt, t_drift=1, turn_limit=8,
                       expected_weight_shift={"sl_ratio": "down", "deal_rate": "up"})
    assert out["adapted"] is False


def test_t2da_subthreshold_shift_not_adapted():
    wt = [{"turn": 0, "weight_vector": [0.50, 0.30, 0.20]},
          {"turn": 1, "weight_vector": [0.55, 0.28, 0.17]}]  # ||Δ||₁ = 0.10 < 0.25
    out = compute_t2da(wt, t_drift=1, turn_limit=8)
    assert out["adapted"] is False and out["t2da"] == 8 - 1 + 1  # penalty
