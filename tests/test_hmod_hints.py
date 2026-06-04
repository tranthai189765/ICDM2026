"""Tests for H-MOD hint training (HintStore, distiller, hint injection)."""

import json
from unittest.mock import MagicMock

from hmod.hint_distiller import LLMHintDistiller, build_episode_digest
from hmod.hints import METRIC_GLOSSARY, HintStore
from hmod.runner import HMODEvaluator


def test_hintstore_dedup_persist_and_reload(tmp_path):
    path = tmp_path / "hints.json"
    store = HintStore(path=str(path), max_hints=5)
    assert store.is_empty()
    assert store.as_text() is None

    store.update(
        [
            "Raise deal_rate on final_offer if price within ceiling.",
            "   ",  # blank dropped
            "Raise deal_rate on final_offer if price within ceiling.",  # dup dropped
            "Keep CVR at zero: never offer above ceiling.",
        ],
        metrics={"gsr": 0.5},
    )
    assert len(store.hints) == 2
    assert store.meta["iterations"] == 1

    reloaded = HintStore(path=str(path))
    assert reloaded.hints == store.hints
    # persisted file carries the glossary for provenance
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "metric_glossary" in data and data["hints"] == store.hints


def test_hintstore_caps_to_max_hints(tmp_path):
    store = HintStore(path=str(tmp_path / "h.json"), max_hints=3)
    store.update([f"hint number {i}" for i in range(10)])
    assert len(store.hints) == 3


def test_build_episode_digest_orders_failures_first():
    dialogues = [
        {
            "drift_mode": "gradual_firming", "gsr": 1,
            "t2da": {"t2da": 2}, "cvr": {"cvr": 0.0},
            "judge_result": {"deal": True, "deal_price": 90},
            "gsr_components": {"constraint_price": 100},
            "weight_trace": [
                {"intent_state": "neutral", "weight_vector": [0.5, 0.3, 0.2]},
                {"intent_state": "firm", "weight_vector": [0.2, 0.3, 0.5]},
            ],
        },
        {
            "drift_mode": "abrupt_final_offer", "gsr": 0,
            "t2da": {"t2da": 8}, "cvr": {"cvr": 0.0},
            "judge_result": {"deal": False, "deal_price": None},
            "gsr_components": {"constraint_price": 80},
            "weight_trace": [{"intent_state": "neutral", "weight_vector": [0.6, 0.2, 0.2]}],
        },
    ]
    digest = build_episode_digest(dialogues, max_episodes=10)
    assert [d["gsr"] for d in digest] == [0, 1]  # failures first
    assert digest[0]["seller_intents_seen"] == ["neutral"]
    assert digest[1]["w_start"] == [0.5, 0.3, 0.2]
    assert digest[1]["w_end"] == [0.2, 0.3, 0.5]


def test_distiller_parses_hints_and_includes_glossary():
    reflector = MagicMock()
    reflector._complete.return_value = json.dumps({
        "analysis": "final-offer episodes adapted too slowly",
        "hints": [
            "On final_offer within ceiling, push deal_rate high at once.",
            "Do not raise sl_ratio after walkaway_risk.",
        ],
    })
    distiller = LLMHintDistiller(reflector, max_hints=5)
    hints = distiller.distill(["old hint"], [{"gsr": 0}], {"gsr": 0.5}, epoch=1)
    assert len(hints) == 2 and all(isinstance(h, str) for h in hints)

    messages = reflector._complete.call_args[0][0]
    user_content = messages[1]["content"]
    assert METRIC_GLOSSARY[:30] in user_content


def test_hint_provider_feeds_controller_experience(tmp_path):
    store = HintStore(path=str(tmp_path / "h.json"), max_hints=5)
    store.update(["Never exceed the buyer ceiling (CVR must stay zero)."])

    evaluator = HMODEvaluator(
        mode="hmod_dynamic",
        controller_mode="llm_reflection",
        llm_fallback_to_rule=True,
        hint_provider=store.provider(),
    )
    # the composed experience_provider should surface the hint text
    provider = evaluator.meta_controller.experience_provider
    assert provider is not None
    text = provider("any macro goal", "gradual_firming")
    assert "buyer ceiling" in text
