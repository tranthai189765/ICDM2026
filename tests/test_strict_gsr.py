"""Tests for the strict GSR rule: deal-cuoi-quyet-dinh, mid-turn overshoot
does not veto if seller closes within ceiling."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PPDPP_DIR = os.path.join(ROOT, "PPDPP")
for path in (PPDPP_DIR, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from hmod.judge import rule_judge_deal  # noqa: E402


def _seller(text):
    return {"role": "user", "content": text}


def _buyer(text):
    return {"role": "assistant", "content": text}


def _gsr(deal_success, deal_price, max_price, turns, turn_limit):
    """Replicates Env.get_episode_record's GSR rule (post-tighten)."""
    if not deal_success:
        return 0
    if max_price is None:
        price_ok = True
    else:
        price_ok = deal_price is not None and float(deal_price) <= float(max_price)
    return int(bool(deal_success and price_ok and turns <= turn_limit))


def test_final_deal_within_ceiling_credits_gsr():
    dialogue = [
        _buyer("How about $250?"),
        _seller("That is too high above my budget."),
        _buyer("Final $190."),
        _seller("Deal at $190."),
    ]
    res = rule_judge_deal(dialogue)
    assert res["deal"] is True
    assert res["deal_price"] == 190.0
    assert _gsr(True, res["deal_price"], max_price=200.0, turns=4, turn_limit=8) == 1


def test_mid_turn_overshoot_does_not_veto_final_ok_deal():
    """Buyer briefly overshoots private ceiling at mid-turn, but seller
    eventually closes within the ceiling. GSR must still credit the goal."""
    dialogue = [
        _buyer("How about $250?"),  # over ceiling 200 - mid-turn overshoot
        _seller("Still too high."),
        _buyer("Final offer $180."),
        _seller("I accept your offer of $180."),
    ]
    res = rule_judge_deal(dialogue)
    assert res["deal"] is True
    assert res["deal_price"] == 180.0
    assert _gsr(True, res["deal_price"], max_price=200.0, turns=4, turn_limit=8) == 1


def test_final_deal_above_ceiling_blocks_gsr():
    dialogue = [
        _buyer("$190?"),
        _seller("I can come down to $220."),
        _buyer("OK $220."),
        _seller("Deal at $220."),
    ]
    res = rule_judge_deal(dialogue)
    assert res["deal"] is True
    assert res["deal_price"] == 220.0
    assert _gsr(True, res["deal_price"], max_price=200.0, turns=4, turn_limit=8) == 0


def test_unknown_deal_price_blocks_gsr_when_ceiling_exists():
    """If we cannot identify the deal price but a ceiling exists, GSR=0."""
    dialogue = [
        _buyer("Lets settle this."),
        _seller("Deal."),
    ]
    res = rule_judge_deal(dialogue)
    assert res["deal"] is True
    assert res["deal_price"] is None  # nothing committed in dialogue
    assert _gsr(True, res["deal_price"], max_price=200.0, turns=2, turn_limit=8) == 0


def test_no_ceiling_allows_gsr():
    """Scenarios without a private ceiling fall back to deal_success only."""
    res = rule_judge_deal([
        _buyer("$200?"),
        _seller("Deal at $200."),
    ])
    assert res["deal"] is True
    assert _gsr(True, res["deal_price"], max_price=None, turns=2, turn_limit=8) == 1


def test_turn_limit_violation_blocks_gsr():
    res = rule_judge_deal([
        _buyer("Final $180."),
        _seller("Deal at $180."),
    ])
    assert res["deal"] is True
    assert _gsr(True, res["deal_price"], max_price=200.0, turns=10, turn_limit=8) == 0


def test_judge_extracts_dollar_price_not_year():
    dialogue = [
        _buyer("$180 for the 1970 model bike?"),
        _seller("Deal at $180."),
    ]
    res = rule_judge_deal(dialogue)
    assert res["deal"] is True
    assert res["deal_price"] == 180.0


def test_judge_extracts_verb_context_price():
    """Seller's "I will sell it to you for $X" is a whitelisted accept
    pattern, so this closes a deal at $X. The 2016 in buyer's text is a
    year and must not be mistaken for a price.
    """
    dialogue = [
        _buyer("I want this 2016 model phone."),
        _seller("I will sell it to you for 350."),
    ]
    res = rule_judge_deal(dialogue)
    assert res["deal"] is True
    assert res["deal_price"] == 350.0
