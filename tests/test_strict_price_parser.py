"""Tests for the tightened price parser in PPDPP/ppdpp_rewards.py."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PPDPP_DIR = os.path.join(ROOT, "PPDPP")
for path in (ROOT, PPDPP_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from ppdpp_rewards import (  # noqa: E402  (path adjusted above)
    PRICE_COMMITTING_ACTIONS,
    bounded_price_from_text,
    compute_price_objectives,
    extract_committed_prices,
)


CASE = {"buyer_price": 100.0, "seller_price": 200.0}


def test_confirm_not_in_committing_actions():
    assert "confirm" not in PRICE_COMMITTING_ACTIONS
    assert "inquire" not in PRICE_COMMITTING_ACTIONS
    assert "inform" not in PRICE_COMMITTING_ACTIONS
    assert "propose" in PRICE_COMMITTING_ACTIONS
    assert "counter" in PRICE_COMMITTING_ACTIONS
    assert "agree" in PRICE_COMMITTING_ACTIONS


def test_extract_committed_requires_context():
    assert extract_committed_prices("The Nishiki Altron 7000 is great.") == []
    assert extract_committed_prices("It was the 1970 model in mint condition.") == []


def test_extract_committed_accepts_dollar_marker():
    assert extract_committed_prices("Final offer is $145.") == [145.0]


def test_extract_committed_accepts_verb_context():
    assert extract_committed_prices("I will pay 150 for it.") == [150.0]
    assert extract_committed_prices("I can come down to 125.") == [125.0]
    assert extract_committed_prices("How about I offer 110?") == [110.0]


def test_extract_committed_ignores_year():
    assert extract_committed_prices("I will accept 150 for this 1970 model.") == [150.0]


def test_bounded_price_respects_context():
    text = "The 2018 vintage Nishiki 7000 is yours."
    assert bounded_price_from_text(text, CASE) is None


def test_bounded_price_accepts_in_range_committed():
    text = "I can sell it to you for 165 dollars."
    assert bounded_price_from_text(text, CASE) == 165.0


def test_compute_price_objectives_skips_confirm():
    text = "Final offer 150."
    sl, fairness, price = compute_price_objectives(CASE, text, action="confirm")
    assert sl == 0.0 and fairness == 0.0
    assert price == 150.0


def test_compute_price_objectives_credits_counter():
    text = "I will counter at $150."
    sl, fairness, price = compute_price_objectives(CASE, text, action="counter")
    assert price == 150.0
    assert sl != 0.0
