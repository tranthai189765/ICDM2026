"""Regression tests for the tightened hmod.judge.rule_judge_deal."""

from hmod.judge import rule_judge_deal


def _seller(text):
    return {"role": "user", "content": text}


def _buyer(text):
    return {"role": "assistant", "content": text}


def test_seller_explicit_accept_closes_deal():
    dialogue = [
        _buyer("Can we do $70?"),
        _seller("Deal, the bike is yours for $70."),
    ]
    result = rule_judge_deal(dialogue)
    assert result["deal"] is True
    assert result["deal_price"] == 70.0


def test_descriptive_good_deal_does_not_close():
    dialogue = [
        _buyer("Will you sell at $55?"),
        _seller("That would be a good deal but I cannot accept $55."),
    ]
    result = rule_judge_deal(dialogue)
    assert result["deal"] is False


def test_buyer_accepting_alone_is_no_deal():
    dialogue = [
        _seller("Lowest I can do is $120."),
        _buyer("I accept your offer of $120."),
        _seller("Sorry, I am not willing to go that low."),
    ]
    result = rule_judge_deal(dialogue)
    assert result["deal"] is False


def test_seller_reject_in_last_turn_vetoes_prior_accept():
    dialogue = [
        _buyer("Will you take $40?"),
        _seller("Deal, the chair is yours for $40."),
        _buyer("Actually I meant $30."),
        _seller("No, that is too low, I have to decline."),
    ]
    result = rule_judge_deal(dialogue)
    assert result["deal"] is False


def test_seller_accept_after_negotiation_uses_last_price():
    dialogue = [
        _buyer("How about $250?"),
        _seller("Lowest I can do is $260."),
        _buyer("$255 final."),
        _seller("Sold, the chair is yours for $255."),
    ]
    result = rule_judge_deal(dialogue)
    assert result["deal"] is True
    assert result["deal_price"] == 255.0


def test_great_deal_is_descriptive_not_acceptance():
    dialogue = [
        _buyer("$1900?"),
        _seller("$1900 would be a great deal but I cannot accept."),
    ]
    result = rule_judge_deal(dialogue)
    assert result["deal"] is False


def test_no_seller_turn_is_no_deal():
    dialogue = [_buyer("Hello?")]
    result = rule_judge_deal(dialogue)
    assert result["deal"] is False
