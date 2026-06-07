"""Resisting strategies from Dutt et al. (2021), reused by TRIP user simulators.

Tables 7/13/14 of the TRIP paper map these strategies into the comprehensive
user-simulator prompt. We keep CB (seller) and P4G (persuadee) variants
separately so the persona-aware prompt builder can pick the correct block.
"""

CB_RESISTING_STRATEGIES = [
    ("Source Derogation",
     "Attacks the other party or questions the item."),
    ("Counter Argument",
     "Provides a non-personal argument or factual response to refute a previous "
     "claim or to justify a new claim."),
    ("Personal Choice",
     "Provides a personal reason for disagreeing with the current situation or "
     "chooses to agree only if a specific condition is met."),
    ("Information Inquiry",
     "Requests for clarification or asks additional information about the item "
     "or situation."),
    ("Self Pity",
     "Provides a reason (meant to elicit sympathy) for disagreeing with the "
     "current terms."),
    ("Hesitance",
     "Stalls for time and is hesitant to commit; specifically, seeks to further "
     "the conversation and give the other party a chance to make a better offer."),
    ("Self-assertion",
     "Asserts a new claim or refutes a previous claim with an air of "
     "finality and confidence."),
    ("Others",
     "Does not explicitly foil the negotiation attempts."),
]


P4G_RESISTING_STRATEGIES = [
    ("Donate", "Show your willingness to donate."),
    ("Source Derogation", "Attack or doubt the organisation's credibility."),
    ("Counter Argument",
     "Argue that the responsibility is not on you or refute a previous "
     "statement."),
    ("Personal Choice",
     "Save face by asserting a personal preference such as your own choice of "
     "charity or your own donation plans."),
    ("Information Inquiry",
     "Ask for factual information about the organisation, either for "
     "clarification or as an attempt to stall."),
    ("Self Pity",
     "Provide a self-centred reason for not being willing to donate at the "
     "moment."),
    ("Hesitance",
     "Stall the conversation by stating you would donate later or that you are "
     "currently unsure."),
    ("Self-assertion",
     "Explicitly refuse to donate without providing a personal reason."),
    ("Others", "Do not explicitly foil the persuasion attempts."),
]


def format_strategy_block(strategies):
    """Render a strategy list as a numbered block for use inside a system prompt."""
    lines = []
    for i, (name, desc) in enumerate(strategies, start=1):
        lines.append(f'{i}. "{name}": {desc}')
    return "\n".join(lines)
