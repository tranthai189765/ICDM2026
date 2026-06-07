"""Deterministic persona pool reproducing TRIP's 20 categories x 2 variants.

The paper (Zhang et al., 2024, Appendix A.1.1) uses GPT-4 to generate 300
persona descriptions over 20 Big-Five x Decision-Making categories. For
reproducibility we instead generate 40 deterministic descriptions (2 per
category) using hand-crafted templates derived from the paper's persona type
definitions. The category distribution stays balanced, matching the paper's
"frequency-based initial sampling distribution".
"""

from __future__ import annotations

from typing import Dict, List


BIG_FIVE: List[str] = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

DECISION_STYLES: List[str] = [
    "directive",
    "analytical",
    "conceptual",
    "behavioral",
]


# Concise behavioural descriptors used to compose persona paragraphs.
_BIG_FIVE_DESC: Dict[str, str] = {
    "openness": (
        "curious, imaginative, and willing to try new arrangements; you stay "
        "open to creative proposals but reject pressure tactics"
    ),
    "conscientiousness": (
        "organized, disciplined, and goal-oriented; you reason carefully and "
        "expect every claim to be supported by concrete facts"
    ),
    "extraversion": (
        "talkative, energetic, and assertive; you respond quickly and rarely "
        "back down without negotiating extra value"
    ),
    "agreeableness": (
        "cooperative, trusting, and conflict-averse; you look for win-win "
        "compromises but resist offers that feel one-sided"
    ),
    "neuroticism": (
        "easily stressed and emotionally reactive; you ask for reassurance, "
        "second-guess offers, and pull back when you feel rushed"
    ),
}


_DECISION_DESC: Dict[str, str] = {
    "directive": (
        "you prefer fast, authoritative decisions and dislike long deliberation"
    ),
    "analytical": (
        "you analyse every detail before committing and request supporting "
        "information for any new claim"
    ),
    "conceptual": (
        "you take a broad, long-term view and weigh many alternatives before "
        "settling on a position"
    ),
    "behavioral": (
        "you value other people's opinions, seek consensus, and avoid open "
        "conflict whenever possible"
    ),
}


_VARIANT_FLAVOURS: List[str] = [
    "You often signal your stance early so the other party knows where you stand.",
    "You stay measured at first and only reveal your true limits late in the dialogue.",
]


def _compose_description(big_five: str, decision: str, variant: int) -> str:
    return (
        f"You are characterised by {big_five}: {_BIG_FIVE_DESC[big_five]}. "
        f"Your decision-making style is {decision}, which means {_DECISION_DESC[decision]}. "
        f"{_VARIANT_FLAVOURS[variant % len(_VARIANT_FLAVOURS)]}"
    )


def build_persona_pool(size: int = 40) -> List[Dict[str, str]]:
    """Return `size` personas balanced across the 20 BigFive x Decision combos.

    Each persona is a dict with `id`, `big_five`, `decision`, and `description`.
    Size is rounded up to a multiple of 20 to keep category balance.
    """
    n_categories = len(BIG_FIVE) * len(DECISION_STYLES)
    variants_per_cat = max(1, (size + n_categories - 1) // n_categories)

    personas: List[Dict[str, str]] = []
    for big_five in BIG_FIVE:
        for decision in DECISION_STYLES:
            for variant in range(variants_per_cat):
                personas.append({
                    "id": f"{big_five}-{decision}-v{variant + 1}",
                    "big_five": big_five,
                    "decision": decision,
                    "description": _compose_description(big_five, decision, variant),
                })
    return personas[:size]
