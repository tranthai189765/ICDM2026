"""PBTP wiring: persona-aware user-prompt + per-episode persona sampling.

Monkey-patches PPDPP/env.message_format['cb'] with a persona-aware seller
prompt that follows TRIP Table 13, then wraps the standard Env so that each
reset() draws a persona from the population pool.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_PPDPP_DIR = os.path.join(_REPO_ROOT, "PPDPP")
for path in (_REPO_ROOT, _PPDPP_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import env as _ppdpp_env  # type: ignore  -- PPDPP/env.py
from env import Env  # type: ignore  -- PPDPP/env.py
from prompt import CBMessages as _OriginalCBMessages  # type: ignore -- PPDPP/prompt.py

from trip_baseline.resisting import (
    CB_RESISTING_STRATEGIES,
    P4G_RESISTING_STRATEGIES,
    format_strategy_block,
)


_CB_PERSONA_INSTRUCTIONS = (
    "You must follow the instructions below during chat:\n"
    "1. Your utterances and bargain behaviour must strictly follow your persona; "
    "vary your wording and avoid repeating yourself verbatim.\n"
    "2. You may flexibly adjust your target price based on your persona and the conversation."
)


def _build_persona_user_prompt_cb(case: Dict[str, Any], persona: Dict[str, str]) -> List[Dict[str, str]]:
    persona_text = persona.get("description", "") if persona else ""
    strategy_block = format_strategy_block(CB_RESISTING_STRATEGIES)
    system_text = (
        "Now enter the role-playing mode. In the following conversation, you will play as a "
        "seller in a price bargaining game."
    )
    user_intro = (
        f"Your persona: {persona_text}\n"
        f"{_CB_PERSONA_INSTRUCTIONS}\n"
        "Your Response Strategy (resisting strategies):\n"
        f"{strategy_block}\n"
        f"You are the seller who is trying to sell the {case['item_name']} with the initial price of "
        f"{case['seller_price']}. Product description: {case.get('seller_item_description', '')}.\n"
        "Please reply with only one short and succinct sentence. Are you ready to play the game?"
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "Buyer", "content": user_intro},
        {"role": "Seller", "content": "Yes, I'm ready to play the game!"},
        {"role": "Buyer", "content": f"Hi, how much is the {case['item_name']}?"},
        {"role": "Seller", "content": (
            f"Hi, this is a good {case['item_name']} and its price is {case['seller_price']}."
        )},
    ]


class PersonaHolder:
    """Mutable per-process slot the patched message builder reads from."""

    def __init__(self) -> None:
        self.current: Optional[Dict[str, str]] = None


_PERSONA_HOLDER = PersonaHolder()


def _persona_aware_cb_messages(case, role, conversation, action=None):
    """Drop-in replacement for PPDPP.prompt.CBMessages.

    Mirrors the original behaviour for system/critic prompts; for the user
    (seller) prompt we inject the current persona + resisting strategies and
    then append the actual dialogue history so the LLM continues from there.
    """
    if role != "user":
        return _OriginalCBMessages(case, role, conversation, action)

    persona = _PERSONA_HOLDER.current
    base = _build_persona_user_prompt_cb(case, persona) if persona else _OriginalCBMessages(
        case, role, conversation, action,
    )
    if persona is None:
        return base

    # Append the dialogue history after the seeded preamble, preserving the
    # alternating Buyer/Seller turn structure PPDPP expects downstream.
    msgs = list(base)
    for turn in conversation:
        msgs.append({"role": turn.get("role", ""), "content": turn.get("content", "")})
    return msgs


def install_persona_aware_message_format(data_name: str) -> None:
    """Patch the module-level message_format dict used by PPDPP/env.py."""
    if data_name == "cb":
        _ppdpp_env.message_format["cb"] = _persona_aware_cb_messages
    # ESC / CIMA TRIP variants can be added analogously when needed.


def restore_default_message_format(data_name: str) -> None:
    if data_name == "cb":
        _ppdpp_env.message_format["cb"] = _OriginalCBMessages


class TRIPEnv(Env):
    """Env wrapper that samples a persona per episode for PBTP."""

    def __init__(
        self,
        args,
        dataset,
        mode,
        env_model=None,
        env_tokenizer=None,
        persona_pool: Optional[List[Dict[str, str]]] = None,
        rng: Optional[random.Random] = None,
        persona_distribution: Optional[List[float]] = None,
    ) -> None:
        super().__init__(args, dataset, mode, env_model=env_model, env_tokenizer=env_tokenizer)
        self.persona_pool = list(persona_pool or [])
        self._rng = rng or random.Random(getattr(args, "seed", 1))
        self.persona_distribution = persona_distribution
        self.last_persona: Optional[Dict[str, str]] = None
        install_persona_aware_message_format(self.args.data_name)

    def _sample_persona(self) -> Optional[Dict[str, str]]:
        if not self.persona_pool:
            return None
        if self.persona_distribution:
            return self._rng.choices(
                self.persona_pool,
                weights=self.persona_distribution[: len(self.persona_pool)],
                k=1,
            )[0]
        return self._rng.choice(self.persona_pool)

    def reset(self):  # type: ignore[override]
        persona = self._sample_persona()
        self.last_persona = persona
        _PERSONA_HOLDER.current = persona
        return super().reset()
