"""Theory-of-Mind helper for TRIP's user-aware strategic planning module.

Calls the same OpenAI-compatible endpoint already configured by PPDPP (see
PPDPP/env.py `_resolve_openai_config`) with the TRIP ToM prompt (Tables 15/16).
Results are cached by a hash of the canonical dialogue so repeated turns in
RL self-play do not re-burn LLM tokens.

A `--trip_disable_tom_llm` flag (wired in run.py) lets users substitute a
deterministic heuristic ToM string for smoke tests / cost-sensitive runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_PPDPP_DIR = os.path.join(_REPO_ROOT, "PPDPP")
for path in (_REPO_ROOT, _PPDPP_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def _import_ppdpp_openai_helpers():
    """Lazy import so unit tests can avoid pulling fastchat at import time."""
    from env import _resolve_openai_config, query_openai_model  # type: ignore

    return _resolve_openai_config, query_openai_model


_TOM_PROMPTS = {
    "cb": (
        "You are an expert in price bargain. Now given the following conversation "
        "history between a buyer and a seller, infer the SELLER's mental states "
        "(such as their internal target price and willingness to close) and the "
        "SELLER's likely next actions. Respond with two short paragraphs labelled "
        "'Mental States:' and 'Future Actions:'."
    ),
    "p4g": (
        "You are an expert in charity persuasion. Now given the following "
        "conversation history between a persuader and a persuadee, infer the "
        "PERSUADEE's mental states (e.g., willingness to donate) and likely "
        "next actions. Respond with two short paragraphs labelled "
        "'Mental States:' and 'Future Actions:'."
    ),
    "esc": (
        "You are an expert clinical counsellor. Given the following conversation "
        "between a therapist and a patient, infer the PATIENT's mental states "
        "(emotion, latent concerns) and likely next actions. Respond with two "
        "short paragraphs labelled 'Mental States:' and 'Future Actions:'."
    ),
}


def _format_dialogue(conversation: List[Dict[str, str]]) -> str:
    lines = []
    for turn in conversation:
        role = turn.get("role", "")
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _dialogue_hash(conversation: List[Dict[str, str]]) -> str:
    payload = json.dumps(conversation, ensure_ascii=False, sort_keys=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _heuristic_tom(conversation: List[Dict[str, str]], task: str) -> str:
    """Light deterministic fallback when LLM ToM is disabled."""
    user_turns = [t for t in conversation if t.get("role", "").lower() not in {"buyer", "assistant", "system"}]
    last_user = user_turns[-1]["content"] if user_turns else ""
    return (
        "Mental States: the other party appears engaged but has not committed. "
        "Future Actions: likely to probe for more information or restate their "
        f"preferred terms. Recent cue: '{last_user[:120]}'"
    )


class TheoryOfMind:
    """Bounded LRU cache + OpenAI-compatible ToM caller."""

    def __init__(
        self,
        task: str = "cb",
        cache_size: int = 256,
        use_llm: bool = True,
        max_tokens: int = 128,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        if task not in _TOM_PROMPTS:
            raise ValueError(f"Unknown ToM task {task!r}; expected one of {sorted(_TOM_PROMPTS)}")
        self.task = task
        self.use_llm = use_llm
        self.max_tokens = max_tokens
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        self._cache_size = max(1, int(cache_size))
        self._api_overrides = {
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
        }
        self.calls = 0
        self.cache_hits = 0
        self.fallback_uses = 0

    def clear(self) -> None:
        self._cache.clear()

    def describe_endpoint(self) -> str:
        """Return a redacted description of the LLM endpoint that will be used."""
        if not self.use_llm:
            return "heuristic (LLM disabled)"
        try:
            resolve_cfg, _ = _import_ppdpp_openai_helpers()
            api_key, base_url, model = resolve_cfg()
        except Exception as exc:  # noqa: BLE001
            return f"unresolved ({exc})"
        api_key = self._api_overrides["api_key"] or api_key
        base_url = self._api_overrides["base_url"] or base_url
        model = self._api_overrides["model"] or model
        key_tail = (api_key or "")[-4:] if api_key else "MISSING"
        return f"model={model} base_url={base_url or 'default-openai'} key=***{key_tail}"

    def _cache_get(self, key: str) -> Optional[str]:
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        return None

    def _cache_put(self, key: str, value: str) -> None:
        self._cache[key] = value
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def infer(self, conversation: List[Dict[str, str]]) -> str:
        if not conversation:
            return ""
        key = _dialogue_hash(conversation)
        cached = self._cache_get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        if not self.use_llm:
            text = _heuristic_tom(conversation, self.task)
            self.fallback_uses += 1
            self._cache_put(key, text)
            return text

        try:
            resolve_cfg, query = _import_ppdpp_openai_helpers()
            api_key, base_url, model = resolve_cfg()
            api_key = self._api_overrides["api_key"] or api_key
            base_url = self._api_overrides["base_url"] or base_url
            model = self._api_overrides["model"] or model
            messages = [
                {"role": "system", "content": _TOM_PROMPTS[self.task]},
                {"role": "user", "content": _format_dialogue(conversation)},
            ]
            text = query(
                api_key=api_key,
                base_url=base_url,
                messages=messages,
                model=model,
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            text = (text or "").strip()
            if not text:
                raise RuntimeError("empty ToM response")
            self.calls += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[trip-tom] fallback to heuristic: {exc}")
            text = _heuristic_tom(conversation, self.task)
            self.fallback_uses += 1

        self._cache_put(key, text)
        return text
