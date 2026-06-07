"""Lightweight DPDP planner: top-K policy candidates + 1-ply lookahead.

This is a tractable analogue of the System-2 MCTS described in the DPDP paper:
the policy LM proposes a prior, we expand the top-K candidate actions, simulate
ONE environment step for each candidate (LLM-driven turn + critic reward), then
pick the candidate with the best simulated scalar reward.

A bounded action-result cache keys on (dialogue_hash, action) so that repeated
candidates inside an episode do not re-burn LLM tokens.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple


def _state_hash(conversation: List[Dict[str, str]]) -> str:
    payload = json.dumps(conversation, ensure_ascii=False, sort_keys=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class EnvSnapshot:
    """Capture mutable fields of a PPDPP Env so we can roll back after a sim."""

    __slots__ = (
        "conversation",
        "turn_records",
        "last_reward_info",
        "cur_conver_step",
        "case",
    )

    def __init__(self, env):
        self.conversation = copy.deepcopy(env.conversation)
        self.turn_records = copy.deepcopy(env.turn_records)
        self.last_reward_info = copy.deepcopy(env.last_reward_info)
        self.cur_conver_step = env.cur_conver_step
        self.case = env.case  # cases are read-only inside step()

    def restore(self, env) -> None:
        env.conversation = copy.deepcopy(self.conversation)
        env.turn_records = copy.deepcopy(self.turn_records)
        env.last_reward_info = copy.deepcopy(self.last_reward_info)
        env.cur_conver_step = self.cur_conver_step
        env.case = self.case


class DPDPPlanner:
    """1-ply Monte Carlo lookahead with a top-K candidate frontier."""

    def __init__(
        self,
        top_k: int = 2,
        cache_size: int = 256,
    ):
        self.top_k = max(1, int(top_k))
        self._cache: "OrderedDict[Tuple[str, str], float]" = OrderedDict()
        self._cache_size = max(1, int(cache_size))
        self.usage = {"policy": 0, "mcts": 0, "sim_calls": 0, "cache_hits": 0}

    def reset_usage(self) -> None:
        self.usage = {"policy": 0, "mcts": 0, "sim_calls": 0, "cache_hits": 0}

    def _cache_get(self, key) -> Optional[float]:
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        return None

    def _cache_put(self, key, value: float) -> None:
        self._cache[key] = value
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        self._cache.clear()

    @staticmethod
    def confidence_margin(probs: List[float]) -> float:
        if probs is None or len(probs) == 0:
            return 1.0
        ordered = sorted(probs, reverse=True)
        top1 = float(ordered[0])
        top2 = float(ordered[1]) if len(ordered) > 1 else 0.0
        return top1 - top2

    def _simulate_action(self, env, action: str, base_hash: str) -> float:
        key = (base_hash, action)
        cached = self._cache_get(key)
        if cached is not None:
            self.usage["cache_hits"] += 1
            return cached

        snapshot = EnvSnapshot(env)
        try:
            self.usage["sim_calls"] += 1
            _state, reward, _done = env.step(action)
            value = float(reward)
        except Exception:
            value = float("-inf")
        finally:
            snapshot.restore(env)

        self._cache_put(key, value)
        return value

    def select_action(
        self,
        env,
        action_labels: List[str],
        probs: List[float],
        eta: float,
        force_mcts: bool = False,
    ) -> Dict[str, Any]:
        """Return action + metadata. Falls back to policy argmax if uncertain
        margin >= eta (and not force_mcts), otherwise expands top-K via 1-ply.
        """
        if not action_labels:
            raise ValueError("action_labels must be non-empty")

        margin = self.confidence_margin(probs)
        ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        if not force_mcts and margin >= float(eta):
            best_idx = ranked[0]
            self.usage["policy"] += 1
            return {
                "action": action_labels[best_idx],
                "planner": "policy",
                "margin": margin,
                "candidates": [action_labels[best_idx]],
                "values": {action_labels[best_idx]: float(probs[best_idx])},
            }

        candidates = [action_labels[i] for i in ranked[: self.top_k]]
        base_hash = _state_hash(env.conversation)
        values: Dict[str, float] = {}
        for candidate in candidates:
            values[candidate] = self._simulate_action(env, candidate, base_hash)

        best_action = max(values, key=lambda act: values[act])
        self.usage["mcts"] += 1
        return {
            "action": best_action,
            "planner": "mcts",
            "margin": margin,
            "candidates": candidates,
            "values": values,
        }
