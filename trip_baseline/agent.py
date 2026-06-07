"""TRIPAgent: PPDPP policy with a ToM-prefixed input for the UASP module."""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_PPDPP_DIR = os.path.join(_REPO_ROOT, "PPDPP")
for path in (_REPO_ROOT, _PPDPP_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from agent import PPDPP  # noqa: E402  -- PPDPP/agent.py

from trip_baseline.tom import TheoryOfMind  # noqa: E402


class TRIPAgent(PPDPP):
    """Strategy planner with optional ToM prefix and full-distribution probe."""

    def __init__(self, args, config, tokenizer, tom: Optional[TheoryOfMind] = None) -> None:
        super().__init__(args, config, tokenizer)
        self.tom = tom

    # ──────────────────────────────────────────────────────────────────────
    # Input construction with ToM prefix
    # ──────────────────────────────────────────────────────────────────────
    def build_input(self, state, max_seq_length=None, tom_text: Optional[str] = None):  # type: ignore[override]
        seq_limit = max_seq_length or self.args.max_seq_length
        dial_id: List[int] = []
        last = None
        for turn in state[::-1]:
            s = self.tokenizer.encode("%s: %s" % (turn['role'], turn['content']))
            if len(dial_id) + len(s) > seq_limit:
                break
            dial_id = s[1:] + dial_id
            last = s
        if last is None:
            last = self.tokenizer.encode("")
        inp = last[:1] + dial_id

        if tom_text:
            tom_ids = self.tokenizer.encode("ToM: " + tom_text)
            # Reserve at least 32 tokens for dialogue; ToM gets the remainder.
            budget = max(0, seq_limit - max(32, len(inp)))
            tom_ids = tom_ids[1:1 + budget]
            if tom_ids:
                inp = inp[:1] + tom_ids + inp[1:]
        return [inp]

    # ──────────────────────────────────────────────────────────────────────
    # Inference helpers
    # ──────────────────────────────────────────────────────────────────────
    def _resolve_tom_text(self, state) -> Optional[str]:
        if self.tom is None:
            return None
        try:
            return self.tom.infer(state)
        except Exception as exc:  # noqa: BLE001
            print(f"[trip-agent] ToM inference failed, dropping prefix: {exc}")
            return None

    def compute_probs(self, state) -> Tuple[List[str], List[float]]:
        tom_text = self._resolve_tom_text(state)
        seq_limits = [self.args.max_seq_length]
        if self.runtime_device.type == "mps":
            for fallback_limit in (384, 256, 192, 128):
                if fallback_limit < self.args.max_seq_length:
                    seq_limits.append(fallback_limit)

        device = next(self.parameters()).device
        last_exc = None
        for seq_limit in seq_limits:
            try:
                inp = self.build_input(state, max_seq_length=seq_limit, tom_text=tom_text)
                inp = torch.tensor(inp).long().to(device)
                with torch.no_grad():
                    outputs = self._policy_forward(inp)
                    pooled_output = outputs[1]
                    pooled_output = self.dropout(pooled_output)
                    logits = self.classifier(pooled_output)
                    probs = self._stable_probs(logits)
                return list(self.act), probs.detach().cpu().view(-1).tolist()
            except RuntimeError as exc:
                last_exc = exc
                is_mps_oom = (
                    self.runtime_device.type == "mps"
                    and "out of memory" in str(exc).lower()
                )
                if not is_mps_oom:
                    raise
                if hasattr(torch, "mps"):
                    torch.mps.empty_cache()
                if seq_limit == seq_limits[-1]:
                    raise
        raise last_exc  # type: ignore[misc]

    def select_action(self, state, is_test=False):  # type: ignore[override]
        tom_text = self._resolve_tom_text(state)
        seq_limits = [self.args.max_seq_length]
        if self.runtime_device.type == "mps":
            for fallback_limit in (384, 256, 192, 128):
                if fallback_limit < self.args.max_seq_length:
                    seq_limits.append(fallback_limit)

        device = next(self.parameters()).device
        last_exc = None
        for seq_limit in seq_limits:
            try:
                inp = self.build_input(state, max_seq_length=seq_limit, tom_text=tom_text)
                inp = torch.tensor(inp).long().to(device)
                if is_test:
                    with torch.no_grad():
                        outputs = self._policy_forward(inp)
                        pooled_output = outputs[1]
                        pooled_output = self.dropout(pooled_output)
                        logits = self.classifier(pooled_output)
                        probs = self._stable_probs(logits)
                        action_idx = probs.argmax().item()
                    return self.act[action_idx]
                outputs = self._policy_forward(inp)
                pooled_output = outputs[1]
                pooled_output = self.dropout(pooled_output)
                logits = self.classifier(pooled_output)
                probs = self._stable_probs(logits)
                from torch.distributions import Categorical
                dist = Categorical(probs)
                action_idx = dist.sample()
                self.saved_log_probs.append(dist.log_prob(action_idx))
                return self.act[action_idx]
            except RuntimeError as exc:
                last_exc = exc
                is_mps_oom = (
                    self.runtime_device.type == "mps"
                    and "out of memory" in str(exc).lower()
                )
                if not is_mps_oom:
                    raise
                if hasattr(torch, "mps"):
                    torch.mps.empty_cache()
                if seq_limit == seq_limits[-1]:
                    raise
        raise last_exc  # type: ignore[misc]
