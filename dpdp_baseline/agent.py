"""DPDP-style agent: reuse PPDPP policy network, add planner-aware action API."""

from __future__ import annotations

from typing import List, Tuple

import torch

# PPDPP modules live at repo_root/PPDPP/ and import siblings without package
# prefix; bootstrap their directory onto sys.path so we can reuse them.
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_PPDPP_DIR = os.path.join(_REPO_ROOT, "PPDPP")
for path in (_REPO_ROOT, _PPDPP_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from agent import PPDPP  # noqa: E402  -- from PPDPP/agent.py


class DPDPAgent(PPDPP):
    """PPDPP policy with an additional `compute_probs` helper for the planner."""

    def compute_probs(self, state) -> Tuple[List[str], List[float]]:
        """Return (action_labels, list_of_probabilities) for the current state.

        Mirrors PPDPP.select_action(is_test=True) but exposes the full
        distribution so the DPDP planner can apply the control gate.
        """
        seq_limits = [self.args.max_seq_length]
        if self.runtime_device.type == "mps":
            for fallback_limit in (384, 256, 192, 128):
                if fallback_limit < self.args.max_seq_length:
                    seq_limits.append(fallback_limit)

        device = next(self.parameters()).device
        last_exc = None
        for seq_limit in seq_limits:
            try:
                inp = self.build_input(state, max_seq_length=seq_limit)
                inp = torch.tensor(inp).long().to(device)
                with torch.no_grad():
                    outputs = self._policy_forward(inp)
                    pooled_output = outputs[1]
                    pooled_output = self.dropout(pooled_output)
                    logits = self.classifier(pooled_output)
                    probs = self._stable_probs(logits)
                prob_list = probs.detach().cpu().view(-1).tolist()
                return list(self.act), prob_list
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
