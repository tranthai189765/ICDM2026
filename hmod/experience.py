"""Cross-episode experience accumulation for the H-MOD LLM controller.

The LLM meta-controller produces the dynamic weight w_t from the macro goal and
the visible dialogue. On its own it has no memory across episodes. This buffer
gives it that memory: after each episode we record the macro goal, the seller
drift, the final reflected w_t, and the outcome (success / deal price vs the
buyer ceiling). Before each reflection we summarise the most relevant past
episodes into a short natural-language hint that is injected into the LLM
prompt, so w_local generation improves with accumulated experience.

The buffer persists to JSON so experience carries over between runs.
"""

import json
import os
from typing import Any, Dict, List, Optional

from hmod.scenario import OBJECTIVE_ORDER, coerce_objective_weight


def _dominant_objective(weight: List[float]) -> str:
    if not weight:
        return "balanced"
    idx = max(range(len(weight)), key=lambda i: weight[i])
    return OBJECTIVE_ORDER[idx] if idx < len(OBJECTIVE_ORDER) else "balanced"


def _mean_weight(weights: List[List[float]]) -> List[float]:
    if not weights:
        return [1.0 / len(OBJECTIVE_ORDER)] * len(OBJECTIVE_ORDER)
    n = len(weights)
    acc = [0.0] * len(OBJECTIVE_ORDER)
    for w in weights:
        cw = coerce_objective_weight(w)
        for i in range(len(OBJECTIVE_ORDER)):
            acc[i] += cw[i]
    return [round(x / n, 3) for x in acc]


class ExperienceBuffer:
    """Accumulates per-episode outcomes and summarises them for the LLM."""

    def __init__(self, path: Optional[str] = None, max_records: int = 500,
                 max_examples_in_prompt: int = 6):
        self.path = path
        self.max_records = max_records
        self.max_examples_in_prompt = max_examples_in_prompt
        self.records: List[Dict[str, Any]] = []
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
            except Exception:
                self.records = []

    # ── writing ──────────────────────────────────────────────────────────
    def record_episode(
        self,
        macro_goal: str,
        drift_mode: str,
        final_weight: List[float],
        gsr: float,
        deal: bool,
        deal_price: Optional[float],
        max_acceptable_price: Optional[float],
        intent_states: Optional[List[str]] = None,
    ) -> None:
        within_budget = (
            deal_price is not None
            and max_acceptable_price is not None
            and float(deal_price) <= float(max_acceptable_price)
        )
        self.records.append({
            "macro_goal": macro_goal,
            "drift_mode": drift_mode,
            "final_weight": [round(float(x), 3) for x in coerce_objective_weight(final_weight)],
            "gsr": float(gsr),
            "deal": bool(deal),
            "deal_price": deal_price,
            "max_acceptable_price": max_acceptable_price,
            "within_budget": bool(within_budget),
            "intent_states": list(dict.fromkeys(intent_states or [])),
        })
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records:]
        self.save()

    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)

    # ── reading / summarising ────────────────────────────────────────────
    def _relevant(self, macro_goal: str, drift_mode: Optional[str]) -> List[Dict[str, Any]]:
        goal = (macro_goal or "").strip().lower()
        same_goal = [r for r in self.records if (r.get("macro_goal") or "").strip().lower() == goal]
        pool = same_goal if same_goal else self.records
        if drift_mode:
            drift_pool = [r for r in pool if r.get("drift_mode") == drift_mode]
            if drift_pool:
                pool = drift_pool
        return pool[-self.max_examples_in_prompt * 4:]

    def summarize(self, macro_goal: str, drift_mode: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return a compact experience summary for the LLM prompt, or None."""
        pool = self._relevant(macro_goal, drift_mode)
        if not pool:
            return None
        successes = [r for r in pool if r.get("gsr", 0) >= 1.0 or (r.get("deal") and r.get("within_budget"))]
        failures = [r for r in pool if r not in successes]
        succ_w = _mean_weight([r["final_weight"] for r in successes])
        fail_w = _mean_weight([r["final_weight"] for r in failures]) if failures else None

        lines = [
            f"{len(pool)} past episodes for a similar goal"
            + (f" under drift '{drift_mode}'" if drift_mode else "")
            + f": {len(successes)} succeeded within budget, {len(failures)} did not."
        ]
        if successes:
            lines.append(
                f"Successful runs used average w_t={dict(zip(OBJECTIVE_ORDER, succ_w))} "
                f"(emphasis on {_dominant_objective(succ_w)})."
            )
        if fail_w is not None:
            lines.append(
                f"Unsuccessful runs averaged w_t={dict(zip(OBJECTIVE_ORDER, fail_w))} "
                f"(emphasis on {_dominant_objective(fail_w)})."
            )
        lines.append(
            "Lean toward the successful balance when the current seller intent is similar, "
            "but never exceed the buyer price ceiling."
        )
        return {
            "n_examples": len(pool),
            "n_success": len(successes),
            "avg_success_w": dict(zip(OBJECTIVE_ORDER, succ_w)) if successes else None,
            "avg_failure_w": dict(zip(OBJECTIVE_ORDER, fail_w)) if fail_w is not None else None,
            "text": " ".join(lines),
        }
