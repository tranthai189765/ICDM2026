"""General-hint store for the H-MOD LLM meta-controller.

H-MOD hint training lets the LLM controller self-play against the drift
simulator, read back the full metric feedback (GSR / T2DA / CVR / llm_sr), and
distil what it learns into a small set of *general* natural-language hints —
reusable rules of thumb for choosing the dynamic weight w_t under different
seller intents. Unlike the per-(goal, drift) ExperienceBuffer, these hints are
global: one shared playbook.

The store persists to JSON so a training run can write the playbook and a later
eval run can load it back and inject it into the reflection prompt.
"""

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional


# ── Metric glossary injected into the distiller prompt ───────────────────────
# The LLM only writes useful hints if it understands what each metric rewards.
METRIC_GLOSSARY = (
    "Metric meanings for a BUYER agent negotiating a lower price:\n"
    "- GSR (Goal Success Rate, 0/1, higher better): 1 only if ALL hold — (a) a deal "
    "closed, (b) the deal price is <= the buyer's max acceptable price (the buyer must "
    "NOT overpay), and (c) the deal closed within the turn limit. This is the primary "
    "success signal.\n"
    "- llm_sr (0..1, higher better): the judge's raw 'a deal was reached' verdict, "
    "ignoring price/turn constraints. If llm_sr is high but GSR is low, deals closed "
    "but the buyer overpaid or took too long.\n"
    "- T2DA (Turns-To-Drift-Adaptation, turns, LOWER better): how many turns after the "
    "seller's intent drifts (e.g. turns firm, issues a final offer, threatens to walk "
    "away) before the buyer's weight vector w_t shifts meaningfully (L1 change >= 0.25 "
    "from its pre-drift value) AND in the right direction (lower sl_ratio, raise "
    "deal_rate). Large T2DA = the controller reacted too slowly to the seller's change.\n"
    "- CVR (Constraint Violation Rate, 0..1, must stay ~0): fraction of turns the buyer "
    "offered above the price ceiling. The ceiling is a HARD constraint — never breach it.\n"
    "Weight vector w_t = [sl_ratio, fairness, deal_rate], each >= 0, summing to 1:\n"
    "- sl_ratio: bargain harder for a lower price (protect buyer surplus).\n"
    "- fairness: keep offers reasonable / preserve the relationship (avoid lowballing).\n"
    "- deal_rate: prioritise closing the deal when the price is acceptable.\n"
    "Good policy: raise deal_rate when the seller signals a final offer or walk-away risk "
    "AND the price is within the ceiling; raise sl_ratio early or when the seller is "
    "cooperative; keep fairness moderate to avoid antagonising; adapt quickly (low T2DA) "
    "and never exceed the ceiling (CVR=0)."
)


class HintStore:
    """Holds, persists and serves the general hint playbook."""

    def __init__(self, path: Optional[str] = None, max_hints: int = 12):
        self.path = path
        self.max_hints = max_hints
        self.hints: List[str] = []
        self.meta: Dict[str, Any] = {"iterations": 0, "history": []}
        if path and os.path.exists(path):
            self.load()

    # ── persistence ──────────────────────────────────────────────────────
    def load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.hints = list(data.get("hints", []))
            self.meta = data.get("meta", {"iterations": 0, "history": []})
        except Exception:
            self.hints = []
            self.meta = {"iterations": 0, "history": []}

    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload = {
            "hints": self.hints,
            "meta": self.meta,
            "metric_glossary": METRIC_GLOSSARY,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    # ── update from a distillation step ──────────────────────────────────
    def update(self, hints: List[str], metrics: Optional[Dict[str, Any]] = None) -> None:
        cleaned = [h.strip() for h in hints if isinstance(h, str) and h.strip()]
        # de-duplicate preserving order, cap to max_hints
        seen, deduped = set(), []
        for h in cleaned:
            key = h.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(h)
        self.hints = deduped[: self.max_hints]
        self.meta["iterations"] = int(self.meta.get("iterations", 0)) + 1
        self.meta.setdefault("history", []).append({
            "iteration": self.meta["iterations"],
            "n_hints": len(self.hints),
            "metrics": metrics or {},
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        self.save()

    # ── review update (two-agent trainer) ───────────────────────────────
    def review_update(
        self,
        remove_candidates: List[str],
        add_hints: List[str],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply a review step: a hint proposed for removal on two consecutive
        epochs is dropped; otherwise its removal streak resets. New hints are
        appended (deduplicated). Returns a small change report.
        """
        streaks: Dict[str, int] = self.meta.setdefault("removal_streak", {})
        remove_set = {h.strip() for h in remove_candidates if isinstance(h, str) and h.strip()}

        dropped: List[str] = []
        kept: List[str] = []
        for h in self.hints:
            if h in remove_set:
                streaks[h] = int(streaks.get(h, 0)) + 1
                if streaks[h] >= 2:          # removed on 2 consecutive epochs -> drop
                    dropped.append(h)
                    streaks.pop(h, None)
                    continue
            else:
                streaks.pop(h, None)         # reset streak
            kept.append(h)

        existing = {h.lower() for h in kept}
        added: List[str] = []
        for h in add_hints:
            h = h.strip() if isinstance(h, str) else ""
            if h and h.lower() not in existing:
                kept.append(h)
                existing.add(h.lower())
                added.append(h)

        self.hints = kept[: self.max_hints]
        self.meta["iterations"] = int(self.meta.get("iterations", 0)) + 1
        self.meta.setdefault("history", []).append({
            "iteration": self.meta["iterations"],
            "n_hints": len(self.hints),
            "dropped": dropped,
            "added": added,
            "remove_proposed": sorted(remove_set),
            "metrics": metrics or {},
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        self.save()
        return {"dropped": dropped, "added": added, "n_hints": len(self.hints)}

    # ── reading ──────────────────────────────────────────────────────────
    def is_empty(self) -> bool:
        return not self.hints

    def as_text(self) -> Optional[str]:
        if not self.hints:
            return None
        lines = ["Learned general hints (from past self-play; apply when relevant):"]
        lines += [f"{i + 1}. {h}" for i, h in enumerate(self.hints)]
        return "\n".join(lines)

    def provider(self) -> Callable[[str, Optional[str]], Optional[str]]:
        """Return a callable(macro_goal, drift_mode) -> hint text (drift-agnostic)."""
        def _provider(macro_goal: str, drift_mode: Optional[str] = None) -> Optional[str]:
            return self.as_text()
        return _provider
