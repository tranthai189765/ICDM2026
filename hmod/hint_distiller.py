"""LLM hint distiller for H-MOD hint training.

After a self-play epoch, the distiller reads the metric glossary, a digest of
the epoch's episodes (which w_t was used under which seller intents, and the
resulting GSR / T2DA / CVR), and the current hint playbook, then rewrites a
small set of GENERAL, reusable hints for choosing w_t. It reuses the project
LLM backend through an existing LLMWeightReflector (so FPT creds from .env are
shared with the controller).
"""

import json
from typing import Any, Dict, List, Optional

from hmod.hints import METRIC_GLOSSARY
from hmod.llm_reflection import LLMWeightReflector, parse_reflection_json


class LLMHintDistiller:
    def __init__(self, reflector: LLMWeightReflector, max_hints: int = 12):
        # Reuse the reflector's backend (.​_complete handles FPT/DeepInfra routing).
        self.reflector = reflector
        self.max_hints = max_hints

    def distill(
        self,
        current_hints: List[str],
        episode_digest: List[Dict[str, Any]],
        aggregate_metrics: Dict[str, Any],
        epoch: int,
    ) -> List[str]:
        messages = self._build_messages(
            current_hints, episode_digest, aggregate_metrics, epoch
        )
        content = self.reflector._complete(messages)
        parsed = parse_reflection_json(content)
        hints = parsed.get("hints", [])
        if not isinstance(hints, list):
            raise ValueError(f"distiller returned non-list hints: {hints!r}")
        out = [str(h).strip() for h in hints if str(h).strip()]
        return out[: self.max_hints]

    def _build_messages(
        self,
        current_hints: List[str],
        episode_digest: List[Dict[str, Any]],
        aggregate_metrics: Dict[str, Any],
        epoch: int,
    ) -> List[Dict[str, str]]:
        system = (
            "You are a negotiation coach improving a BUYER agent's dynamic-weight "
            "meta-controller. The controller emits a weight vector w_t = "
            "[sl_ratio, fairness, deal_rate] each few turns from the visible dialogue. "
            "You are given the metric definitions, this epoch's self-play results, and "
            "the current hint playbook. Rewrite the playbook into a small set of GENERAL, "
            "reusable hints (rules of thumb) that would raise GSR, lower T2DA, and keep "
            "CVR at zero across future episodes. Hints must be transferable (about seller "
            "intents and when to shift w_t), NOT about a specific item or price."
        )
        user = {
            "metric_glossary": METRIC_GLOSSARY,
            "epoch": epoch,
            "current_aggregate_metrics": aggregate_metrics,
            "current_hints": current_hints or [],
            "self_play_episodes": episode_digest,
            "instructions": (
                f"Analyse why episodes succeeded or failed (look at w_t vs seller intent "
                f"vs GSR/T2DA/CVR). Produce AT MOST {self.max_hints} concise, general, "
                "non-overlapping hints. Keep good existing hints, drop ones contradicted "
                "by the data, add new ones the results suggest. Each hint <= 30 words."
            ),
            "required_json_schema": {
                "analysis": "2-4 sentences on what drove success/failure this epoch",
                "hints": ["general hint string", "..."],
            },
            "output_instruction": "Return ONLY valid JSON with keys 'analysis' and 'hints'.",
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]


def build_episode_digest(
    dialogues: List[Dict[str, Any]],
    max_episodes: int = 14,
) -> List[Dict[str, Any]]:
    """Condense per-dialogue records into a compact, LLM-friendly digest.

    Failures (GSR=0) are prioritised so the distiller sees what to fix.
    """
    rows = []
    for rec in dialogues:
        weight_trace = rec.get("weight_trace", [])
        intents = list(dict.fromkeys(
            w.get("intent_state") for w in weight_trace if w.get("intent_state")
        ))
        w_first = weight_trace[0]["weight_vector"] if weight_trace else None
        w_last = weight_trace[-1]["weight_vector"] if weight_trace else None
        judge = rec.get("judge_result", {})
        t2da = (rec.get("t2da") or {}).get("t2da")
        cvr = (rec.get("cvr") or {}).get("cvr")
        rows.append({
            "drift_mode": rec.get("drift_mode"),
            "seller_intents_seen": intents,
            "w_start": [round(float(x), 2) for x in w_first] if w_first else None,
            "w_end": [round(float(x), 2) for x in w_last] if w_last else None,
            "gsr": rec.get("gsr"),
            "t2da": t2da,
            "cvr": cvr,
            "deal": judge.get("deal"),
            "deal_price": judge.get("deal_price"),
            "max_acceptable_price": (rec.get("gsr_components") or {}).get("constraint_price"),
        })
    # failures first, then successes; cap
    rows.sort(key=lambda r: (r.get("gsr") or 0))
    return rows[:max_episodes]
