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


def build_detector_digest(
    detector_records: List[Dict[str, Any]],
    max_examples: int = 12,
) -> Dict[str, Any]:
    """Summarise the detector's per-turn predictions vs gold for the review."""
    n = len(detector_records)
    if n == 0:
        return {"n_turns": 0, "intent_accuracy": None, "drift_accuracy": None,
                "confusion": {}, "mistakes": []}
    intent_ok = sum(1 for r in detector_records if r.get("intent_correct"))
    drift_ok = sum(1 for r in detector_records if r.get("drift_correct"))
    confusion: Dict[str, Dict[str, int]] = {}
    mistakes: List[Dict[str, Any]] = []
    for r in detector_records:
        g, p = r.get("gold_intent"), r.get("pred_intent")
        confusion.setdefault(g, {}).setdefault(p, 0)
        confusion[g][p] += 1
        if not r.get("intent_correct") or not r.get("drift_correct"):
            mistakes.append({
                "turn": r.get("turn"), "gold_intent": g, "pred_intent": p,
                "gold_drift": r.get("gold_drift"), "pred_drift": r.get("pred_drift"),
            })
    return {
        "n_turns": n,
        "intent_accuracy": round(intent_ok / n, 3),
        "drift_accuracy": round(drift_ok / n, 3),
        "confusion_gold_to_pred": confusion,
        "mistakes": mistakes[:max_examples],
    }


class ReviewHintDistiller:
    """Reviewing distiller for the two-agent trainer.

    Each epoch it reviews the whole epoch's feedback plus the current hints and
    proposes which hints to REMOVE (no longer helping its task) and which to ADD.
    `kind` selects the task framing: 'high_policy' (w_local quality, judged by
    GSR/T2DA/CVR) or 'intent_detection' (drift/intent accuracy).
    """

    def __init__(self, reflector: LLMWeightReflector, kind: str):
        if kind not in {"high_policy", "intent_detection"}:
            raise ValueError("kind must be 'high_policy' or 'intent_detection'")
        self.reflector = reflector
        self.kind = kind

    def review(
        self,
        current_hints: List[str],
        feedback: Dict[str, Any],
        epoch: int,
    ) -> Dict[str, Any]:
        messages = self._build_messages(current_hints, feedback, epoch)
        content = self.reflector._complete(messages)
        parsed = parse_reflection_json(content)
        remove = [str(h).strip() for h in parsed.get("remove", []) if str(h).strip()]
        add = [str(h).strip() for h in parsed.get("add", []) if str(h).strip()]
        return {"remove": remove, "add": add, "analysis": str(parsed.get("analysis", ""))}

    def _build_messages(self, current_hints, feedback, epoch) -> List[Dict[str, str]]:
        if self.kind == "high_policy":
            system = (
                "You coach the High-Policy weight controller of a BUYER agent. It sets "
                "w_local = [sl_ratio, fairness, deal_rate] from the current seller intent. "
                "Review this epoch's metric feedback and the current hint playbook, then "
                "propose which hints to REMOVE (not helping GSR/T2DA/CVR) and which general, "
                "transferable hints to ADD. Hints are about how to set w_local for a given "
                "seller intent, never about a specific item or price."
            )
            payload = {
                "metric_glossary": METRIC_GLOSSARY,
                "epoch": epoch,
                "current_hints": current_hints or [],
                "aggregate_metrics": feedback.get("aggregate_metrics"),
                "self_play_episodes": feedback.get("episode_digest"),
            }
        else:
            system = (
                "You coach the Intent-Drift Detector of a negotiation buyer agent. It reads "
                "the dialogue and predicts whether the SELLER intent drifted and what it is "
                "(neutral|firm|final_offer|walkaway_risk). Review this epoch's accuracy "
                "feedback (vs gold) and the current hint playbook, then propose which hints "
                "to REMOVE (not improving accuracy) and which general detection hints to ADD "
                "(cues that distinguish the intents)."
            )
            payload = {
                "epoch": epoch,
                "current_hints": current_hints or [],
                "detection_feedback": feedback.get("detector_digest"),
                "intent_definitions": feedback.get("intent_definitions"),
            }
        payload["instructions"] = (
            "A hint proposed for removal on two consecutive epochs is dropped automatically. "
            "Keep good hints (do not list them in remove), drop ones contradicted by the "
            "feedback, add new ones the data suggests. Each hint <= 30 words, general."
        )
        payload["required_json_schema"] = {
            "analysis": "2-4 sentences on what the feedback shows",
            "remove": ["exact current-hint strings to remove", "..."],
            "add": ["new general hint", "..."],
        }
        payload["output_instruction"] = "Return ONLY valid JSON with keys analysis, remove, add."
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
