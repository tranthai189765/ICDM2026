"""Replay an existing PPDPP dialogues.jsonl through the strict judge.

Compares the strict rule judge against the original metrics.json so we can
quantify how many deals were false positives under the old loose judge.
Read-only: does not retrain or re-call any LLM.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Drop PPDPP from sys.path if it leaks in via cwd; we only need hmod here.
for entry in list(sys.path):
    if entry.endswith("/PPDPP"):
        sys.path.remove(entry)

from hmod.judge import rule_judge_deal


def _conv_to_dialogue(turns):
    rows = []
    for turn in turns:
        role = str(turn.get("role", "")).lower()
        rows.append({
            "role": "assistant" if role in {"buyer", "assistant", "system"} else "user",
            "content": turn.get("content", ""),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Path to an output run dir containing dialogues.jsonl + metrics.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    dialogues_path = run_dir / "dialogues.jsonl"
    metrics_path = run_dir / "metrics.json"
    if not dialogues_path.exists():
        raise SystemExit(f"missing {dialogues_path}")

    old_metrics = {}
    if metrics_path.exists():
        with metrics_path.open() as fh:
            old_metrics = json.load(fh).get("metrics", {})

    total = 0
    strict_deal = 0
    old_deal = 0
    disagreements = 0
    examples = []
    for line in dialogues_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        dialogue = _conv_to_dialogue(row.get("dialogue") or [])
        result = rule_judge_deal(dialogue)
        total += 1
        if result["deal"]:
            strict_deal += 1

        # Per-dialogue old verdict comes from cumulative reward_vector.deal_rate
        # > 0 in the turn_records (1.0 if old judge said deal, else -0.1).
        turn_records = row.get("turn_records") or []
        old_judge_deal = any(
            (t.get("deal_success") is True)
            or (t.get("reward_vector", {}).get("deal_rate", 0.0) >= 0.999)
            for t in turn_records
        )
        if old_judge_deal:
            old_deal += 1
        if old_judge_deal != result["deal"]:
            disagreements += 1
            if len(examples) < 5:
                examples.append({
                    "scenario_id": row.get("scenario_id"),
                    "old": old_judge_deal,
                    "strict": result["deal"],
                    "strict_evidence": result.get("evidence"),
                    "last_seller_turn": next(
                        (t["content"] for t in reversed(dialogue) if t["role"] == "user"),
                        "",
                    )[:160],
                })

    print(f"run_dir: {run_dir}")
    print(f"dialogues: {total}")
    print(f"old judge deal_rate (from metrics.json): {old_metrics.get('deal_rate')}")
    if total:
        print(f"old judge deal_rate (recomputed from turn_records): {old_deal / total:.4f}")
        print(f"strict judge deal_rate: {strict_deal / total:.4f}")
        delta = (strict_deal - old_deal) / total
        print(f"delta (strict - old): {delta:+.4f}")
    print(f"disagreements: {disagreements}")
    if examples:
        print("examples (showing up to 5):")
        for ex in examples:
            print(json.dumps(ex, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
