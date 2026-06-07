"""Monitor PPDPP output_dir and print metrics whenever a new metrics.json appears.

Watches a directory for new sub-run dirs ending in *epoch-*-* and, for each
one, prints a one-line summary of key metrics. Designed to be run in a
separate terminal while PPDPP training is in progress, so the user can see
quick/full eval results land in real time without scrolling through stdout.
"""

import argparse
import json
import os
import time
from pathlib import Path


KEYS = ["deal_rate", "gsr", "avg_turn", "sl_ratio", "fairness", "cvr", "weighted_return"]


def _format_metric(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_summary(metrics_path):
    try:
        payload = json.loads(metrics_path.read_text())
    except Exception as exc:
        print(f"  ! failed to parse {metrics_path.name}: {exc}")
        return
    m = payload.get("metrics", {})
    tag = "full" if "-full-" in metrics_path.parent.name else "quick" if "-quick-" in metrics_path.parent.name else "?"
    epoch_chunk = ""
    for piece in metrics_path.parent.name.split("-"):
        if piece.startswith("epoch") and "epoch-" in metrics_path.parent.name:
            continue
    # crude epoch extract
    parts = metrics_path.parent.name.split("epoch-")
    epoch = parts[1].split("-")[0] if len(parts) > 1 else "?"
    n = m.get("num_dialogues", "?")
    print(f"\n=== {metrics_path.parent.name}")
    print(f"  type={tag}  epoch={epoch}  n={n}")
    for k in KEYS:
        print(f"    {k:<16} {_format_metric(m.get(k))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", help="PPDPP run output dir to watch")
    ap.add_argument("--poll", type=float, default=15.0, help="poll interval seconds")
    args = ap.parse_args()

    root = Path(args.output_dir)
    if not root.exists():
        print(f"waiting for {root} to appear...")
    seen = set()
    while True:
        if root.exists():
            for sub in sorted(root.iterdir()):
                metrics_path = sub / "metrics.json"
                if not metrics_path.exists():
                    continue
                key = (sub.name, metrics_path.stat().st_mtime)
                if key in seen:
                    continue
                seen.add(key)
                _print_summary(metrics_path)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
