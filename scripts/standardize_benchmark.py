"""Standardize benchmark metrics across PPDPP and HMOD runs.

This script reads one or more metrics.json files (or directories containing them),
normalizes them into a shared schema, and prints a compact comparison table.

Shared schema:
- framework: ppdpp | hmod | unknown
- run_name
- num_dialogues
- deal_rate: deal reached rate (apples-to-apples with llm_sr/deal_rate)
- gsr: strict goal success rate (deal + price constraint + turn limit)
- avg_turn
- cvr: actual constraint violation rate
- t2da
- objective
- judge_model

Example:
  python scripts/standardize_benchmark.py --glob "outputs/**/metrics.json"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _collect_metric_files(inputs: List[str], pattern: str) -> List[Path]:
    files: List[Path] = []

    if inputs:
        for raw in inputs:
            p = Path(raw)
            if p.is_file() and p.name == "metrics.json":
                files.append(p)
            elif p.is_dir():
                files.extend(sorted(p.glob("**/metrics.json")))
    else:
        files.extend(sorted(Path(".").glob(pattern)))

    dedup = []
    seen = set()
    for p in files:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        dedup.append(p)
    return dedup


def _load_json(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _infer_framework(payload: Dict) -> str:
    model = str(payload.get("model", "")).upper()
    if model == "PPDPP":
        return "ppdpp"
    if model == "DPDP":
        return "dpdp"
    if model == "TRIP":
        return "trip"
    if payload.get("mode") is not None and payload.get("controller_mode") is not None:
        return "hmod"
    return "unknown"


def _pick_deal_rate(metrics: Dict) -> Optional[float]:
    # Prefer explicit deal_rate. For HMOD, llm_sr is the deal-reached rate.
    return _safe_float(
        metrics.get("deal_rate", metrics.get("llm_sr", metrics.get("sr")))
    )


def _pick_cvr(metrics: Dict) -> Optional[float]:
    return _safe_float(metrics.get("cvr", metrics.get("actual_cvr")))


def _normalize(path: Path, payload: Dict) -> Dict:
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    framework = _infer_framework(payload)

    row = {
        "path": str(path),
        "run_name": path.parent.name,
        "framework": framework,
        "num_dialogues": metrics.get("num_dialogues", payload.get("num_dialogues")),
        "deal_rate": _pick_deal_rate(metrics),
        "gsr": _safe_float(metrics.get("gsr")),
        "avg_turn": _safe_float(metrics.get("avg_turn")),
        "cvr": _pick_cvr(metrics),
        "t2da": _safe_float(metrics.get("t2da")),
        "objective": payload.get("objective") or metrics.get("objective"),
        "judge_model": payload.get("judge_model"),
        "scenario_file": payload.get("scenario_file"),
        "sr_legacy": _safe_float(metrics.get("sr")),
        "llm_sr": _safe_float(metrics.get("llm_sr")),
    }

    # Diagnostic: PPDPP often names "sr" as deal rate in aggregate outputs.
    if framework == "ppdpp":
        sr = row["sr_legacy"]
        gsr = row["gsr"]
        if sr is not None and gsr is not None:
            row["sr_minus_gsr"] = sr - gsr
        else:
            row["sr_minus_gsr"] = None
    else:
        row["sr_minus_gsr"] = None

    return row


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_table(rows: List[Dict], limit: int) -> None:
    if not rows:
        print("No metrics.json found.")
        return

    print("\nStandardized Benchmark Table")
    print("framework | run_name | n | deal_rate | gsr | avg_turn | cvr | t2da | objective | judge")
    print("-" * 120)
    shown = rows[:limit] if limit > 0 else rows
    for row in shown:
        print(
            " | ".join(
                [
                    _fmt(row["framework"]),
                    _fmt(row["run_name"]),
                    _fmt(row["num_dialogues"]),
                    _fmt(row["deal_rate"]),
                    _fmt(row["gsr"]),
                    _fmt(row["avg_turn"]),
                    _fmt(row["cvr"]),
                    _fmt(row["t2da"]),
                    _fmt(row["objective"]),
                    _fmt(row["judge_model"]),
                ]
            )
        )

    if len(rows) > len(shown):
        print(f"... {len(rows) - len(shown)} more rows not shown (use --limit 0)")


def _print_summary(rows: List[Dict]) -> None:
    by_framework: Dict[str, List[Dict]] = {}
    for row in rows:
        by_framework.setdefault(row["framework"], []).append(row)

    print("\nFramework Summary (mean over discovered runs)")
    print("framework | runs | deal_rate | gsr | avg_turn | cvr | t2da")
    print("-" * 80)

    for fw, group in sorted(by_framework.items()):
        def mean(key: str) -> Optional[float]:
            vals = [_safe_float(r.get(key)) for r in group]
            vals = [v for v in vals if v is not None]
            if not vals:
                return None
            return sum(vals) / len(vals)

        print(
            " | ".join(
                [
                    fw,
                    str(len(group)),
                    _fmt(mean("deal_rate")),
                    _fmt(mean("gsr")),
                    _fmt(mean("avg_turn")),
                    _fmt(mean("cvr")),
                    _fmt(mean("t2da")),
                ]
            )
        )


def _write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "path",
        "run_name",
        "framework",
        "num_dialogues",
        "deal_rate",
        "gsr",
        "avg_turn",
        "cvr",
        "t2da",
        "objective",
        "judge_model",
        "scenario_file",
        "sr_legacy",
        "llm_sr",
        "sr_minus_gsr",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "inputs",
        nargs="*",
        help="metrics.json files or directories containing run outputs",
    )
    ap.add_argument(
        "--glob",
        default="outputs/**/metrics.json",
        help="glob to scan when no positional inputs are provided",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=30,
        help="max rows to print in the detailed table (0 = all)",
    )
    ap.add_argument(
        "--sort-by",
        default="path",
        choices=["path", "framework", "deal_rate", "gsr", "avg_turn", "cvr", "t2da"],
        help="sort key for printed rows",
    )
    ap.add_argument(
        "--descending",
        action="store_true",
        help="reverse sort order",
    )
    ap.add_argument(
        "--csv-out",
        default=None,
        help="optional path to write normalized rows as CSV",
    )
    args = ap.parse_args()

    metric_files = _collect_metric_files(args.inputs, args.glob)
    rows = []
    for path in metric_files:
        payload = _load_json(path)
        if payload is None:
            continue
        rows.append(_normalize(path, payload))

    def sort_key(row: Dict):
        value = row.get(args.sort_by)
        if value is None:
            return float("-inf") if args.descending else float("inf")
        return value

    rows.sort(key=sort_key, reverse=args.descending)

    _print_table(rows, args.limit)
    _print_summary(rows)

    if args.csv_out:
        out_path = Path(args.csv_out)
        _write_csv(rows, out_path)
        print(f"\nWrote normalized CSV: {out_path}")


if __name__ == "__main__":
    main()
