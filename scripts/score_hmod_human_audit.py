"""Score human verification labels exported by eval_hmod.py.

Expected input is outputs/.../human_audit.jsonl after annotators fill:
  - human_deal: true/false
  - human_success: true/false
  - human_deal_price: number|null
"""

import argparse
import json
from typing import Any, Dict, Iterable, List


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_labeled(row: Dict[str, Any]) -> bool:
    return row.get("human_deal") is not None or row.get("human_success") is not None


def score_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    labeled = [row for row in rows if _is_labeled(row)]
    if not labeled:
        return {
            "total_samples": len(rows),
            "labeled_samples": 0,
            "deal_agreement": None,
            "success_agreement": None,
            "avg_abs_price_error": None,
        }

    deal_compared = [
        row for row in labeled
        if row.get("human_deal") is not None and row.get("llm_judge", {}).get("deal") is not None
    ]
    success_compared = [
        row for row in labeled
        if row.get("human_success") is not None and row.get("llm_judge", {}).get("success") is not None
    ]
    price_errors = []
    for row in labeled:
        human_price = row.get("human_deal_price")
        llm_price = row.get("llm_judge", {}).get("deal_price")
        if human_price is None or llm_price is None:
            continue
        price_errors.append(abs(float(human_price) - float(llm_price)))

    return {
        "total_samples": len(rows),
        "labeled_samples": len(labeled),
        "deal_agreement": (
            sum(bool(row["human_deal"]) == bool(row["llm_judge"]["deal"]) for row in deal_compared)
            / len(deal_compared)
            if deal_compared else None
        ),
        "success_agreement": (
            sum(bool(row["human_success"]) == bool(row["llm_judge"]["success"]) for row in success_compared)
            / len(success_compared)
            if success_compared else None
        ),
        "avg_abs_price_error": (
            sum(price_errors) / len(price_errors) if price_errors else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_file")
    args = parser.parse_args()
    print(json.dumps(score_rows(_read_jsonl(args.audit_file)), indent=2))


if __name__ == "__main__":
    main()

