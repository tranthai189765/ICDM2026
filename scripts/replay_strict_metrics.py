"""Recompute PPDPP metrics on existing dialogues.jsonl with the strict judge.

Reads each per-turn record's `system_response` to recompute price-based
objectives with the tightened `extract_committed_prices`, then re-judges the
whole dialogue with the strict `rule_judge_deal`. Writes a new
`metrics_strict.json` next to the original metrics file. Read-only with
respect to the original outputs.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PPDPP_DIR = ROOT / "PPDPP"
# Insert PPDPP first so its modules are importable, then prepend ROOT so the
# repo-level `utils` package wins over PPDPP/utils.py (which shadows it).
for path in (PPDPP_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Load FPT/OpenAI-compatible credentials from .env so llm_judge_deal can
# reach the same provider PPDPP uses at training time. utils.prompt also
# calls load_dotenv() at import time but defaults to cwd, so we force the
# repo root .env here BEFORE any utils.prompt import below.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(ROOT / ".env", override=True)
except Exception:
    pass

from hmod.judge import llm_judge_deal, rule_judge_deal  # noqa: E402
from ppdpp_rewards import (  # noqa: E402
    OBJECTIVE_WEIGHTS,
    aggregate_episode_records,
    compute_price_objectives,
    scalarize,
    subgroup_metrics,
)
from scenario_loader import load_scenario_cases  # noqa: E402


def _conv_to_dialogue(turns):
    rows = []
    for turn in turns:
        role = str(turn.get("role", "")).lower()
        rows.append({
            "role": "assistant" if role in {"buyer", "assistant", "system"} else "user",
            "content": turn.get("content", ""),
        })
    return rows


def _dialogue_hash(dialogue):
    payload = json.dumps(dialogue, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_cache(cache_path):
    if cache_path is None or not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(cache_path, cache):
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False)
    tmp.replace(cache_path)


def _judge(dialogue, judge_kind, cache, judge_model_name):
    """Return a normalised judge result and update the cache for LLM calls."""
    canonical = _conv_to_dialogue(dialogue)
    if judge_kind == "rule":
        return rule_judge_deal(canonical)
    # call_llm matches on the lowercase constants in config.constants
    # (CHATGPT='chatgpt', LLAMA3='llama3', FPT='fpt'); accept either case
    # so users can pass --judge_model FPT.
    if judge_model_name:
        judge_model_name = str(judge_model_name).lower()
    key = f"{judge_model_name or 'default'}::{_dialogue_hash(canonical)}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    if judge_model_name:
        result = llm_judge_deal(canonical, model_type=judge_model_name, fallback_to_rule=False)
    else:
        result = llm_judge_deal(canonical, fallback_to_rule=False)
    cache[key] = result
    return result


def _rebuild_episode_record(
    case_meta, dialogue, turn_records, objective, judge_kind, cache, judge_model_name
):
    """Recompute reward_vector / deal_success on each turn with the strict
    judge + strict price parser, then build the per-dialogue episode record
    in the same schema as `Env.get_episode_record`.
    """
    seller_price = case_meta.get("seller_price")
    buyer_price = case_meta.get("buyer_price")
    if seller_price is None or buyer_price is None:
        # We need at least the case price endpoints; without them we cannot
        # recompute sl_ratio. Fall back to old reward vector.
        seller_price = 0.0
        buyer_price = 0.0
    case = {"buyer_price": float(buyer_price), "seller_price": float(seller_price)}

    judge_result = _judge(dialogue, judge_kind, cache, judge_model_name)
    deal_success = bool(judge_result.get("deal"))
    deal_price = judge_result.get("deal_price")

    cum_vec = {"sl_ratio": 0.0, "fairness": 0.0, "deal_rate": 0.0}
    weighted_return = 0.0
    price_attempt_count = 0
    per_turn_violations = 0

    for row in turn_records:
        action = row.get("action")
        sys_response = row.get("system_response") or ""
        sl, fairness, system_price = compute_price_objectives(
            case if buyer_price != seller_price else {"buyer_price": 1.0, "seller_price": 0.0},
            sys_response,
            action=action,
        )
        deal_reward = 1.0 if deal_success else -0.1
        vec = [sl, fairness, deal_reward]
        scalar = scalarize(vec, objective)
        cum_vec["sl_ratio"] += sl
        cum_vec["fairness"] += fairness
        cum_vec["deal_rate"] += deal_reward
        weighted_return += scalar
        if system_price is not None:
            price_attempt_count += 1
        max_price = case_meta.get("max_acceptable_price")
        if (
            max_price is not None
            and system_price is not None
            and float(system_price) > float(max_price)
        ):
            per_turn_violations += 1

    max_price = case_meta.get("max_acceptable_price")
    turn_limit = int(case_meta.get("turn_limit") or 0) or len(turn_records)
    turns = len(turn_records)
    if not deal_success:
        price_ok = False
    elif max_price is None:
        price_ok = True
    else:
        price_ok = deal_price is not None and float(deal_price) <= float(max_price)
    gsr = int(bool(deal_success and price_ok and turns <= turn_limit))

    final_deal_violation = bool(
        deal_success
        and max_price is not None
        and deal_price is not None
        and float(deal_price) > float(max_price)
    )
    violation_attempts = turns + (1 if final_deal_violation else 0)
    actual_violation_count = per_turn_violations + (1 if final_deal_violation else 0)
    denom = max(violation_attempts, 1)

    return {
        "scenario_id": case_meta.get("scenario_id"),
        "source_dataset": case_meta.get("source_dataset"),
        "recommendation_domain": case_meta.get("recommendation_domain"),
        "drift_mode": case_meta.get("drift_mode"),
        "seller_persona_type": case_meta.get("seller_persona_type"),
        "buyer_intent_id": case_meta.get("buyer_intent_id"),
        "objective": objective,
        "static_w": case_meta.get("static_w"),
        "success": deal_success,
        "gsr": gsr,
        "turns": turns,
        "weighted_return": weighted_return,
        "cumulative_reward_vector": cum_vec,
        "deal_price": deal_price,
        "max_acceptable_price": max_price,
        "price_violation": bool(per_turn_violations or final_deal_violation),
        "price_attempt_count": price_attempt_count,
        "blocked_cvr": 0.0,
        "actual_cvr": actual_violation_count / denom,
        "cvr": actual_violation_count / denom,
        "blocked_violation_count": 0,
        "actual_violation_count": actual_violation_count,
        "violation_attempt_count": violation_attempts,
        "final_deal_violation": final_deal_violation,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Run dir containing dialogues.jsonl + metrics.json")
    ap.add_argument(
        "--output",
        default=None,
        help="Filename inside run_dir to write the new metrics. Defaults to metrics_strict.json for rule judge and metrics_strict_llm.json for llm judge.",
    )
    ap.add_argument(
        "--scenario_file",
        default=None,
        help=(
            "Optional HMOD scenario YAML used to fill in case_meta (buyer_price, "
            "seller_price, max_acceptable_price, turn_limit) for legacy dialogues "
            "that did not persist this metadata."
        ),
    )
    ap.add_argument(
        "--judge",
        choices=["rule", "llm"],
        default="rule",
        help="Which judge to use to relabel dialogues (default: rule).",
    )
    ap.add_argument(
        "--judge_model",
        default=None,
        help="Optional LLM model name forwarded to llm_judge_deal (e.g. LLAMA3, FPT). Defaults to hmod.judge default.",
    )
    ap.add_argument(
        "--cache_dir",
        default=str(ROOT / "cache" / "judge_replay"),
        help="Directory to store LLM judge cache files (per run_dir).",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    dialogues_path = run_dir / "dialogues.jsonl"
    metrics_path = run_dir / "metrics.json"
    if not dialogues_path.exists():
        raise SystemExit(f"missing {dialogues_path}")

    old_metrics_payload = {}
    if metrics_path.exists():
        with metrics_path.open() as fh:
            old_metrics_payload = json.load(fh)
    objective = old_metrics_payload.get("objective") or "uniform"

    # Build a lookup from scenario_id -> case_meta from per-dialogue records.
    case_meta_by_id = {}
    for line in dialogues_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case_meta = row.get("case_meta") or {}
        sid = row.get("scenario_id")
        if sid is not None and sid not in case_meta_by_id:
            case_meta_by_id[sid] = case_meta

    # Augment from a scenario YAML when dialogues lack case_meta (legacy runs).
    if args.scenario_file:
        for case in load_scenario_cases(args.scenario_file):
            sid = case.get("scenario_id")
            if sid is None:
                continue
            existing = case_meta_by_id.get(sid) or {}
            merged = {
                "scenario_id": sid,
                "source_dataset": existing.get("source_dataset") or case.get("source_dataset"),
                "recommendation_domain": existing.get("recommendation_domain") or case.get("recommendation_domain"),
                "drift_mode": existing.get("drift_mode") or case.get("drift_mode"),
                "seller_persona_type": existing.get("seller_persona_type") or case.get("seller_persona_type"),
                "buyer_intent_id": existing.get("buyer_intent_id") or case.get("buyer_intent_id"),
                "static_w": existing.get("static_w") or case.get("static_w"),
                "buyer_price": existing.get("buyer_price") if existing.get("buyer_price") is not None else case.get("buyer_price"),
                "seller_price": existing.get("seller_price") if existing.get("seller_price") is not None else case.get("seller_price"),
                "max_acceptable_price": existing.get("max_acceptable_price") if existing.get("max_acceptable_price") is not None else case.get("max_acceptable_price"),
                "turn_limit": existing.get("turn_limit") or case.get("turn_limit"),
                "item_name": existing.get("item_name") or case.get("item_name"),
            }
            case_meta_by_id[sid] = merged

    out_filename = args.output or (
        "metrics_strict.json" if args.judge == "rule" else "metrics_strict_llm.json"
    )

    cache_path = None
    cache = {}
    if args.judge == "llm":
        cache_path = Path(args.cache_dir) / f"{run_dir.name}.json"
        cache = _load_cache(cache_path)
        print(f"[llm judge] cache: {cache_path} ({len(cache)} entries)")

    episode_records = []
    t0 = time.time()
    for idx, line in enumerate(dialogues_path.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row.get("scenario_id")
        case_meta = row.get("case_meta") or case_meta_by_id.get(sid) or {}
        # Some old dialogues files did not persist case_meta. Pull from any
        # field in the dialogue that hints at price endpoints; fall back to
        # zero-span case which disables price metrics but keeps deal/GSR.
        case_meta = {
            "scenario_id": sid,
            "source_dataset": row.get("source_dataset"),
            "recommendation_domain": row.get("recommendation_domain"),
            "drift_mode": row.get("drift_mode"),
            "seller_persona_type": row.get("seller_persona_type"),
            "buyer_intent_id": row.get("buyer_intent_id"),
            "static_w": row.get("static_w"),
            "max_acceptable_price": row.get("max_acceptable_price"),
            "turn_limit": row.get("turn_limit"),
            "buyer_price": case_meta.get("buyer_price"),
            "seller_price": case_meta.get("seller_price"),
            **{k: v for k, v in case_meta.items() if k not in {"buyer_price", "seller_price"}},
        }
        dialogue = row.get("dialogue") or []
        turn_records = row.get("turn_records") or []
        episode_records.append(
            _rebuild_episode_record(
                case_meta,
                dialogue,
                turn_records,
                objective,
                args.judge,
                cache,
                args.judge_model,
            )
        )
        # Persist the LLM cache periodically so a long replay can resume.
        if args.judge == "llm" and (idx + 1) % 25 == 0:
            _save_cache(cache_path, cache)
            print(f"[llm judge] {idx + 1} dialogues processed ({time.time() - t0:.1f}s)")

    if args.judge == "llm":
        _save_cache(cache_path, cache)

    strict_metrics = aggregate_episode_records(episode_records, objective)
    metrics_payload = {
        **{k: v for k, v in old_metrics_payload.items() if k != "metrics"},
        "objective": objective,
        "objective_weight": OBJECTIVE_WEIGHTS[objective],
        "metrics": strict_metrics,
        "metrics_by_drift_mode": subgroup_metrics(episode_records, objective, "drift_mode"),
        "metrics_by_persona": subgroup_metrics(episode_records, objective, "seller_persona_type"),
        "metrics_by_objective": subgroup_metrics(episode_records, objective, "buyer_intent_id"),
        "metrics_by_recommendation_domain": subgroup_metrics(
            episode_records, objective, "recommendation_domain"
        ),
        "num_dialogues": len(episode_records),
        "regenerated_with": (
            "strict_rule_judge + extract_committed_prices"
            if args.judge == "rule"
            else f"strict_llm_judge({args.judge_model or 'default'}) + extract_committed_prices"
        ),
    }

    out_path = run_dir / out_filename
    with out_path.open("w") as fh:
        json.dump(metrics_payload, fh, indent=2, ensure_ascii=False)

    old_metrics = old_metrics_payload.get("metrics", {})
    keys = ["deal_rate", "gsr", "avg_turn", "sl_ratio", "fairness", "cvr", "weighted_return"]
    print(f"run_dir: {run_dir}")
    print(f"wrote:   {out_path.name}")
    print(f"{'metric':<18}{'old':>12}{'strict':>12}{'delta':>12}")
    for key in keys:
        old_val = old_metrics.get(key)
        new_val = strict_metrics.get(key)
        if old_val is None or new_val is None:
            delta = ""
        else:
            delta = f"{new_val - old_val:+.4f}"
        print(f"{key:<18}{(old_val if old_val is not None else '-'):>12}{(new_val if new_val is not None else '-'):>12}{delta:>12}")


if __name__ == "__main__":
    main()
