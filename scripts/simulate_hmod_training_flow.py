"""Dry-run an end-to-end H-MOD buyer-agent training/evaluation flow.

This script is intentionally lightweight: it does not update neural network
weights or require pretrained checkpoints. It materializes the exact data
transformations around H-MOD:

1. Load buyer-drift scenarios and ambiguous buyer objectives.
2. Map each objective to an initial W plus stage-specific adaptation rules.
3. Build the Phase-1 skill library used by the DMORL/H-MOD trainer.
4. Run the Phase-2-style dynamic W evaluation loop and export metrics/logs.

Use the heavy `run_dmorl.py` path when you want actual RL optimization.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hmod.objectives import BuyerObjectiveLibrary
from hmod.runner import run_and_write
from hmod.scenario import load_scenarios
from hmod.training import HMODController, HMOD_OBJECTIVE_ORDER


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=_json_default)


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _scenario_to_training_case(scenario, library: BuyerObjectiveLibrary) -> Dict[str, Any]:
    mapping = library.map_objective(
        macro_goal=scenario.macro_goal,
        scenario_objective_id=scenario.buyer_intent_id,
        fallback_weight=scenario.static_w,
    )
    objective = library.get(mapping.objective_id)
    return {
        "case_id": scenario.id,
        "role_setup": {
            "agent": "buyer",
            "simulator": "seller",
        },
        "raw_input": {
            "macro_goal": scenario.macro_goal,
            "buyer_intent_id": scenario.buyer_intent_id,
            "drift_mode": scenario.drift_mode,
            "seller_persona": {
                "type": scenario.seller_persona.type,
                "description": scenario.seller_persona.description,
            },
            "case": scenario.case,
            "buyer_constraints": {
                "max_acceptable_price": scenario.max_acceptable_price(),
                "target_price": scenario.target_price(),
                "turn_limit": scenario.turn_limit,
            },
        },
        "transformed_for_training": {
            "objective_id": mapping.objective_id,
            "objective_cluster": mapping.cluster,
            "initial_w": mapping.weight_vector,
            "static_baseline_w": scenario.static_w,
            "stage_weights": objective.stage_weights if objective else {},
            "adaptation_rules": objective.adaptation_rules if objective else [],
            "expected_weight_shift": scenario.expected_weight_shift,
        },
    }


def _extract_weight_examples(eval_run_dir: str, max_rows: int = 30) -> List[Dict[str, Any]]:
    path = os.path.join(eval_run_dir, "weight_trace.jsonl")
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if idx >= max_rows:
                break
            row = json.loads(line)
            rows.append(
                {
                    "scenario_id": row.get("scenario_id"),
                    "turn": row.get("turn"),
                    "intent_state": row.get("intent_state"),
                    "w_t": row.get("w_t", row.get("weight_vector")),
                    "reflection_step": row.get("reflection_step"),
                    "decision_summary": row.get("decision_summary"),
                }
            )
    return rows


def simulate_flow(
    scenario_file: str,
    objective_file: str,
    output_dir: str,
    num_cases: Optional[int],
    reflection_horizon: int,
    audit_sample_size: int,
    compare_baseline: bool,
    controller_mode: str,
    llm_model: Optional[str],
    llm_api_key: Optional[str],
    llm_api_key_env: str,
    llm_base_url: Optional[str],
    llm_temperature: float,
    llm_max_tokens: int,
    llm_fallback_to_rule: bool,
) -> Dict[str, Any]:
    run_id = f"hmod_training_flow_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    scenarios = load_scenarios(scenario_file, limit=num_cases)
    objective_library = BuyerObjectiveLibrary.from_file(objective_file)

    training_cases = [
        _scenario_to_training_case(scenario, objective_library)
        for scenario in scenarios
    ]
    _write_json(
        os.path.join(run_dir, "01_input_training_cases.json"),
        {
            "source_files": {
                "scenario_file": scenario_file,
                "objective_file": objective_file,
            },
            "num_cases": len(training_cases),
            "cases": training_cases,
        },
    )
    _write_jsonl(os.path.join(run_dir, "01_input_training_cases.jsonl"), training_cases)

    controller = HMODController(
        n_objectives=len(HMOD_OBJECTIVE_ORDER),
        objective_names=HMOD_OBJECTIVE_ORDER,
        scenario="negotiation",
        objective_file=objective_file,
        objective_id=None,
        n_basic_skills=min(8, len(objective_library.intents)),
        n_advanced_skills=min(8, len(objective_library.clusters)),
        dynamic_weight_horizon=reflection_horizon,
        skills_file=os.path.join(run_dir, "03_hmod_skills.json"),
        hints_file=os.path.join(run_dir, "03_hmod_hints.json"),
        skill_log_file=os.path.join(run_dir, "03_skill_discovery.txt"),
    )
    controller.initialize_skills(force_rediscover=True)

    phase1_payload = {
        "phase": "phase_1_skill_training_dry_run",
        "objective_order": HMOD_OBJECTIVE_ORDER,
        "basic_skills": controller.skill_library.basic_skills,
        "advanced_skills": controller.skill_library.advanced_skills,
        "how_real_training_uses_this": [
            "Phase 1a samples one basic skill W per episode and trains the buyer agent under that fixed W.",
            "Phase 1b samples composite advanced skill W values and trains the shared policy/GPI library.",
            "The dry-run records the exact W set that the heavy RL trainer would consume.",
        ],
    }
    _write_json(os.path.join(run_dir, "03_phase1_skill_library.json"), phase1_payload)

    eval_output_dir = os.path.join(run_dir, "04_eval")
    hmod_eval = run_and_write(
        scenario_file=scenario_file,
        mode="hmod_dynamic",
        num_cases=num_cases,
        output_dir=eval_output_dir,
        audit_sample_size=audit_sample_size,
        judge_model="rule",
        use_llm_simulator=False,
        objective_file=objective_file,
        objective_id=None,
        reflection_horizon=reflection_horizon,
        controller_mode=controller_mode,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_api_key_env=llm_api_key_env,
        llm_base_url=llm_base_url,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        llm_fallback_to_rule=llm_fallback_to_rule,
    )

    baseline_eval = None
    if compare_baseline:
        baseline_eval = run_and_write(
            scenario_file=scenario_file,
            mode="padpp_static",
            num_cases=num_cases,
            output_dir=eval_output_dir,
            audit_sample_size=audit_sample_size,
            judge_model="rule",
            use_llm_simulator=False,
            objective_file=objective_file,
            objective_id=None,
            reflection_horizon=reflection_horizon,
            controller_mode="rule_scaffold",
        )

    dynamic_examples = _extract_weight_examples(hmod_eval["run_dir"])
    _write_json(
        os.path.join(run_dir, "04_phase2_dynamic_weight_examples.json"),
        {
            "phase": "phase_2_dynamic_weight_dry_run",
            "reflection_horizon": reflection_horizon,
            "examples": dynamic_examples,
            "full_weight_trace": os.path.join(hmod_eval["run_dir"], "weight_trace.jsonl"),
        },
    )

    report = {
        "run_dir": run_dir,
        "summary": {
            "num_cases": len(scenarios),
            "num_objectives": len(objective_library.intents),
            "num_basic_skills": len(controller.skill_library.basic_skills),
            "num_advanced_skills": len(controller.skill_library.advanced_skills),
            "reflection_horizon": reflection_horizon,
            "controller_mode": controller_mode,
        },
        "data_flow": [
            {
                "step": "Input data",
                "artifact": os.path.join(run_dir, "01_input_training_cases.json"),
                "description": "Scenario YAML + objective file are loaded as buyer-agent cases.",
            },
            {
                "step": "Objective transformation",
                "artifact": os.path.join(run_dir, "01_input_training_cases.jsonl"),
                "description": "Each ambiguous buyer objective is mapped to initial_w, stage_weights, and adaptation_rules.",
            },
            {
                "step": "Phase 1 skill library",
                "artifact": os.path.join(run_dir, "03_phase1_skill_library.json"),
                "description": "Basic and advanced W skills that the real DMORL/H-MOD trainer consumes.",
            },
            {
                "step": "Phase 2 dynamic W",
                "artifact": os.path.join(run_dir, "04_phase2_dynamic_weight_examples.json"),
                "description": (
                    "Self-reflection every T turns produces W_t. In llm_reflection mode, "
                    "the controller uses only macro_goal and visible dialogue; simulator drift "
                    "trace is kept for metric computation."
                ),
            },
            {
                "step": "Evaluation",
                "artifact": hmod_eval["run_dir"],
                "description": "Dialogue, weight_trace, violation_trace, human_audit, and aggregate metrics.",
            },
        ],
        "hmod_dynamic_metrics": hmod_eval["metrics"],
        "hmod_dynamic_run_dir": hmod_eval["run_dir"],
        "hmod_dynamic_controller_mode": hmod_eval["controller_mode"],
        "padpp_static_metrics": baseline_eval["metrics"] if baseline_eval else None,
        "padpp_static_run_dir": baseline_eval["run_dir"] if baseline_eval else None,
    }
    _write_json(os.path.join(run_dir, "flow_report.json"), report)
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario_file",
        default="config/scenario/hmod_buyer_drift_scenarios.yaml",
    )
    parser.add_argument(
        "--objective_file",
        default="config/scenario/hmod_buyer_objectives.py",
    )
    parser.add_argument("--output_dir", default="outputs/hmod_training_flow")
    parser.add_argument("--num_cases", type=int, default=10)
    parser.add_argument("--reflection_horizon", type=int, default=2)
    parser.add_argument("--audit_sample_size", type=int, default=5)
    parser.add_argument("--compare_baseline", action="store_true")
    parser.add_argument(
        "--controller_mode",
        choices=["rule_scaffold", "llm_reflection"],
        default="rule_scaffold",
        help="Use rule_scaffold for fast offline tests or llm_reflection for the paper path.",
    )
    parser.add_argument("--llm_model", default=None, help="Optional override for DEEPINFRA_MODEL.")
    parser.add_argument("--llm_api_key", default=None)
    parser.add_argument("--llm_api_key_env", default="DEEPINFRA_API_KEY")
    parser.add_argument("--llm_base_url", default=None, help="Optional override for DEEPINFRA_BASE_URL.")
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument("--llm_max_tokens", type=int, default=500)
    parser.add_argument("--llm_fallback_to_rule", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    report = simulate_flow(
        scenario_file=args.scenario_file,
        objective_file=args.objective_file,
        output_dir=args.output_dir,
        num_cases=args.num_cases,
        reflection_horizon=args.reflection_horizon,
        audit_sample_size=args.audit_sample_size,
        compare_baseline=args.compare_baseline,
        controller_mode=args.controller_mode,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        llm_api_key_env=args.llm_api_key_env,
        llm_base_url=args.llm_base_url,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        llm_fallback_to_rule=args.llm_fallback_to_rule,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
