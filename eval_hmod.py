"""Run H-MOD buyer-agent drift evaluation.

This runner is intentionally separate from run_dmorl.py. It evaluates the new
seller-drift simulator and H-MOD metrics while preserving the legacy
PADPP/DMORL buyer-agent role.
"""

import argparse
import json
import os
import time

from loguru import logger

from hmod.runner import run_and_write

# Tee loguru to a file in addition to the console. The default stderr sink is
# kept so --verbose dialogues still print live; a copy lands in logs/ so the
# turn-by-turn trace (seller intent, w_t, utterances) is persisted.
os.makedirs("logs", exist_ok=True)
_log_file = os.path.join("logs", f"hmod_eval_{time.strftime('%Y%m%d_%H%M%S')}.log")
logger.add(_log_file, level="INFO", encoding="utf-8",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario_file",
        default="config/scenario/hmod_buyer_drift_scenarios.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=["padpp_static", "hmod_dynamic", "hmod_no_mask"],
        default="hmod_dynamic",
    )
    parser.add_argument("--num_cases", type=int, default=None)
    parser.add_argument("--output_dir", default="outputs/hmod_eval")
    parser.add_argument("--audit_sample_size", type=int, default=50)
    parser.add_argument(
        "--judge_model",
        default="rule",
        help="Use 'rule' for deterministic offline judging, or an existing LLM backend name.",
    )
    parser.add_argument(
        "--use_llm_simulator",
        action="store_true",
        help="Use LLM to verbalize buyer responses; deterministic templates are used by default.",
    )
    parser.add_argument(
        "--objective_file",
        default="config/scenario/hmod_buyer_objectives.py",
        help=(
            "Optional Python assignment file defining BUYER_STRATEGY_INTENTS and "
            "BUYER_STRATEGY_MACRO_CLUSTERS."
        ),
    )
    parser.add_argument(
        "--objective_id",
        default=None,
        help="Optional buyer strategy intent id to apply to all scenarios.",
    )
    parser.add_argument(
        "--reflection_horizon",
        type=int,
        default=4,
        help="Macro-step length: hold w_local fixed and let the low policy take T "
        "actions, then re-select w_local. T=4 means 4 low-policy actions per w_local.",
    )
    parser.add_argument(
        "--controller_mode",
        choices=["rule_scaffold", "llm_reflection"],
        default="rule_scaffold",
        help=(
            "rule_scaffold keeps the deterministic test path; llm_reflection uses "
            "the paper path: one NL macro_goal -> LLM-reflected W_t."
        ),
    )
    parser.add_argument(
        "--llm_model",
        default=None,
        help="Optional override. Defaults to DEEPINFRA_MODEL from .env.",
    )
    parser.add_argument(
        "--llm_api_key",
        default=None,
        help="Optional override. Defaults to DEEPINFRA_API_KEY from .env.",
    )
    parser.add_argument(
        "--llm_api_key_env",
        default="DEEPINFRA_API_KEY",
        help="Environment variable containing the LLM API key.",
    )
    parser.add_argument(
        "--llm_base_url",
        default=None,
        help="Optional override. Defaults to DEEPINFRA_BASE_URL from .env.",
    )
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument("--llm_max_tokens", type=int, default=500)
    parser.add_argument(
        "--llm_fallback_to_rule",
        action="store_true",
        help="If LLM reflection fails, fall back to rule_scaffold and log llm_error.",
    )
    # ── Merged pipeline: use the trained R-PADPP neural low policy ──────────
    parser.add_argument(
        "--low_policy_checkpoint",
        default=None,
        help=(
            "Path to a trained R-PADPP/DMORL checkpoint (e.g. dmorl_phase2.pth). "
            "When set, the buyer turns are driven by the neural low policy "
            "(w_t -> action) instead of the rule scaffold buyer."
        ),
    )
    parser.add_argument("--low_policy_scenario", default="negotiation")
    parser.add_argument("--low_policy_datasets", default="craigslist_bargain")
    parser.add_argument("--low_policy_models", default="dmorl")
    parser.add_argument("--low_policy_gen_models", default="fpt")
    parser.add_argument("--low_policy_model_type", default="fpt")
    # ── Experience accumulation for the LLM controller ─────────────────────
    parser.add_argument(
        "--use_experience_buffer",
        action="store_true",
        help="Feed past episode outcomes into the LLM reflection to improve w_t over time.",
    )
    parser.add_argument(
        "--experience_path",
        default="outputs/hmod_experience.json",
        help="Where the cross-episode experience buffer is persisted.",
    )
    parser.add_argument(
        "--hints_file",
        default=None,
        help="Load a distilled general-hint playbook (from train_hmod.py) and "
        "inject it into the LLM reflection prompt during inference.",
    )
    # ── Sample-dialogue logging + dialogue length ──────────────────────────
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each sample dialogue turn-by-turn: seller intent_state, the "
        "LLM-chosen w_t, and both buyer/seller utterances.",
    )
    parser.add_argument(
        "--turn_limit_mult",
        type=float,
        default=1.0,
        help="Multiply each scenario's turn limit (e.g. 3.0 = triple the dialogue "
        "length). Metrics (GSR/T2DA) use the scaled limit too.",
    )
    return parser.parse_args()


def _build_neural_buyer_policy(args):
    """Construct the neural low policy buyer when a checkpoint is provided."""
    from hmod.low_policy import NeuralLowPolicy
    from hmod.policy import NeuralBuyerPolicy

    low_policy = NeuralLowPolicy(
        checkpoint=args.low_policy_checkpoint,
        scenario=args.low_policy_scenario,
        datasets=args.low_policy_datasets,
        models=args.low_policy_models,
        gen_models=args.low_policy_gen_models,
        model_type=args.low_policy_model_type,
    )
    return NeuralBuyerPolicy(act_fn=low_policy.act)


def main():
    args = parse_args()
    buyer_policy = None
    if args.low_policy_checkpoint:
        buyer_policy = _build_neural_buyer_policy(args)
    experience_buffer = None
    if args.use_experience_buffer:
        from hmod.experience import ExperienceBuffer
        experience_buffer = ExperienceBuffer(path=args.experience_path)
    hint_provider = None
    if args.hints_file:
        from hmod.hints import HintStore
        hint_store = HintStore(path=args.hints_file)
        if hint_store.is_empty():
            logger.warning(f"--hints_file {args.hints_file} has no hints; running without a playbook.")
        else:
            logger.info(f"Loaded {len(hint_store.hints)} general hints from {args.hints_file}")
            hint_provider = hint_store.provider()
    result = run_and_write(
        scenario_file=args.scenario_file,
        mode=args.mode,
        num_cases=args.num_cases,
        output_dir=args.output_dir,
        audit_sample_size=args.audit_sample_size,
        judge_model=args.judge_model,
        use_llm_simulator=args.use_llm_simulator,
        objective_file=args.objective_file,
        objective_id=args.objective_id,
        reflection_horizon=args.reflection_horizon,
        controller_mode=args.controller_mode,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        llm_api_key_env=args.llm_api_key_env,
        llm_base_url=args.llm_base_url,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        llm_fallback_to_rule=args.llm_fallback_to_rule,
        buyer_policy=buyer_policy,
        experience_buffer=experience_buffer,
        verbose=args.verbose,
        turn_limit_mult=args.turn_limit_mult,
        hint_provider=hint_provider,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info(f"Console log saved to: {_log_file}")
    logger.info(f"Structured outputs (dialogues/weight_trace/metrics) in: {result.get('run_dir')}")


if __name__ == "__main__":
    main()
