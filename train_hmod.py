"""H-MOD hint training: self-play the LLM meta-controller, distil general hints.

The LLM controller plays repeated drift negotiations with the trained R-PADPP
low policy, reads back the metric feedback (GSR / T2DA / CVR / llm_sr) with an
explanatory glossary, and distils a reusable playbook of general hints. The
playbook is saved to JSON; pass it to eval_hmod.py via --hints_file so the same
hints ground inference at evaluation time.

Example:
    python train_hmod.py --epochs 5 --llm_model fpt \
        --low_policy_checkpoint checkpoints/dmorl_phase2_best.pth \
        --low_policy_gen_models fpt --low_policy_model_type fpt \
        --judge_model fpt --turn_limit_mult 2.0 --hints_out outputs/hmod_hints.json
"""

import argparse
import os
import time

from loguru import logger

from hmod.hint_distiller import LLMHintDistiller
from hmod.hint_trainer import HMODHintTrainer
from hmod.hints import HintStore
from hmod.llm_reflection import LLMWeightReflector
from hmod.scenario import load_scenarios

# Tee loguru to a training log file (keep console for live progress + hints).
os.makedirs("logs", exist_ok=True)
_log_file = os.path.join("logs", f"hmod_train_{time.strftime('%Y%m%d_%H%M%S')}.log")
logger.add(_log_file, level="INFO", encoding="utf-8",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario_file", default="config/scenario/hmod_buyer_drift_scenarios.yaml")
    p.add_argument("--num_cases", type=int, default=None,
                   help="Limit number of training scenarios (default: all).")
    p.add_argument("--mode", default="hmod_dynamic",
                   choices=["hmod_dynamic", "hmod_no_mask"],
                   help="Drift eval mode used during self-play.")
    p.add_argument("--epochs", type=int, default=5,
                   help="Self-play + distillation rounds.")
    p.add_argument("--judge_model", default="rule",
                   help="Deal judge for self-play feedback (e.g. fpt or rule).")
    p.add_argument("--reflection_horizon", type=int, default=4,
                   help="Macro-step length: hold w_local fixed for T low-policy "
                        "actions, then re-select w_local (default 4).")
    p.add_argument("--turn_limit_mult", type=float, default=1.0)
    p.add_argument("--use_llm_simulator", action="store_true")
    p.add_argument("--verbose", action="store_true",
                   help="Print each self-play dialogue turn-by-turn.")
    # ── LLM controller / distiller backend ─────────────────────────────────
    p.add_argument("--llm_model", default=None,
                   help="Controller+distiller model. Use 'fpt' to reuse FPT_* from .env.")
    p.add_argument("--llm_api_key", default=None)
    p.add_argument("--llm_api_key_env", default="DEEPINFRA_API_KEY")
    p.add_argument("--llm_base_url", default=None)
    p.add_argument("--llm_temperature", type=float, default=0.0)
    p.add_argument("--llm_max_tokens", type=int, default=700)
    p.add_argument("--no_fallback_to_rule", action="store_true",
                   help="Disable the rule controller fallback during self-play.")
    # ── Low policy (trained R-PADPP) ───────────────────────────────────────
    p.add_argument("--low_policy_checkpoint", default=None,
                   help="R-PADPP checkpoint. If omitted, the rule buyer policy is used.")
    p.add_argument("--low_policy_scenario", default="negotiation")
    p.add_argument("--low_policy_datasets", default="craigslist_bargain")
    p.add_argument("--low_policy_models", default="dmorl")
    p.add_argument("--low_policy_gen_models", default="fpt")
    p.add_argument("--low_policy_model_type", default="fpt")
    # ── Hint playbook output ───────────────────────────────────────────────
    p.add_argument("--hints_out", default="outputs/hmod_hints.json",
                   help="Where the distilled general hints are saved.")
    p.add_argument("--max_hints", type=int, default=12)
    p.add_argument("--resume_hints", action="store_true",
                   help="Continue from the existing --hints_out playbook instead of starting empty.")
    return p.parse_args()


def _build_neural_buyer_policy(args):
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
        logger.info(f"Loading R-PADPP low policy from {args.low_policy_checkpoint}")
        buyer_policy = _build_neural_buyer_policy(args)
    else:
        logger.warning("No --low_policy_checkpoint given; self-play uses the rule buyer policy.")

    scenarios = load_scenarios(args.scenario_file, limit=args.num_cases)
    logger.info(f"Loaded {len(scenarios)} training scenarios from {args.scenario_file}")

    # Shared LLM backend for the distiller (controller builds its own internally
    # with the same flags, so both hit the same FPT/DeepInfra endpoint).
    distiller_reflector = LLMWeightReflector(
        model=args.llm_model,
        api_key=args.llm_api_key,
        api_key_env=args.llm_api_key_env,
        base_url=args.llm_base_url,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens,
    )
    distiller = LLMHintDistiller(distiller_reflector, max_hints=args.max_hints)

    # Start fresh unless resuming. (HintStore auto-loads if the file exists.)
    if not args.resume_hints and os.path.exists(args.hints_out):
        os.remove(args.hints_out)
    hint_store = HintStore(path=args.hints_out, max_hints=args.max_hints)

    trainer = HMODHintTrainer(
        scenarios=scenarios,
        buyer_policy=buyer_policy,
        hint_store=hint_store,
        distiller=distiller,
        mode=args.mode,
        judge_model=args.judge_model,
        reflection_horizon=args.reflection_horizon,
        turn_limit_mult=args.turn_limit_mult,
        verbose=args.verbose,
        use_llm_simulator=args.use_llm_simulator,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        llm_api_key_env=args.llm_api_key_env,
        llm_base_url=args.llm_base_url,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        llm_fallback_to_rule=not args.no_fallback_to_rule,
    )
    trainer.train(args.epochs)

    logger.info(f"Hints playbook saved to: {args.hints_out}")
    logger.info(f"Training console log: {_log_file}")
    print(f"\nSaved {len(hint_store.hints)} hints to {args.hints_out}")
    print("Use it at eval time:  python eval_hmod.py ... --hints_file " + args.hints_out)


if __name__ == "__main__":
    main()
