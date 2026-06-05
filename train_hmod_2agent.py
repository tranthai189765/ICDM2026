"""Two-agent H-MOD hint training.

Trains two independent LLM agents on the intent-drift environment and distils
one hint playbook each:
  * Intent-Drift Detector  -> --detector_hints_out
  * High-Policy w_local LLM -> --policy_hints_out

During training the GOLD seller intent drives w_local (the detector is scored
against it but does not steer w_local). Use the playbooks at eval time with
eval_hmod.py --two_agent --detector_hints_file ... --policy_hints_file ...

Example:
    python train_hmod_2agent.py --epochs 6 --llm_model fpt \
        --scenario_file config/scenario/generated/hmod_bargain_train_scenarios.yaml \
        --num_cases 150 \
        --low_policy_checkpoint checkpoints/dmorl_phase2_best.pth \
        --low_policy_gen_models fpt --low_policy_model_type fpt \
        --judge_model rule
"""

import argparse
import os
import time

from loguru import logger

from hmod.hints import HintStore
from hmod.llm_reflection import LLMWeightReflector
from hmod.scenario import load_scenarios
from hmod.two_agent_trainer import TwoAgentHintTrainer

os.makedirs("logs", exist_ok=True)
_log_file = os.path.join("logs", f"hmod_train2_{time.strftime('%Y%m%d_%H%M%S')}.log")
logger.add(_log_file, level="INFO", encoding="utf-8",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario_file",
                   default="config/scenario/generated/hmod_bargain_train_scenarios.yaml")
    p.add_argument("--fewshot_scenario_file", default=None,
                   help="Train file used to build detector few-shots (defaults to --scenario_file).")
    p.add_argument("--num_cases", type=int, default=None)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--judge_model", default="fpt",
                   help="Deal judge for self-play feedback. Default 'fpt' (LLM); use 'rule' to save tokens.")
    p.add_argument("--turn_limit_mult", type=float, default=1.5,
                   help="Scale each scenario turn limit. Default 1.5 ensures the intent "
                        "drift fires and leaves room to adapt (1.0 = raw scenario limit).")
    p.add_argument("--use_llm_simulator", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no_fallback_to_rule", action="store_true")
    # LLM backend (shared by both agents + both distillers)
    p.add_argument("--llm_model", default=None,
                   help="Use 'fpt' to reuse FPT_* from .env.")
    p.add_argument("--llm_api_key", default=None)
    p.add_argument("--llm_api_key_env", default="DEEPINFRA_API_KEY")
    p.add_argument("--llm_base_url", default=None)
    p.add_argument("--llm_temperature", type=float, default=0.0)
    p.add_argument("--llm_max_tokens", type=int, default=700)
    # Low policy (frozen R-PADPP)
    p.add_argument("--low_policy_checkpoint", default=None)
    p.add_argument("--low_policy_scenario", default="negotiation")
    p.add_argument("--low_policy_datasets", default="craigslist_bargain")
    p.add_argument("--low_policy_models", default="dmorl")
    p.add_argument("--low_policy_gen_models", default="fpt")
    p.add_argument("--low_policy_model_type", default="fpt")
    # Hint outputs
    p.add_argument("--policy_hints_out", default="outputs/hmod_policy_hints.json")
    p.add_argument("--detector_hints_out", default="outputs/hmod_detector_hints.json")
    p.add_argument("--max_hints", type=int, default=12)
    p.add_argument("--resume_hints", action="store_true")
    return p.parse_args()


def _build_neural_buyer_policy(args):
    from hmod.low_policy import NeuralLowPolicy
    from hmod.policy import NeuralBuyerPolicy
    low = NeuralLowPolicy(
        checkpoint=args.low_policy_checkpoint,
        scenario=args.low_policy_scenario,
        datasets=args.low_policy_datasets,
        models=args.low_policy_models,
        gen_models=args.low_policy_gen_models,
        model_type=args.low_policy_model_type,
    )
    return NeuralBuyerPolicy(act_fn=low.act)


def main():
    args = parse_args()

    buyer_policy = None
    if args.low_policy_checkpoint:
        logger.info(f"Loading R-PADPP low policy from {args.low_policy_checkpoint}")
        buyer_policy = _build_neural_buyer_policy(args)
    else:
        logger.warning("No --low_policy_checkpoint; self-play uses the rule buyer policy.")

    scenarios = load_scenarios(args.scenario_file, limit=args.num_cases)
    logger.info(f"Loaded {len(scenarios)} scenarios from {args.scenario_file}")

    reflector = LLMWeightReflector(
        model=args.llm_model, api_key=args.llm_api_key,
        api_key_env=args.llm_api_key_env, base_url=args.llm_base_url,
        temperature=args.llm_temperature, max_tokens=args.llm_max_tokens,
    )

    for path in (args.policy_hints_out, args.detector_hints_out):
        if not args.resume_hints and os.path.exists(path):
            os.remove(path)
    policy_hints = HintStore(path=args.policy_hints_out, max_hints=args.max_hints)
    detector_hints = HintStore(path=args.detector_hints_out, max_hints=args.max_hints)

    trainer = TwoAgentHintTrainer(
        scenarios=scenarios,
        buyer_policy=buyer_policy,
        reflector=reflector,
        policy_hints=policy_hints,
        detector_hints=detector_hints,
        fewshot_scenario_file=args.fewshot_scenario_file or args.scenario_file,
        judge_model=args.judge_model,
        turn_limit_mult=args.turn_limit_mult,
        verbose=args.verbose,
        use_llm_simulator=args.use_llm_simulator,
        llm_fallback_to_rule=not args.no_fallback_to_rule,
    )
    trainer.train(args.epochs)

    logger.info(f"Policy hints   -> {args.policy_hints_out}")
    logger.info(f"Detector hints -> {args.detector_hints_out}")
    logger.info(f"Training log    -> {_log_file}")
    print(f"\nSaved {len(policy_hints.hints)} policy hints to {args.policy_hints_out}")
    print(f"Saved {len(detector_hints.hints)} detector hints to {args.detector_hints_out}")
    print("Eval:  python eval_hmod.py --two_agent "
          f"--policy_hints_file {args.policy_hints_out} "
          f"--detector_hints_file {args.detector_hints_out} ...")


if __name__ == "__main__":
    main()
