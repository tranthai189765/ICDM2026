"""TRIP baseline runner for HMOD bargain / recommendation scenarios.

Mirrors PPDPP/run.py + dpdp_baseline/run.py and adds TRIP-specific flags:
- `--trip_use_uasp`         enable Theory-of-Mind prefix (User-Aware Strategic Planning)
- `--trip_disable_tom_llm`  use heuristic ToM (saves LLM tokens for smoke runs)
- `--trip_use_pbtp`         enable population-based persona sampling per episode
- `--trip_population_size`  size of the persona pool (default 40)
- `--trip_tom_cache`        bounded LRU size for ToM responses

Example (HMOD bargain test, eval only, no LLM ToM):
    .venv/bin/python -m trip_baseline.run \\
        --data_name cb --system chatgpt --user chatgpt --critic chatgpt \\
        --test_scenario_file config/scenario/generated/hmod_bargain_test_scenarios.yaml \\
        --num_cases 40 --objective uniform --judge_model rule \\
        --trip_use_uasp --trip_disable_tom_llm --trip_use_pbtp \\
        --max_turn 8 --cache_dir cache/hf \\
        --output_dir outputs/trip_eval_bargain --do_eval
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from itertools import count

import torch
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_PPDPP_DIR = os.path.join(_REPO_ROOT, "PPDPP")
for path in (_REPO_ROOT, _PPDPP_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from env import Env, clear_llm_judge_cache  # noqa: E402  -- PPDPP/env.py
from utils import (  # noqa: E402  -- PPDPP/utils.py
    TMP_DIR,
    enablePrint,
    load_dataset,
    save_rl_mtric,
    set_random_seed,
)
from transformers import BertConfig, BertTokenizer, RobertaConfig, RobertaTokenizer  # noqa: E402
from fastchat.model import add_model_args  # noqa: E402
from scenario_loader import load_custom_dataset  # noqa: E402
from ppdpp_rewards import (  # noqa: E402
    OBJECTIVE_WEIGHTS,
    aggregate_episode_records,
    subgroup_metrics,
    write_evaluation_outputs,
)

from trip_baseline.agent import TRIPAgent  # noqa: E402
from trip_baseline.env_wrapper import (  # noqa: E402
    TRIPEnv,
    install_persona_aware_message_format,
    restore_default_message_format,
)
from trip_baseline.personas import build_persona_pool  # noqa: E402
from trip_baseline.tom import TheoryOfMind  # noqa: E402


_TOK = {"bert": BertTokenizer, "roberta": RobertaTokenizer}
_CFG = {"bert": BertConfig, "roberta": RobertaConfig}


def _memory_cleanup(clear_judge_cache: bool = False) -> None:
    if clear_judge_cache:
        clear_llm_judge_cache()
    gc.collect()
    if torch.backends.mps.is_available() and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _resolve_runtime_device(requested: str) -> str:
    req = str(requested or "").lower()
    if req == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if req == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if req == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _make_env(args, dataset, mode, persona_pool, train_env=None):
    env_model = getattr(train_env, "vicuna_model", None) if train_env else None
    env_tokenizer = getattr(train_env, "vicuna_tokenizer", None) if train_env else None
    if args.trip_use_pbtp and persona_pool:
        return TRIPEnv(
            args, dataset, mode=mode,
            env_model=env_model, env_tokenizer=env_tokenizer,
            persona_pool=persona_pool,
        )
    return Env(args, dataset, mode=mode, env_model=env_model, env_tokenizer=env_tokenizer)


def evaluate(args, dataset, policy, filename, i_episode, train_env, persona_pool,
             eval_limit=None, eval_tag=None):
    test_env = _make_env(args, dataset, "test", persona_pool, train_env=train_env)
    clear_llm_judge_cache()
    if policy.tom is not None:
        policy.tom.clear()
    set_random_seed(args.seed)

    SR, AvgT, total_reward = 0, 0, 0
    SR_turn = [0] * args.max_turn
    episode_records = []
    dialogue_records = []
    test_size = len(test_env.dataset)
    if eval_limit is not None:
        test_size = min(test_size, int(eval_limit))
    if test_size <= 0:
        print("[trip] empty test split; skipping eval")
        return [0.0, 0.0, 0.0]

    print("Test size: ", test_size)
    prefix = f"{eval_tag}-" if eval_tag else ""
    test_filename = "{}Evaluate-epoch-{}-".format(prefix, i_episode) + filename
    record_filename = "{}Record-epoch-{}-".format(prefix, i_episode) + filename
    rec_dir = os.path.join(TMP_DIR[args.data_name], "eval_result")
    os.makedirs(rec_dir, exist_ok=True)
    rec_path = os.path.join(rec_dir, record_filename + ".txt")

    with open(rec_path, "w") as rec_file:
        for test_num in tqdm(range(test_size)):
            print("\n================test tuple:{}====================".format(test_num))
            epi_reward = 0
            done = 0
            state = test_env.reset()
            for t in count():
                action = policy.select_action(state, is_test=True)
                state, reward, done = test_env.step(action)
                if (
                    args.data_name == "cb"
                    and getattr(args, "objective", None) not in OBJECTIVE_WEIGHTS
                    and reward < 0
                ):
                    reward = 0
                epi_reward += reward
                if done:
                    if done == 1:
                        SR_turn = [v + 1 if i > t else v for i, v in enumerate(SR_turn)]
                        SR += 1
                    total_reward += epi_reward
                    AvgT += t + 1
                    persona_id = (getattr(test_env, "last_persona", None) or {}).get("id")
                    rec_file.write("%s\n\n" % str({
                        "dialog": state,
                        "reward": epi_reward,
                        "persona_id": persona_id,
                    }))
                    if (
                        args.data_name == "cb"
                        and getattr(args, "objective", None) in OBJECTIVE_WEIGHTS
                    ):
                        episode_records.append(test_env.get_episode_record())
                        dialogue_records.append(test_env.get_dialogue_record())
                    break
            enablePrint()

    if episode_records:
        metrics = aggregate_episode_records(episode_records, args.objective)
        SR_mean = metrics["sr"]
        AvgT_mean = metrics["avg_turn"]
        reward_mean = metrics["weighted_return"]
    else:
        SR_mean = float(SR) / test_size
        AvgT_mean = float(AvgT) / test_size
        reward_mean = total_reward / test_size
        metrics = {
            "num_dialogues": test_size,
            "objective": getattr(args, "objective", None),
            "sr": SR_mean,
            "deal_rate": SR_mean,
            "avg_turn": AvgT_mean,
            "weighted_return": reward_mean,
            "t2da": None,
            "t2da_status": "not_applicable",
        }

    SR_all = [SR_mean, AvgT_mean, reward_mean]
    save_rl_mtric(
        dataset=args.data_name,
        filename=test_filename,
        epoch=test_size,
        SR=SR_all,
        mode="test",
    )

    tom_stats = (
        {"calls": policy.tom.calls,
         "cache_hits": policy.tom.cache_hits,
         "fallback_uses": policy.tom.fallback_uses}
        if policy.tom is not None
        else None
    )

    if episode_records:
        metrics_payload = {
            "model": "TRIP",
            "epoch": i_episode,
            "filename": filename,
            "data_name": args.data_name,
            "scenario_file": (
                getattr(args, "test_scenario_file", None)
                or getattr(args, "scenario_file", None)
            ),
            "objective": args.objective,
            "objective_weight": OBJECTIVE_WEIGHTS[args.objective],
            "metrics": metrics,
            "metrics_by_drift_mode": subgroup_metrics(
                episode_records, args.objective, "drift_mode"
            ),
            "metrics_by_persona": subgroup_metrics(
                episode_records, args.objective, "seller_persona_type"
            ),
            "metrics_by_objective": subgroup_metrics(
                episode_records, args.objective, "buyer_intent_id"
            ),
            "metrics_by_recommendation_domain": subgroup_metrics(
                episode_records, args.objective, "recommendation_domain"
            ),
            "num_dialogues": len(episode_records),
            "judge_model": getattr(args, "judge_model", None),
            "trip_use_uasp": args.trip_use_uasp,
            "trip_use_pbtp": args.trip_use_pbtp,
            "trip_disable_tom_llm": args.trip_disable_tom_llm,
            "trip_population_size": len(persona_pool) if persona_pool else 0,
            "tom_stats": tom_stats,
        }
        run_dir = os.path.join(
            args.output_dir,
            f"{filename}-{eval_tag or 'eval'}-epoch-{i_episode}-{time.strftime('%Y%m%d_%H%M%S')}",
        )
        write_evaluation_outputs(run_dir, metrics_payload, episode_records, dialogue_records)
        print(f"TRIP structured outputs written to {run_dir}")

    print("TRIP eval -> SR:{}, AvgT:{}, reward:{}".format(SR_mean, AvgT_mean, reward_mean))
    if tom_stats is not None:
        print("ToM usage:", tom_stats)

    del episode_records
    del dialogue_records
    del test_env
    _memory_cleanup(clear_judge_cache=True)
    return SR_all


def train(args, config, dataset, filename, tokenizer):
    persona_pool = build_persona_pool(args.trip_population_size) if args.trip_use_pbtp else None
    if args.trip_use_pbtp:
        print(f"[trip] PBTP enabled with {len(persona_pool)} personas")
        # Pre-install the patched CB message format even before TRIPEnv
        # exists (eval-only runs still need it for the eval env construction).
        install_persona_aware_message_format(args.data_name)
    else:
        restore_default_message_format(args.data_name)

    tom = None
    if args.trip_use_uasp:
        tom = TheoryOfMind(
            task=args.trip_tom_task,
            cache_size=args.trip_tom_cache,
            use_llm=not args.trip_disable_tom_llm,
            max_tokens=args.trip_tom_max_tokens,
            model=args.trip_tom_model,
            base_url=args.trip_tom_base_url,
        )
        endpoint_info = tom.describe_endpoint()
        print(f"[trip] UASP enabled (use_llm={not args.trip_disable_tom_llm}, task={args.trip_tom_task})")
        print(f"[trip] ToM endpoint -> {endpoint_info}")

    train_env = _make_env(args, dataset, "train", persona_pool)

    set_random_seed(args.seed)
    policy = TRIPAgent(args, config, tokenizer, tom=tom)

    if args.sft_dir is not None:
        print("Loading policy from {}".format(args.sft_dir))
        policy.load_model(data_name=args.data_name, filename=args.sft_dir)

    if args.load_rl_epoch > 0:
        print("Loading RL checkpoint epoch {}".format(args.load_rl_epoch))
        policy.load_model(
            data_name=args.data_name,
            filename=filename,
            epoch_user=args.load_rl_epoch,
        )

    test_performance = []
    if args.do_eval:
        initial_full_eval = args.do_eval and (not args.do_train)
        SR15_mean = evaluate(
            args, dataset, policy, filename, 0, train_env, persona_pool,
            eval_limit=None if initial_full_eval else args.quick_eval_num_cases,
            eval_tag="full" if initial_full_eval else "quick",
        )
        test_performance = [SR15_mean]
        _memory_cleanup(clear_judge_cache=True)

    if not args.do_train:
        return

    start_step = max(int(getattr(args, "load_rl_epoch", 0) or 0) + 1, 1)
    for train_step in range(start_step, args.max_steps + 1):
        SR, AvgT, total_reward = 0.0, 0.0, 0.0
        loss = torch.tensor(0, dtype=torch.float, device=args.device)
        for i_episode in tqdm(range(args.sample_times), desc="sampling"):
            print("\n================new tuple:{}====================".format(i_episode))
            state = train_env.reset()
            epi_reward = 0
            for t in count():
                action = policy.select_action(state, is_test=False)
                state, reward, done = train_env.step(action)
                epi_reward += reward
                reward = torch.tensor([reward], device=args.device, dtype=torch.float)
                policy.rewards.append(reward)
                if done:
                    if done == 1:
                        SR += 1
                    AvgT += t + 1
                    total_reward += epi_reward
                    break
            newloss = policy.optimize_model()
            if newloss is not None:
                loss += newloss
            if i_episode % 10 == 0:
                _memory_cleanup(clear_judge_cache=False)

        enablePrint()
        print("loss : {} epoch {}".format(loss.item() / max(args.sample_times, 1), train_step))
        print("SR:{} AvgT:{} reward:{} sample_times:{}".format(
            SR / max(args.sample_times, 1),
            AvgT / max(args.sample_times, 1),
            total_reward / max(args.sample_times, 1),
            args.sample_times,
        ))

        if train_step % args.eval_num == 0:
            is_full_eval = (train_step % args.full_eval_every == 0)
            eval_limit = None if is_full_eval else args.quick_eval_num_cases
            eval_tag = "full" if is_full_eval else "quick"
            SR_all = evaluate(
                args, dataset, policy, filename, train_step, train_env, persona_pool,
                eval_limit=eval_limit, eval_tag=eval_tag,
            )
            test_performance.append(SR_all)
            _memory_cleanup(clear_judge_cache=True)
        if train_step % args.save_num == 0:
            policy.save_model(
                data_name=args.data_name,
                filename=filename,
                epoch_user=train_step,
            )
        _memory_cleanup(clear_judge_cache=True)
    print(test_performance)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TRIP baseline runner")
    parser.add_argument("--seed", "-seed", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--epochs", "-me", type=int, default=50000)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--data_name", type=str, default="cb",
                        choices=["esc", "cima", "cb"])
    parser.add_argument("--system", type=str, default="chatgpt",
                        choices=["vicuna", "chatgpt", "llama2"])
    parser.add_argument("--user", type=str, default="chatgpt",
                        choices=["vicuna", "chatgpt", "llama2"])
    parser.add_argument("--critic", type=str, default="chatgpt",
                        choices=["vicuna", "chatgpt", "llama2"])
    parser.add_argument("--sft_dir", default=None, type=str)
    parser.add_argument("--max_turn", type=int, default=8)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--load_rl_epoch", type=int, default=0)
    parser.add_argument("--scenario_file", type=str, default=None)
    parser.add_argument("--test_scenario_file", type=str, default=None)
    parser.add_argument("--valid_scenario_file", type=str, default=None)
    parser.add_argument("--objective", type=str, default="uniform",
                        choices=sorted(OBJECTIVE_WEIGHTS.keys()))
    parser.add_argument("--num_cases", type=int, default=None)
    parser.add_argument("--train_num_cases", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/trip_eval")
    parser.add_argument("--full_eval_every", type=int, default=3)
    parser.add_argument("--quick_eval_num_cases", type=int, default=20)
    parser.add_argument("--judge_model", type=str, default="rule",
                        choices=["critic", "rule", "llm"])
    parser.add_argument("--use_case_turn_limit", action="store_true")

    # TRIP-specific flags
    parser.add_argument("--trip_use_uasp", action="store_true",
                        help="Enable Theory-of-Mind prefix injection (UASP).")
    parser.add_argument("--trip_disable_tom_llm", action="store_true",
                        help="Use heuristic ToM (skip LLM calls, useful for smoke runs).")
    parser.add_argument("--trip_use_pbtp", action="store_true",
                        help="Enable population-based persona sampling (PBTP).")
    parser.add_argument("--trip_population_size", type=int, default=40,
                        help="Size of the persona pool (paper default: 40).")
    parser.add_argument("--trip_tom_cache", type=int, default=256,
                        help="Bounded LRU size for ToM response cache.")
    parser.add_argument("--trip_tom_task", type=str, default="cb",
                        choices=["cb", "p4g", "esc"],
                        help="Task tag picking the ToM prompt template.")
    parser.add_argument("--trip_tom_max_tokens", type=int, default=128,
                        help="Max tokens for ToM LLM responses.")
    parser.add_argument("--trip_tom_model", type=str, default=None,
                        help="Override the ToM LLM model (else uses FPT_MODEL / POLICY_LLM_MODEL from .env).")
    parser.add_argument("--trip_tom_base_url", type=str, default=None,
                        help="Override the ToM LLM base URL (else uses FPT_API_URL from .env).")

    # PPDPP infra knobs reused as-is
    parser.add_argument("--cache_dir", default="/storage_fast/ydeng/plm", type=str)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_path", type=str,
                        default="/storage_fast/ydeng/llm/vicuna_hf/7B")
    parser.add_argument("--model_name", type=str, default="roberta")
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str)
    parser.add_argument("--do_lower_case", action="store_false")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--sample_times", type=int, default=100)
    parser.add_argument("--eval_num", type=int, default=1)
    parser.add_argument("--save_num", type=int, default=1)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    add_model_args(parser)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    args.device = _resolve_runtime_device(getattr(args, "device", "cuda"))
    print("device:", args.device)
    print("data_set:", args.data_name)
    print("trip flags -> uasp:", args.trip_use_uasp,
          "pbtp:", args.trip_use_pbtp,
          "pop_size:", args.trip_population_size,
          "disable_tom_llm:", args.trip_disable_tom_llm)

    if args.scenario_file or args.test_scenario_file or args.valid_scenario_file:
        if args.data_name != "cb":
            raise ValueError(
                "HMOD scenarios use the Craigslist Bargain interface; set --data_name cb."
            )
        args.use_case_turn_limit = True
        dataset = load_custom_dataset(
            train_path=args.scenario_file,
            test_path=args.test_scenario_file,
            valid_path=args.valid_scenario_file,
            train_limit=args.train_num_cases,
            test_limit=args.num_cases,
            valid_limit=args.num_cases,
        )
    else:
        dataset = load_dataset(args.data_name)

    filename = "trip-{}-{}-{}-{}-{}-{}".format(
        args.data_name, args.objective, args.sft_dir or "noSFT",
        args.system, args.user, args.critic,
    )

    config = _CFG[args.model_name].from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir,
    )
    tokenizer = _TOK[args.model_name].from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir,
    )

    if args.sft_dir:
        args.sft_dir = os.path.join(args.sft_dir, args.data_name, args.model_name, "best_checkpoint")
    if args.sft_dir and not os.path.exists(args.sft_dir):
        print("no sft model, randomly initialize policy model")
        args.sft_dir = None

    train(args, config, dataset, filename, tokenizer)


if __name__ == "__main__":
    main()
