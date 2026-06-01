"""eval_dmorl.py — Standalone PADPP Table-2-style evaluator for a trained DMORL checkpoint.

Loads a saved model, runs N episodes per basic skill on the test set with the
skill's fixed weight vector, and prints per-skill + averaged metrics:
  SR (success rate), SL ratio, Fairness, DealRate, AvgTurns.

Example:
  python eval_dmorl.py \
      --scenario negotiation \
      --datasets craigslist_bargain \
      --models dmorl \
      --gen_models fpt --model_type fpt \
      --metrics sr,deal_rate,sl_ratio,fairness,avg_turn \
      --loggers terminal \
      --checkpoint checkpoints/negotiation/craigslist_bargain/DMORLModel_42/dmorl_phase1a.pth \
      --skills_file dmorl_skills_neg.json \
      --n_eval_episodes 20
"""
import argparse
import copy
import json
import os
import random
import sys
import time
from collections import defaultdict
from itertools import count

# ── Quiet terminal ──────────────────────────────────────────────────────────
os.environ["TQDM_DISABLE"] = "1"
import warnings as _warnings  # noqa: E402
import transformers as _transformers  # noqa: E402

_transformers.logging.set_verbosity_error()
_warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from accelerate import Accelerator, DistributedDataParallelKwargs  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from loguru import logger  # noqa: E402

from utils.utils import (  # noqa: E402
    get_datasets_by_names, get_model_by_names, reformat_args,
    get_metrics_by_names, load_config_from_yaml_file, get_scenario_by_name,
    get_text_generation_model_by_name,
    load_user_simulators, set_seed, parse_args,
)
from config.config import DatasetConfigForRecommendation  # noqa: E402
from config.constants import (  # noqa: E402
    BART_GENERATION, VICUNA, RECOMMENDATION, NEGOTIATION, EMOTIONAL_SUPPORT,
    SUCCESS_RATE, SL_RATIO, FAIRNESS, DEAL_RATE, AVG_TURN,
)
from eval.offline import OfflineEvaluator  # noqa: E402
from eval.online import OnlineEvaluator  # noqa: E402

load_dotenv()


# ── Loguru to file (and tee a short summary to stdout at the end) ──────────
logger.remove()
_log_dir = "logs"
os.makedirs(_log_dir, exist_ok=True)
_log_file = os.path.join(_log_dir, f"eval_{time.strftime('%Y%m%d_%H%M%S')}.log")
logger.add(_log_file, level="INFO", encoding="utf-8",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")


def parse_eval_args():
    """Standard PADPP args + a few eval-only flags."""
    base_args = parse_args()
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--checkpoint', type=str, required=True,
                       help='Path to a saved model .pth (full pickled model).')
    extra.add_argument('--skills_file', type=str, default=None,
                       help='Path to dmorl_skills_*.json. Default: model_config.skills_file '
                            'resolved relative to current dir.')
    extra.add_argument('--n_eval_episodes', type=int, default=20,
                       help='Episodes per skill.')
    extra.add_argument('--output_dir', type=str, default='eval_results',
                       help='Where to write the per-skill JSON + summary.')
    extra.add_argument('--include_advanced', action='store_true',
                       help='Also evaluate the advanced skills in the JSON.')
    extra.add_argument('--use_llm_price_extraction', action='store_true',
                       help='Use the LLM (gen model) to extract the buyer price in compute_reward')
    extra.add_argument('--use_price_tag', action='store_true',
                       help='Buyer/seller LLMs append a [[PRICE: x]] tag; compute_reward parses it directly')
    extra.add_argument('--fairness_train_scale', type=float, default=1.0,
                       help='Keep 1.0 at eval so reported r_fair matches the paper (max 0.5).')
    extra_args, _ = extra.parse_known_args(sys.argv[1:])
    return base_args, vars(extra_args)


def _episode(trainer, case, simulator, action_mapping, w, skill_name, max_horizon):
    """Run a single inference episode with fixed weight `w`. Returns metrics dict."""
    state = trainer.game.reset(case, simulator)
    state['w'] = np.array(w)

    epi_rewards = []
    conv_turn = 0
    done = 0

    for t in count():
        # is_test=True → greedy argmax (no epsilon exploration). Without this
        # the eval ran epsilon-greedy (10% random actions), injecting random
        # deny/greet/high-counter turns that depressed r_gain and added noise.
        action, _, _ = trainer.predict(
            state, torch.FloatTensor(w).to(trainer.device),
            action_mapping, is_test=True, is_computing_reward=False, use_gpi=False,
        )
        state, reward, done, _ = trainer.game.step(
            state, action, trainer.generation_method, simulator,
        )
        if isinstance(reward, list):
            epi_rewards.append(reward)
        else:
            epi_rewards.append([float(reward)])
        conv_turn = t + 1
        if done or t >= max_horizon - 1:
            break

    arr = np.array(epi_rewards)                 # [n_turns, n_obj]
    # PADPP paper convention: MEAN of each reward dimension across all agent
    # decision turns of the episode. See PADPP_original/padpp/trainer.py:1254
    # `epi_reward.mean(dim=0)`. Reward shaping is OFF (base/game.py was
    # reverted to PADPP-original), so no un-shaping is needed.
    mean_per_obj = arr.mean(axis=0)
    terminal = arr[-1]
    n_obj = arr.shape[1]
    sl   = float(mean_per_obj[0]) if n_obj >= 1 else 0.0
    fair = float(mean_per_obj[1]) if n_obj >= 2 else 0.0
    deal_rate = float(mean_per_obj[2]) if n_obj >= 3 else 0.0
    sl_terminal   = float(terminal[0]) if n_obj >= 1 else 0.0
    fair_terminal = float(terminal[1]) if n_obj >= 2 else 0.0
    deal_terminal = float(terminal[2]) if n_obj >= 3 else 0.0
    utterance_count = len(state['dialogue_context'])
    return {
        SUCCESS_RATE: int(done == 1),
        AVG_TURN: [conv_turn, 0],
        SL_RATIO: sl,
        FAIRNESS: fair,
        DEAL_RATE: deal_rate,
        'sl_terminal':   sl_terminal,
        'fair_terminal': fair_terminal,
        'deal_terminal': deal_terminal,
        'skill': skill_name,
        'n_turns': conv_turn,
        'n_utterances': utterance_count,
        'done_flag': int(done),
    }


def _aggregate(results):
    """Aggregate per-episode results into PADPP Table-2 columns.
    Column names follow the paper: SR, avg.turn, r_gain, r_fair, r_deal.

    avg_turn here is the UTTERANCE count (= 2 initial + 2*agent_steps),
    which matches what the PADPP paper reports in Table 2 (values 5-10).
    For the raw agent decision count use `avg_steps` instead.
    """
    n = max(len(results), 1)
    return {
        'SR':        sum(r[SUCCESS_RATE]    for r in results) / n,
        'avg_turn':  sum(r['n_utterances'] for r in results) / n,
        'avg_steps': sum(r[AVG_TURN][0]    for r in results) / n,
        # PADPP MEAN convention
        'r_gain':    sum(r[SL_RATIO]       for r in results) / n,
        'r_fair':    sum(r[FAIRNESS]       for r in results) / n,
        'r_deal':    sum(r[DEAL_RATE]      for r in results) / n,
        # Terminal-only convention (diagnostic)
        'r_gain_terminal': sum(r['sl_terminal']   for r in results) / n,
        'r_fair_terminal': sum(r['fair_terminal'] for r in results) / n,
        'r_deal_terminal': sum(r['deal_terminal'] for r in results) / n,
        'n':         n,
    }


# PADPP Table 2 reference numbers (Negotiation, MEAN convention).
# Keys are skill names in the eval skills_file. None = focal row, not measured.
_PADPP_REF_BY_SKILL = {
    "Uniform Balancer":      {'SR': 0.427, 'avg_turn': 9.638,
                              'r_gain': 0.622, 'r_fair': 0.287, 'r_deal': 0.142},
    "Price Gain Maximizer":  {'SR': 0.085, 'avg_turn': 9.898,
                              'r_gain': 0.944, 'r_fair': None,  'r_deal': None},
    "Fairness Maximizer":    {'SR': 0.126, 'avg_turn': 9.914,
                              'r_gain': None, 'r_fair': 0.368,  'r_deal': None},
    "Deal Maximizer":        {'SR': 0.489, 'avg_turn': 9.531,
                              'r_gain': None, 'r_fair': None,   'r_deal': 0.165},
}


def _print_table(per_skill, average, n_ep):
    sep = "=" * 105
    print()
    print(sep)
    print(f" DMORL Table-2 evaluation (MEAN conv.)   (n={n_ep} episodes per skill)")
    print(sep)
    print(f"{'Skill / row':<32} {'SR':>8} {'avg.turn':>10} {'r_gain':>9} {'r_fair':>9} {'r_deal':>9}")
    print("-" * 105)
    for name, r in per_skill.items():
        ref = _PADPP_REF_BY_SKILL.get(name)
        # DMORL row
        gain_s = f"{r['r_gain']:>9.3f}"
        fair_s = f"{r['r_fair']:>9.3f}"
        deal_s = f"{r['r_deal']:>9.3f}"
        print(f"{name + ' [DMORL]':<32} {r['SR']:>8.3f} {r['avg_turn']:>10.2f} "
              f"{gain_s} {fair_s} {deal_s}")
        # PADPP reference row, if available
        if ref is not None:
            gp = f"{ref['r_gain']:>9.3f}" if ref['r_gain'] is not None else f"{'-':>9}"
            fp = f"{ref['r_fair']:>9.3f}" if ref['r_fair'] is not None else f"{'-':>9}"
            dp = f"{ref['r_deal']:>9.3f}" if ref['r_deal'] is not None else f"{'-':>9}"
            print(f"{name + ' [PADPP paper]':<32} {ref['SR']:>8.3f} {ref['avg_turn']:>10.3f} "
                  f"{gp} {fp} {dp}")
        print("-" * 105)
    print(f"{'AVERAGE [DMORL]':<32} {average['SR']:>8.3f} {average['avg_turn']:>10.2f} "
          f"{average['r_gain']:>9.3f} {average['r_fair']:>9.3f} {average['r_deal']:>9.3f}")
    print(sep)


if __name__ == '__main__':
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    args, eval_overrides = parse_eval_args()
    args = reformat_args(vars(args))
    set_seed(args['seed'])

    accelerator = Accelerator(device_placement=True, kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    # ── Scenario / game ────────────────────────────────────────────────────
    game_config_file, game_config_class, game_class, game_simulator_class = \
        get_scenario_by_name(args['scenario'])
    game_params = load_config_from_yaml_file(game_config_file)
    game_config = game_config_class(game_params)
    game_config.set_params({
        'seed': args['seed'],
        'model_type': args['model_type'],
        'use_llm_price_extraction': eval_overrides.get('use_llm_price_extraction', False),
        'use_price_tag': eval_overrides.get('use_price_tag', False),
        'fairness_train_scale': eval_overrides.get('fairness_train_scale', 1.0),
    })

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset_config_classes_and_paths = get_datasets_by_names(
        args['scenario'], args['datasets'])
    data_config_path, dataset_class, dataset_scenario_config_class = \
        dataset_config_classes_and_paths[0]
    dataset_params = load_config_from_yaml_file(data_config_path)
    dataset_config = dataset_scenario_config_class(dataset_params)
    if isinstance(dataset_config, DatasetConfigForRecommendation):
        dataset_config.set_params({"domain": args['domain']})
    dataset = dataset_class(dataset_config)

    # ── Test simulators (must exist; load from cached file) ───────────────
    if not os.path.exists(dataset_config.save_test_simulator_path):
        raise FileNotFoundError(
            f"Test simulators not found at {dataset_config.save_test_simulator_path}. "
            "Run training once (or `--overwrite_sim`) to build them."
        )
    test_sims = load_user_simulators(dataset_config.save_test_simulator_path)
    for sim in test_sims:
        sim.set_model_type(game_config.model_type)
        sim.is_using_persona(args['use_persona'])

    # ── Model + trainer skeleton (no training) ────────────────────────────
    model_classes_and_pipelines = get_model_by_names(args['scenario'], args['models'])
    config_file, config_class, model_class, pipeline_class, trainer_class = \
        model_classes_and_pipelines[0]

    model_params = load_config_from_yaml_file(config_file)
    model_config = config_class(model_params)
    model_config.set_params({'model_type': game_config.model_type})

    scenario_params = {
        'n_goals': dataset.n_goals,
        'n_objectives': game_config.n_objectives,
        'device': device,
        'scenario_name': args['scenario'],
    }
    if args['scenario'] == RECOMMENDATION:
        scenario_params['n_topics'] = dataset.n_topics
        scenario_params['domain'] = args['domain']
    model_config.set_params(scenario_params)
    # Inference-only: never run SFT/RL
    model_config.set_params({'run_sft': False, 'run_rlt': False, 'test_phase': True})

    game = game_class(game_config=game_config, dataset_config=dataset_config)

    # ── Generation method (LLM that produces utterances) ──────────────────
    generation_packages = get_text_generation_model_by_name(
        args['scenario'], args['gen_models'])
    gen_name = args['gen_models'].split(',')[0].strip()
    generation_package = generation_packages[0]

    if gen_name == BART_GENERATION:
        raise NotImplementedError("BART generation not supported by eval_dmorl.py; "
                                  "use --gen_models fpt|llama3|chatgpt.")
    else:
        (generation_config_path, generation_prompt,
         generation_config_class, generation_class) = generation_package
        gen_params = load_config_from_yaml_file(generation_config_path)
        generation_config = generation_config_class(gen_params)
        generation_config.set_params({
            'prompt': generation_prompt,
            'scenario_name': game_config.name,
            'dataset': dataset_config.dataset_name,
        })
        if gen_name == VICUNA:
            generation_config.set_params({
                'device': args['device'],
                'num_gpus': args['num_gpus'],
                'load_8bit': args['load_8bit'],
                'cpu_offloading': args['cpu_offloading'],
                'max_gpu_memory': args['max_gpu_memory'],
            })
        generation_method = generation_class(generation_config, None, None)

    # Build model + trainer (model will be REPLACED by checkpoint below)
    model = model_class(model_config)
    offline_metrics, online_metrics = get_metrics_by_names(args['scenario'], args['metrics'])
    offline_evaluator = OfflineEvaluator(offline_metrics)
    online_evaluator = OnlineEvaluator(online_metrics)
    online_evaluator.reset()

    trainer = trainer_class(
        accelerator=accelerator,
        game_config=game_config,
        model_config=model_config,
        game=game,
        model=model,
        generation_method=generation_method,
        offline_evaluator=offline_evaluator,
        online_evaluator=online_evaluator,
        loggers=[],
    )

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt_path = eval_overrides['checkpoint']
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    trainer.load_model(ckpt_path)
    trainer.model.to(device)
    trainer.model.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    # ── Action mapping + test cases ───────────────────────────────────────
    action_mapping = dataset.construct_action_mapping(
        combine=model_config.combined_action)

    from utils.game import create_cases
    n_test = eval_overrides['n_eval_episodes']
    test_cases = create_cases(
        test_instances=dataset.test_instances,
        num_cases=n_test,
    )
    if len(test_sims) > len(test_cases):
        random.seed(args['seed'])
        test_sims_sample = random.sample(test_sims, len(test_cases))
    else:
        test_sims_sample = test_sims[:len(test_cases)]
    logger.info(f"Test cases: {len(test_cases)}, simulators: {len(test_sims_sample)}")

    # ── Load skills ───────────────────────────────────────────────────────
    skills_path = (eval_overrides['skills_file']
                   or getattr(model_config, 'skills_file', 'dmorl_skills_neg.json'))
    if not os.path.exists(skills_path):
        raise FileNotFoundError(
            f"Skills file not found: {skills_path}. Run training once first "
            "or pass --skills_file."
        )
    with open(skills_path, 'r', encoding='utf-8') as f:
        skills_data = json.load(f)
    skills_to_eval = list(skills_data.get('basic', []))
    if eval_overrides['include_advanced']:
        skills_to_eval += list(skills_data.get('advanced', []))
    if not skills_to_eval:
        raise RuntimeError(f"No skills found in {skills_path}")

    # ── Evaluation loop ───────────────────────────────────────────────────
    max_horizon = getattr(game_config, 'max_horizon', 10)
    per_skill_results = {}
    per_skill_raw = {}

    with torch.no_grad():
        for skill in skills_to_eval:
            name = skill['name']
            w = skill['weight_vector']
            logger.info(f"=== Evaluating {name}  w={w} ===")
            raw = []
            for idx, (case, sim) in enumerate(zip(test_cases, test_sims_sample)):
                r = _episode(trainer, case, sim, action_mapping, w, name, max_horizon)
                raw.append(r)
                logger.info(
                    f"[{name}] ep{idx}: turns={r['n_turns']} success={r[SUCCESS_RATE]} "
                    f"sl={r[SL_RATIO]:.3f} fair={r[FAIRNESS]:.3f} dr={r[DEAL_RATE]:.3f}"
                )
            per_skill_raw[name] = raw
            per_skill_results[name] = _aggregate(raw)

    # ── Average across skills (replaces PADPP's "uniform" row) ───────────
    keys = ['SR', 'avg_turn', 'avg_steps', 'r_gain', 'r_fair', 'r_deal']
    average = {k: sum(r[k] for r in per_skill_results.values()) / len(per_skill_results)
               for k in keys}

    # ── Report ────────────────────────────────────────────────────────────
    _print_table(per_skill_results, average, n_test)

    out_dir = eval_overrides['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"eval_{local_time}.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            'checkpoint': ckpt_path,
            'skills_file': skills_path,
            'n_episodes_per_skill': n_test,
            'per_skill': per_skill_results,
            'average': average,
            'episodes': per_skill_raw,
        }, f, indent=2, default=str)
    print(f"\nSaved detailed results: {out_file}\n")
