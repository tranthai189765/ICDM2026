"""eval_llm_baseline.py — LLM-as-policy baseline (no RL).

For each test case, an LLM directly selects the next action index from the
masked action space, conditioned on (skill name + description + weight vector,
dialogue context, available actions with prices, game state). The selected
action is then run through the standard game/simulator pipeline.

Useful for comparing the trained DMORL policy against a strong frontier
LLM that uses the SAME action space + skill conditioning but no RL.

Two LLMs are involved (can be the same):
  - POLICY LLM: chooses action index. Configured via env POLICY_LLM_* or
    falls back to FPT_* (see utils/prompt.py).
  - GENERATION LLM: produces the buyer utterance for the chosen action.
    Uses --gen_models (fpt / llama3 / qwen / chatgpt).

Usage:
  python eval_llm_baseline.py \
      --scenario negotiation \
      --datasets craigslist_bargain \
      --models dmorl \
      --gen_models fpt --model_type fpt \
      --metrics sr,deal_rate,sl_ratio,fairness,avg_turn \
      --loggers terminal \
      --skills_file dmorl_skills_neg.json \
      --n_eval_episodes 20
"""
import argparse
import copy
import json
import os
import random
import re
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
import openai  # noqa: E402
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
from utils.game import create_cases  # noqa: E402
from config.config import DatasetConfigForRecommendation  # noqa: E402
from config.constants import (  # noqa: E402
    BART_GENERATION, VICUNA, RECOMMENDATION, NEGOTIATION, EMOTIONAL_SUPPORT,
    SUCCESS_RATE, SL_RATIO, FAIRNESS, DEAL_RATE, AVG_TURN,
)
from dmorl.pipeline import SCENARIO_OBJECTIVE_NAMES, OBJECTIVE_DESCRIPTIONS  # noqa: E402

load_dotenv()


# ── Loguru to file ─────────────────────────────────────────────────────────
logger.remove()
_log_dir = "logs"
os.makedirs(_log_dir, exist_ok=True)
_log_file = os.path.join(_log_dir, f"eval_llm_baseline_{time.strftime('%Y%m%d_%H%M%S')}.log")
logger.add(_log_file, level="INFO", encoding="utf-8",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")


# ── Constants matching Option B action space ───────────────────────────────
_PRICE_BEARING = {'propose', 'counter', 'final_offer'}

ACTION_DESCRIPTIONS = {
    'propose':         "Make an initial price offer.",
    'counter':         "Counter the seller's offer with a NEW price.",
    'counter-noprice': "Push back on the seller without naming a price (e.g. 'still too high').",
    'agree':           "Accept the seller's most recent offered price (closes the deal).",
    'disagree':        "Reject the seller's offer without proposing a new price.",
    'walk_away':       "Announce you are walking away from this negotiation (BATNA).",
    'final_offer':     "Take-it-or-leave-it ultimatum at the specified price.",
    'inquire':         "Ask a question about the product, condition, market price, etc.",
    'inform':          "Provide information that justifies your position (market refs, defects).",
    'greet':           "Greet or make small talk.",
    'deny':            "Deny information the seller stated.",
    'affirm':          "Acknowledge agreement on a NON-price point.",
    'confirm':         "Confirm details about the deal (terms, delivery, etc.).",
}


# ── Policy LLM ────────────────────────────────────────────────────────────
_policy_client = None


def _get_policy_client():
    """Build the OpenAI-compatible client for the POLICY LLM.
    Falls back to FPT_* env vars if POLICY_LLM_* not set."""
    global _policy_client
    if _policy_client is not None:
        return _policy_client
    key = os.getenv("POLICY_LLM_API_KEY") or os.getenv("FPT_API_KEY")
    url = (os.getenv("POLICY_LLM_BASE_URL")
           or os.getenv("FPT_API_URL", "").rsplit("/chat/completions", 1)[0])
    if not key or not url:
        raise RuntimeError(
            "POLICY_LLM_API_KEY / POLICY_LLM_BASE_URL (or FPT_*) must be set in .env"
        )
    _policy_client = openai.OpenAI(api_key=key, base_url=url)
    return _policy_client


def _policy_model_name():
    return os.getenv("POLICY_LLM_MODEL") or os.getenv("FPT_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")


def call_policy_llm(messages, temperature=0.2, max_tokens=15):
    client = _get_policy_client()
    resp = client.chat.completions.create(
        model=_policy_model_name(),
        messages=messages,
        temperature=max(temperature, 0.01),
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# ── Build action list with prices ──────────────────────────────────────────
def _build_valid_actions(action_mapping, state, bin_num=5):
    """Return list of (idx, (strategy, topic), price_or_None) for valid actions
    under Option B mask. Sorted by idx."""
    if isinstance(action_mapping, tuple):
        am = action_mapping[0]
    else:
        am = action_mapping

    seller_price = state['task_background']['seller_price']
    buyer_price = state['task_background']['buyer_price']
    bin_width = (seller_price - buyer_price) / bin_num

    out = []
    for (strategy, topic), idx in am.items():
        if strategy not in _PRICE_BEARING and topic != 0:
            continue  # Option B mask
        if strategy in _PRICE_BEARING:
            price = int(buyer_price + bin_width * topic)
        else:
            price = None
        out.append((idx, (strategy, topic), price))
    out.sort(key=lambda x: x[0])
    return out


def _format_action_list(actions):
    lines = []
    for idx, (strategy, topic), price in actions:
        desc = ACTION_DESCRIPTIONS.get(strategy, strategy)
        if price is not None:
            lines.append(f"  {idx}: ({strategy}, bin {topic}) at ${price}  — {desc}")
        else:
            lines.append(f"  {idx}: ({strategy})  — {desc}")
    return "\n".join(lines)


def _format_history(state):
    dialogue = state.get('dialogue_context', [])
    lines = []
    for turn in dialogue:
        role = "SELLER" if turn.get('role') == 'user' else "YOU (BUYER)"
        lines.append(f"  {role}: {turn.get('content', '')}")
    return "\n".join(lines) or "  (conversation just started — seller will speak first)"


# ── LLM policy: action selection ───────────────────────────────────────────
def llm_select_action(state, skill, action_mapping, objective_names,
                       objective_descriptions, max_horizon, step_in_episode):
    """Ask the policy LLM to pick an action index for this turn."""
    actions = _build_valid_actions(action_mapping, state)
    action_text = _format_action_list(actions)
    valid_ids = [a[0] for a in actions]
    id_to_action = {a[0]: a[1] for a in actions}

    weights = skill['weight_vector']
    weight_text = "\n".join(
        f"  - {name}: {w:.2f}  ({objective_descriptions.get(name, '')})"
        for name, w in zip(objective_names, weights)
    )

    task = state['task_background']
    seller_price = task['seller_price']
    buyer_price = task['buyer_price']
    mid_price = (seller_price + buyer_price) / 2
    remaining_turns = max(0, (max_horizon - len(state.get('dialogue_context', []))) // 2)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert dialogue policy for a BUYER in a single-item price "
                "negotiation. Your job is to choose ONE action from a numbered list per "
                "turn. Respond with ONLY the action index as a single integer (no prose, "
                "no JSON, no quotes)."
            ),
        },
        {
            "role": "user",
            "content": (
                f"## Your Skill\n"
                f"Name: {skill['name']}\n"
                f"Description: {skill.get('description', '')}\n"
                f"Weights (higher = more important):\n{weight_text}\n\n"
                f"## Task Background\n"
                f"Item: {task['item_name']}\n"
                f"Seller's listed price: ${seller_price} (highest realistic; you should pay LESS)\n"
                f"Your target price:    ${buyer_price} (lowest; ideal outcome)\n"
                f"Midpoint price:       ${mid_price:.0f}\n\n"
                f"## Action Mechanics\n"
                f"- Each price-bearing action (propose / counter / final_offer) has 5 bins.\n"
                f"  bin 0 = your target price (lowest, BEST for r_gain),\n"
                f"  bin 2 = ~midpoint (BEST for r_fair, balanced for r_gain),\n"
                f"  bin 4 = near seller's ask (BEST for closing fast).\n"
                f"- agree closes the deal at the seller's most recent offered price.\n"
                f"- walk_away signals withdrawal; useful as a pressure tactic.\n"
                f"- final_offer is a hard ultimatum at the specified bin.\n\n"
                f"## Dialogue So Far\n{_format_history(state)}\n\n"
                f"## Available Actions (choose ONE by index)\n{action_text}\n\n"
                f"## Constraints\n"
                f"- This is decision turn #{step_in_episode + 1}.\n"
                f"- You have ~{remaining_turns} more turns before timeout.\n"
                f"- Pick the action that BEST advances your skill's weighted objectives.\n\n"
                f"Respond with ONLY the integer action index."
            ),
        },
    ]

    try:
        raw = call_policy_llm(messages, temperature=0.2, max_tokens=15)
        m = re.search(r"\d+", raw)
        if m is None:
            raise ValueError(f"no integer in '{raw}'")
        idx = int(m.group(0))
        if idx not in id_to_action:
            raise ValueError(f"idx {idx} not in valid set (top: {valid_ids[:8]}...)")
        return id_to_action[idx], idx
    except Exception as e:
        logger.warning(f"[LLM policy] selection failed ({e}); using random valid action.")
        idx = int(np.random.choice(valid_ids))
        return id_to_action[idx], idx


# ── Episode runner (no trainer, no checkpoint) ────────────────────────────
def _episode(game, generation_method, simulator, case, action_mapping,
             skill, objective_names, objective_descriptions, max_horizon):
    state = game.reset(case, simulator)
    state['w'] = np.array(skill['weight_vector'])

    turns = []
    epi_rewards = []
    conv_turn = 0
    done = 0

    for t in count():
        action, action_idx = llm_select_action(
            state, skill, action_mapping, objective_names,
            objective_descriptions, max_horizon, step_in_episode=t,
        )

        pre_dialogue = list(state.get('dialogue_context', []))
        state, reward, done, _ = game.step(state, action, generation_method, simulator)
        new_dialogue = list(state.get('dialogue_context', []))
        new_utts = new_dialogue[len(pre_dialogue):]

        epi_rewards.append(reward if isinstance(reward, list) else [float(reward)])
        conv_turn = t + 1
        turns.append({
            "step": t,
            "action": str(action),
            "action_idx": action_idx,
            "utterances": new_utts,
            "reward": reward if isinstance(reward, list) else [float(reward)],
            "done": int(bool(done)),
        })

        if done or t >= max_horizon - 1:
            break

    arr = np.array(epi_rewards)
    terminal = arr[-1]
    n_obj = arr.shape[1]
    sl = float(terminal[0]) if n_obj >= 1 else 0.0
    fair = float(terminal[1]) if n_obj >= 2 else 0.0
    deal_rate = float(terminal[2]) if n_obj >= 3 else 0.0
    utterance_count = len(state['dialogue_context'])

    return {
        SUCCESS_RATE: int(done == 1),
        AVG_TURN: [conv_turn, 0],
        SL_RATIO: sl,
        FAIRNESS: fair,
        DEAL_RATE: deal_rate,
        'skill': skill['name'],
        'n_turns': conv_turn,
        'n_utterances': utterance_count,
        'done_flag': int(done),
        'outcome': "success" if done == 1 else ("failure" if done == -1 else "ongoing"),
        'weight_vector': [float(x) for x in skill['weight_vector']],
        'final_reward': arr[-1].tolist(),
        'turns': turns,
        'task_background': dict(state['task_background']),
    }


def _aggregate(results):
    n = max(len(results), 1)
    return {
        'SR':        sum(r[SUCCESS_RATE]    for r in results) / n,
        'avg_turn':  sum(r['n_utterances'] for r in results) / n,
        'avg_steps': sum(r[AVG_TURN][0]    for r in results) / n,
        'r_gain':    sum(r[SL_RATIO]       for r in results) / n,
        'r_fair':    sum(r[FAIRNESS]       for r in results) / n,
        'r_deal':    sum(r[DEAL_RATE]      for r in results) / n,
        'n':         n,
    }


def _print_table(per_skill, average, n_ep, policy_model):
    sep = "=" * 100
    print()
    print(sep)
    print(f" LLM-as-policy baseline  (policy: {policy_model})  (n={n_ep} ep/skill)")
    print(sep)
    print(f"{'Skill':<28} {'SR':>8} {'avg.turn':>10} {'steps':>7} "
          f"{'r_gain':>8} {'r_fair':>8} {'r_deal':>8}")
    print("-" * 100)
    for name, r in per_skill.items():
        print(f"{name:<28} {r['SR']:>8.3f} {r['avg_turn']:>10.2f} {r['avg_steps']:>7.2f} "
              f"{r['r_gain']:>8.3f} {r['r_fair']:>8.3f} {r['r_deal']:>8.3f}")
    print("-" * 100)
    print(f"{'AVERAGE':<28} {average['SR']:>8.3f} {average['avg_turn']:>10.2f} "
          f"{average['avg_steps']:>7.2f} "
          f"{average['r_gain']:>8.3f} {average['r_fair']:>8.3f} {average['r_deal']:>8.3f}")
    print(sep)


def parse_eval_args():
    base_args = parse_args()
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--skills_file', type=str, default='dmorl_skills_neg.json')
    extra.add_argument('--n_eval_episodes', type=int, default=20)
    extra.add_argument('--output_dir', type=str, default='eval_results')
    extra.add_argument('--include_advanced', action='store_true')
    extra_args, _ = extra.parse_known_args(sys.argv[1:])
    return base_args, vars(extra_args)


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
    game_config.set_params({'seed': args['seed'], 'model_type': args['model_type']})

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset_config_classes_and_paths = get_datasets_by_names(args['scenario'], args['datasets'])
    data_config_path, dataset_class, dataset_scenario_config_class = \
        dataset_config_classes_and_paths[0]
    dataset_params = load_config_from_yaml_file(data_config_path)
    dataset_config = dataset_scenario_config_class(dataset_params)
    if isinstance(dataset_config, DatasetConfigForRecommendation):
        dataset_config.set_params({"domain": args['domain']})
    dataset = dataset_class(dataset_config)

    # ── Test simulators ───────────────────────────────────────────────────
    if not os.path.exists(dataset_config.save_test_simulator_path):
        raise FileNotFoundError(
            f"Test simulators not found at {dataset_config.save_test_simulator_path}"
        )
    test_sims = load_user_simulators(dataset_config.save_test_simulator_path)
    for sim in test_sims:
        sim.set_model_type(game_config.model_type)
        sim.is_using_persona(args['use_persona'])

    # ── Model config (for combined_action only — no model loading) ────────
    model_classes_and_pipelines = get_model_by_names(args['scenario'], args['models'])
    config_file, config_class, model_class, pipeline_class, trainer_class = \
        model_classes_and_pipelines[0]
    model_params = load_config_from_yaml_file(config_file)
    model_config = config_class(model_params)
    model_config.set_params({'model_type': game_config.model_type})

    game = game_class(game_config=game_config, dataset_config=dataset_config)

    # ── Generation method (buyer utterance generator) ─────────────────────
    generation_packages = get_text_generation_model_by_name(args['scenario'], args['gen_models'])
    gen_name = args['gen_models'].split(',')[0].strip()
    generation_package = generation_packages[0]

    if gen_name == BART_GENERATION:
        raise NotImplementedError("BART generation not supported; use fpt/llama3/qwen/chatgpt.")
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
                'device': args['device'], 'num_gpus': args['num_gpus'],
                'load_8bit': args['load_8bit'], 'cpu_offloading': args['cpu_offloading'],
                'max_gpu_memory': args['max_gpu_memory'],
            })
        generation_method = generation_class(generation_config, None, None)

    # ── Action mapping + test cases ───────────────────────────────────────
    action_mapping = dataset.construct_action_mapping(combine=model_config.combined_action)

    n_test = eval_overrides['n_eval_episodes']
    test_cases = create_cases(test_instances=dataset.test_instances, num_cases=n_test)
    if len(test_sims) > len(test_cases):
        random.seed(args['seed'])
        test_sims_sample = random.sample(test_sims, len(test_cases))
    else:
        test_sims_sample = test_sims[:len(test_cases)]
    logger.info(f"Test cases: {len(test_cases)}, simulators: {len(test_sims_sample)}")

    # ── Skills ────────────────────────────────────────────────────────────
    skills_path = (eval_overrides['skills_file']
                   or getattr(model_config, 'skills_file', 'dmorl_skills_neg.json'))
    if not os.path.exists(skills_path):
        raise FileNotFoundError(f"Skills file not found: {skills_path}")
    with open(skills_path, 'r', encoding='utf-8') as f:
        skills_data = json.load(f)
    skills_to_eval = list(skills_data.get('basic', []))
    if eval_overrides['include_advanced']:
        skills_to_eval += list(skills_data.get('advanced', []))
    if not skills_to_eval:
        raise RuntimeError("No skills found in skills file")

    # ── Objective metadata for prompt ─────────────────────────────────────
    objective_names = SCENARIO_OBJECTIVE_NAMES.get(args['scenario'],
                                                    [f"obj_{i}" for i in range(len(skills_to_eval[0]['weight_vector']))])
    objective_descriptions = {n: OBJECTIVE_DESCRIPTIONS.get(n, '') for n in objective_names}

    # ── Eval loop ─────────────────────────────────────────────────────────
    max_horizon = getattr(game_config, 'max_horizon', 10)
    policy_model = _policy_model_name()
    logger.info(f"Policy LLM: {policy_model}")
    print(f"Policy LLM: {policy_model}\n")

    per_skill_results = {}
    per_skill_raw = {}

    for skill in skills_to_eval:
        name = skill['name']
        logger.info(f"=== LLM baseline evaluating {name}  w={skill['weight_vector']} ===")
        raw = []
        for idx, (case, sim) in enumerate(zip(test_cases, test_sims_sample)):
            try:
                r = _episode(game, generation_method, sim, case, action_mapping,
                             skill, objective_names, objective_descriptions, max_horizon)
                raw.append(r)
                logger.info(
                    f"[LLM/{name}] ep{idx}: turns={r['n_turns']} success={r[SUCCESS_RATE]} "
                    f"sl={r[SL_RATIO]:.3f} fair={r[FAIRNESS]:.3f} dr={r[DEAL_RATE]:.3f}"
                )
            except Exception as e:
                logger.error(f"[LLM/{name}] ep{idx} crashed: {e}")
                continue
        if not raw:
            logger.warning(f"Skill {name}: all episodes failed; skipping aggregate.")
            continue
        per_skill_raw[name] = raw
        per_skill_results[name] = _aggregate(raw)

    keys = ['SR', 'avg_turn', 'avg_steps', 'r_gain', 'r_fair', 'r_deal']
    average = {k: sum(r[k] for r in per_skill_results.values()) / max(len(per_skill_results), 1)
               for k in keys}

    _print_table(per_skill_results, average, n_test, policy_model)

    out_dir = eval_overrides['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"llm_baseline_{local_time}.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            'baseline': 'llm-as-policy',
            'policy_model': policy_model,
            'gen_model': args['gen_models'],
            'skills_file': skills_path,
            'n_episodes_per_skill': n_test,
            'per_skill': per_skill_results,
            'average': average,
            'episodes': per_skill_raw,
        }, f, indent=2, default=str)
    print(f"\nSaved detailed results: {out_file}\n")
