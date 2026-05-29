"""eval_llm_uniform.py — LLM-as-policy baseline with UNIFORM preference + reward feedback.

This is a variant of eval_llm_baseline.py that:

1. Uses a single UNIFORM weight vector across all objectives (matches the
   PADPP paper Table 2 'uniform' row convention), instead of per-skill weights.

2. Prepends a Pareto-strategy briefing to the prompt so the LLM understands
   that under uniform weights, the buyer-side Pareto front has the property
   sl_ratio + fairness ~ 1.0 for system_price <= midpoint, and that the
   optimal play is to close the deal near the midpoint quickly.

3. Provides per-turn REWARD FEEDBACK to the LLM. After every action, the
   raw reward vector and the running weighted return are appended to the
   prompt of the next turn. This lets the LLM adjust within an episode
   (in-context "learning").

4. Reports metrics in the same RAW-reward convention as eval_llm_baseline.py
   (un-shaped sl_ratio / fairness) so numbers can be compared directly with
   the paper baseline.

Usage:
  python eval_llm_uniform.py \
      --scenario negotiation --datasets craigslist_bargain --models dmorl \
      --gen_models fpt --model_type fpt \
      --metrics sr,deal_rate,sl_ratio,fairness,avg_turn \
      --loggers terminal \
      --n_eval_episodes 20

Optional:
  --no_feedback         disable per-turn reward feedback (compare to baseline)
  --no_pareto_hint      disable the Pareto-strategy briefing
  --uniform_3d          use 3D weight [1/3, 1/3, 1/3, 0]  to match PADPP exactly
                        (default: 4D weight [1/4]*4 with avg_turn included)
"""
import argparse
import json
import os
import random
import re
import sys
import time
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


# ── Loguru ─────────────────────────────────────────────────────────────────
logger.remove()
_log_dir = "logs"
os.makedirs(_log_dir, exist_ok=True)
_log_file = os.path.join(_log_dir, f"eval_llm_uniform_{time.strftime('%Y%m%d_%H%M%S')}.log")
logger.add(_log_file, level="INFO", encoding="utf-8",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")


_PRICE_BEARING = {'propose', 'counter', 'final_offer'}

ACTION_DESCRIPTIONS = {
    'propose':         "Make an initial price offer.",
    'counter':         "Counter the seller's offer with a NEW price.",
    'counter-noprice': "Push back without naming a price (e.g. 'still too high').",
    'agree':           "Accept the seller's most recent offered price (closes the deal).",
    'disagree':        "Reject the seller's offer without proposing a new price.",
    'walk_away':       "Announce you are walking away from this negotiation (BATNA).",
    'final_offer':     "Take-it-or-leave-it ultimatum at the specified price.",
    'inquire':         "Ask a question about the product, condition, or seller's reasoning.",
    'inform':          "Provide information that justifies your position.",
    'greet':           "Greet or make small talk.",
    'deny':            "Deny information the seller stated.",
    'affirm':          "Acknowledge agreement on a NON-price point.",
    'confirm':         "Confirm details about the deal.",
}

# Pareto strategy briefing for negotiation (uniform-weight specific)
PARETO_HINT_NEG = """## Pareto Strategy Guide (you MUST internalise these RULES)

### Core math
With uniform weights, the buyer-side Pareto front has the property:

    r_gain + r_fair ~ 1.0  for any system_price <= midpoint

So bins 0, 1, 2 yield the SAME (r_gain + r_fair) sum when the deal closes.
Bins 3 and 4 are strictly worse. The right choice between bins 0/1/2 depends
on what the SELLER will accept, NOT on which objective matters more.

### HARD DECISION RULES (apply in order — AGGRESSIVE r_gain mode)

R1. NEVER agree at a price strictly ABOVE bin 1 (= buyer_target + 20% of range).
    - If seller's latest offer > bin 1 price, you MUST counter (not agree).
    - This is stricter than midpoint — pushes for strong buyer-side outcomes.

R2. Anchor bin 0 (buyer_target price) for the FIRST 2 turns.
    Use `counter, 0` or `final_offer, 0` to make seller move down.

R3. If 3+ turns remain and seller hasn't moved below bin 2 (~midpoint),
    use `final_offer, 1` to commit at bin 1 with credible walk-threat.

R4. If seller's offer is at or below bin 1 price, AGREE immediately
    (max r_gain outcome — top of Pareto front).

R5. If seller's offer is between bin 1 and bin 2 AND only 1 turn remains,
    AGREE (avoids r_deal=0 catastrophe). Else counter at bin 1.

R6. If seller refuses your final_offer at bin 1 twice, use `walk_away`.
    Timeout at bin 1 anchor gives r_gain ~ 0.8 (better than capitulating).

### Seller simulator pattern (learned from prior runs)
- Seller REFUSES bin 0 anchors -> always leads to timeout. Avoid as opener.
- Seller usually counters at 75-85% of listed price after one round.
- Seller often accepts bin 1 or bin 2 after 2-3 polite counters.
- `final_offer, 2` + persistence has high acceptance rate.
- `agree, 0` is correct ONLY when seller has explicitly offered <= midpoint.

### Target Pareto-optimal outcome
Close the deal at ~midpoint in 2-3 turns:
  -> r_gain ~ 0.6, r_fair ~ 0.4, r_deal = 1.0, r_turn ~ -0.2
  -> WEIGHTED RETURN ~ 0.45 per turn  (this is your maximisation target)

Anything below weighted 0.30/turn means you are off the Pareto front.
"""

# Chain-of-thought scaffold (toggle via --cot)
COT_INSTRUCTIONS = """## Reasoning Protocol

Reason step-by-step, then output your choice in the EXACT format below.

Reasoning (briefly):
  Q1. Seller's most recent stated price (number only). Is it above / below /
      equal to midpoint?
  Q2. Which of the HARD DECISION RULES R1-R5 applies right now?
  Q3. Among rule-conformant actions, which yields the HIGHEST expected
      weighted reward this turn?

After reasoning, output the FINAL answer on its own line, using the EXACT
format (no extra prose, no quotes, no parentheses):

    Action index: <N>

where <N> is the integer ID of the chosen action FROM THE NUMBERED LIST
ABOVE — NOT a price, NOT a bin number, NOT a weighted score. The ID must
be one of the integers shown next to actions in the 'Available Actions'
section (e.g. 0, 5, 10, 15, ...).
"""


# ── Policy LLM ────────────────────────────────────────────────────────────
_policy_client = None


def _get_policy_client():
    global _policy_client
    if _policy_client is not None:
        return _policy_client
    key = os.getenv("POLICY_LLM_API_KEY") or os.getenv("FPT_API_KEY")
    url = (os.getenv("POLICY_LLM_BASE_URL")
           or os.getenv("FPT_API_URL", "").rsplit("/chat/completions", 1)[0])
    if not key or not url:
        raise RuntimeError("POLICY_LLM_* / FPT_* must be set in .env")
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


# ── Action helpers ─────────────────────────────────────────────────────────
def _build_valid_actions(action_mapping, state, bin_num=5):
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
            continue
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


def _format_feedback_history(feedback_log, weight):
    """Render the per-turn feedback block summarising past actions, raw
    rewards, and the running weighted return."""
    if not feedback_log:
        return "  (no actions taken yet)"
    lines = []
    cum = 0.0
    for fb in feedback_log:
        r_raw = fb['r_raw']
        weighted = float(np.dot(weight, r_raw))
        cum += weighted
        r_str = ", ".join(f"{name}={v:+.3f}" for name, v in zip(fb['names'], r_raw))
        lines.append(
            f"  Turn {fb['step']}: action={fb['action']} -> raw rewards [{r_str}] "
            f"-> weighted={weighted:+.3f} (cumulative={cum:+.3f})"
        )
    return "\n".join(lines)


# ── LLM policy with uniform weight + feedback ─────────────────────────────
def llm_select_action_uniform(state, weight, action_mapping, objective_names,
                                objective_descriptions, max_horizon,
                                step_in_episode, feedback_log,
                                include_pareto_hint=True,
                                include_feedback=True,
                                use_cot=False):
    actions = _build_valid_actions(action_mapping, state)
    action_text = _format_action_list(actions)
    valid_ids = [a[0] for a in actions]
    id_to_action = {a[0]: a[1] for a in actions}

    weight_text = "\n".join(
        f"  - {name}: {w:.2f}  ({objective_descriptions.get(name, '')})"
        for name, w in zip(objective_names, weight)
    )

    task = state['task_background']
    seller_price = task['seller_price']
    buyer_price = task['buyer_price']
    mid_price = (seller_price + buyer_price) / 2
    remaining_turns = max(0, (max_horizon - len(state.get('dialogue_context', []))) // 2)

    sections = [
        "## Your Objective: UNIFORM Preference",
        ("You are evaluated under the UNIFORM weight vector — every objective "
         "matters equally. Your goal is NOT to specialise; it is to reach a "
         "Pareto-optimal outcome where no single metric can be improved without "
         "sacrificing another."),
        "",
        "Weights (all equal):",
        weight_text,
    ]

    if include_pareto_hint:
        sections += ["", PARETO_HINT_NEG]

    sections += [
        "",
        "## Task Background",
        f"Item: {task['item_name']}",
        f"Seller's listed price: ${seller_price}  (highest realistic)",
        f"Your target price:    ${buyer_price}  (lowest realistic)",
        f"Midpoint price:       ${mid_price:.0f}  ← Pareto sweet spot",
        "",
        "## Dialogue So Far",
        _format_history(state),
    ]

    if include_feedback:
        sections += [
            "",
            "## Past Turns: Reward Feedback (RAW rewards, no shaping)",
            _format_feedback_history(feedback_log, weight),
        ]

    valid_ids_text = ", ".join(str(i) for i in valid_ids)
    sections += [
        "",
        "## Available Actions",
        action_text,
        "",
        "## Constraints",
        f"- Decision turn #{step_in_episode + 1}.",
        f"- ~{remaining_turns} more turns before timeout.",
        "- Pick the action that BEST advances the weighted uniform objective.",
        f"- The action ID you output MUST be one of: [{valid_ids_text}]",
    ]

    if use_cot:
        sections += ["", COT_INSTRUCTIONS]
    else:
        sections += [
            "",
            "Respond with ONLY a single integer — the chosen action ID from "
            "the list above (NOT a price, NOT a bin number).",
        ]

    user_content = "\n".join(sections)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert dialogue policy for a BUYER in a price negotiation. "
                "You must choose ONE action per turn from a numbered list to MAXIMISE the "
                "weighted sum of objectives. Respond with ONLY the action index as an integer."
            ),
        },
        {"role": "user", "content": user_content},
    ]

    try:
        max_tok = 400 if use_cot else 15
        raw = call_policy_llm(messages, temperature=0.2, max_tokens=max_tok)
        idx = None
        # Priority 1: explicit "Action index: N" marker (CoT) — most reliable.
        marker = re.search(r"[Aa]ction[ _]?index\s*[:=]\s*(\d+)", raw)
        if marker is not None:
            candidate = int(marker.group(1))
            if candidate in id_to_action:
                idx = candidate
        # Priority 2: scan ALL integers in the response from the end, picking
        # the last one that is a valid action_id. This skips prices ($1800),
        # bin numbers ("bin 2"), weighted return scores, etc.
        if idx is None:
            for tok in reversed(re.findall(r"\d+", raw)):
                cand = int(tok)
                if cand in id_to_action:
                    idx = cand
                    break
        if idx is None:
            raise ValueError(f"no valid action index in response (raw: {raw[:200]!r})")
        return id_to_action[idx], idx
    except Exception as e:
        logger.warning(f"[LLM uniform] selection failed ({e}); using random valid action.")
        idx = int(np.random.choice(valid_ids))
        return id_to_action[idx], idx


# ── Reward un-shaping ─────────────────────────────────────────────────────
def _unshape_reward(reward, done):
    """Return RAW reward vector (un-shape sl_ratio and fairness)."""
    if isinstance(reward, list):
        r = list(reward)
    else:
        r = [float(reward)]
    if done != 1:
        # shaping = 0.3 was applied to r0 (sl_ratio) and r1 (fairness)
        if len(r) >= 1:
            r[0] = r[0] / 0.3
        if len(r) >= 2:
            r[1] = r[1] / 0.3
    return r


# ── Episode runner ────────────────────────────────────────────────────────
def _episode(game, generation_method, simulator, case, action_mapping,
             weight, objective_names, objective_descriptions, max_horizon,
             include_pareto_hint, include_feedback, use_cot=False):
    state = game.reset(case, simulator)
    state['w'] = np.array(weight)

    turns = []
    epi_rewards_raw = []
    feedback_log = []  # entries: {step, action, r_raw, names}
    conv_turn = 0
    done = 0

    for t in count():
        action, action_idx = llm_select_action_uniform(
            state, weight, action_mapping, objective_names,
            objective_descriptions, max_horizon,
            step_in_episode=t, feedback_log=feedback_log,
            include_pareto_hint=include_pareto_hint,
            include_feedback=include_feedback,
            use_cot=use_cot,
        )

        pre_dialogue = list(state.get('dialogue_context', []))
        state, reward, done, _ = game.step(state, action, generation_method, simulator)
        new_dialogue = list(state.get('dialogue_context', []))
        new_utts = new_dialogue[len(pre_dialogue):]

        r_raw = _unshape_reward(reward, done)
        epi_rewards_raw.append(r_raw)
        conv_turn = t + 1
        turns.append({
            "step": t,
            "action": str(action),
            "action_idx": action_idx,
            "utterances": new_utts,
            "reward_shaped": reward if isinstance(reward, list) else [float(reward)],
            "reward_raw": r_raw,
            "done": int(bool(done)),
        })
        # Append to feedback log AFTER step so next turn can see it
        feedback_log.append({
            "step": t,
            "action": f"({action[0]}, {action[1]})" if isinstance(action, tuple) else str(action),
            "r_raw": r_raw,
            "names": list(objective_names),
        })

        if done or t >= max_horizon - 1:
            break

    arr = np.array(epi_rewards_raw)
    terminal = arr[-1]  # already raw
    n_obj = arr.shape[1]
    sl = float(terminal[0]) if n_obj >= 1 else 0.0
    fair = float(terminal[1]) if n_obj >= 2 else 0.0
    deal_rate = float(terminal[2]) if n_obj >= 3 else 0.0
    utterance_count = len(state['dialogue_context'])

    # Cumulative weighted return (paper-style sanity check)
    cum_weighted = float(sum(np.dot(weight, r) for r in epi_rewards_raw))

    return {
        SUCCESS_RATE: int(done == 1),
        AVG_TURN: [conv_turn, 0],
        SL_RATIO: sl,
        FAIRNESS: fair,
        DEAL_RATE: deal_rate,
        'n_turns': conv_turn,
        'n_utterances': utterance_count,
        'done_flag': int(done),
        'outcome': "success" if done == 1 else ("failure" if done == -1 else "ongoing"),
        'weight_vector': list(weight),
        'final_reward_raw': arr[-1].tolist(),
        'cumulative_weighted_return': cum_weighted,
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
        'weighted':  sum(r['cumulative_weighted_return'] for r in results) / n,
        'n':         n,
    }


def _print_table(average, n_ep, policy_model, weight, objective_names, padpp_ref=None):
    sep = "=" * 100
    print()
    print(sep)
    print(f" LLM-as-policy UNIFORM baseline  (policy: {policy_model})  (n={n_ep} episodes)")
    print(sep)
    weight_str = " ".join(f"{n}={w:.2f}" for n, w in zip(objective_names, weight))
    print(f" Weight vector: [{weight_str}]")
    print(f" Cumulative weighted return per episode: {average['weighted']:+.4f}")
    print(sep)
    print(f"{'Method':<28} {'SR':>8} {'avg.turn':>10} {'steps':>7} "
          f"{'r_gain':>8} {'r_fair':>8} {'r_deal':>8}")
    print("-" * 100)
    if padpp_ref:
        print(f"{'PADPP (paper Table 2)':<28} "
              f"{padpp_ref['SR']:>8.3f} {padpp_ref['avg_turn']:>10.3f} {'-':>7} "
              f"{padpp_ref['r_gain']:>8.3f} {padpp_ref['r_fair']:>8.3f} {padpp_ref['r_deal']:>8.3f}")
    print(f"{'LLM uniform (this run)':<28} {average['SR']:>8.3f} {average['avg_turn']:>10.2f} "
          f"{average['avg_steps']:>7.2f} "
          f"{average['r_gain']:>8.3f} {average['r_fair']:>8.3f} {average['r_deal']:>8.3f}")
    if padpp_ref:
        print("-" * 100)
        print(f"{'Δ vs PADPP':<28} "
              f"{average['SR'] - padpp_ref['SR']:>+8.3f} "
              f"{average['avg_turn'] - padpp_ref['avg_turn']:>+10.3f} {'':>7} "
              f"{average['r_gain'] - padpp_ref['r_gain']:>+8.3f} "
              f"{average['r_fair'] - padpp_ref['r_fair']:>+8.3f} "
              f"{average['r_deal'] - padpp_ref['r_deal']:>+8.3f}")
    print(sep)


def parse_eval_args():
    base_args = parse_args()
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--n_eval_episodes', type=int, default=20)
    extra.add_argument('--output_dir', type=str, default='eval_results')
    extra.add_argument('--no_feedback', action='store_true',
                       help='Disable per-turn reward feedback to LLM.')
    extra.add_argument('--no_pareto_hint', action='store_true',
                       help='Disable the Pareto-strategy briefing.')
    extra.add_argument('--uniform_3d', action='store_true',
                       help='Use 3D uniform weight [1/3,1/3,1/3,0] (match PADPP paper).')
    extra.add_argument('--cot', action='store_true',
                       help='Enable chain-of-thought reasoning before action selection.')
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

    if not os.path.exists(dataset_config.save_test_simulator_path):
        raise FileNotFoundError(f"Test simulators not found")
    test_sims = load_user_simulators(dataset_config.save_test_simulator_path)
    for sim in test_sims:
        sim.set_model_type(game_config.model_type)
        sim.is_using_persona(args['use_persona'])

    # ── Model config (only need combined_action) ──────────────────────────
    model_classes_and_pipelines = get_model_by_names(args['scenario'], args['models'])
    config_file, config_class, model_class, pipeline_class, trainer_class = \
        model_classes_and_pipelines[0]
    model_params = load_config_from_yaml_file(config_file)
    model_config = config_class(model_params)
    model_config.set_params({'model_type': game_config.model_type})

    game = game_class(game_config=game_config, dataset_config=dataset_config)

    # ── Generation method ─────────────────────────────────────────────────
    generation_packages = get_text_generation_model_by_name(args['scenario'], args['gen_models'])
    gen_name = args['gen_models'].split(',')[0].strip()
    generation_package = generation_packages[0]
    if gen_name == BART_GENERATION:
        raise NotImplementedError()
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

    # ── Action mapping + cases ────────────────────────────────────────────
    action_mapping = dataset.construct_action_mapping(combine=model_config.combined_action)

    n_test = eval_overrides['n_eval_episodes']
    test_cases = create_cases(test_instances=dataset.test_instances, num_cases=n_test)
    if len(test_sims) > len(test_cases):
        random.seed(args['seed'])
        test_sims_sample = random.sample(test_sims, len(test_cases))
    else:
        test_sims_sample = test_sims[:len(test_cases)]
    logger.info(f"Test cases: {len(test_cases)}, simulators: {len(test_sims_sample)}")

    # ── Objective setup ──────────────────────────────────────────────────
    objective_names = SCENARIO_OBJECTIVE_NAMES.get(args['scenario'], [])
    objective_descriptions = {n: OBJECTIVE_DESCRIPTIONS.get(n, '') for n in objective_names}

    n_obj = len(objective_names)
    if eval_overrides['uniform_3d'] and 'avg_turn' in objective_names:
        # Match PADPP paper exactly: ignore avg_turn (weight = 0)
        weight = []
        n_active = sum(1 for o in objective_names if o != 'avg_turn')
        for o in objective_names:
            weight.append(0.0 if o == 'avg_turn' else 1.0 / n_active)
    else:
        # 4D uniform
        weight = [1.0 / n_obj] * n_obj

    include_pareto_hint = not eval_overrides['no_pareto_hint']
    include_feedback = not eval_overrides['no_feedback']
    use_cot = eval_overrides['cot']

    policy_model = _policy_model_name()
    logger.info(f"Policy LLM: {policy_model}")
    logger.info(f"Weight vector: {weight}")
    logger.info(f"Pareto hint: {include_pareto_hint} | Reward feedback: {include_feedback} | CoT: {use_cot}")
    print(f"Policy LLM: {policy_model}")
    print(f"Weight vector: {weight}")
    print(f"Pareto hint: {include_pareto_hint}, Reward feedback: {include_feedback}, CoT: {use_cot}\n")

    max_horizon = getattr(game_config, 'max_horizon', 10)
    raw_results = []
    for idx, (case, sim) in enumerate(zip(test_cases, test_sims_sample)):
        try:
            r = _episode(game, generation_method, sim, case, action_mapping,
                         weight, objective_names, objective_descriptions, max_horizon,
                         include_pareto_hint, include_feedback, use_cot=use_cot)
            raw_results.append(r)
            logger.info(
                f"[Uniform] ep{idx}: turns={r['n_turns']} success={r[SUCCESS_RATE]} "
                f"sl={r[SL_RATIO]:.3f} fair={r[FAIRNESS]:.3f} dr={r[DEAL_RATE]:.3f} "
                f"weighted={r['cumulative_weighted_return']:+.3f}"
            )
        except Exception as e:
            logger.error(f"[Uniform] ep{idx} crashed: {e}")
            continue

    if not raw_results:
        raise RuntimeError("All episodes failed.")
    average = _aggregate(raw_results)

    # PADPP paper Table 2 negotiation 'uniform' row for reference
    padpp_ref = {
        'SR': 0.427, 'avg_turn': 9.638,
        'r_gain': 0.622, 'r_fair': 0.287, 'r_deal': 0.142,
    } if args['scenario'] == NEGOTIATION else None

    _print_table(average, n_test, policy_model, weight, objective_names, padpp_ref)

    out_dir = eval_overrides['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"llm_uniform_{local_time}.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            'baseline': 'llm-uniform-with-feedback',
            'policy_model': policy_model,
            'gen_model': args['gen_models'],
            'weight_vector': weight,
            'objective_names': objective_names,
            'pareto_hint': include_pareto_hint,
            'reward_feedback': include_feedback,
            'n_episodes': n_test,
            'average': average,
            'padpp_ref': padpp_ref,
            'episodes': raw_results,
        }, f, indent=2, default=str)
    print(f"\nSaved detailed results: {out_file}\n")
