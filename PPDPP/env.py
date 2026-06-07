import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastchat.model import load_model, get_conversation_template

import openai
import os
from dotenv import load_dotenv

from utils import *
from prompt import *
#from unidecode import unidecode
import nltk
import re
import time
from ppdpp_rewards import (
    OBJECTIVE_WEIGHTS,
    compute_reward_info,
    rule_judge_dialogue,
)

# Strict LLM judge (shared with HMOD baseline) for --judge_model llm.
import hashlib
import sys as _sys
from collections import OrderedDict
_HMOD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _HMOD_ROOT not in _sys.path:
    _sys.path.insert(0, _HMOD_ROOT)
try:
    from hmod.judge import llm_judge_deal as _strict_llm_judge_deal
except Exception:
    _strict_llm_judge_deal = None

# Map --judge_model llm to a default FPT-backed model_type. The actual API
# key/base_url/model are picked up by utils.prompt.call_llm via .env.
from config.constants import FPT as _FPT_MODEL_TYPE  # noqa: E402
_LLM_JUDGE_CACHE_MAX = int(os.getenv("PPDPP_LLM_JUDGE_CACHE_SIZE", "512"))
_LLM_JUDGE_CACHE: "OrderedDict[str, tuple]" = OrderedDict()


def _llm_judge_cache_put(key, value):
    """Insert/refresh an entry and keep cache size bounded."""
    _LLM_JUDGE_CACHE[key] = value
    _LLM_JUDGE_CACHE.move_to_end(key)
    while len(_LLM_JUDGE_CACHE) > _LLM_JUDGE_CACHE_MAX:
        _LLM_JUDGE_CACHE.popitem(last=False)


def clear_llm_judge_cache():
    """Public helper for run.py to release cached judge results between phases."""
    _LLM_JUDGE_CACHE.clear()

system_role = {'esc':'Therapist', 'cima': 'Teacher', 'cb': 'Buyer'}
user_role = {'esc':'Patient', 'cima': 'Student', 'cb': 'Seller'}
message_format = {'esc': ESConvMessages, 'cima': CIMAMessages, 'cb': CBMessages}

YOUR_API_KEY = ""

_ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, ".env"))
if os.path.exists(_ENV_PATH):
    load_dotenv(_ENV_PATH, override=False)


def _resolve_openai_config():
    api_key = (
        os.getenv("PPDPP_OPENAI_API_KEY")
        or os.getenv("FPT_API_KEY")
        or os.getenv("POLICY_LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or YOUR_API_KEY
    )
    base_url = (
        os.getenv("PPDPP_OPENAI_BASE_URL")
        or os.getenv("FPT_API_URL")
        or os.getenv("POLICY_LLM_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
    )
    model = (
        os.getenv("PPDPP_OPENAI_MODEL")
        or os.getenv("FPT_MODEL")
        or os.getenv("POLICY_LLM_MODEL")
        or "gpt-3.5-turbo-0613"
    )
    return api_key, base_url, model


PPDPP_OPENAI_API_KEY, PPDPP_OPENAI_BASE_URL, PPDPP_OPENAI_MODEL = _resolve_openai_config()

class Env(object):
    def __init__(self, args, dataset, mode, env_model=None, env_tokenizer=None):
        if 'vicuna' in [args.system, args.user, args.critic] or 'llama2' in [args.system, args.user, args.critic]:
            if mode == 'train':
                self.vicuna_model, self.vicuna_tokenizer = load_model(
                    args.model_path,
                    args.device,
                    args.num_gpus,
                    args.max_gpu_memory,
                    args.load_8bit,
                    args.cpu_offloading,
                    debug=args.debug,
                )
            else:
                self.vicuna_model = env_model
                self.vicuna_tokenizer = env_tokenizer
        
        
        self.args = args
        self.dataset = dataset[mode]
        self.max_turn = args.max_turn
        self.conversation = []
        self.turn_records = []
        self.last_reward_info = None
        self.cur_conver_step = 0
        self.test_num = 0
        self.mode = mode

        self.reward_dict = {
            'esc': {
                'worse': -1.0,
                'same': -0.5,
                'better': 0.5,
                'solved': 1.0,
            },
            'cima': {
                'incorrect': -1.0,
                'did not': -0.5,
                'part': 0.5,
                'whole': 1.0,
            },
        }

        set_random_seed(args.seed)

        
    def reset(self):
        self.cur_conver_step = 0
        self.turn_records = []
        self.last_reward_info = None
        if self.mode == 'train':
            self.case = np.random.choice(self.dataset)
        elif self.mode == 'test':
            self.case = self.dataset[self.test_num]
            self.test_num += 1
        
        if self.args.data_name == 'esc':
            self.conversation = [{"role":"Patient", "content":self.case['situation']}]
        elif self.args.data_name == 'cima':
            self.conversation = [{"role":"Teacher", "content":self.case['dialog'][0]['text']}, {"role":"Student", "content":self.case['dialog'][1]['text']}]
        elif self.args.data_name == 'cb':
            self.conversation = [{"role":"Buyer", "content":"Hi, how much is the %s?" % self.case['item_name']}, {"role":"Seller", "content":"Hi, this is a good %s and its price is %s." % (self.case['item_name'], self.case['seller_price'])}]
        print(self.conversation)
        return self.conversation

    def _current_turn_limit(self):
        if (getattr(self.args, "use_case_turn_limit", False)
                and self.args.data_name == "cb"
                and self.case.get("turn_limit")):
            return int(self.case["turn_limit"])
        return self.max_turn


    def step(self, action):
        done = 0
        print('---------------step:{}-------------'.format(self.cur_conver_step))
        
        print(action)
        messages = message_format[self.args.data_name](self.case, 'system', self.conversation, action)
        response = self.generate_response(self.args.system, messages, system_role[self.args.data_name])
        response = self.postprocess_response(response, user_role[self.args.data_name])
        self.conversation.append({"role":system_role[self.args.data_name],"content":response})
        print(self.conversation[-1])

        messages = message_format[self.args.data_name](self.case, 'user', self.conversation)
        user_response = self.generate_response(self.args.user, messages, user_role[self.args.data_name])
        user_response = self.postprocess_response(user_response, system_role[self.args.data_name])
        self.conversation.append({"role":user_role[self.args.data_name], "content":user_response})
        print(self.conversation[-1])

        messages = message_format[self.args.data_name](self.case, 'critic', self.conversation)
        reward = self.compute_reward(
            self.args.critic, messages, self.case,
            system_response=response, action=action)

        if self.args.data_name == "cb" and self.last_reward_info is not None:
            turn_record = {
                "turn": self.cur_conver_step,
                "action": action,
                "system_response": response,
                "user_response": user_response,
                **self.last_reward_info,
            }
            self.turn_records.append(turn_record)

        if self.args.data_name == 'esc':
            if reward > 0.5:
                print('--> Goal completed !')
                done = 1
            else:
                if self.cur_conver_step == self._current_turn_limit() - 1:
                    print('--> Maximum number of turns reached !')
                    done = -1
                else:
                    print('--> On-going !')
        elif self.args.data_name == 'cima':
            if reward == 1:
                print('--> Goal completed !')
                done = 1
            else:
                if self.cur_conver_step == self._current_turn_limit() - 1:
                    print('--> Maximum number of turns reached !')
                    done = -1
                else:
                    print('--> On-going !')
        elif self.args.data_name == 'cb':
            use_objective_reward = getattr(self.args, "objective", None) in OBJECTIVE_WEIGHTS
            if use_objective_reward:
                goal_completed = bool(
                    self.last_reward_info and self.last_reward_info.get("deal_success"))
            else:
                goal_completed = reward >= 0
            if goal_completed:
                print('--> Goal completed !')
                done = 1
            else:
                if self.cur_conver_step == self._current_turn_limit() - 1:
                    print('--> Maximum number of turns reached !')
                    done = -1
                else:
                    print('--> On-going !')
                
        self.cur_conver_step += 1
        return self.conversation, reward, done

    def get_episode_record(self):
        rewards = {"sl_ratio": 0.0, "fairness": 0.0, "deal_rate": 0.0}
        weighted_return = 0.0
        success = False
        deal_price = None
        price_attempt_count = 0
        per_turn_violations = 0

        for row in self.turn_records:
            vector = row.get("reward_vector", {})
            for key in rewards:
                rewards[key] += float(vector.get(key, 0.0) or 0.0)
            weighted_return += float(row.get("scalar_reward", 0.0) or 0.0)
            success = success or bool(row.get("deal_success"))
            if row.get("deal_price") is not None:
                deal_price = row["deal_price"]
            if row.get("system_price") is not None:
                price_attempt_count += 1
            if bool(row.get("actual_violation") or row.get("price_violation")):
                per_turn_violations += 1

        max_price = self.case.get("max_acceptable_price")
        turn_limit = int(self.case.get("turn_limit") or self._current_turn_limit())
        turns = len(self.turn_records)
        # GSR rule: only credit the goal when (a) seller accepted, (b) the
        # final deal price is known and within the buyer's private ceiling
        # (if a ceiling exists at all), and (c) the dialogue did not exceed
        # the turn budget. Per-turn "overshoot" still counts toward CVR but
        # no longer vetoes GSR -- a successful final deal is what matters.
        if not success:
            price_ok = False
        elif max_price is None:
            price_ok = True
        else:
            price_ok = (
                deal_price is not None
                and float(deal_price) <= float(max_price)
            )
        gsr = int(bool(success and price_ok and turns <= turn_limit))

        # Mirror hmod.compute_cvr: 1 violation_trace entry per buyer turn, plus
        # 1 extra entry when the closed deal price exceeds the ceiling.
        final_deal_violation = bool(
            success
            and max_price is not None
            and deal_price is not None
            and float(deal_price) > float(max_price)
        )
        violation_attempts = turns + (1 if final_deal_violation else 0)
        actual_violation_count = per_turn_violations + (1 if final_deal_violation else 0)
        blocked_violation_count = 0  # PPDPP cannot mask actions before emission.
        denom = max(violation_attempts, 1)
        blocked_cvr = blocked_violation_count / denom
        actual_cvr = actual_violation_count / denom

        return {
            "scenario_id": self.case.get("scenario_id"),
            "source_dataset": self.case.get("source_dataset"),
            "recommendation_domain": self.case.get("recommendation_domain"),
            "drift_mode": self.case.get("drift_mode"),
            "seller_persona_type": self.case.get("seller_persona_type"),
            "buyer_intent_id": self.case.get("buyer_intent_id"),
            "objective": getattr(self.args, "objective", None),
            "static_w": self.case.get("static_w"),
            "success": bool(success),
            "gsr": gsr,
            "turns": turns,
            "weighted_return": weighted_return,
            "cumulative_reward_vector": rewards,
            "deal_price": deal_price,
            "max_acceptable_price": max_price,
            "price_violation": bool(per_turn_violations or final_deal_violation),
            "price_attempt_count": price_attempt_count,
            "blocked_cvr": blocked_cvr,
            "actual_cvr": actual_cvr,
            "cvr": actual_cvr,
            "blocked_violation_count": blocked_violation_count,
            "actual_violation_count": actual_violation_count,
            "violation_attempt_count": violation_attempts,
            "final_deal_violation": final_deal_violation,
        }

    def _strict_llm_judge(self):
        """Run hmod.judge.llm_judge_deal on the current PPDPP dialogue.

        Returns the (deal_success, deal_price) tuple PPDPP expects, falling
        back to the strict rule judge on LLM failure. Results are cached by a
        hash of the canonical dialogue so identical histories within a run do
        not pay for repeated API calls.
        """
        if _strict_llm_judge_deal is None:
            return rule_judge_dialogue(self.conversation)
        canonical = []
        for turn in self.conversation:
            role = str(turn.get("role", "")).lower()
            mapped = "assistant" if role in {"buyer", "assistant", "system"} else "user"
            canonical.append({"role": mapped, "content": turn.get("content", "")})
        import json as _json
        key = hashlib.sha1(
            _json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cached = _LLM_JUDGE_CACHE.get(key)
        if cached is not None:
            _LLM_JUDGE_CACHE.move_to_end(key)
            return cached
        try:
            result = _strict_llm_judge_deal(canonical, model_type=_FPT_MODEL_TYPE)
            tup = (bool(result.get("deal")), result.get("deal_price"))
        except Exception as exc:
            print(f"[llm judge] fallback to rule on error: {exc}")
            tup = rule_judge_dialogue(self.conversation)
        _llm_judge_cache_put(key, tup)
        return tup

    def get_dialogue_record(self):
        return {
            "scenario_id": self.case.get("scenario_id"),
            "objective": getattr(self.args, "objective", None),
            "dialogue": self.conversation,
            "turn_records": self.turn_records,
            # Snapshot the per-case scalar fields so offline tools (judge
            # replay, parser regression, metric recomputation) do not need
            # the original scenario YAML to rebuild metrics.
            "case_meta": {
                "scenario_id": self.case.get("scenario_id"),
                "source_dataset": self.case.get("source_dataset"),
                "recommendation_domain": self.case.get("recommendation_domain"),
                "drift_mode": self.case.get("drift_mode"),
                "seller_persona_type": self.case.get("seller_persona_type"),
                "buyer_intent_id": self.case.get("buyer_intent_id"),
                "static_w": self.case.get("static_w"),
                "buyer_price": self.case.get("buyer_price"),
                "seller_price": self.case.get("seller_price"),
                "max_acceptable_price": self.case.get("max_acceptable_price"),
                "turn_limit": self.case.get("turn_limit"),
                "item_name": self.case.get("item_name"),
            },
        }
    
    def postprocess_response(self, response, role):
        #print(response)
        if role in response:
            response = response.split(role)[0].strip()
        sents = nltk.sent_tokenize(response)
        if len(sents) == 1:
            if response[-1] not in ['.','!','?',':']:
                return response + '.'
            return response.strip()
        try:
            if sents[-1].strip()[-1] not in ['.','!','?',':']:
                return ' '.join(sents[:-1]).strip()
            else:
                return response.strip()
        except Exception as e:
            return response.strip()

    def generate_response(self, model, messages, role):
        if self.mode == 'test':
            temperature = 0
        else:
            temperature = 0.7
        if model == 'vicuna':
            prompt = vicuna_prompt(messages, role)
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            #print(len(input_ids[0]))
            max_new_tokens = self.args.max_new_tokens
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=max_new_tokens,
                temperature = temperature,
                early_stopping=True
            )
            output_ids = output_ids[0][len(input_ids[0]):]
            output = self.vicuna_tokenizer.decode(output_ids, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
        elif model == 'llama2':
            prompt = llama2_prompt(messages, role)
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            #print(len(input_ids[0]))
            max_new_tokens = self.args.max_new_tokens
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=max_new_tokens,
                temperature = temperature,
                early_stopping=True
            )
            output_ids = output_ids[0][len(input_ids[0]):]
            output = self.vicuna_tokenizer.decode(output_ids, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
        elif model == 'chatgpt':
            messages = chatgpt_prompt(messages, role)
            #print(messages)
            output = query_openai_model(
                api_key=PPDPP_OPENAI_API_KEY,
                base_url=PPDPP_OPENAI_BASE_URL,
                messages=messages,
                model=PPDPP_OPENAI_MODEL,
                max_tokens=self.args.max_new_tokens,
                temperature=temperature
            )
        return output
    
    def compute_reward(self, model, messages, case, system_response=None, action=None):
        judge_model = getattr(self.args, "judge_model", "critic")
        judge_result = None
        outputs = []

        if self.args.data_name == "cb" and judge_model == "rule":
            judge_result = rule_judge_dialogue(self.conversation)
        elif self.args.data_name == "cb" and judge_model == "llm":
            judge_result = self._strict_llm_judge()
        elif model == 'vicuna':
            prompt = vicuna_prompt(messages, 'critic')
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=16,
                temperature = 1.1,
                do_sample = True,
                early_stopping=True,
                num_return_sequences=10,
            )
            for o in output_ids:
                output_id = o[len(input_ids[0]):]
                output = self.vicuna_tokenizer.decode(output_id, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
                outputs.append(output)
        elif model == 'llama2':
            prompt = llama2_prompt(messages, 'critic')
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=16,
                temperature = 1.1,
                do_sample = True,
                early_stopping=True,
                num_return_sequences=10,
            )
            for o in output_ids:
                output_id = o[len(input_ids[0]):]
                output = self.vicuna_tokenizer.decode(output_id, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
                outputs.append(output)
        elif model == 'chatgpt':
            messages = chatgpt_prompt(messages, user_role[self.args.data_name])
            outputs = query_openai_model(
                api_key=PPDPP_OPENAI_API_KEY,
                base_url=PPDPP_OPENAI_BASE_URL,
                messages=messages,
                model=PPDPP_OPENAI_MODEL,
                max_tokens=self.args.max_new_tokens,
                temperature=1.1,
                n=10
            )
        
        if self.args.data_name in ['esc','cima']:
            rewards = []
            print(outputs)
            for output in outputs:
                for key in self.reward_dict[self.args.data_name]:
                    if key in output.lower():
                        rewards.append(self.reward_dict[self.args.data_name][key])
                        break
            if len(rewards) == 0:
                reward = 0
            else:
                reward = sum(rewards)/len(rewards)
            print(reward)
        elif self.args.data_name == 'cb':
            deals = []
            rewards = []
            print(outputs)
            if judge_result is not None:
                deal_success, deal_price = judge_result
                if deal_success and deal_price is not None:
                    reward = (deal_price - case['seller_price']) / (case['buyer_price'] - case['seller_price'])
                    rewards.append(reward)
                else:
                    deals.append(-1)
            else:
                for output in outputs:
                    if 'have not' in output.lower():
                        deals.append(-1)
                    elif 'have reached' in output.lower():
                        deals.append(1)

                    prices = re.findall(r"[-+]?\d*\.?\d+", output.replace(",",""))
                    if len(prices) > 0:
                        deal_price = float(prices[0])
                        reward = (deal_price - case['seller_price']) / (case['buyer_price'] - case['seller_price'])
                        rewards.append(reward)

            if -1 in deals:
                reward = -0.1
            else:
                if len(rewards) == 0:
                    reward = 0
                else:
                    reward = max(set(rewards), key = rewards.count)
            print(reward)

            objective = getattr(self.args, "objective", "legacy")
            if objective in OBJECTIVE_WEIGHTS:
                self.last_reward_info = compute_reward_info(
                    case=case,
                    conversation=self.conversation,
                    system_response=system_response or "",
                    action=action,
                    objective=objective,
                    judge_outputs=outputs,
                    judge_result=judge_result,
                )
                self.last_reward_info["legacy_reward"] = reward
                reward = self.last_reward_info["scalar_reward"]
            else:
                self.last_reward_info = {
                    "scalar_reward": reward,
                    "reward_vector": {},
                    "deal_success": reward >= 0,
                    "deal_price": None,
                    "objective": objective,
                    "legacy_reward": reward,
                }

        return reward



def query_openai_model(api_key: str, messages: str, model: str = "gpt-3.5-turbo-0613", max_tokens: int = 128, temperature: float = 0, n: int = 1, base_url: str = None):
    if not api_key:
        raise RuntimeError(
            "Missing API key for chatgpt mode. Set one of: "
            "PPDPP_OPENAI_API_KEY, POLICY_LLM_API_KEY, FPT_API_KEY, OPENAI_API_KEY."
        )

    # Prefer OpenAI v1+ client; keep legacy fallback for older SDKs.
    client = None
    if hasattr(openai, "OpenAI"):
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = openai.OpenAI(**kwargs)
    else:
        openai.api_key = api_key
        if base_url:
            openai.api_base = base_url

    flag = True
    while flag:
        try:
            if client is not None:
                completions = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    n=n,
                    temperature=temperature,
                    timeout=10,
                )
                if n == 1:
                    output = (completions.choices[0].message.content or "").strip()
                else:
                    output = [
                        (choice.message.content or "").strip()
                        for choice in completions.choices
                    ]
            else:
                completions = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    n=n,
                    stop=None,
                    temperature=temperature,
                    request_timeout=10,
                )
                if n == 1:
                    output = completions.choices[0].message.content.strip()
                else:
                    output = []
                    for choice in completions.choices:
                        output.append(choice.message.content.strip())

            flag = False
        except Exception as e:
            print(f"Some error happened here: {e}")
            time.sleep(5)
    return output
