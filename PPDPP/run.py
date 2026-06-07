from env import Env, clear_llm_judge_cache
from agent import PPDPP
from utils import *
from itertools import count
from tqdm import tqdm
import argparse
import gc
import json
import os
import time
import torch
from transformers import BertTokenizer, RobertaTokenizer, BertConfig, RobertaConfig
from fastchat.model import add_model_args
from scenario_loader import load_custom_dataset
from ppdpp_rewards import (
    OBJECTIVE_WEIGHTS,
    aggregate_episode_records,
    subgroup_metrics,
    write_evaluation_outputs,
)

tok = {'bert': BertTokenizer, 'roberta': RobertaTokenizer}
cfg = {'bert': BertConfig, 'roberta': RobertaConfig}


def _memory_cleanup(clear_judge_cache=False):
    """Best-effort host + MPS memory cleanup between long phases."""
    if clear_judge_cache:
        clear_llm_judge_cache()
    gc.collect()
    if torch.backends.mps.is_available() and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _resolve_runtime_device(requested):
    """Select an available torch device, with macOS MPS fallback support."""
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



def train(args, config, dataset, filename, tokenizer):
    env = Env(args, dataset, mode='train') # env init
    set_random_seed(args.seed)
    policy = PPDPP(args, config, tokenizer) # policy network init

    # load policy parameters
    if args.sft_dir is not None:
        print('Staring loading policy model from {}'.format(args.sft_dir))
        policy.load_model(data_name=args.data_name, filename=args.sft_dir)
    
    if args.load_rl_epoch > 0:
        print('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
        policy.load_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)
    

    test_performance = []
    if args.do_eval:
        initial_full_eval = args.do_eval and (not args.do_train)
        SR15_mean = evaluate(
            args,
            dataset,
            policy,
            filename,
            0,
            env,
            eval_limit=None if initial_full_eval else args.quick_eval_num_cases,
            eval_tag='full' if initial_full_eval else 'quick',
        )
        test_performance = [SR15_mean]
        _memory_cleanup(clear_judge_cache=True)
    if not args.do_train:
        return
    start_step = max(int(getattr(args, 'load_rl_epoch', 0) or 0) + 1, 1)
    for train_step in range(start_step, args.max_steps+1):
        SR, AvgT, total_reward = 0., 0., 0.
        loss = torch.tensor(0, dtype=torch.float, device=args.device)
        for i_episode in tqdm(range(args.sample_times),desc='sampling'):
            #blockPrint()
            print('\n================new tuple:{}===================='.format(i_episode))
            state = env.reset()

            epi_reward = 0
            done = False
            for t in count():   # user  dialog
                action = policy.select_action(state)
                state, reward, done = env.step(action)
                epi_reward += reward
                reward = torch.tensor([reward], device=args.device, dtype=torch.float)
                policy.rewards.append(reward)

                if done:
                    if done == 1:
                        SR += 1
                    AvgT += t+1
                    total_reward += epi_reward
                    break

            newloss = policy.optimize_model()
            if newloss is not None:
                loss += newloss
            # Prevent growth from temporary dialogue/model tensors over long runs.
            if i_episode % 10 == 0:
                _memory_cleanup(clear_judge_cache=False)
            
        enablePrint() # Enable print function
        print('loss : {} in epoch_uesr {}'.format(loss.item()/args.sample_times, args.sample_times))
        print('SR:{}, AvgT:{}, rewards:{} Total epoch_uesr:{}'.format(SR / args.sample_times,
                    AvgT / args.sample_times, total_reward / args.sample_times, args.sample_times))

        if train_step % args.eval_num == 0:
            is_full_eval = (train_step % args.full_eval_every == 0)
            eval_limit = None if is_full_eval else args.quick_eval_num_cases
            eval_tag = 'full' if is_full_eval else 'quick'
            SR_all = evaluate(args, dataset, policy, filename, train_step, env, eval_limit=eval_limit, eval_tag=eval_tag)
            test_performance.append(SR_all)
            _memory_cleanup(clear_judge_cache=True)
        if train_step % args.save_num == 0:
            policy.save_model(data_name=args.data_name, filename=filename, epoch_user=train_step)
        _memory_cleanup(clear_judge_cache=True)
    print(test_performance)

def evaluate(args, dataset, policy, filename, i_episode, train_env, eval_limit=None, eval_tag=None):
    if 'vicuna' in [args.system, args.user, args.critic] or 'llama2' in [args.system, args.user, args.critic]:
        test_env = Env(args, dataset, mode='test', env_model=train_env.vicuna_model, env_tokenizer=train_env.vicuna_tokenizer)
    else:
        test_env = Env(args, dataset, mode='test') # env init
    # Keep per-phase judge cache bounded and reset before each eval split.
    clear_llm_judge_cache()
    set_random_seed(args.seed)

    SR, AvgT, total_reward = 0, 0, 0
    SR_turn = [0]* args.max_turn
    episode_records = []
    dialogue_records = []
    test_size = len(test_env.dataset)
    if eval_limit is not None:
        test_size = min(test_size, int(eval_limit))
    print('Test size: ', test_size)
    prefix = f'{eval_tag}-' if eval_tag else ''
    test_filename = '{}Evaluate-epoch-{}-'.format(prefix, i_episode) + filename
    record_filename = '{}Record-epoch-{}-'.format(prefix, i_episode) + filename
    REC_PATH = TMP_DIR[args.data_name] + '/eval_result/' + record_filename + '.txt'
    if not os.path.isdir(TMP_DIR[args.data_name] + '/eval_result/'):
        os.makedirs(TMP_DIR[args.data_name] + '/eval_result/')
    rec_file = open(REC_PATH, 'w')
    for test_num in tqdm(range(test_size)):  #test_size
        #blockPrint()
        print('\n================test tuple:{}===================='.format(test_num))
        epi_reward = 0
        done = 0
        is_last_turn = False
        state = test_env.reset()
        for t in count():  # user  dialog
            action = policy.select_action(state, is_test=True)
            state, reward, done = test_env.step(action)
            if (args.data_name == 'cb'
                    and getattr(args, "objective", None) not in OBJECTIVE_WEIGHTS
                    and reward < 0): # reward = Sale-to-List Ratio
                reward = 0
            epi_reward += reward

            if done:
                if done == 1:  
                    SR_turn = [v+1 if i>t  else v for i, v in enumerate(SR_turn) ]
                    SR += 1
                total_reward += epi_reward
                AvgT += t+1

                rec_file.write('%s\n\n' % str({'dialog':state, 'reward':epi_reward}))
                if args.data_name == 'cb' and getattr(args, "objective", None) in OBJECTIVE_WEIGHTS:
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
        SR_mean = float(SR)/test_size
        AvgT_mean = float(AvgT)/test_size
        reward_mean = total_reward/test_size
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
    save_rl_mtric(dataset=args.data_name, filename=test_filename, epoch=test_num, SR=SR_all, mode='test')  # save RL SR
    print('save test evaluate successfully!')

    if episode_records:
        metrics_payload = {
            "model": "PPDPP",
            "epoch": i_episode,
            "filename": filename,
            "data_name": args.data_name,
            "scenario_file": getattr(args, "test_scenario_file", None) or getattr(args, "scenario_file", None),
            "objective": args.objective,
            "objective_weight": OBJECTIVE_WEIGHTS[args.objective],
            "metrics": metrics,
            "metrics_by_drift_mode": subgroup_metrics(episode_records, args.objective, "drift_mode"),
            "metrics_by_persona": subgroup_metrics(episode_records, args.objective, "seller_persona_type"),
            "metrics_by_objective": subgroup_metrics(episode_records, args.objective, "buyer_intent_id"),
            "metrics_by_recommendation_domain": subgroup_metrics(
                episode_records, args.objective, "recommendation_domain"),
            "num_dialogues": len(episode_records),
        }
        run_dir = os.path.join(
            args.output_dir,
            f"{filename}-{eval_tag or 'eval'}-epoch-{i_episode}-{time.strftime('%Y%m%d_%H%M%S')}",
        )
        write_evaluation_outputs(run_dir, metrics_payload, episode_records, dialogue_records)
        print(f"Structured evaluation outputs written to {run_dir}")

    SRturn_all = [0] * args.max_turn
    for i in range(len(SRturn_all)):
        SRturn_all[i] = float(SR_turn[i])/test_size
    print('success turn:{}'.format(SRturn_all))
    print('SR:{}, AvgT:{}, reward:{}'.format(SR_mean, AvgT_mean, reward_mean))
    PATH = TMP_DIR[args.data_name] + '/eval_result/' + test_filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('Training epocch:{}\n'.format(i_episode))
        f.write('===========Test Turn===============\n')
        f.write('Testing {} user tuples\n'.format(test_num))
        for i in range(len(SRturn_all)):
            f.write('Testing SR-turn@{}: {}\n'.format(i, SRturn_all[i]))
        f.write('================================\n')
    with open(PATH, 'a') as f:
        f.write('{}\t{}\t{}\t{}\n'.format(i_episode, SR_mean, AvgT_mean, reward_mean))
    # Drop heavy lists/environment refs immediately after metrics are written.
    del episode_records
    del dialogue_records
    del test_env
    _memory_cleanup(clear_judge_cache=True)
    return SR_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=1, help='random seed.')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus.')
    parser.add_argument('--epochs', '-me', type=int, default=50000, help='the number of RL train epoch')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='learning rate.')

    parser.add_argument('--data_name', type=str, default='cb', choices=['esc','cima','cb'],
                        help='One of {esc, cima, cb}.')
    parser.add_argument('--system', type=str, default='vicuna', choices=['vicuna','chatgpt','llama2'],
                        help='One of {vicuna, chatgpt, llama2}.')
    parser.add_argument('--user', type=str, default='vicuna', choices=['vicuna','chatgpt','llama2'],
                        help='One of {vicuna, chatgpt, llama2}.')
    parser.add_argument('--critic', type=str, default='vicuna', choices=['vicuna','chatgpt','llama2'],
                        help='One of {vicuna, chatgpt, llama2}.')
    parser.add_argument('--sft_dir', default='sft', #../pretrain/outputs/best_pretrain.pt
                        type=str, help="Pretrain model path.")
    parser.add_argument('--max_turn', type=int, default=8, help='max conversation turn')
    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')
    parser.add_argument('--load_rl_epoch', type=int, default=0, help='load agent from epoch')
    parser.add_argument('--scenario_file', type=str, default=None,
                        help='Custom HMOD-style scenario YAML/JSONL for training.')
    parser.add_argument('--test_scenario_file', type=str, default=None,
                        help='Custom HMOD-style scenario YAML/JSONL for evaluation.')
    parser.add_argument('--valid_scenario_file', type=str, default=None,
                        help='Optional custom HMOD-style validation scenario YAML/JSONL.')
    parser.add_argument('--objective', type=str, default='uniform',
                        choices=sorted(OBJECTIVE_WEIGHTS.keys()),
                        help='Fixed PPDPP scalarization objective for CB custom experiments.')
    parser.add_argument('--num_cases', type=int, default=None,
                        help='Limit custom test/eval scenarios for smoke runs.')
    parser.add_argument('--train_num_cases', type=int, default=None,
                        help='Limit custom train scenarios for smoke runs.')
    parser.add_argument('--output_dir', type=str, default='outputs/ppdpp_eval',
                        help='Directory for metrics.json, dialogues.jsonl, and summary.csv.')
    parser.add_argument('--full_eval_every', type=int, default=3,
                        help='Run full test-set evaluation every N training epochs.')
    parser.add_argument('--quick_eval_num_cases', type=int, default=20,
                        help='Representative test subset size for non-full evaluations.')
    parser.add_argument('--judge_model', type=str, default='critic', choices=['critic', 'rule', 'llm'],
                        help="'rule' = strict hmod rule judge, 'llm' = strict hmod LLM judge (FPT), otherwise use args.critic.")
    parser.add_argument('--use_case_turn_limit', action='store_true',
                        help='Use each custom scenario turn_limit instead of --max_turn.')


    parser.add_argument("--cache_dir", default='/storage_fast/ydeng/plm', type=str, help="The cache directory.")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_path", type=str, default="/storage_fast/ydeng/llm/vicuna_hf/7B")
    parser.add_argument("--model_name", type=str, default="roberta")
    parser.add_argument("--model_name_or_path", default='roberta-large', type=str, help="model name or path")

    parser.add_argument("--do_lower_case", action='store_false', help="Set this flag if you are using an uncased model.")

    parser.add_argument('--max_steps', type=int, default=10, help='max training steps')
    parser.add_argument('--sample_times', type=int, default=100, help='the epoch of sampling')
    parser.add_argument('--eval_num', type=int, default=1, help='the number of steps to evaluate RL model and metric')
    parser.add_argument('--save_num', type=int, default=1, help='the number of steps to save RL model and metric')


    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval.")

    add_model_args(parser)
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    args.device = _resolve_runtime_device(getattr(args, "device", "cuda"))
    
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(args.device)
    print('data_set:{}'.format(args.data_name))

    if args.scenario_file or args.test_scenario_file or args.valid_scenario_file:
        if args.data_name != 'cb':
            raise ValueError("Custom HMOD scenarios use the Craigslist Bargain interface; set --data_name cb.")
        # HMOD scenarios carry per-case turn_limit; mirror hmod runner so GSR
        # uses the same horizon and PPDPP/HMOD numbers stay comparable.
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
    filename = '{}-{}-{}-{}-{}-{}'.format(
        args.data_name, args.objective, args.sft_dir, args.system, args.user, args.critic)

    config = cfg[args.model_name].from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = tok[args.model_name].from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir)

    if args.sft_dir:
        args.sft_dir = os.path.join(args.sft_dir, args.data_name, args.model_name, 'best_checkpoint')
    if args.sft_dir and not os.path.exists(args.sft_dir):
        print("no sft model, randomly initialize policy model")
        args.sft_dir = None

    train(args, config, dataset, filename, tokenizer)

if __name__ == '__main__':
    main()
