"""
run_dmorl.py – Entry point for DMORL experiments.

Usage examples:

  # Negotiation (full pipeline)
  python run_dmorl.py \
      --scenario negotiation \
      --datasets craigslist_bargain \
      --models dmorl \
      --gen_models chatgpt \
      --metrics sr,deal_rate,sl_ratio,fairness,avg_turn \
      --loggers terminal,file \
      --n_basic_skills 5 \
      --n_advanced_skills 5 \
      --dynamic_weight_horizon 3 \
      --use_dynamic_weight

  # Recommendation (inference only with saved checkpoint)
  python run_dmorl.py \
      --scenario recommendation \
      --datasets durecdial \
      --models dmorl \
      --gen_models chatgpt \
      --metrics sr,user_reward,item_freq,avg_turn \
      --loggers terminal \
      --test_phase \
      --use_dynamic_weight

  # Emotional Support
  python run_dmorl.py \
      --scenario emotional_support \
      --datasets es_conv \
      --models dmorl \
      --gen_models chatgpt \
      --metrics sr,user_reward,toxicity,avg_turn \
      --loggers terminal,file \
      --n_basic_skills 5 \
      --n_advanced_skills 5

  # ── R-PADPP Phase 1 only (anchor curriculum + Table 2 eval) ────────────────
  python run_dmorl.py \
      --scenario negotiation \
      --datasets craigslist_bargain \
      --models dmorl \
      --gen_models fpt --model_type fpt \
      --metrics sr,deal_rate,sl_ratio,fairness,avg_turn \
      --loggers terminal,file \
      --n_basic_skills 7 \
      --n_skill_train_epochs 15 \
      --run_curriculum \
      --phase1_only \
      --debug

  # ── R-PADPP Phase 2 only (load Phase 1 checkpoint, run regret-gated GPI) ───
  python run_dmorl.py \
      --scenario negotiation \
      --datasets craigslist_bargain \
      --models dmorl \
      --gen_models fpt --model_type fpt \
      --metrics sr,deal_rate,sl_ratio,fairness,avg_turn \
      --loggers terminal,file \
      --n_rpadpp_epochs 30 \
      --epsilon_threshold 0.05 \
      --alpha_rpadpp 0.5 \
      --phase2_only \
      --debug
"""

import os
import sys
import time

# ── Silence terminal output ───────────────────────────────────────────────────
os.environ["TQDM_DISABLE"] = "1"   # disable all tqdm bars

# Silence misleading HuggingFace tokenizer warning "Token indices sequence
# length is longer than the specified maximum sequence length (NNN > 512)".
# The tokenize() call below manually truncates with input_ids[-(max-2):], so
# there will be no indexing errors — the warning is just preemptive noise.
import warnings as _warnings
import transformers as _transformers
_transformers.logging.set_verbosity_error()
_warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

from dotenv import load_dotenv
from accelerate import Accelerator, DistributedDataParallelKwargs
from loguru import logger
import wandb

# Redirect loguru to file only (INFO+); removes default stderr handler
logger.remove()
_log_dir = "logs"
os.makedirs(_log_dir, exist_ok=True)
_log_file = os.path.join(_log_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")
logger.add(_log_file, level="INFO", encoding="utf-8",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")

from utils.utils import (
    get_datasets_by_names, get_model_by_names, reformat_args,
    get_metrics_by_names, load_config_from_yaml_file, get_scenario_by_name,
    get_loggers_by_names, get_text_generation_model_by_name,
    load_user_simulators, create_user_simulators, set_seed, parse_args
)

from config.config import DatasetConfigForRecommendation
from eval.offline import OfflineEvaluator
from eval.online import OnlineEvaluator
from config.constants import BART_GENERATION, VICUNA, RECOMMENDATION, NEGOTIATION, EMOTIONAL_SUPPORT

from dmorl.config import DMORLConfig
from dmorl.trainer import DMORLTrainer

import argparse

load_dotenv()


def parse_dmorl_args():
    """Parse all PADPP args plus DMORL-specific overrides."""
    # Start with the standard PADPP parser
    base_args = parse_args()

    # Parse DMORL extras from remaining argv (allows --dmorl_xxx flags)
    dmorl_parser = argparse.ArgumentParser(add_help=False)
    dmorl_parser.add_argument('--n_basic_skills', type=int, default=None)
    dmorl_parser.add_argument('--n_advanced_skills', type=int, default=None)
    dmorl_parser.add_argument('--n_skill_train_epochs', type=int, default=None)
    dmorl_parser.add_argument('--n_advanced_train_epochs', type=int, default=None)
    dmorl_parser.add_argument('--dynamic_weight_horizon', type=int, default=None)
    dmorl_parser.add_argument('--use_dynamic_weight', action='store_true', default=None)
    dmorl_parser.add_argument('--no_dynamic_weight', dest='use_dynamic_weight',
                               action='store_false')
    dmorl_parser.add_argument('--use_hints', action='store_true', default=None)
    dmorl_parser.add_argument('--run_curriculum', action='store_true', default=None)
    dmorl_parser.add_argument('--no_curriculum', dest='run_curriculum', action='store_false')
    dmorl_parser.add_argument('--force_rediscover_skills', action='store_true')
    dmorl_parser.add_argument('--phase1_only', action='store_true', default=False,
                               help='Stop after Phase 1 anchor curriculum (save dmorl_phase1.pth and eval Table 2)')
    dmorl_parser.add_argument('--phase2_only', action='store_true', default=False,
                               help='Skip SFT + Phase 1 (load dmorl_phase1.pth), run R-PADPP Phase 2 only')
    dmorl_parser.add_argument('--n_rpadpp_epochs', type=int, default=None,
                               help='Phase 2 RL epoch count override')
    dmorl_parser.add_argument('--epsilon_threshold', type=float, default=None,
                               help='Regret threshold for admitting w to W_converged')
    dmorl_parser.add_argument('--alpha_rpadpp', type=float, default=None,
                               help='Mixing weight: L = (1-α)·L_self + α·L_know')
    dmorl_parser.add_argument('--use_active_sampling', action='store_true', default=None,
                               help='Pick highest-regret candidate w for rollout each epoch')
    dmorl_parser.add_argument('--use_llm_price_extraction', action='store_true', default=False,
                               help='Use the LLM (gen model) to extract the buyer price in compute_reward')
    dmorl_parser.add_argument('--use_price_tag', action='store_true', default=False,
                               help='Buyer/seller LLMs append a [[PRICE: x]] tag; compute_reward parses it directly')
    dmorl_parser.add_argument('--fairness_train_scale', type=float, default=None,
                               help='Multiply fairness reward by this during training (e.g. 2.0). Keep 1.0 at eval.')
    dmorl_parser.add_argument('--turn_penalty', type=float, default=None,
                               help='Per-turn penalty subtracted from every reward component during training (e.g. 0.05). Keep 0.0 at eval.')
    dmorl_parser.add_argument('--no_mask_redundant', dest='mask_redundant_actions',
                               action='store_false', default=None,
                               help='Disable masking of bin-redundant duplicate actions (default: masked)')
    dmorl_parser.add_argument('--eps_start', type=float, default=None,
                               help='Epsilon at epoch 0 (default 1.0)')
    dmorl_parser.add_argument('--eps_end', type=float, default=None,
                               help='Floor epsilon after decay (default 0.05)')
    dmorl_parser.add_argument('--eps_decay_epochs', type=int, default=None,
                               help='Epochs to linearly decay eps_start→eps_end before the floor (default 15)')
    dmorl_parser.add_argument('--debug', action='store_true', default=False,
                               help='Print LLM prompts/outputs, per-step rewards/losses, save eval dialogues')
    dmorl_parser.add_argument('--debug_output_dir', type=str, default=None,
                               help='Directory for debug artefacts (default: debug_output)')
    dmorl_parser.add_argument('--skip_sft', action='store_true', default=False,
                               help='Skip SFT phase (use existing checkpoint)')
    dmorl_parser.add_argument('--skip_offline_eval', action='store_true', default=False,
                               help='Skip offline evaluation phase')
    dmorl_parser.add_argument('--sampled_times', type=int, default=None,
                               help='Episodes collected per RL epoch (override config, e.g. 5 for fast debug)')

    import sys
    dmorl_extra, _ = dmorl_parser.parse_known_args(sys.argv[1:])
    return base_args, vars(dmorl_extra)


if __name__ == '__main__':
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    args, dmorl_overrides = parse_dmorl_args()
    args = reformat_args(vars(args))
    set_seed(args['seed'])

    accelerator = Accelerator(device_placement=True, kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    # ── Scenario setup ────────────────────────────────────────────────────────
    game_config_file, game_config_class, game_class, game_simulator_class = \
        get_scenario_by_name(args['scenario'])
    game_params = load_config_from_yaml_file(game_config_file)
    game_config = game_config_class(game_params)
    game_config.set_params({
        'seed': args['seed'],
        'model_type': args['model_type'],
        'use_llm_price_extraction': dmorl_overrides.get('use_llm_price_extraction', False),
        'use_price_tag': dmorl_overrides.get('use_price_tag', False),
        'fairness_train_scale': (dmorl_overrides.get('fairness_train_scale')
                                 if dmorl_overrides.get('fairness_train_scale') is not None else 1.0),
        'turn_penalty': (dmorl_overrides.get('turn_penalty')
                         if dmorl_overrides.get('turn_penalty') is not None else 0.0),
    })

    # ── Datasets ──────────────────────────────────────────────────────────────
    dataset_config_classes_and_paths = get_datasets_by_names(
        args['scenario'], args['datasets'])

    # ── Models ───────────────────────────────────────────────────────────────
    model_classes_and_pipelines = get_model_by_names(
        args['scenario'], args['models'])

    # ── Metrics ───────────────────────────────────────────────────────────────
    offline_metrics, online_metrics = get_metrics_by_names(
        args['scenario'], args['metrics'])
    offline_evaluator = OfflineEvaluator(offline_metrics)
    online_evaluator = OnlineEvaluator(online_metrics)

    # ── Generation models ─────────────────────────────────────────────────────
    generation_packages = get_text_generation_model_by_name(
        args['scenario'], args['gen_models'])

    # ── Loop over datasets ────────────────────────────────────────────────────
    for data_config_path, dataset_class, dataset_scenario_config_class in \
            dataset_config_classes_and_paths:

        dataset_params = load_config_from_yaml_file(data_config_path)
        dataset_config = dataset_scenario_config_class(dataset_params)

        if isinstance(dataset_config, DatasetConfigForRecommendation):
            dataset_config.set_params({"domain": args['domain']})

        dataset = dataset_class(dataset_config)

        # User simulators
        if not os.path.exists(dataset_config.save_dev_simulator_path) \
                or args['overwrite_sim']:
            train_profiles, dev_profiles, test_profiles = dataset.get_user_profiles()
            logger.info("Creating dev simulators ...")
            dev_sims = create_user_simulators(
                game_simulator_class, dev_profiles,
                saved_filed_path=dataset_config.save_dev_simulator_path)
            logger.info("Creating test simulators ...")
            test_sims = create_user_simulators(
                game_simulator_class, test_profiles,
                saved_filed_path=dataset_config.save_test_simulator_path)
        else:
            dev_sims = load_user_simulators(dataset_config.save_dev_simulator_path)
            test_sims = load_user_simulators(dataset_config.save_test_simulator_path)

        for sim in dev_sims + test_sims:
            sim.set_model_type(game_config.model_type)
            sim.is_using_persona(args['use_persona'])

        # ── Loop over models ──────────────────────────────────────────────────
        for (config_file, config_class, model_class, pipeline_class,
             trainer_class) in model_classes_and_pipelines:

            game = game_class(game_config=game_config, dataset_config=dataset_config)

            model_params = load_config_from_yaml_file(config_file)
            model_config = config_class(model_params)

            model_config.set_params({'model_type': game_config.model_type})

            if args['scenario'] == RECOMMENDATION:
                model_config.set_params({
                    'n_goals': dataset.n_goals,
                    'n_topics': dataset.n_topics,
                    'n_objectives': game_config.n_objectives,
                    'device': device,
                    'scenario_name': RECOMMENDATION,
                    'domain': args['domain'],
                })
            elif args['scenario'] == NEGOTIATION:
                model_config.set_params({
                    'n_goals': dataset.n_goals,
                    'n_objectives': game_config.n_objectives,
                    'device': device,
                    'scenario_name': NEGOTIATION,
                })
            elif args['scenario'] == EMOTIONAL_SUPPORT:
                model_config.set_params({
                    'n_goals': dataset.n_goals,
                    'n_objectives': game_config.n_objectives,
                    'device': device,
                    'scenario_name': EMOTIONAL_SUPPORT,
                })

            # Apply DMORL-specific overrides from CLI
            if isinstance(model_config, DMORLConfig):
                dmorl_params = {
                    k: v for k, v in dmorl_overrides.items() if v is not None
                }
                if args.get('objective_weight') is not None:
                    dmorl_params['objective_weight'] = [
                        float(x) for x in args['objective_weight']]
                dmorl_params['prioritized_objective'] = args.get(
                    'prioritized_objective', 'uniform')
                dmorl_params['test_phase'] = args.get('test_phase', False)
                dmorl_params['num_train_rl_epochs'] = args.get('num_train_rl_epochs', 50)
                # --skip_sft / --skip_offline_eval convenience flags
                if dmorl_overrides.get('skip_sft'):
                    dmorl_params['run_sft'] = False
                if dmorl_overrides.get('skip_offline_eval'):
                    dmorl_params['run_offline_eval'] = False
                # --phase1_only: stop after Phase 1
                if dmorl_overrides.get('phase1_only'):
                    dmorl_params['phase1_only'] = True
                # --phase2_only: skip SFT + Phase 1, load dmorl_phase1.pth, run R-PADPP
                if dmorl_overrides.get('phase2_only'):
                    dmorl_params['run_sft'] = False
                    dmorl_params['run_offline_eval'] = False
                    dmorl_params['phase2_only'] = True
                model_config.set_params(dmorl_params)

            model = model_class(model_config)
            model_name = str(model.__class__.__name__)

            # ── Generation method loop ────────────────────────────────────────
            for gen_name, generation_package in list(
                    zip(args['gen_models'].split(','), generation_packages)):

                if gen_name.strip() == BART_GENERATION:
                    (generation_config_path, generation_config_class,
                     generation_model_class, generation_trainer_class,
                     generation_pipeline_class, generation_class) = generation_package
                    gen_params = load_config_from_yaml_file(generation_config_path)
                    generation_config = generation_config_class(gen_params)
                    generation_config.set_params({
                        'device': device,
                        'special_tokens_dict': model_config.special_tokens_dict,
                    })
                    generation_model = generation_model_class(generation_config)
                    generation_trainer = generation_trainer_class(
                        dataset_config=dataset_config, accelerator=accelerator,
                        game_config=game_config, model_config=generation_config,
                        game=game, model=generation_model,
                        offline_evaluator=offline_evaluator,
                        online_evaluator=None, loggers=None)
                    generation_pipeline = generation_pipeline_class(
                        dataset_config=dataset_config, dataset=dataset,
                        trainer=generation_trainer)
                    generation_method = generation_class(
                        generation_config, generation_pipeline, is_test=True)
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

                loggers = get_loggers_by_names(
                    args['loggers'],
                    game_config=game_config,
                    dataset_config=dataset_config,
                    model_config=model_config,
                    local_time=local_time,
                    random_seed=args['seed'],
                    model_name=model_name,
                    exp_name=args.get('exp_name', ''),
                    log_dir=args.get('log_dir', 'logs'),
                    wandb_key=os.getenv('WANDB_KEY'),
                    project_name=args.get('project_name', 'DMORL'),
                )

                trainer = trainer_class(
                    accelerator=accelerator,
                    game_config=game_config,
                    model_config=model_config,
                    game=game,
                    model=model,
                    generation_method=generation_method,
                    offline_evaluator=offline_evaluator,
                    online_evaluator=online_evaluator,
                    loggers=loggers,
                )

                pipeline = pipeline_class(
                    dataset_config=dataset_config,
                    dataset=dataset,
                    trainer=trainer,
                )

                pipeline.set_user_simulators(
                    dev_simulators=dev_sims,
                    test_simulators=test_sims,
                )

                start_time = time.time()
                pipeline.execute()
                logger.info(f"Execution time: {time.time() - start_time:.1f}s")

                wandb.finish()
