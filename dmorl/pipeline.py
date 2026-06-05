"""
DMORL Pipeline (R-PADPP).

Two-phase pipeline:
  Phase 1: Basic-skill anchor curriculum + Table 2 evaluation
  Phase 2: R-PADPP (regret-gated GPI) training
"""

import os
import random
import torch

from loguru import logger
from sklearn.model_selection import train_test_split

from padpp.pipeline import (
    PADPPPipeline,
    PADPPPipelineForRecommendation,
    PADPPPipelineForNegotiation,
    PADPPPipelineForEmotionalSupport,
)
from dmorl.llm_controller import DMORLController
from dmorl.trainer import DMORLTrainer
from utils.game import create_target_set, create_cases
from text_gen.bart_generation import BARTGeneration

SCENARIO_OBJECTIVE_NAMES = {
    "recommendation":   ["user_reward", "item_freq"],
    "negotiation":      ["sl_ratio", "fairness", "deal_rate"],
    "emotional_support": ["user_reward", "toxicity", "avg_turn"],
}

OBJECTIVE_DESCRIPTIONS = {
    "sl_ratio":    "Reward in [0, 1]: 1.0 = agent (buyer) secured a very low price; 0.5 = midpoint; 0.0 = paid the seller's full ask.",
    "fairness":    "Reward in [-0.5, 0.5], max 0.5 when price equals the midpoint between buyer's target and seller's ask.",
    "deal_rate":   "Binary terminal reward, 1.0 if deal closed, 0.0 if timed out.",
    "avg_turn":    "Constant per-turn penalty of -0.1.",
    "user_reward": "User satisfaction with recommended item.",
    "item_freq":   "Coverage of the long-tail item space.",
    "toxicity":    "Negative toxicity score.",
}


class DMORLPipeline(PADPPPipeline):
    """R-PADPP pipeline: Phase 1 (anchor curriculum) → Phase 2 (regret-gated GPI)."""

    def _init_dmorl_controller(self):
        scenario = self.game_config.name
        objective_names = SCENARIO_OBJECTIVE_NAMES.get(
            scenario, [f"obj_{i}" for i in range(self.model_config.n_objectives)]
        )
        objective_descriptions = {
            n: OBJECTIVE_DESCRIPTIONS[n] for n in objective_names if n in OBJECTIVE_DESCRIPTIONS
        }
        saved_dir = getattr(self.model_config, "saved_dir", "checkpoints")
        skill_log_file = os.path.join(saved_dir, "skill_discovery.txt")
        controller = DMORLController(
            n_objectives=self.model_config.n_objectives,
            objective_names=objective_names,
            objective_descriptions=objective_descriptions,
            scenario=scenario,
            n_basic_skills=self.model_config.n_basic_skills,
            n_advanced_skills=getattr(self.model_config, "n_advanced_skills", 0),
            dynamic_weight_horizon=getattr(self.model_config, "dynamic_weight_horizon", 3),
            skills_file=self.model_config.skills_file,
            hints_file=getattr(self.model_config, "hints_file", "dmorl_hints.json"),
            skill_log_file=skill_log_file,
        )
        controller.initialize_skills(
            force_rediscover=self.model_config.force_rediscover_skills
        )
        self.trainer.model.set_skill_library(controller.skill_library)
        self.trainer.dmorl_controller = controller
        logger.info("[DMORL] Controller initialised and attached to trainer.")
        return controller

    def execute(self):
        """
        Full DMORL R-PADPP pipeline:
          SFT → offline eval → Phase 1 (anchor curriculum) → Phase 2 (R-PADPP) → online eval
        """
        offline_eval_results, online_eval_results = None, None

        if self.model_config.test_phase:
            if self.model_config.run_online_eval:
                self._init_dmorl_controller()
                if self.model_config.ablation not in (None, ''):
                    self.load_pretrained_model(is_rl=False)
                else:
                    self.load_pretrained_model(is_rl=True)
                if not isinstance(self.trainer.generation_method, BARTGeneration):
                    self.trainer.generation_method.generation_config.set_params(
                        {'temperature': 0.0})
                online_eval_results = self.run_online_test()
            return offline_eval_results, online_eval_results

        if self.model_config.run_sft:
            logger.info("[DMORL] Running SFT ...")
            self.run_sft()
            self.trainer.global_step = 0

        if self.model_config.run_offline_eval:
            self.load_pretrained_model(is_rl=False)
            logger.info("[DMORL] Running offline evaluation ...")
            offline_eval_results = self.run_offline_test()

        if self.model_config.run_rlt:
            self.trainer.global_step = 0
            phase2_only = getattr(self.model_config, "phase2_only", False)
            phase1_only = getattr(self.model_config, "phase1_only", False)

            if phase2_only:
                saved_dir = getattr(self.model_config, "saved_dir", "checkpoints")
                # Which Phase 1 checkpoint to start Phase 2 from. Defaults to the
                # last-epoch dmorl_phase1.pth; set --phase1_ckpt_name to load a
                # best-checkpoint (dmorl_phase1_best.pth / *_best_wsum.pth).
                ckpt_name = getattr(self.model_config, "phase1_ckpt_name", "dmorl_phase1.pth")
                phase1_ckpt = os.path.join(saved_dir, ckpt_name)
                if not os.path.exists(phase1_ckpt):
                    raise FileNotFoundError(
                        f"[DMORL phase2_only] Phase 1 checkpoint not found at "
                        f"{phase1_ckpt}. Run Phase 1 first."
                    )
                logger.info(f"[DMORL phase2_only] Loading Phase 1 checkpoint → {phase1_ckpt}")
                self.trainer.load_model(phase1_ckpt)
                self.model = self.trainer.model
            else:
                self.load_pretrained_model(is_rl=False)

            # Phase 0: build skill library (7 anchors)
            self._init_dmorl_controller()

            if self.model_config.run_curriculum and not phase2_only:
                logger.info("[DMORL] === Phase 1: Anchor Curriculum ===")
                self.run_phase1()

                if phase1_only:
                    logger.info("[DMORL] phase1_only=True — stopping after Phase 1.")
                    return offline_eval_results, None

            logger.info("[DMORL] === Phase 2: R-PADPP ===")
            self.run_phase2()

        if self.model_config.run_online_eval:
            # Load final checkpoint (Phase 2 if it exists, else Phase 1)
            saved_dir = getattr(self.model_config, "saved_dir", "checkpoints")
            phase2_ckpt = os.path.join(saved_dir, "dmorl_phase2.pth")
            phase1_ckpt = os.path.join(saved_dir, "dmorl_phase1.pth")
            if os.path.exists(phase2_ckpt):
                logger.info(f"[DMORL] Loading Phase 2 checkpoint for eval → {phase2_ckpt}")
                self.trainer.load_model(phase2_ckpt)
                self.model = self.trainer.model
            elif os.path.exists(phase1_ckpt):
                logger.info(f"[DMORL] Loading Phase 1 checkpoint for eval → {phase1_ckpt}")
                self.trainer.load_model(phase1_ckpt)
                self.model = self.trainer.model
            else:
                self.load_pretrained_model(is_rl=True)

            logger.info("[DMORL] Running online evaluation ...")
            if not isinstance(self.trainer.generation_method, BARTGeneration):
                self.trainer.generation_method.generation_config.set_params(
                    {'temperature': 0.0})
            online_eval_results = self.run_online_test()

        return offline_eval_results, online_eval_results

    # Subclasses provide run_phase1, run_phase2, run_online_test
    def run_phase1(self):
        raise NotImplementedError

    def run_phase2(self):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation
# ─────────────────────────────────────────────────────────────────────────────

class DMORLPipelineForRecommendation(DMORLPipeline, PADPPPipelineForRecommendation):

    def load_pretrained_model(self, is_rl=False, is_last=False):
        if not is_rl:
            path = os.path.join(self.model_config.saved_dir, f"model_{self.model_config.domain}.pth")
        else:
            path = os.path.join(self.model_config.saved_dir, f"rl_model_{self.model_config.domain}.pth")
        if not os.path.exists(path):
            raise Exception(f"No pretrained model at {path}")
        self.model = self.trainer.load_model(path)

    def _get_rlt_splits(self, dev_ratio=0.1):
        dev_target_items = create_target_set(
            self.dataset.train_convs,
            test_instances=self.dataset.dev_instances,
            num_items=self.dataset_config.num_dev_items,
            domain=self.dataset_config.domain,
        )
        action_mapping = self.dataset.construct_action_mapping(
            combine=self.model_config.combined_action)
        assert self.dev_simulators is not None
        train_items, dev_items = train_test_split(
            dev_target_items, test_size=dev_ratio, random_state=self.game_config.seed)
        train_sims, dev_sims = train_test_split(
            self.dev_simulators, test_size=dev_ratio, random_state=self.game_config.seed)
        if len(dev_sims) > len(dev_items):
            dev_sims = random.sample(dev_sims, len(dev_items))
        return train_items, dev_items, train_sims, dev_sims, action_mapping

    def run_phase1(self):
        train_items, _, train_sims, _, action_mapping = self._get_rlt_splits()
        assert isinstance(self.trainer, DMORLTrainer)
        self.trainer.train_phase1(
            cases=train_items, device=self.device,
            simulators=train_sims, action_mapping=action_mapping)

    def run_phase2(self):
        train_items, _, train_sims, _, action_mapping = self._get_rlt_splits()
        assert isinstance(self.trainer, DMORLTrainer)
        self.trainer.train_phase2(
            cases=train_items, device=self.device,
            simulators=train_sims, action_mapping=action_mapping)

    def run_online_test(self, target_items=None, simulators=None):
        if target_items is None:
            target_items = create_target_set(
                self.dataset.train_convs,
                test_instances=self.dataset.test_instances,
                num_items=self.dataset_config.num_test_items,
                domain=self.dataset_config.domain,
            )
            simulators = self.test_simulators
            if self.model_config.num_test_cases:
                random.seed(self.game_config.seed)
                target_items = random.sample(target_items, self.model_config.num_test_cases)

        action_mapping = self.dataset.construct_action_mapping(
            combine=self.model_config.combined_action)
        if len(simulators) > len(target_items):
            simulators = random.sample(simulators, len(target_items))

        assert self.test_simulators is not None
        with torch.no_grad():
            results = self.trainer.online_test_dmorl(
                target_items, device=self.device,
                simulators=simulators, action_mapping=action_mapping,
                stage='test')
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Negotiation
# ─────────────────────────────────────────────────────────────────────────────

class DMORLPipelineForNegotiation(DMORLPipeline, PADPPPipelineForNegotiation):

    def load_pretrained_model(self, is_rl=False, is_last=False):
        if not is_rl:
            path = os.path.join(self.model_config.saved_dir, "model.pth")
        else:
            path = os.path.join(self.model_config.saved_dir, "rl_model.pth")
        if not os.path.exists(path):
            raise Exception(f"No pretrained model at {path}")
        self.model = self.trainer.load_model(path)

    def _get_rlt_splits(self, dev_ratio=0.1):
        dev_cases = create_cases(
            test_instances=self.dataset.dev_instances,
            num_cases=self.dataset_config.num_dev_cases)
        action_mapping = self.dataset.construct_action_mapping(
            combine=self.model_config.combined_action)
        assert self.dev_simulators is not None
        train_cases, dev_cases_split = train_test_split(
            dev_cases, test_size=dev_ratio, random_state=self.game_config.seed)
        train_sims, dev_sims = train_test_split(
            self.dev_simulators, test_size=dev_ratio, random_state=self.game_config.seed)
        if len(dev_sims) > len(dev_cases_split):
            dev_sims = random.sample(dev_sims, len(dev_cases_split))
        return train_cases, dev_cases_split, train_sims, dev_sims, action_mapping

    def run_phase1(self):
        train_cases, _, train_sims, _, action_mapping = self._get_rlt_splits()
        self.trainer.train_phase1(
            cases=train_cases, device=self.device,
            simulators=train_sims, action_mapping=action_mapping)

    def run_phase2(self):
        train_cases, _, train_sims, _, action_mapping = self._get_rlt_splits()
        self.trainer.train_phase2(
            cases=train_cases, device=self.device,
            simulators=train_sims, action_mapping=action_mapping)

    def run_online_test(self, cases=None, simulators=None):
        if cases is None:
            cases = create_cases(
                test_instances=self.dataset.test_instances,
                num_cases=self.dataset_config.num_test_cases)
            simulators = self.test_simulators
        action_mapping = self.dataset.construct_action_mapping(
            combine=self.model_config.combined_action)
        if len(simulators) > len(cases):
            simulators = random.sample(simulators, len(cases))
        assert self.test_simulators is not None
        with torch.no_grad():
            results = self.trainer.online_test_dmorl(
                cases, device=self.device,
                simulators=simulators, action_mapping=action_mapping,
                stage='test')
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Emotional Support
# ─────────────────────────────────────────────────────────────────────────────

class DMORLPipelineForEmotionalSupport(DMORLPipeline, PADPPPipelineForEmotionalSupport):

    def load_pretrained_model(self, is_rl=False, is_last=False):
        if not is_rl:
            path = os.path.join(self.model_config.saved_dir, "model.pth")
        else:
            path = os.path.join(self.model_config.saved_dir, "rl_model.pth")
        if not os.path.exists(path):
            raise Exception(f"No pretrained model at {path}")
        self.model = self.trainer.load_model(path)

    def _get_rlt_splits(self, dev_ratio=0.1):
        dev_cases = create_cases(
            test_instances=self.dataset.dev_instances,
            num_cases=self.dataset_config.num_dev_cases)
        action_mapping = self.dataset.construct_action_mapping(
            combine=self.model_config.combined_action)
        assert self.dev_simulators is not None
        train_cases, dev_cases_split = train_test_split(
            dev_cases, test_size=dev_ratio, random_state=self.game_config.seed)
        train_sims, dev_sims = train_test_split(
            self.dev_simulators, test_size=dev_ratio, random_state=self.game_config.seed)
        if len(dev_sims) > len(dev_cases_split):
            dev_sims = random.sample(dev_sims, len(dev_cases_split))
        return train_cases, dev_cases_split, train_sims, dev_sims, action_mapping

    def run_phase1(self):
        train_cases, _, train_sims, _, action_mapping = self._get_rlt_splits()
        self.trainer.train_phase1(
            cases=train_cases, device=self.device,
            simulators=train_sims, action_mapping=action_mapping)

    def run_phase2(self):
        train_cases, _, train_sims, _, action_mapping = self._get_rlt_splits()
        self.trainer.train_phase2(
            cases=train_cases, device=self.device,
            simulators=train_sims, action_mapping=action_mapping)

    def run_online_test(self, cases=None, simulators=None):
        if cases is None:
            cases = create_cases(
                test_instances=self.dataset.test_instances,
                num_cases=self.dataset_config.num_test_cases)
            simulators = self.test_simulators
        action_mapping = self.dataset.construct_action_mapping(
            combine=self.model_config.combined_action)
        if len(simulators) > len(cases):
            simulators = random.sample(simulators, len(cases))
        assert self.test_simulators is not None
        with torch.no_grad():
            results = self.trainer.online_test_dmorl(
                cases, device=self.device,
                simulators=simulators, action_mapping=action_mapping,
                stage='test')
        return results
