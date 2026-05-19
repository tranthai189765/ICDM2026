"""
DMORL Trainer
Implements 3-phase training on top of PADPPTrainer:

  Phase 1a – Basic Skills Curriculum
    For each LLM-discovered basic skill (weight vector w_k):
      Run n_skill_train_epochs epochs with w fixed to w_k.

  Phase 1b – Advanced Skills
    Run n_advanced_train_epochs RLT epochs where weights are drawn
    preferentially from the advanced skill weight vectors.

  Phase 2 – Full PADPP RLT (random weight sampling)
    Normal PADPP train_rlt to generalise across the whole simplex.

  Phase 3 – Post-Dialogue Refinement (inference-time)
    After each inference dialogue, call DMORLController.refine_after_dialogue()
    to accumulate tactical hints fed back to the DynamicWeightController.
"""

import copy
import datetime
import json
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from itertools import count
from collections import deque, defaultdict
from tqdm import tqdm

from loguru import logger as loguru_logger

from padpp.trainer import PADPPTrainer
import dmorl.llm_controller as _llm_ctrl
from dmorl.llm_controller import DMORLController
from dmorl.model import DMORLModel
from utils.game import random_weights
from config.constants import RECOMMENDATION, NEGOTIATION, EMOTIONAL_SUPPORT, SUCCESS_RATE


class DebugLogger:
    """Appended to trainer.loggers when debug=True; echoes loss/metrics to console."""

    def record(self, results: dict, step: int) -> None:
        items = []
        for k, v in results.items():
            val = v.item() if hasattr(v, "item") else v
            items.append(f"{k}={val:.6g}" if isinstance(val, float) else f"{k}={val}")
        loguru_logger.debug(f"[DEBUG|step={step}] {', '.join(items)}")

    def close(self) -> None:
        pass


class DMORLTrainer(PADPPTrainer):

    def __init__(self, game_config, model_config, accelerator, game, model,
                 offline_evaluator, online_evaluator, loggers,
                 generation_method=None, dmorl_controller: DMORLController = None):
        super().__init__(game_config, model_config, accelerator, game, model,
                         offline_evaluator, online_evaluator, loggers,
                         generation_method)
        self.dmorl_controller = dmorl_controller

        if getattr(model_config, "debug", False):
            if self.loggers is None:
                self.loggers = []
            self.loggers.append(DebugLogger())
            _llm_ctrl.enable_debug(True)
            loguru_logger.info("[DMORL] Debug mode ON — LLM prompts, rewards, and losses will be printed.")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1a: Basic Skills Curriculum
    # ─────────────────────────────────────────────────────────────────────────

    def train_basic_skills(self, cases, device=None, simulators=None,
                           action_mapping=None):
        """
        For each basic skill, run `n_skill_train_epochs` RL epochs with
        the skill's weight vector held fixed.
        """
        if not self.dmorl_controller:
            return
        basic_skills = self.dmorl_controller.skill_library.basic_skills
        if not basic_skills:
            loguru_logger.warning("[DMORL Phase-1a] No basic skills found. Skipping.")
            return

        loguru_logger.info(
            f"[DMORL Phase-1a] Training {len(basic_skills)} basic skills "
            f"x {self.model_config.n_skill_train_epochs} epochs each."
        )

        original_epochs = self.model_config.num_train_rl_epochs
        self.model_config.num_train_rl_epochs = self.model_config.n_skill_train_epochs

        for skill in basic_skills:
            w_fixed = np.array(skill["weight_vector"])
            loguru_logger.info(
                f"[DMORL Phase-1a] Skill: '{skill['name']}' | w={w_fixed.round(3)}"
            )
            self._run_curriculum_rlt(cases, device, simulators, action_mapping,
                                     fixed_weight=w_fixed)

        self.model_config.num_train_rl_epochs = original_epochs
        loguru_logger.info("[DMORL Phase-1a] Basic skill curriculum complete.")

        # Save Phase 1a checkpoint so later phases / reruns can load from it
        saved_dir = getattr(self.model_config, "saved_dir", "checkpoints")
        os.makedirs(saved_dir, exist_ok=True)
        ckpt_path = os.path.join(saved_dir, "dmorl_phase1a.pth")
        self.save_model(ckpt_path)
        loguru_logger.info(f"[DMORL Phase-1a] Checkpoint saved → {ckpt_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1b: Advanced Skills
    # ─────────────────────────────────────────────────────────────────────────

    def train_advanced_skills(self, cases, device=None, simulators=None,
                              action_mapping=None):
        """
        Run `n_advanced_train_epochs` RLT epochs with episode weights sampled
        preferentially from the advanced skill weight vectors (p=0.6).
        """
        if not self.dmorl_controller:
            return
        advanced_skills = self.dmorl_controller.skill_library.advanced_skills
        if not advanced_skills:
            loguru_logger.warning("[DMORL Phase-1b] No advanced skills. Skipping.")
            return

        adv_weights = np.array([s["weight_vector"] for s in advanced_skills])
        loguru_logger.info(
            f"[DMORL Phase-1b] Training with {len(advanced_skills)} advanced skills "
            f"over {self.model_config.n_advanced_train_epochs} epochs."
        )

        original_epochs = self.model_config.num_train_rl_epochs
        self.model_config.num_train_rl_epochs = self.model_config.n_advanced_train_epochs

        self._run_curriculum_rlt(cases, device, simulators, action_mapping,
                                 skill_weights=adv_weights, p_skill=0.6)

        self.model_config.num_train_rl_epochs = original_epochs
        loguru_logger.info("[DMORL Phase-1b] Advanced skill training complete.")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: Curriculum RLT (shared by Phase 1a and 1b)
    # ─────────────────────────────────────────────────────────────────────────

    def _run_curriculum_rlt(self, cases, device, simulators, action_mapping,
                             fixed_weight=None, skill_weights=None, p_skill=1.0):
        """
        Run `num_train_rl_epochs` of GPI TD-learning.
        - fixed_weight: use this w for every episode (Phase 1a)
        - skill_weights + p_skill: probabilistically sample from skill_weights (Phase 1b)
        """
        self.model.to(self.device)

        max_training_steps = self.model_config.num_train_rl_epochs * int(
            self.model_config.buffer_length // self.model_config.train_rl_batch_size
        )
        optimizer = self.create_optimizer(self.model, self.model_config.actor_learning_rate)
        scheduler = self.create_scheduler(optimizer,
                                          num_warmup_steps=self.model_config.warmup_steps,
                                          max_train_steps=max_training_steps)

        # Buffer format matches PADPP: [state_dict, reward_tensor, 1, done_flag]
        buffer = deque(maxlen=self.model_config.buffer_length)
        self.memory_buffer = deque(maxlen=self.model_config.preference_buffer_length)

        best_metric = -math.inf

        for train_step in range(self.model_config.num_train_rl_epochs):
            self.model.train()

            # ── Collect trajectories ──────────────────────────────────────────
            for _ in range(self.model_config.sampled_times):
                case = np.random.choice(cases)
                simulator = np.random.choice(simulators)
                state = self.game.reset(case, simulator)

                # Weight selection
                if fixed_weight is not None:
                    w = fixed_weight
                elif skill_weights is not None and np.random.random() < p_skill:
                    idx = np.random.randint(0, len(skill_weights))
                    w = skill_weights[idx]
                else:
                    w = random_weights(self.model_config.n_objectives)[0]

                state['w'] = w
                done = False

                for t in count():
                    old_state = copy.deepcopy(state)

                    action, _, _ = self.predict(
                        state, torch.FloatTensor(w).to(self.device),
                        action_mapping, is_computing_reward=False,
                        use_gpi=(fixed_weight is None)   # use GPI in Phase 1b
                    )

                    state, reward, done, _ = self.game.step(
                        state, action, self.generation_method, simulator
                    )

                    reward = torch.tensor([reward], device=self.device, dtype=torch.float)
                    old_state['next_state'] = copy.deepcopy(state)
                    old_state['act'] = action

                    # Normalise done flag to match PADPP convention
                    done_flag = 1 if done in (1, -1) else 0
                    if done == -1:
                        done = 1

                    # Buffer element: [state_dict, reward, 1, abs(done)]
                    buffer.append([old_state, reward, 1, abs(done_flag)])

                    if getattr(self.model_config, "debug", False):
                        r_val = reward.item() if hasattr(reward, "item") else float(reward)
                        loguru_logger.debug(
                            f"[DEBUG|Curriculum] epoch={train_step} t={t} "
                            f"action={action} reward={r_val:.4f} done={done_flag}"
                        )

                    if done:
                        break

            # ── RL update ─────────────────────────────────────────────────────
            if train_step >= 0 and len(buffer) >= self.model_config.train_rl_batch_size:
                loguru_logger.warning(
                    f"[Curriculum] Epoch {train_step}, updating Q-network ..."
                )
                self.train_rl_step(buffer, action_mapping, optimizer, scheduler)

        loguru_logger.info("[Curriculum] Phase complete.")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: Inference with Dynamic Weights
    # ─────────────────────────────────────────────────────────────────────────

    def predict_dynamic(self, state, dialogue_history, action_mapping,
                        is_test=True, step_in_episode=0):
        """
        Like predict() but replaces the static w with an LLM-chosen w
        every `dynamic_weight_horizon` turns (Phase 2).
        Also applies GPI over the full skill library when available.
        """
        use_dynamic = (
            self.model_config.use_dynamic_weight
            and self.dmorl_controller is not None
        )
        horizon = getattr(self.model_config, 'dynamic_weight_horizon', 3)

        if use_dynamic and step_in_episode % horizon == 0:
            w_list = self.dmorl_controller.get_dynamic_weight(dialogue_history)
            w = np.array(w_list)
        else:
            w = np.array(state.get('w', [1.0 / self.model_config.n_objectives]
                                   * self.model_config.n_objectives))

        state['w'] = w

        # If model has a skill library, use skill-library GPI
        skill_weights = None
        if isinstance(self.model, DMORLModel):
            skill_weights = self.model.get_skill_weight_tensor(self.device)

        if skill_weights is not None and skill_weights.size(0) > 0:
            action, log_prob = self._predict_with_skill_gpi(
                state, w, skill_weights, action_mapping, is_test
            )
            return action, log_prob, None
        else:
            return self.predict(state, torch.FloatTensor(w).to(self.device),
                                action_mapping, is_test=is_test, use_gpi=True)

    def _predict_with_skill_gpi(self, state, w, skill_weights, action_mapping, is_test):
        """
        GPI over the full skill library: for each action, take the max
        scalarised Q-value across all skill weight vectors.
        """
        if isinstance(action_mapping, tuple):
            inverse_mapping = {v: k for k, v in action_mapping[0].items()}
        else:
            inverse_mapping = {v: k for k, v in action_mapping.items()}

        data_loader = self.construct_dataloaders(
            [state], batch_size=1, goal2id=action_mapping,
            shuffle=False, num_workers=self.model_config.num_workers
        )

        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                w_tensor = torch.FloatTensor(w).to(self.device)
                state_rep, _, w_emb = self.model.compute_state_resp(batch, w_tensor)
                feature = torch.cat([state_rep, w_emb.unsqueeze(0)], dim=-1)

                # GPI over skill library
                gpi_logits = self.model.gpi_action_values(feature, skill_weights)  # [1, n_actions]

                if is_test:
                    action_idx = gpi_logits.argmax().item()
                else:
                    eps = getattr(self.model_config, 'epsilon', 0.1)
                    if np.random.random() < eps:
                        action_idx = np.random.randint(0, gpi_logits.size(-1))
                    else:
                        action_idx = gpi_logits.argmax().item()

                action = inverse_mapping[action_idx]
                return action, None

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: Post-Dialogue Refinement
    # ─────────────────────────────────────────────────────────────────────────

    def post_dialogue_refinement(self, dialogue_history, outcome: str):
        """Call LLM to generate and store tactical hints after a dialogue."""
        if self.dmorl_controller and self.model_config.use_hints:
            hints = self.dmorl_controller.refine_after_dialogue(dialogue_history, outcome)
            loguru_logger.info(f"[DMORL Phase-3] Generated {len(hints)} new hints.")

    # ─────────────────────────────────────────────────────────────────────────
    # Override online_test: dynamic weights + hints + refinement
    # ─────────────────────────────────────────────────────────────────────────

    def online_test_dmorl(self, cases, device=None, simulators=None,
                          action_mapping=None, stage='dev', obj='uniform'):
        """
        Inference loop with:
          - Dynamic weight selection every `horizon` turns (Phase 2)
          - Skill-library GPI for action selection
          - Post-dialogue hint generation (Phase 3)
        """
        loguru_logger.warning("[DMORL] Online testing with dynamic weights + hints ...")

        turn_level_results = defaultdict(list)
        SR = 0.
        self.model.to(device or self.device)
        convs = []
        test_cases = list(zip(cases[:10], simulators))

        for idx, (case, simulator) in tqdm(enumerate(test_cases)):
            loguru_logger.info(f"\n====== Dialogue {idx} ======")

            # Determine base preference weight for this episode
            if stage == 'test':
                if self.model_config.objective_weight is not None:
                    w = np.array(self.model_config.objective_weight)
                elif self.model_config.prioritized_objective == "uniform":
                    w = random_weights(self.model_config.n_objectives, dist="uniform")[0]
                else:
                    w = np.array(
                        self.model_config.obj_to_weight[
                            self.model_config.prioritized_objective.strip()])
            else:
                w = random_weights(self.model_config.n_objectives, dist="uniform")[0]

            state = self.game.reset(case, simulator)
            state['w'] = w
            dialogue_history = list(state.get('dialogue_context', []))
            done = False

            for t in count():
                # Phase 2: dynamic weight selection + skill-GPI action
                action, _, _ = self.predict_dynamic(
                    state, dialogue_history, action_mapping,
                    is_test=True, step_in_episode=t
                )

                state, reward, done, _ = self.game.step(
                    state, action, self.generation_method, simulator
                )

                dialogue_history = list(state.get('dialogue_context', []))

                if done or t >= self.game_config.max_horizon:
                    break

            # Outcome classification for hint generation
            outcome = "success" if done == 1 else ("failure" if done == -1 else "partial_success")

            # Phase 3: post-dialogue refinement
            self.post_dialogue_refinement(dialogue_history, outcome)

            turn_level_results['turn'].append(t + 1)
            turn_level_results['outcome'].append(1 if done == 1 else 0)
            SR += 1 if done == 1 else 0
            convs.append(copy.deepcopy(dialogue_history))

        n = max(len(test_cases), 1)
        results = {
            SUCCESS_RATE: SR / n,
            'avg_turn': np.mean(turn_level_results['turn']),
        }
        loguru_logger.warning(f"[DMORL] Results: {results}")

        for lgr in self.loggers:
            lgr.record(results, self.ppo_global_step)

        # Save evaluation dialogues when debug mode is active
        if getattr(self.model_config, "debug", False) and convs:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(
                getattr(self.model_config, "debug_output_dir", "debug_output"),
                f"eval_dialogues_{ts}",
            )
            os.makedirs(out_dir, exist_ok=True)
            for i, conv in enumerate(convs):
                with open(os.path.join(out_dir, f"dialogue_{i:04d}.json"), "w", encoding="utf-8") as fh:
                    json.dump(conv, fh, ensure_ascii=False, indent=2)
            loguru_logger.info(f"[DMORL DEBUG] Saved {len(convs)} eval dialogues → {out_dir}")

        return results
