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
import csv
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


class TrainingCSVLogger:
    """
    Writes two CSV files to log_dir:
      - training_losses.csv  : global_step, loss
      - training_rewards.csv : epoch, episode, step, phase, skill, action, done, r0, r1, ...
    Added to trainer.loggers automatically; loss rows come from train_rl_step;
    reward rows are written via log_reward() called inside _run_curriculum_rlt.
    """

    def __init__(self, log_dir: str, n_objectives: int):
        os.makedirs(log_dir, exist_ok=True)

        loss_path = os.path.join(log_dir, "training_losses.csv")
        self._loss_f = open(loss_path, "w", newline="", encoding="utf-8")
        self._loss_w = csv.writer(self._loss_f)
        self._loss_w.writerow(["global_step", "loss"])

        reward_path = os.path.join(log_dir, "training_rewards.csv")
        self._reward_f = open(reward_path, "w", newline="", encoding="utf-8")
        self._reward_w = csv.writer(self._reward_f)
        obj_cols = [f"r{i}" for i in range(n_objectives)]
        w_cols = [f"w{i}" for i in range(n_objectives)]
        self._reward_w.writerow(
            ["epoch", "episode", "step", "phase", "skill", "action", "done"]
            + obj_cols + w_cols + ["weighted_sum"]
        )
        self._n_obj = n_objectives

        loguru_logger.info(f"[DMORL CSV] Writing logs → {loss_path}")
        loguru_logger.info(f"[DMORL CSV] Writing logs → {reward_path}")

    def record(self, results: dict, step: int) -> None:
        loss = results.get("loss", None)
        if loss is not None:
            val = loss.item() if hasattr(loss, "item") else float(loss)
            self._loss_w.writerow([step, f"{val:.8f}"])
            self._loss_f.flush()

    def log_reward(self, epoch: int, episode: int, step: int,
                   phase: str, skill: str, action, rewards, done: int,
                   weight=None) -> None:
        if not isinstance(rewards, list):
            rewards = [rewards]
        r_vec = [float(r) for r in rewards[:self._n_obj]]
        # Pad short reward vectors with zeros so columns line up
        if len(r_vec) < self._n_obj:
            r_vec = r_vec + [0.0] * (self._n_obj - len(r_vec))

        if weight is None:
            w_vec = [float("nan")] * self._n_obj
            weighted_sum = float("nan")
        else:
            w_list = list(weight) if hasattr(weight, "__iter__") else [weight]
            w_vec = [float(x) for x in w_list[:self._n_obj]]
            if len(w_vec) < self._n_obj:
                w_vec = w_vec + [0.0] * (self._n_obj - len(w_vec))
            weighted_sum = float(np.dot(w_vec, r_vec))

        obj_vals = [f"{v:.6f}" for v in r_vec]
        w_vals = [f"{v:.6f}" for v in w_vec]
        ws_val = f"{weighted_sum:.6f}" if not math.isnan(weighted_sum) else "nan"
        self._reward_w.writerow(
            [epoch, episode, step, phase, skill, str(action), done]
            + obj_vals + w_vals + [ws_val]
        )
        self._reward_f.flush()

    def close(self) -> None:
        self._loss_f.close()
        self._reward_f.close()


class DMORLTrainer(PADPPTrainer):

    def __init__(self, game_config, model_config, accelerator, game, model,
                 offline_evaluator, online_evaluator, loggers,
                 generation_method=None, dmorl_controller: DMORLController = None):
        super().__init__(game_config, model_config, accelerator, game, model,
                         offline_evaluator, online_evaluator, loggers,
                         generation_method)
        self.dmorl_controller = dmorl_controller

        if self.loggers is None:
            self.loggers = []

        # Always log rewards + losses to CSV
        csv_log_dir = getattr(model_config, "saved_dir", "checkpoints")
        self.csv_logger = TrainingCSVLogger(csv_log_dir, model_config.n_objectives)
        self.loggers.append(self.csv_logger)

        if getattr(model_config, "debug", False):
            self.loggers.append(DebugLogger())
            _llm_ctrl.enable_debug(True)
            loguru_logger.info("[DMORL] Debug mode ON — LLM prompts, rewards, and losses will be printed.")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1a: Basic Skills Curriculum
    # ─────────────────────────────────────────────────────────────────────────

    def train_basic_skills(self, cases, device=None, simulators=None,
                           action_mapping=None):
        """
        Centralized Phase-1a training: at each episode randomly pick one of
        the basic skills and run the agent with that weight vector. All N
        skills share the same `n_skill_train_epochs` total budget.
        After training, run a few evaluation dialogues per skill and dump
        them to {saved_dir}/phase1a_eval/<skill_name>.json.
        """
        if not self.dmorl_controller:
            return
        basic_skills = self.dmorl_controller.skill_library.basic_skills
        if not basic_skills:
            loguru_logger.warning("[DMORL Phase-1a] No basic skills found. Skipping.")
            return

        basic_weights = np.array([s["weight_vector"] for s in basic_skills])
        skill_names = [s["name"] for s in basic_skills]

        loguru_logger.info(
            f"[DMORL Phase-1a] Centralized training on {len(basic_skills)} basic skills "
            f"({self.model_config.n_skill_train_epochs} epochs total, skill sampled per episode)."
        )

        original_epochs = self.model_config.num_train_rl_epochs
        self.model_config.num_train_rl_epochs = self.model_config.n_skill_train_epochs

        self._run_curriculum_rlt(
            cases, device, simulators, action_mapping,
            skill_weights=basic_weights, skill_names=skill_names, p_skill=1.0,
            phase="1a", skill_name="centralized",
        )

        self.model_config.num_train_rl_epochs = original_epochs
        loguru_logger.info("[DMORL Phase-1a] Centralized basic skill training complete.")

        # Save Phase 1a checkpoint
        saved_dir = getattr(self.model_config, "saved_dir", "checkpoints")
        os.makedirs(saved_dir, exist_ok=True)
        ckpt_path = os.path.join(saved_dir, "dmorl_phase1a.pth")
        self.save_model(ckpt_path)
        loguru_logger.info(f"[DMORL Phase-1a] Checkpoint saved → {ckpt_path}")

        # Per-skill evaluation: run each skill, save dialogues to JSON
        self._eval_basic_skills_per_skill(cases, device, simulators, action_mapping, basic_skills)

    def _eval_basic_skills_per_skill(self, cases, device, simulators,
                                       action_mapping, basic_skills):
        """For each basic skill, run N eval episodes with that fixed weight
        and dump dialogues to {saved_dir}/phase1a_eval/<skill>.json."""
        out_dir = os.path.join(
            getattr(self.model_config, "saved_dir", "checkpoints"),
            "phase1a_eval",
        )
        os.makedirs(out_dir, exist_ok=True)
        n_eval = getattr(self.model_config, "phase1a_eval_episodes", 3)
        max_horizon = getattr(self.game_config, "max_horizon", 10)

        self.model.to(self.device)
        self.model.eval()

        for skill in basic_skills:
            w_fixed = np.array(skill["weight_vector"])
            episodes = []
            for ep_idx in range(n_eval):
                case = np.random.choice(cases)
                simulator = np.random.choice(simulators)
                state = self.game.reset(case, simulator)
                state['w'] = w_fixed

                turns = []
                done = 0
                final_reward = None
                for t in count():
                    pre_dialogue = list(state.get('dialogue_context', []))
                    action, _, _ = self.predict(
                        state, torch.FloatTensor(w_fixed).to(self.device),
                        action_mapping, is_computing_reward=False, use_gpi=False,
                    )
                    state, reward, done, _ = self.game.step(
                        state, action, self.generation_method, simulator
                    )
                    new_dialogue = list(state.get('dialogue_context', []))
                    new_utts = new_dialogue[len(pre_dialogue):]
                    final_reward = reward if isinstance(reward, list) else [float(reward)]
                    turns.append({
                        "step": t,
                        "action": str(action),
                        "utterances": new_utts,
                        "reward": final_reward,
                        "done": int(bool(done)),
                    })
                    if done or t >= max_horizon:
                        break

                episodes.append({
                    "episode": ep_idx,
                    "skill": skill["name"],
                    "weight_vector": [float(x) for x in w_fixed.tolist()],
                    "outcome": "success" if done == 1 else ("failure" if done == -1 else "ongoing"),
                    "n_turns": len(turns),
                    "final_reward": final_reward,
                    "turns": turns,
                })

            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in skill["name"])
            out_path = os.path.join(out_dir, f"{safe}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(episodes, f, indent=2, default=str, ensure_ascii=False)
            loguru_logger.info(
                f"[Phase-1a Eval] {skill['name']}: {n_eval} dialogues → {out_path}"
            )

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

        adv_names = [s["name"] for s in advanced_skills]
        self._run_curriculum_rlt(cases, device, simulators, action_mapping,
                                 skill_weights=adv_weights, skill_names=adv_names,
                                 p_skill=0.6,
                                 phase="1b", skill_name="advanced")

        self.model_config.num_train_rl_epochs = original_epochs
        loguru_logger.info("[DMORL Phase-1b] Advanced skill training complete.")

        # Save Phase 1b checkpoint so evaluation can load the final trained model.
        saved_dir = getattr(self.model_config, "saved_dir", "checkpoints")
        os.makedirs(saved_dir, exist_ok=True)
        ckpt_path = os.path.join(saved_dir, "dmorl_phase1b.pth")
        self.save_model(ckpt_path)
        loguru_logger.info(f"[DMORL Phase-1b] Checkpoint saved → {ckpt_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: Curriculum RLT (shared by Phase 1a and 1b)
    # ─────────────────────────────────────────────────────────────────────────

    def _run_curriculum_rlt(self, cases, device, simulators, action_mapping,
                             fixed_weight=None, skill_weights=None, skill_names=None,
                             p_skill=1.0, phase="1a", skill_name=""):
        """
        Run `num_train_rl_epochs` of GPI TD-learning.
        - fixed_weight: use this w for every episode
        - skill_weights + p_skill: probabilistically sample from skill_weights
        - skill_names: names for each row of skill_weights (used in CSV log)
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

        episode_counter = 0
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
                    current_skill = skill_name
                elif skill_weights is not None and np.random.random() < p_skill:
                    idx = np.random.randint(0, len(skill_weights))
                    w = skill_weights[idx]
                    current_skill = skill_names[idx] if skill_names else f"skill_{idx}"
                else:
                    w = random_weights(self.model_config.n_objectives)[0]
                    current_skill = "random"

                state['w'] = w
                done = False

                for t in count():
                    old_state = copy.deepcopy(state)

                    action, _, _ = self.predict(
                        state, torch.FloatTensor(w).to(self.device),
                        action_mapping, is_computing_reward=False,
                        use_gpi=(phase == "1b")   # GPI only in Phase 1b
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

                    r_list = reward.squeeze().tolist()
                    if not isinstance(r_list, list):
                        r_list = [r_list]

                    # CSV reward logging (always on)
                    self.csv_logger.log_reward(
                        epoch=train_step, episode=episode_counter, step=t,
                        phase=phase, skill=current_skill,
                        action=action, rewards=r_list, done=done_flag,
                        weight=w,
                    )

                    if getattr(self.model_config, "debug", False):
                        r_str = "[" + ", ".join(f"{v:.4f}" for v in r_list) + "]"
                        loguru_logger.debug(
                            f"[DEBUG|Curriculum] epoch={train_step} t={t} "
                            f"action={action} reward={r_str} done={done_flag}"
                        )

                    if done:
                        break

                episode_counter += 1

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

                # Mask redundant actions before selection
                from padpp.trainer import _build_action_mask
                action_mask = _build_action_mask(
                    action_mapping, gpi_logits.size(-1), gpi_logits.device)
                gpi_logits = gpi_logits.masked_fill(~action_mask, float('-inf'))

                if is_test:
                    action_idx = gpi_logits.argmax().item()
                else:
                    eps = getattr(self.model_config, 'epsilon', 0.1)
                    if np.random.random() < eps:
                        # sample uniformly from VALID actions only
                        valid_idx = action_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                        action_idx = int(np.random.choice(valid_idx))
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
