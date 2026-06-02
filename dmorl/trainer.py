"""
DMORL Trainer — R-PADPP (Regret-Gated PADPP) variant.

Two-phase pipeline on top of PADPPTrainer:

  Phase 1 — Basic Skill Curriculum (Anchor Training)
    Train Q at 7 anchor preferences spanning the simplex:
      * 3 corners      : [1,0,0], [0,1,0], [0,0,1]
      * 1 uniform      : [1/3, 1/3, 1/3]
      * 3 edge midpts  : [1/2,1/2,0], [0,1/2,1/2], [1/2,0,1/2]
    Centralized: each episode samples one anchor and rolls out under it.
    Standard PADPP TD loss (random Dirichlet preferences in the inner update).
    Save dmorl_phase1.pth and evaluate against PADPP Table 2 (4 specific w's).

  Phase 2 — R-PADPP (Regret-Gated GPI Knowledge Reuse)
    Load dmorl_phase1.pth, initialise W_converged with the 7 anchors,
    initialise Q_old as a snapshot of the current model. For each RL epoch:
      1. Sample N candidate preferences w (Dirichlet over the simplex).
      2. (Optional) Active sampling: pick the w with highest current regret
         for rollout; otherwise pick uniformly at random.
      3. Roll out trajectories under rollout_w and add to the replay buffer.
      4. Update Q with the R-PADPP dual loss
            L = (1-α) · L_self + α · L_know
         where
            L_self uses a DDQN-style self target (no GPI)
            L_know uses a GPI target whose envelope is W_converged only.
      5. Re-evaluate regret of each candidate w (and the rollout w); admit
         to W_converged if  regret(w) < epsilon_threshold.
      6. Every q_old_update_freq epochs, snapshot Q_old ← current model.
    Save dmorl_phase2.pth at the end.
"""

import copy
import csv
import json
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from itertools import count
from collections import deque
from loguru import logger as loguru_logger

from padpp.trainer import PADPPTrainer
import dmorl.llm_controller as _llm_ctrl
from dmorl.llm_controller import DMORLController
from utils.game import random_weights


# ─────────────────────────────────────────────────────────────────────────────
# CSV / Debug loggers
# ─────────────────────────────────────────────────────────────────────────────

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
    Writes to log_dir:
      - training_losses.csv  : global_step, phase, loss, L_self, L_know
      - training_rewards.csv : epoch, episode, step, phase, skill, action, done,
                               r0..rK, w0..wK, weighted_sum
      - regret_log.csv       : epoch, w_str, regret, admitted_to_Wconv
    """

    def __init__(self, log_dir: str, n_objectives: int):
        os.makedirs(log_dir, exist_ok=True)

        loss_path = os.path.join(log_dir, "training_losses.csv")
        self._loss_f = open(loss_path, "w", newline="", encoding="utf-8")
        self._loss_w = csv.writer(self._loss_f)
        self._loss_w.writerow(["global_step", "phase", "loss", "L_self", "L_know"])

        reward_path = os.path.join(log_dir, "training_rewards.csv")
        self._reward_f = open(reward_path, "w", newline="", encoding="utf-8")
        self._reward_w = csv.writer(self._reward_f)
        obj_cols = [f"r{i}" for i in range(n_objectives)]
        w_cols = [f"w{i}" for i in range(n_objectives)]
        self._reward_w.writerow(
            ["epoch", "episode", "step", "phase", "skill", "action", "done"]
            + obj_cols + w_cols + ["weighted_sum"]
        )

        regret_path = os.path.join(log_dir, "regret_log.csv")
        self._regret_f = open(regret_path, "w", newline="", encoding="utf-8")
        self._regret_w = csv.writer(self._regret_f)
        self._regret_w.writerow(["epoch", "w", "regret", "admitted"])

        self._n_obj = n_objectives

        loguru_logger.info(f"[DMORL CSV] losses  → {loss_path}")
        loguru_logger.info(f"[DMORL CSV] rewards → {reward_path}")
        loguru_logger.info(f"[DMORL CSV] regret  → {regret_path}")

    def record(self, results: dict, step: int) -> None:
        loss = results.get("loss", None)
        if loss is None:
            return
        val = loss.item() if hasattr(loss, "item") else float(loss)
        l_self = results.get("L_self", float("nan"))
        l_know = results.get("L_know", float("nan"))
        if hasattr(l_self, "item"):
            l_self = l_self.item()
        if hasattr(l_know, "item"):
            l_know = l_know.item()
        phase = results.get("phase", "")
        self._loss_w.writerow([step, phase, f"{val:.8f}",
                               f"{float(l_self):.8f}", f"{float(l_know):.8f}"])
        self._loss_f.flush()

    def log_reward(self, epoch, episode, step, phase, skill,
                    action, rewards, done, weight=None) -> None:
        if not isinstance(rewards, list):
            rewards = [rewards]
        r_vec = [float(r) for r in rewards[:self._n_obj]]
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

    def log_regret(self, epoch, w, regret, admitted: bool):
        w_str = "[" + ",".join(f"{x:.3f}" for x in w) + "]"
        self._regret_w.writerow([epoch, w_str, f"{regret:.6f}", int(admitted)])
        self._regret_f.flush()

    def close(self) -> None:
        self._loss_f.close()
        self._reward_f.close()
        self._regret_f.close()


# ─────────────────────────────────────────────────────────────────────────────
# DMORL Trainer (R-PADPP)
# ─────────────────────────────────────────────────────────────────────────────

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

        csv_log_dir = getattr(model_config, "saved_dir", "checkpoints")
        self.csv_logger = TrainingCSVLogger(csv_log_dir, model_config.n_objectives)
        self.loggers.append(self.csv_logger)

        if getattr(model_config, "debug", False):
            self.loggers.append(DebugLogger())
            _llm_ctrl.enable_debug(True)
            loguru_logger.info("[DMORL] Debug ON — LLM prompts and per-step rewards/losses printed.")

        # R-PADPP state (populated in Phase 2)
        self.q_old_network = None
        self.W_converged = []          # list[ list[float] ]

        # Phase 1 anchor sampling: when set, train_rl_step samples loss
        # preferences from these basic-skill weights with replacement instead
        # of the PADPP default random Dirichlet.
        self._phase1_basic_weights_np = None     # np.ndarray [K, n_obj] or None

    # ─────────────────────────────────────────────────────────────────────────
    # Override: Phase 1 loss preference source = 7 basic anchors (Option B)
    # ─────────────────────────────────────────────────────────────────────────

    def train_rl_step(self, buffer, action_mapping, optimizer, scheduler):
        """
        Thin wrapper around PADPP train_rl_step. If self._phase1_basic_weights_np
        is set (Phase 1 active), the module-level `random_weights` in
        padpp.trainer is swapped for a sampler that draws from the 7 basic
        skill weights with replacement. This makes Q train *only* at the
        anchor preferences during Phase 1, matching the Option-B design:

          sampled_preferences = random.choice(basic_weights, k=n_preferences)

        Outside Phase 1 (Phase 2 or any other caller) we fall through to the
        unmodified PADPP behaviour.
        """
        basic = self._phase1_basic_weights_np
        if basic is None:
            return super().train_rl_step(buffer, action_mapping, optimizer, scheduler)

        import padpp.trainer as _padpp_trainer
        _orig_random_weights = _padpp_trainer.random_weights

        def _basic_sampler(dim, n=1, dist="dirichlet", seed=None, rng=None, p=0.01):
            # Draw `n` preferences from the basic anchors with replacement.
            idx = np.random.randint(0, len(basic), size=n)
            samples = [basic[i].tolist() for i in idx]
            if n == 1:
                return samples[0]
            return samples

        _padpp_trainer.random_weights = _basic_sampler
        try:
            return super().train_rl_step(buffer, action_mapping, optimizer, scheduler)
        finally:
            _padpp_trainer.random_weights = _orig_random_weights

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Basic Skill Curriculum (Anchor Pre-training)
    # ═════════════════════════════════════════════════════════════════════════

    def train_phase1(self, cases, device=None, simulators=None, action_mapping=None):
        """
        Centralized curriculum training at the N anchor preferences.
        Each episode samples one anchor; standard PADPP TD loss runs on the
        collected buffer (random Dirichlet preferences inside train_rl_step).
        Saves dmorl_phase1.pth and emits per-anchor eval dialogues.
        """
        if not self.dmorl_controller:
            raise RuntimeError("[DMORL Phase-1] No DMORL controller — cannot train.")
        basic_skills = self.dmorl_controller.skill_library.basic_skills
        if not basic_skills:
            loguru_logger.warning("[DMORL Phase-1] No basic skills found. Skipping.")
            return

        basic_weights = np.array([s["weight_vector"] for s in basic_skills])
        skill_names = [s["name"] for s in basic_skills]

        loguru_logger.info(
            f"[DMORL Phase-1] Centralized anchor training: {len(basic_skills)} anchors "
            f"over {self.model_config.n_skill_train_epochs} RL epochs."
        )
        loguru_logger.info(f"[DMORL Phase-1] Anchors = {basic_weights.tolist()}")

        original_epochs = self.model_config.num_train_rl_epochs
        self.model_config.num_train_rl_epochs = self.model_config.n_skill_train_epochs

        # Phase 1 is PURE self-learning (DDQN/PI), no GPI envelope in the loss.
        # Force use_gpi=False so train_rl_step takes the PI branch:
        #   loss = MSE(w·Q1, w·(r + γ·Q_next_target))
        # where Q_next_target comes from successor-feature scoring under the
        # *current* preference (no convex envelope over past w's).
        original_use_gpi = self.model_config.use_gpi
        self.model_config.use_gpi = False

        # Option B: loss preferences for Phase 1 are sampled with replacement
        # from the 7 basic anchors (not random Dirichlet). The overridden
        # train_rl_step picks this up via self._phase1_basic_weights_np.
        self._phase1_basic_weights_np = basic_weights.astype(np.float32)

        try:
            self._run_curriculum_rlt(
                cases, device, simulators, action_mapping,
                skill_weights=basic_weights, skill_names=skill_names, p_skill=1.0,
                phase="phase1", skill_name="anchor",
            )
        finally:
            self.model_config.num_train_rl_epochs = original_epochs
            self.model_config.use_gpi = original_use_gpi
            self._phase1_basic_weights_np = None

        loguru_logger.info("[DMORL Phase-1] Anchor curriculum training complete (self-learning only, no GPI).")

        # Save Phase 1 checkpoint
        saved_dir = getattr(self.model_config, "saved_dir", "checkpoints")
        os.makedirs(saved_dir, exist_ok=True)
        ckpt_path = os.path.join(saved_dir, "dmorl_phase1.pth")
        self.save_model(ckpt_path)
        loguru_logger.info(f"[DMORL Phase-1] Checkpoint saved → {ckpt_path}")

        # Per-anchor evaluation dialogues
        self._eval_anchors_per_skill(cases, device, simulators, action_mapping, basic_skills)

    def _eval_anchors_per_skill(self, cases, device, simulators, action_mapping,
                                  basic_skills):
        """Run n_eval episodes per anchor and dump dialogues to JSON."""
        out_dir = os.path.join(
            getattr(self.model_config, "saved_dir", "checkpoints"),
            "phase1_eval",
        )
        os.makedirs(out_dir, exist_ok=True)
        n_eval = getattr(self.model_config, "phase1_eval_episodes", 3)
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
                    # is_test=True → greedy (no epsilon) for the per-anchor eval.
                    action, _, _ = self.predict(
                        state, torch.FloatTensor(w_fixed).to(self.device),
                        action_mapping, is_test=True, is_computing_reward=False, use_gpi=False,
                    )
                    state, reward, done, _ = self.game.step(
                        state, action, self.generation_method, simulator
                    )
                    new_dialogue = list(state.get('dialogue_context', []))
                    new_utts = new_dialogue[len(pre_dialogue):]
                    final_reward = reward if isinstance(reward, list) else [float(reward)]
                    turns.append({
                        "step": t, "action": str(action),
                        "utterances": new_utts, "reward": final_reward,
                        "done": int(bool(done)),
                    })
                    if done or t >= max_horizon:
                        break

                episodes.append({
                    "episode": ep_idx, "skill": skill["name"],
                    "weight_vector": [float(x) for x in w_fixed.tolist()],
                    "outcome": "success" if done == 1 else ("failure" if done == -1 else "ongoing"),
                    "n_turns": len(turns), "final_reward": final_reward, "turns": turns,
                })

            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in skill["name"])
            out_path = os.path.join(out_dir, f"{safe}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(episodes, f, indent=2, default=str, ensure_ascii=False)
            loguru_logger.info(
                f"[Phase-1 Eval] {skill['name']}: {n_eval} dialogues → {out_path}"
            )

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 2 — R-PADPP (Regret-Gated GPI)
    # ═════════════════════════════════════════════════════════════════════════

    def train_phase2(self, cases, device=None, simulators=None, action_mapping=None):
        """
        R-PADPP main loop. Assumes the model has already been loaded from
        dmorl_phase1.pth by the pipeline.
        """
        if not self.dmorl_controller:
            raise RuntimeError("[DMORL Phase-2] No DMORL controller — cannot train.")

        n_obj   = self.model_config.n_objectives
        n_epochs = self.model_config.n_rpadpp_epochs
        epsilon = self.model_config.epsilon_threshold
        q_old_freq = self.model_config.q_old_update_freq
        n_cand  = self.model_config.n_candidate_w
        use_active = self.model_config.use_active_sampling
        alpha_r = self.model_config.alpha_rpadpp

        # Init W_converged with all anchors from Phase 1
        basic_skills = self.dmorl_controller.skill_library.basic_skills
        self.W_converged = [list(map(float, s["weight_vector"])) for s in basic_skills]

        # Init Q_old snapshot
        self.model.to(self.device)
        self.q_old_network = copy.deepcopy(self.model).to(self.device)
        self.q_old_network.eval()

        # Replay buffer and optimizer
        max_training_steps = n_epochs * int(
            self.model_config.buffer_length // self.model_config.train_rl_batch_size
        )
        optimizer = self.create_optimizer(self.model, self.model_config.actor_learning_rate)
        scheduler = self.create_scheduler(
            optimizer, num_warmup_steps=self.model_config.warmup_steps,
            max_train_steps=max_training_steps,
        )
        buffer = deque(maxlen=self.model_config.buffer_length)
        self.target_model = copy.deepcopy(self.model).to(self.device)

        loguru_logger.info(
            f"[DMORL Phase-2] R-PADPP start: n_epochs={n_epochs}, "
            f"|W_converged|_init={len(self.W_converged)}, epsilon={epsilon}, "
            f"alpha={alpha_r}, active_sampling={use_active}"
        )

        episode_counter = 0
        for epoch in range(n_epochs):
            self.model.train()

            # epsilon-greedy: linear decay then floor (see _epsilon_for_epoch)
            self.current_eps = self._epsilon_for_epoch(epoch, n_epochs)
            loguru_logger.info(f"[Phase-2] Epoch {epoch}: epsilon={self.current_eps:.3f}")

            # 1. Sample candidate preferences for this epoch
            candidate_ws = [random_weights(n_obj) for _ in range(n_cand)]

            # 2. Choose rollout preference (active-sampling or random)
            if use_active and len(buffer) >= self.model_config.regret_batch_size:
                regrets = [self._evaluate_regret_for_w(w, buffer) for w in candidate_ws]
                rollout_w = candidate_ws[int(np.argmax(regrets))]
            else:
                rollout_w = candidate_ws[np.random.randint(0, len(candidate_ws))]

            # 3. Collect trajectories under rollout_w
            for _ in range(self.model_config.sampled_times):
                case = np.random.choice(cases)
                simulator = np.random.choice(simulators)
                state = self.game.reset(case, simulator)
                state['w'] = rollout_w
                done = False

                for t in count():
                    old_state = copy.deepcopy(state)
                    action, _, _ = self.predict(
                        state, torch.FloatTensor(rollout_w).to(self.device),
                        action_mapping, is_computing_reward=False,
                        use_gpi=True,
                    )
                    state, reward, done, _ = self.game.step(
                        state, action, self.generation_method, simulator
                    )
                    reward_t = torch.tensor([reward], device=self.device, dtype=torch.float)
                    old_state['next_state'] = copy.deepcopy(state)
                    old_state['act'] = action

                    done_flag = 1 if done in (1, -1) else 0
                    if done == -1:
                        done = 1

                    buffer.append([old_state, reward_t, 1, abs(done_flag)])

                    r_list = reward_t.squeeze().tolist()
                    if not isinstance(r_list, list):
                        r_list = [r_list]
                    self.csv_logger.log_reward(
                        epoch=epoch, episode=episode_counter, step=t,
                        phase="phase2", skill="rpadpp",
                        action=action, rewards=r_list, done=done_flag,
                        weight=rollout_w,
                    )
                    if done:
                        break
                episode_counter += 1

            # 4. R-PADPP train step
            if len(buffer) >= self.model_config.train_rl_batch_size:
                loguru_logger.warning(f"[Phase-2] Epoch {epoch}: R-PADPP Q-net update ...")
                self._train_rl_step_rpadpp(buffer, action_mapping, optimizer, scheduler, alpha_r)

            # 5. Re-evaluate regret + update W_converged
            if len(buffer) >= self.model_config.regret_batch_size:
                for w_cand in candidate_ws + [rollout_w]:
                    reg = self._evaluate_regret_for_w(w_cand, buffer)
                    admitted = (reg < epsilon)
                    if admitted and not self._is_in_W_converged(w_cand):
                        self.W_converged.append(list(map(float, w_cand)))
                    self.csv_logger.log_regret(epoch, w_cand, reg, admitted)
                loguru_logger.info(
                    f"[Phase-2] Epoch {epoch}: |W_converged|={len(self.W_converged)}"
                )

            # 6. Snapshot Q_old periodically
            if (epoch + 1) % q_old_freq == 0:
                self.q_old_network.load_state_dict(self.model.state_dict())
                loguru_logger.info(f"[Phase-2] Epoch {epoch}: Q_old snapshot updated.")

        # clear the exploration schedule so eval starts greedy
        self.current_eps = None

        # Save Phase 2 checkpoint
        saved_dir = getattr(self.model_config, "saved_dir", "checkpoints")
        os.makedirs(saved_dir, exist_ok=True)
        ckpt_path = os.path.join(saved_dir, "dmorl_phase2.pth")
        self.save_model(ckpt_path)
        loguru_logger.info(
            f"[DMORL Phase-2] Complete. Final |W_converged|={len(self.W_converged)}. "
            f"Checkpoint saved → {ckpt_path}"
        )

    def _is_in_W_converged(self, w, tol: float = 1e-3) -> bool:
        for w_old in self.W_converged:
            if all(abs(float(a) - float(b)) < tol for a, b in zip(w, w_old)):
                return True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Regret estimation
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_regret_for_w(self, w, buffer) -> float:
        """
        Reg(w) = E_{s in buffer} [ mean_{a, o} |Q_current(s, a, w) - Q_old(s, a, w)| ]

        Sample a small state batch from the replay buffer and average the
        per-action, per-objective absolute Q-difference.
        """
        n = min(self.model_config.regret_batch_size, len(buffer))
        if n <= 0 or self.q_old_network is None:
            return float("inf")

        sample_idx = np.random.choice(len(buffer), n, replace=False)
        states = [buffer[i][0] for i in sample_idx.tolist()]

        loader = self.construct_dataloaders(
            states, batch_size=n, shuffle=False,
            goal2id=None, num_workers=0,
        )

        w_tensor = torch.FloatTensor([list(w)]).to(self.device)  # [1, n_obj]

        total_abs = 0.0
        total_count = 0
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                state_resp, _, w_emb = self.model.compute_state_resp(batch, w_tensor)
                bs = state_resp.size(0)
                # broadcast w_emb [1, w_dim] across bs
                w_emb_b = w_emb.repeat(1, bs).view(-1, w_emb.size(-1))
                feat = torch.cat([state_resp, w_emb_b], dim=-1)

                q_now = self.model.actor(feat).detach()
                q_old = self.q_old_network.actor(feat).detach()
                # both [bs, action_size * n_obj]
                diff = (q_now - q_old).abs()
                total_abs += diff.sum().item()
                total_count += diff.numel()
        self.model.train()
        return total_abs / max(total_count, 1)

    # ─────────────────────────────────────────────────────────────────────────
    # R-PADPP TD step
    # ─────────────────────────────────────────────────────────────────────────

    def _train_rl_step_rpadpp(self, buffer, action_mapping, optimizer, scheduler,
                                alpha_r):
        """
        One inner Q-network update with the R-PADPP dual loss:
            L = (1 - alpha_r) * L_self  +  alpha_r * L_know

          L_self : DDQN-style scalar TD MSE at the sampled preference w
          L_know : vector MSE between Q1 and y_know, where y_know uses the GPI
                    action chosen over W_converged as the next-step action.
        """
        n_obj = self.model_config.n_objectives
        gamma = self.model_config.gamma

        progress = tqdm = range  # avoid extra import; not strictly needed

        mean_total, mean_self, mean_know = [], [], []

        for i in range(self.model_config.num_train_q_network_epochs):
            indices = np.random.choice(
                len(buffer), self.model_config.train_rl_batch_size)
            batch_instances = [buffer[i] for i in indices.tolist()]
            states = [x[0] for x in batch_instances]

            if isinstance(action_mapping, tuple):
                batch_act = [action_mapping[0][s['act']] for s in states]
            else:
                batch_act = [action_mapping[s['act']] for s in states]
            batch_act = torch.LongTensor(batch_act).to(self.device)

            rewards = torch.cat([x[1] for x in batch_instances], dim=0)
            batch_done = torch.Tensor([x[3] for x in batch_instances]).to(self.device)

            train_loader = self.construct_dataloaders(
                states, batch_size=self.model_config.train_rl_batch_size,
                shuffle=False, goal2id=action_mapping,
                num_workers=self.model_config.num_workers,
            )

            for batch in train_loader:
                # Sample K random training preferences (PADPP convention)
                K = self.model_config.n_preferences
                w_sampled_np = random_weights(n_obj, n=K)
                w_sampled = torch.Tensor(w_sampled_np).to(self.device).requires_grad_(False)

                # Encode states once
                state, next_state, w_emb = self.model.compute_state_resp(batch, w_sampled)
                bs = state.size(0)

                # Repeat to [K*bs, ·]
                w_emb_rep    = w_emb.repeat(1, bs).view(-1, w_emb.size(-1))
                state_rep    = state.repeat(K, 1).view(-1, state.size(-1))
                next_st_rep  = next_state.repeat(K, 1).view(-1, next_state.size(-1))
                action_rep   = batch_act.repeat(K, 1).view(-1)
                w_rep        = w_sampled.repeat(1, bs).view(-1, n_obj)        # [K*bs, n_obj]
                rew_rep      = rewards.repeat(K, 1).view(-1, n_obj)            # [K*bs, n_obj]
                done_rep     = batch_done.repeat(K, 1).view(-1, 1)              # [K*bs, 1]

                # === Q1 = Q(s, a, w_sampled) ===
                feature = torch.cat([state_rep, w_emb_rep], dim=-1)
                Q_all = self.model.actor(feature)
                action_size = Q_all.view(Q_all.size(0), -1, n_obj).size(1)
                Q_all = Q_all.view(Q_all.size(0), action_size, n_obj)
                Q1 = Q_all.gather(
                    1, action_rep.view(-1, 1, 1).expand(Q_all.size(0), 1, n_obj)
                ).squeeze(1)                                                    # [K*bs, n_obj]

                # Action mask (drop bin-redundant duplicates from target argmax)
                _use_mask = getattr(self.model_config, 'mask_redundant_actions', True)
                if _use_mask:
                    from padpp.trainer import _build_action_mask
                    _amask = _build_action_mask(action_mapping, action_size, self.device)  # [action_size]

                # === L_self (DDQN scalar TD) ===
                with torch.no_grad():
                    self.model.eval()
                    next_feat = torch.cat([next_st_rep, w_emb_rep], dim=-1)
                    Q_next_T = self.target_model.actor(next_feat).detach()
                    Q_next_T = Q_next_T.view(-1, action_size, n_obj)            # [K*bs, n_a, n_obj]

                    # argmax_a (w · Q_target(s', a, w))
                    w_rep_3d = w_rep.unsqueeze(1).expand(-1, action_size, n_obj)
                    scalar_self = (w_rep_3d * Q_next_T).sum(dim=-1)             # [K*bs, n_a]
                    if _use_mask:
                        scalar_self = scalar_self.masked_fill(~_amask.unsqueeze(0), float('-inf'))
                    a_star = scalar_self.max(dim=1)[1]                          # [K*bs]
                    Q_next_self = Q_next_T.gather(
                        1, a_star.view(-1, 1, 1).expand(-1, 1, n_obj)
                    ).squeeze(1)                                                # [K*bs, n_obj]
                self.model.train()

                y_self_vec = rew_rep + gamma * (1 - done_rep) * Q_next_self     # [K*bs, n_obj]
                wQ1_scalar = (w_rep * Q1).sum(dim=-1)                            # [K*bs]
                wY_self    = (w_rep * y_self_vec).sum(dim=-1)                    # [K*bs]
                L_self = F.mse_loss(wQ1_scalar, wY_self, reduction='mean')

                # === L_know (GPI vector TD, envelope = W_converged) ===
                with torch.no_grad():
                    self.model.eval()
                    W_conv_np = np.array(self.W_converged, dtype=np.float32)
                    W_conv_t = torch.from_numpy(W_conv_np).to(self.device)      # [K_c, n_obj]
                    K_c = W_conv_t.size(0)

                    _, _, w_emb_conv = self.model.compute_state_resp(batch, W_conv_t)
                    # w_emb_conv: [K_c, w_dim]
                    w_emb_conv_rep = w_emb_conv.repeat(1, bs).view(-1, w_emb_conv.size(-1))
                    next_st_conv = next_state.repeat(K_c, 1).view(-1, next_state.size(-1))
                    feat_conv = torch.cat([next_st_conv, w_emb_conv_rep], dim=-1)

                    Q_next_conv = self.target_model.actor(feat_conv).detach()
                    Q_next_conv = Q_next_conv.view(K_c, bs, action_size, n_obj)

                    # score[k_sampled, b, a] = max_{k_c} ( w_sampled[k_sampled] · Q_next_conv[k_c, b, a, :] )
                    # einsum: tmp[k_c, k, b, a] = sum_o w[k,o] * Q[k_c,b,a,o]
                    tmp = torch.einsum('ko,cbao->ckba', w_sampled, Q_next_conv)
                    score, _ = tmp.max(dim=0)                                   # [K, bs, action_size]
                    if _use_mask:
                        score = score.masked_fill(
                            ~_amask.view(1, 1, action_size), float('-inf'))
                    a_teacher = score.max(dim=-1)[1]                            # [K, bs]

                    # Evaluate Q_next at sampled w (not w_i) at a_teacher
                    Q_next_T_kb = Q_next_T.view(K, bs, action_size, n_obj)
                    Q_at_teacher = Q_next_T_kb.gather(
                        2, a_teacher.unsqueeze(-1).unsqueeze(-1).expand(K, bs, 1, n_obj)
                    ).squeeze(2).view(-1, n_obj)                                # [K*bs, n_obj]
                self.model.train()

                y_know = rew_rep + gamma * (1 - done_rep) * Q_at_teacher        # [K*bs, n_obj]
                L_know = F.mse_loss(Q1, y_know, reduction='mean')

                # === Total ===
                loss = (1.0 - alpha_r) * L_self + alpha_r * L_know

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Update target network every inner step
                self.target_model.load_state_dict(self.model.state_dict())

                mean_total.append(loss.detach())
                mean_self.append(L_self.detach())
                mean_know.append(L_know.detach())

        m_total = torch.stack(mean_total).mean()
        m_self  = torch.stack(mean_self).mean()
        m_know  = torch.stack(mean_know).mean()
        results = {"loss": m_total, "L_self": m_self, "L_know": m_know, "phase": "phase2"}
        loguru_logger.info(
            f"[R-PADPP] step={self.ppo_global_step} "
            f"loss={m_total.item():.6f} L_self={m_self.item():.6f} L_know={m_know.item():.6f}"
        )
        for lg in self.loggers:
            lg.record(results, self.ppo_global_step)
        self.ppo_global_step += 1

    # ═════════════════════════════════════════════════════════════════════════
    # Exploration schedule: linear decay then floor
    # ═════════════════════════════════════════════════════════════════════════

    def _epsilon_for_epoch(self, epoch, total_epochs):
        """
        epsilon = eps_start → eps_end linearly over the first eps_decay_epochs
        epochs, then held at eps_end for the remaining epochs.

        e.g. eps_start=1.0, eps_end=0.05, eps_decay_epochs=15, total=25:
          epoch 0  → 1.00
          epoch 15 → 0.05
          epoch 16..24 → 0.05
        """
        eps_start = getattr(self.model_config, 'eps_start', 1.0)
        eps_end = getattr(self.model_config, 'eps_end', 0.05)
        decay = getattr(self.model_config, 'eps_decay_epochs', None)
        if decay is None or decay <= 0:
            decay = max(total_epochs - 1, 1)
        if epoch >= decay:
            return eps_end
        return eps_start + (eps_end - eps_start) * (epoch / decay)

    # ═════════════════════════════════════════════════════════════════════════
    # Internal: Curriculum RLT (Phase 1)
    # ═════════════════════════════════════════════════════════════════════════

    def _run_curriculum_rlt(self, cases, device, simulators, action_mapping,
                             fixed_weight=None, skill_weights=None, skill_names=None,
                             p_skill=1.0, phase="phase1", skill_name=""):
        """
        Phase 1 inner loop. Uses PADPP-original train_rl_step (random Dirichlet
        preferences in the inner update). No GPI hooks, no envelopes — vanilla.
        """
        self.model.to(self.device)

        max_training_steps = self.model_config.num_train_rl_epochs * int(
            self.model_config.buffer_length // self.model_config.train_rl_batch_size
        )
        optimizer = self.create_optimizer(self.model, self.model_config.actor_learning_rate)
        scheduler = self.create_scheduler(
            optimizer, num_warmup_steps=self.model_config.warmup_steps,
            max_train_steps=max_training_steps,
        )

        buffer = deque(maxlen=self.model_config.buffer_length)
        self.memory_buffer = deque(maxlen=self.model_config.preference_buffer_length)

        # Epsilon-greedy exploration: linear decay then floor (see
        # _epsilon_for_epoch). Strong early exploration helps the policy escape
        # degenerate local optima (e.g. deny/agree spam) before exploiting.
        total_eps_epochs = self.model_config.num_train_rl_epochs

        episode_counter = 0
        for train_step in range(self.model_config.num_train_rl_epochs):
            self.model.train()

            self.current_eps = self._epsilon_for_epoch(train_step, total_eps_epochs)
            loguru_logger.info(
                f"[Curriculum] Epoch {train_step}: epsilon={self.current_eps:.3f}"
            )

            for _ in range(self.model_config.sampled_times):
                case = np.random.choice(cases)
                simulator = np.random.choice(simulators)
                state = self.game.reset(case, simulator)

                if fixed_weight is not None:
                    w = fixed_weight
                    current_skill = skill_name
                elif skill_weights is not None and np.random.random() < p_skill:
                    idx = np.random.randint(0, len(skill_weights))
                    w = skill_weights[idx]
                    current_skill = skill_names[idx] if skill_names else f"skill_{idx}"
                else:
                    w = random_weights(self.model_config.n_objectives)
                    current_skill = "random"

                state['w'] = w
                done = False
                for t in count():
                    old_state = copy.deepcopy(state)
                    action, _, _ = self.predict(
                        state, torch.FloatTensor(w).to(self.device),
                        action_mapping, is_computing_reward=False, use_gpi=False,
                    )
                    state, reward, done, _ = self.game.step(
                        state, action, self.generation_method, simulator
                    )
                    reward_t = torch.tensor([reward], device=self.device, dtype=torch.float)
                    old_state['next_state'] = copy.deepcopy(state)
                    old_state['act'] = action

                    done_flag = 1 if done in (1, -1) else 0
                    if done == -1:
                        done = 1
                    buffer.append([old_state, reward_t, 1, abs(done_flag)])

                    r_list = reward_t.squeeze().tolist()
                    if not isinstance(r_list, list):
                        r_list = [r_list]
                    self.csv_logger.log_reward(
                        epoch=train_step, episode=episode_counter, step=t,
                        phase=phase, skill=current_skill,
                        action=action, rewards=r_list, done=done_flag, weight=w,
                    )
                    if done:
                        break
                episode_counter += 1

            if train_step >= 0 and len(buffer) >= self.model_config.train_rl_batch_size:
                loguru_logger.warning(
                    f"[Curriculum] Epoch {train_step}, vanilla PADPP Q-net update ..."
                )
                self.train_rl_step(buffer, action_mapping, optimizer, scheduler)

        # clear the exploration schedule so eval / later phases start fresh
        self.current_eps = None
        loguru_logger.info("[Curriculum] Phase 1 complete.")
