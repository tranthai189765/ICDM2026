# R-PADPP: Regret-Gated Preference-Adaptive Dialogue Policy Planner

Implementation of **R-PADPP**, a regret-aware extension of PADPP for multi-objective
negotiation dialogue policy learning. R-PADPP introduces a **regret-gated GPI
mechanism** that filters out non-converged preference teachers from the knowledge
reuse step, leading to more stable convergence than vanilla PADPP.

Built on top of [PADPP](https://aclanthology.org/2025.emnlp-main.1123/) (EMNLP 2025).
Submitted to ICDM 2026.

---

## 1. Motivation

PADPP trains a single Q-network `Q(s, a, w; θ)` that handles any preference `w`
on the simplex of objectives. At each TD update, it uses **Generalized Policy
Improvement (GPI)**: pick the target action by maximising over a buffer `W` of
previously-sampled preferences.

The problem: `W` is filled with *every* preference ever encountered, including
ones whose Q-values are still far from convergence. Using these noisy teachers
injects unstable gradients (the "moving target" problem).

R-PADPP fixes this with a **regret-gated buffer** `W_converged`: only
preferences whose Q has stabilised (low regret) are admitted as teachers.

---

## 2. Algorithm

### 2.1 Regret tracking

We snapshot a second copy of the network, `Q_old`, every `q_old_update_freq` RL
epochs. The regret of a preference `w` is the average absolute change in Q
across a state batch sampled from the replay buffer:

$$
\mathrm{Reg}(w) \;=\; \mathbb{E}_{s,a} \big[\, \lvert Q_{\text{current}}(s, a, w) - Q_{\text{old}}(s, a, w) \rvert \,\big]
$$

A preference is admitted to `W_converged` iff `Reg(w) < ε`.

### 2.2 GPI knowledge reuse with `W_converged`

Instead of scanning the full preference history, the teacher action is selected
only over the converged set:

$$
\pi_{\text{teacher}}(s') \;=\; \arg\max_{a} \; \max_{w_i \in \mathcal{W}_{\text{converged}}} \; w^{\top} Q(s', a, w_i; \theta_{\text{target}})
$$

where `w` is the **training** preference for this minibatch (not `w_i`).

### 2.3 Loss function

Per minibatch update:

$$
\mathcal{L}(\theta) \;=\; (1 - \alpha)\,\mathcal{L}_{\text{self}}(\theta) \;+\; \alpha\,\mathcal{L}_{\text{know}}(\theta)
$$

**Self loss (DDQN-style scalar TD):**

$$
y^{\text{self}}_w \;=\; w^{\top}\!\left(r + \gamma (1 - d) \, Q_{\text{target}}(s', a^{\star}_{\text{self}}, w) \right), \quad
a^{\star}_{\text{self}} = \arg\max_a w^{\top} Q_{\text{target}}(s', a, w)
$$
$$
\mathcal{L}_{\text{self}}(\theta) \;=\; \mathbb{E}\!\left[\,(y^{\text{self}}_w - w^{\top} Q(s, a, w; \theta))^2\,\right]
$$

**Knowledge-reuse loss (regret-gated GPI vector TD):**

$$
y^{\text{know}}_w \;=\; r + \gamma (1 - d) \, Q_{\text{target}}(s', \pi_{\text{teacher}}(s'), w)
$$
$$
\mathcal{L}_{\text{know}}(\theta) \;=\; \mathbb{E}\!\left[\, \lVert y^{\text{know}}_w - Q(s, a, w; \theta) \rVert_2^2 \,\right]
$$

### 2.4 Active sampling (optional)

Per epoch, sample `n_candidate_w` random Dirichlet preferences. Compute their
regret, then pick the **highest-regret** candidate as the rollout preference
for the next batch of episodes. This forces the agent to collect data where its
own Q is least stable.

---

## 3. Pipeline

Two phases on top of the standard SFT bootstrap:

### Phase 1 — Anchor Curriculum

Pre-train Q at **7 anchor preferences** that span the simplex:

| Type | Weights |
|---|---|
| 3 corners | `[1,0,0]`, `[0,1,0]`, `[0,0,1]` |
| 1 uniform | `[1/3, 1/3, 1/3]` |
| 3 edge midpoints | `[1/2,1/2,0]`, `[0,1/2,1/2]`, `[1/2,0,1/2]` |

Each episode samples one anchor uniformly. The Q-network update uses the
**PI (Policy Improvement) branch** of PADPP — i.e. **pure self-learning, no
GPI envelope, no teacher forcing**. To anchor Q firmly at the 7 basics
(rather than generalising across the whole simplex, which is Phase 2's job),
the inner loss samples its `n_preferences = 128` training preferences **with
replacement from the 7 anchors only** (not from a random Dirichlet over the
simplex). This is implemented as a runtime swap of `random_weights` inside
`train_rl_step` during Phase 1 — `padpp/trainer.py` is unmodified.

$$
\mathcal{L}_{\text{phase1}}(\theta) \;=\; \mathbb{E}\!\left[\,(w^{\top}(r + \gamma(1-d)\, Q_{\text{target}}(s', a^{\star}_{\text{SF}}, w)) - w^{\top} Q(s, a, w; \theta))^2\,\right]
$$

where $a^{\star}_{\text{SF}}$ is chosen by the successor-feature scorer
(`cosine × scalar`) under the *current* preference $w$ — no convex envelope
over past preferences is involved. This matches the PADPP `use_gpi=False`
branch of `train_rl_step`.

After training:

- Save checkpoint → `dmorl_phase1.pth`
- Evaluate against the 4 PADPP-paper Table 2 scenarios (uniform + 3 corners)
- Dump per-anchor evaluation dialogues to `phase1_eval/<anchor>.json`

These 7 anchors **initialise** `W_converged` for Phase 2.

### Phase 2 — R-PADPP (Regret-Gated GPI)

Load `dmorl_phase1.pth`. For each of `n_rpadpp_epochs` RL epochs:

1. Sample `n_candidate_w` random Dirichlet preferences.
2. Choose rollout preference: highest-regret candidate (active sampling) or random.
3. Collect `sampled_times` trajectories under the rollout preference.
4. Run R-PADPP TD update on the buffer (`num_train_q_network_epochs` inner steps).
5. Recompute regret for each candidate; admit to `W_converged` if `< ε`.
6. Every `q_old_update_freq` epochs, snapshot `Q_old ← Q`.

Save final checkpoint → `dmorl_phase2.pth`.

---

## 4. Hyperparameters

Configured in `config/models/DMORL_NEG.yaml`:

| Param | Value | Description |
|---|---|---|
| `n_basic_skills` | 7 | Anchor count for Phase 1 |
| `n_skill_train_epochs` | 15 | RL epochs in Phase 1 |
| `n_rpadpp_epochs` | 30 | RL epochs in Phase 2 |
| `epsilon_threshold` | 0.05 | Regret threshold |
| `q_old_update_freq` | 1 | Q_old snapshot frequency (epochs) |
| `n_candidate_w` | 32 | Candidate preferences per epoch |
| `regret_batch_size` | 64 | Buffer states used for regret estimate |
| `use_active_sampling` | false | Highest-regret rollout if true |
| `alpha_rpadpp` | 0.5 | L = (1-α)·L_self + α·L_know |
| `gamma` | 0.99 | Discount factor |

**Reward-side options** (CLI flags, both `run_dmorl.py` and `eval_dmorl.py`):

| Flag | Effect |
|---|---|
| `--use_llm_price_extraction` | Delegate buyer-price extraction in `compute_reward` to the gen LLM (Buyer-aware, disambiguates years/quantities/seller-ask). Falls back to the regex heuristic on a None/implausible LLM reply. |

Anchor weights live in `dmorl_skills_neg.json`.

---

## 5. Commands

### Full pipeline (SFT → Phase 1 → Phase 2 → online eval)
```bash
python run_dmorl.py \
    --scenario negotiation \
    --datasets craigslist_bargain \
    --models dmorl \
    --gen_models fpt --model_type fpt \
    --metrics sr,deal_rate,sl_ratio,fairness,avg_turn \
    --loggers terminal,file \
    --debug
```

### Phase 1 only (anchor curriculum + Table 2 eval)
```bash
python run_dmorl.py \
    --scenario negotiation \
    --datasets craigslist_bargain \
    --models dmorl \
    --gen_models fpt --model_type fpt \
    --metrics sr,deal_rate,sl_ratio,fairness,avg_turn \
    --loggers terminal,file \
    --phase1_only \
    --debug
```

### Phase 2 only (loads `dmorl_phase1.pth`, runs R-PADPP)
```bash
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
```

### Inspect Table 2 metrics on a checkpoint
```bash
python eval_dmorl.py --checkpoint checkpoints/.../DMORLModel_42/dmorl_phase1.pth
python eval_dmorl.py --checkpoint checkpoints/.../DMORLModel_42/dmorl_phase2.pth
```

---

## 6. Outputs

Per run, the checkpoint directory
(`checkpoints/<scenario>/<dataset>/<Model>_<seed>/`) contains:

| File | Content |
|---|---|
| `dmorl_phase1.pth` | Phase 1 anchor-curriculum weights |
| `dmorl_phase2.pth` | Phase 2 R-PADPP weights |
| `training_losses.csv` | per-update: `global_step, phase, loss, L_self, L_know` |
| `training_rewards.csv` | per-step: `epoch, episode, phase, skill, action, reward vec, weight, weighted_sum` |
| `regret_log.csv` | per epoch: candidate `w`, regret, admitted flag |
| `phase1_eval/<anchor>.json` | Phase 1 per-anchor evaluation dialogues |
| `skill_discovery.txt` | Anchor library audit trail |
