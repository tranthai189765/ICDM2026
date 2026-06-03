"""
DMORL Configuration classes (R-PADPP variant).
Extends PADPPConfig with R-PADPP-specific hyper-parameters.
"""

from padpp.config import (
    PADPPConfig,
    PADPPConfigForRecommendation,
    PADPPConfigForNegotiation,
    PADPPConfigForEmotionalSupport,
)
from config.constants import (
    rec_special_tokens_dict,
    neg_special_tokens_dict,
    es_special_tokens_dict,
)


class DMORLConfig(PADPPConfig):
    # ── Phase 1: Basic skill curriculum ──────────────────────────────────────
    n_basic_skills = 7                 # 3 corners + 1 uniform + 3 edge midpoints
    n_skill_train_epochs = 15          # RL epochs over centralized basic-skill mix
    phase1_eval_episodes = 3           # Eval dialogues per skill after Phase 1
    skills_file = "dmorl_skills.json"
    run_curriculum = True
    force_rediscover_skills = False
    phase1_only = False                # Stop after Phase 1 (+ Table 2 eval)
    phase2_only = False                # Load Phase 1 checkpoint, run Phase 2 only

    # ── Phase 2: R-PADPP (Regret-Gated GPI) ──────────────────────────────────
    n_rpadpp_epochs = 30               # RL epochs of regret-gated training
    epsilon_threshold = 0.05           # Regret threshold for W_converged admission
    q_old_update_freq = 1              # Update Q_old every K RL epochs
    regret_batch_size = 64             # State batch size for regret estimation
    n_candidate_w = 32                 # Random w's drawn per epoch for regret screen
    use_active_sampling = False        # Rollout under highest-regret w if True
    alpha_rpadpp = 0.5                 # L = (1-α)·L_self + α·L_know

    # Mask bin-redundant duplicate actions (non-price strategies at bin>0 are
    # identical to bin 0). Shrinks the 55-action grid to 19 distinct actions.
    mask_redundant_actions = True

    # Class-balanced SFT loss: weight each action by inverse-sqrt strategy
    # frequency so the model learns minority strategies (esp. agree) instead of
    # collapsing to the dominant counter/inquire/greet classes.
    sft_class_balanced = False

    # Deal-weighted agree exploration: during epsilon exploration, steer the
    # random action to ('agree', 0) with probability = the deal-rate weight
    # (w=[0,0,1] -> 100%, uniform -> 33%, gain/fair -> 0%). Encourages
    # deal-caring skills to actually try agree, which is otherwise rare.
    agree_explore_bias = False

    # Best-checkpoint selection in Phase 1: every eval_every_epochs epochs,
    # estimate greedy SR over quick_eval_episodes dialogues per anchor and save
    # dmorl_phase1_best.pth when it improves. 0 disables (only the last epoch is
    # saved). The last-epoch greedy policy can collapse (e.g. counter-only,
    # never agreeing), so a mid-training epoch is often better.
    eval_every_epochs = 0
    quick_eval_episodes = 2

    # ── Exploration schedule (epsilon-greedy: linear decay then floor) ───────
    # epsilon decays linearly from eps_start to eps_end across the first
    # eps_decay_epochs epochs, then stays at eps_end for the rest.
    eps_start = 1.0                    # exploration rate at epoch 0
    eps_end = 0.05                     # floor exploration rate
    eps_decay_epochs = 15              # epochs over which to decay before the floor

    # ── Legacy DMORL fields kept for compatibility with existing code paths ──
    n_advanced_skills = 0
    n_advanced_train_epochs = 0
    use_dynamic_weight = False
    dynamic_weight_horizon = 3
    use_hints = False
    hints_file = "dmorl_hints.json"

    # Debug mode
    debug = False
    debug_output_dir = "debug_output"

    def __init__(self, params):
        super().__init__(params)
        for k, v in params.items():
            setattr(self, k, v)


class DMORLConfigForRecommendation(DMORLConfig):
    combined_action = False
    special_tokens_dict = rec_special_tokens_dict
    learning_rate = 5e-5
    actor_learning_rate = 5e-4
    obj_to_weight = {
        "uniform": None,
        "user_reward": [1.0, 0.0],
        "item_freq": [0.0, 1.0],
    }


class DMORLConfigForNegotiation(DMORLConfig):
    combined_action = True
    special_tokens_dict = neg_special_tokens_dict
    n_topics = 5
    actor_learning_rate = 2e-4
    # 3 objectives (PADPP paper convention): sl_ratio, fairness, deal_rate
    obj_to_weight = {
        "uniform": None,
        "sl_ratio":  [1.0, 0.0, 0.0],
        "fairness":  [0.0, 1.0, 0.0],
        "deal_rate": [0.0, 0.0, 1.0],
    }


class DMORLConfigForEmotionalSupport(DMORLConfig):
    combined_action = False
    special_tokens_dict = es_special_tokens_dict
    obj_to_weight = {
        "uniform": None,
        "user_reward": [1.0, 0.0, 0.0],
        "toxicity": [0.0, 1.0, 0.0],
        "avg_turn": [0.0, 0.0, 1.0],
    }
