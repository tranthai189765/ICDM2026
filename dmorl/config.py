"""
DMORL Configuration classes.
Extends PADPPConfig with DMORL-specific hyper-parameters.
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
    # ── DMORL Phase-1 hyper-parameters ────────────────────────────────────────
    n_basic_skills = 5                  # N basic skills discovered by LLM
    n_advanced_skills = 5              # M advanced skills discovered by LLM
    n_skill_train_epochs = 15          # RL epochs dedicated per basic skill (curriculum)
    n_advanced_train_epochs = 20       # RL epochs for advanced skill phase

    # ── DMORL Phase-2 hyper-parameters ────────────────────────────────────────
    use_dynamic_weight = True          # Replace static w with LLM-selected w at inference
    dynamic_weight_horizon = 3         # T: re-query LLM every T dialogue turns

    # ── DMORL Phase-3 hyper-parameters ────────────────────────────────────────
    use_hints = True                   # Feed accumulated hints to DynamicWeightController
    hints_file = "dmorl_hints.json"    # Where to persist hints
    skills_file = "dmorl_skills.json"  # Where to persist discovered skills

    # Keep full standard PADPP RLT after the curriculum phases
    run_curriculum = True              # If False, skip phases 1a/1b and go straight to PADPP RLT
    force_rediscover_skills = True     # Always re-query LLM for skills each run
    phase1a_only = False               # Stop after Phase 1a (for isolated testing)
    phase1a_eval_episodes = 3          # Eval dialogues per basic skill after Phase 1a
    phase1b_only = False               # Load dmorl_phase1a.pth and only run Phase 1b
    skip_phase2 = False                # Skip Phase 2 (full PADPP RLT generalisation)

    # Debug mode
    debug = False                      # Print LLM prompts/outputs, per-step rewards/losses
    debug_output_dir = "debug_output"  # Directory for debug artefacts (eval dialogues, etc.)

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
    # Lowered from 5e-4 to 2e-4: with sparse intermediate rewards the loss
    # signal is weaker per step, and 5e-4 caused loss to plateau at ~0.07
    # after only 8 epochs (Q overshooting then oscillating).
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
