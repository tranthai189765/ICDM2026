# DMORL: Dynamic Multi-Objective Reinforcement Learning for Dialogue Policy Planning

> **Built on top of [PADPP](https://aclanthology.org/2025.emnlp-main.1123/) (EMNLP 2025)**
> Submitted to **ICDM 2026**

DMORL extends PADPP to address two core limitations of existing multi-objective dialogue policy methods:

| Limitation | Our Solution |
|---|---|
| Fixed objective weights throughout a dialogue | **Dynamic Weighting** — LLM re-selects the optimal weight vector every T turns based on conversation context |
| Random / unstable warm-up during training | **Skill-based Curriculum** — LLM discovers semantically meaningful skills and trains them one at a time before full RL |

---

## Table of Contents

1. [Motivation & Key Ideas](#1-motivation--key-ideas)
2. [System Architecture](#2-system-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Installation](#4-installation)
5. [Configuration](#5-configuration)
6. [Training Pipeline](#6-training-pipeline)
7. [Inference](#7-inference)
8. [Parameters Reference](#8-parameters-reference)
9. [Running Experiments](#9-running-experiments)
10. [PADPP Baseline](#10-padpp-baseline)
11. [Citation](#11-citation)

---

## 1. Motivation & Key Ideas

### Problems with Existing Multi-Objective Methods

**Learning Instability:** In sequential multi-objective RL (e.g. PADPP, MODPL), the agent must balance several objectives (success rate, fairness, user satisfaction, …) simultaneously from the start. Because early policies are far from convergence, using them to bootstrap later objectives accumulates error.

**Fixed Weights:** Traditional methods fix the objective weight vector `w` before a dialogue begins. In practice, user intent *drifts* — a customer initially probing product features may switch to price negotiation. A static `w` cannot adapt to this shift.

### Our Approach — Three Phases

```
Phase 1a  ──  Basic Skills Curriculum
              LLM proposes N "basic skills" (e.g. "Active Listening", "Price Push"),
              each mapped to a weight vector w_k on the objective simplex.
              Agent trains for n_skill_train_epochs epochs per skill with w fixed.
              Result: a stable, convergent policy foundation for each semantic strategy.

Phase 1b  ──  Advanced Skills Training
              LLM proposes M more nuanced "advanced skills".
              Agent trains with 60% probability of sampling advanced skill weights,
              40% random. GPI provides implicit teacher regularisation.
              Result: the policy gains breadth across more diverse strategy profiles.

Phase 2   ──  Full PADPP RLT
              Standard PADPP Reinforcement Learning Tuning with random weight sampling.
              Result: full generalisation across the entire objective simplex.

Phase 3   ──  Post-Dialogue Refinement  (inference-time, continuous)
              After each dialogue, LLM reads the history + outcome and generates
              tactical "Hints" (e.g. "If user hesitates on price, shift towards
              Friendliness for the next 3 turns").
              Hints accumulate across dialogues; the Dynamic Weight Controller
              incorporates them when selecting w at inference time.
```

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DMORL Full Pipeline                         │
│                                                                     │
│  ┌──────────────┐     ┌─────────────────────────────────────────┐  │
│  │  LLM (Qwen3) │     │             DMORLTrainer                 │  │
│  │              │────▶│  Phase 1a: Basic Skill Curriculum        │  │
│  │  Skill       │     │  Phase 1b: Advanced Skill Training       │  │
│  │  Discovery   │     │  Phase 2 : PADPP RLT (generalisation)    │  │
│  └──────────────┘     └─────────────────────────────────────────┘  │
│                                           │                         │
│  ┌──────────────┐     ┌─────────────────────────────────────────┐  │
│  │  LLM (Qwen3) │     │           Inference (online_test_dmorl) │  │
│  │              │◀────│                                         │  │
│  │  Dynamic     │     │  Every T turns:                         │  │
│  │  Weighting   │────▶│    w = LLM(history, hints) → GPI(w, Q) │  │
│  │              │     │                                         │  │
│  │  Hint        │◀────│  After dialogue:                        │  │
│  │  Generation  │     │    hints += LLM(history, outcome)       │  │
│  └──────────────┘     └─────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | Description |
|---|---|---|
| `DMORLController` | `dmorl/llm_controller.py` | Master controller wrapping all LLM interactions |
| `SkillLibrary` | `dmorl/llm_controller.py` | Discovers + stores basic/advanced skills with weight vectors |
| `DynamicWeightController` | `dmorl/llm_controller.py` | Per-turn intent detection → optimal `w` |
| `HintManager` | `dmorl/llm_controller.py` | Generates + persists tactical hints after each dialogue |
| `DMORLModel` | `dmorl/model.py` | Extends PADPPModel; adds `gpi_action_values()` over the full skill library |
| `DMORLTrainer` | `dmorl/trainer.py` | 3-phase training + dynamic inference + post-dialogue refinement |
| `DMORLPipeline*` | `dmorl/pipeline.py` | Scenario-specific orchestration (REC / NEG / ES) |

---

## 3. Repository Structure

```
ICDM2026/
├── .env                          ← API keys (gitignored)
├── .gitignore
├── README.md                     ← This file
├── run.py                        ← Original PADPP entry point
├── run_dmorl.py                  ← DMORL entry point
│
├── dmorl/                        ← DMORL module (new)
│   ├── __init__.py
│   ├── llm_controller.py         ← LLM interface: skill discovery, dynamic weight, hints
│   ├── config.py                 ← DMORLConfig + 3 scenario configs
│   ├── model.py                  ← DMORLModel (PADPPModel + skill-library GPI)
│   ├── trainer.py                ← DMORLTrainer (3-phase training loop)
│   ├── pipeline.py               ← DMORLPipeline for REC, NEG, ES
│   └── README.md                 ← Module-level doc
│
├── padpp/                        ← PADPP baseline (unchanged)
│   ├── config.py
│   ├── data_processor.py
│   ├── model.py
│   ├── trainer.py
│   └── pipeline.py
│
├── base/                         ← Abstract base classes
│   ├── game.py
│   ├── model.py
│   ├── simulator.py
│   ├── trainer.py
│   └── pipeline.py
│
├── config/
│   ├── config.py                 ← Base config classes + game configs
│   ├── constants.py              ← All string constants, prompts, path constants
│   └── models/
│       ├── PADPP_NEG.yaml        ← PADPP negotiation config
│       ├── PADPP_REC.yaml        ← PADPP recommendation config
│       ├── PADPP_ES.yaml         ← PADPP emotional support config
│       ├── DMORL_NEG.yaml        ← DMORL negotiation config (new)
│       ├── DMORL_REC.yaml        ← DMORL recommendation config (new)
│       └── DMORL_ES.yaml         ← DMORL emotional support config (new)
│
├── dataset/
│   ├── rec_datasets/             ← DuRecDial, Inspired
│   └── neg_datasets/             ← CraigslistBargain
│
├── simulator/
│   ├── rec_simulator.py          ← LLM-based recommendation user simulator
│   ├── neg_simulator.py          ← Negotiation user simulator
│   └── es_simulator.py           ← Emotional support user simulator
│
├── text_gen/
│   ├── chatgpt_generation.py     ← ChatGPT response generation
│   ├── llama3_generation.py
│   └── bart_generation.py
│
├── eval/
│   ├── offline.py
│   └── online.py
│
└── utils/
    ├── utils.py                  ← Model/dataset registry, arg parsing
    ├── game.py                   ← Weight sampling, game utilities
    └── prompt.py                 ← LLM call wrappers
```

---

## 4. Installation

```bash
git clone https://github.com/tranthai189765/ICDM2026.git
cd ICDM2026
pip install -r requirements.txt
```

Copy and fill in the environment file:

```bash
cp .env.example .env   # then edit .env with your keys
```

Or set variables manually:

```bash
export DEEPINFRA_API_KEY=your_key_here
export DEEPINFRA_MODEL=Qwen/Qwen3-32B
export DEEPINFRA_BASE_URL=https://api.deepinfra.com/v1/openai
```

Download datasets from the [PADPP data link](https://drive.google.com/drive/folders/1geGSLEuyW2YrCbLMLdyOqaE7n5KRZN4z?usp=drive_link) and place the `data/` directory in the project root.

```bash
mkdir logs checkpoints
```

---

## 5. Configuration

### YAML Config Files

Each model + scenario pair has a YAML file under `config/models/`. Example: `config/models/DMORL_NEG.yaml`.

**PADPP base parameters** (inherited by DMORL):

| Parameter | Default | Description |
|---|---|---|
| `tokenizer` | `roberta-large` | HuggingFace tokenizer name |
| `plm` | `roberta-large` | Pretrained language model backbone |
| `lm_size` | `1024` | Hidden size of the PLM |
| `objective_embedding_size` | `6` | Dimension of the preference embedding layer |
| `mlp_hidden_size` | `128` | Actor/projector hidden size |
| `n_objectives` | set by game config | Number of objectives (2 for REC, 3 for NEG/ES) |
| `learning_rate` | `5e-5` | SFT learning rate |
| `actor_learning_rate` | `5e-4` | RL actor learning rate |
| `num_train_epochs` | varies | Number of SFT epochs |
| `num_train_rl_epochs` | `10` | Number of RL fine-tuning epochs |
| `train_rl_batch_size` | `128` | Batch size for RL updates |
| `buffer_length` | `2000` | Replay buffer capacity |
| `n_preferences` | `128` | Number of preference samples per RL step |
| `alpha` | `0.7` | GPI loss blend: `α·scalar_loss + (1-α)·vector_loss` |
| `gamma` | `0.99` | Discount factor |
| `use_gpi` | `true` | Enable Generalised Policy Improvement |
| `run_sft` | `true` | Run supervised fine-tuning phase |
| `run_rlt` | `true` | Run RL fine-tuning phase |
| `run_online_eval` | `true` | Run online evaluation |
| `run_offline_eval` | `true` | Run offline evaluation |
| `freeze_plm` | `true` | Freeze PLM parameters during RL |
| `saved_dir` | `checkpoints/` | Directory to save checkpoints |

**DMORL-specific parameters** (added by DMORLConfig):

| Parameter | Default | Description |
|---|---|---|
| `n_basic_skills` | `5` | N — number of basic skills the LLM discovers |
| `n_advanced_skills` | `5` | M — number of advanced skills |
| `n_skill_train_epochs` | `10` | RL epochs dedicated to each basic skill in Phase 1a |
| `n_advanced_train_epochs` | `10` | Total RL epochs for Phase 1b |
| `use_dynamic_weight` | `true` | Enable LLM-based dynamic weighting at inference |
| `dynamic_weight_horizon` | `3` | T — re-query LLM every T dialogue turns |
| `use_hints` | `true` | Include accumulated hints in the LLM weight query |
| `run_curriculum` | `true` | Run Phase 1a + 1b before full PADPP RLT |
| `force_rediscover_skills` | `false` | Re-run LLM skill discovery even if file exists |
| `skills_file` | `dmorl_skills_neg.json` | Path to persist discovered skills |
| `hints_file` | `dmorl_hints_neg.json` | Path to persist accumulated hints |

### Objective Definitions

| Scenario | Objectives | Metrics |
|---|---|---|
| **Recommendation** | user_reward, item_freq | SR, Avg_Turn |
| **Negotiation** | sl_ratio, fairness, deal_rate | SR, Deal_Rate, SL_Ratio, Fairness, Avg_Turn |
| **Emotional Support** | user_reward, toxicity, avg_turn | SR, User_Reward, Toxicity, Avg_Turn |

---

## 6. Training Pipeline

### Full DMORL Pipeline

```
Input Data
    │
    ▼
[SFT] Supervised Fine-Tuning on demonstration data
    │  Loss: cross-entropy on goal prediction
    │  Output: model.pth checkpoint
    │
    ▼
[Phase 1a] Basic Skill Curriculum
    │  For each of N basic skills (LLM-discovered):
    │    ├─ Fixed weight w_k for n_skill_train_epochs epochs
    │    └─ GPI disabled (pure w_k optimisation)
    │
    ▼
[Phase 1b] Advanced Skill Training
    │  For n_advanced_train_epochs epochs:
    │    ├─ 60% prob: sample w from advanced skill library
    │    ├─ 40% prob: sample w randomly from simplex
    │    └─ GPI enabled (teacher policy signal)
    │
    ▼
[Phase 2] Full PADPP RLT
    │  For num_train_rl_epochs epochs:
    │    ├─ Sample w randomly from simplex
    │    ├─ GPI TD-learning with preference memory buffer
    │    └─ Evaluate on dev set every 10 epochs; save best checkpoint
    │
    ▼
[Online Eval] Dynamic Inference + Hint Accumulation
    │  For each dialogue:
    │    ├─ Every T turns: w = LLM(history, hints)
    │    ├─ Action = argmax GPI_Q(s, a, w, skill_library)
    │    └─ After dialogue: hints += LLM(history, outcome)
    │
    ▼
Results (SR, Avg_Turn, Objective metrics)
```

### Skill Discovery Process

```
LLM Prompt (Skill Discovery)
─────────────────────────────
Scenario: negotiation
Objectives: [sl_ratio, fairness, deal_rate]

Propose 5 BASIC dialogue skills with weight vectors.

LLM Response (example):
[
  {"name": "Firm Seller",      "description": "Maximise price gain", "weight_vector": [0.85, 0.05, 0.10]},
  {"name": "Fair Dealer",      "description": "Balance gain+fairness", "weight_vector": [0.33, 0.43, 0.24]},
  {"name": "Deal Closer",      "description": "Prioritise reaching deal", "weight_vector": [0.10, 0.15, 0.75]},
  {"name": "Rapport Builder",  "description": "Build trust first",   "weight_vector": [0.20, 0.55, 0.25]},
  {"name": "Flexible Adaptor", "description": "Stay balanced",       "weight_vector": [0.34, 0.33, 0.33]}
]
```

### Dynamic Weight Selection (Inference)

```
Every T turns during a dialogue:

LLM Input:
  - Recent conversation history (last 10 turns)
  - Accumulated hints from past dialogues
  - Current objective names

LLM Output:
  {"weight_vector": [0.15, 0.20, 0.65]}

Interpretation: at this turn, prioritise deal_rate (0.65)
→ Agent selects action = argmax_a max_{k} w_k · Q(s, a, w_k)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                         GPI over skill library
```

---

## 7. Inference

### Test-only Mode

```bash
python run_dmorl.py \
    --scenario negotiation \
    --datasets craigslist_bargain \
    --models dmorl \
    --gen_models chatgpt \
    --metrics sr,deal_rate,sl_ratio,fairness,avg_turn \
    --loggers terminal \
    --test_phase \
    --use_dynamic_weight
```

### Objective-specific Evaluation

```bash
# Evaluate maximising deal_rate only
python run_dmorl.py \
    --scenario negotiation \
    --datasets craigslist_bargain \
    --models dmorl \
    --gen_models chatgpt \
    --metrics sr,deal_rate,sl_ratio,fairness \
    --loggers terminal \
    --test_phase \
    --prioritized_objective deal_rate
```

### Custom Weight Evaluation

```bash
# Test with fixed weight [0.6, 0.2, 0.2] (sl_ratio biased)
python run_dmorl.py \
    --scenario negotiation \
    --datasets craigslist_bargain \
    --models dmorl \
    --gen_models chatgpt \
    --metrics sr,deal_rate,sl_ratio \
    --loggers terminal \
    --test_phase \
    --objective_weight 0.6,0.2,0.2
```

---

## 8. Parameters Reference

### Command-line Arguments

#### Shared with PADPP (`run.py` / `run_dmorl.py`)

| Argument | Default | Description |
|---|---|---|
| `--scenario` | `recommendation` | `recommendation`, `negotiation`, or `emotional_support` |
| `--datasets` | — | Dataset name(s): `durecdial`, `inspired`, `craigslist_bargain`, `es_conv` |
| `--models` | `bert` | Model name(s): `padpp`, `dmorl`, etc. |
| `--gen_models` | `bart` | Response generation: `chatgpt`, `llama3`, `bart`, `vicuna` |
| `--metrics` | — | Comma-separated metric names |
| `--loggers` | `terminal` | `terminal`, `file`, `wandb` |
| `--seed` | `42` | Random seed |
| `--domain` | `movie` | Recommendation domain (durecdial only) |
| `--model_type` | `llama3` | LLM model for simulators |
| `--use_persona` | false | Enable diverse user personas |
| `--test_phase` | false | Skip training; run online evaluation only |
| `--prioritized_objective` | `uniform` | Focus on specific objective at evaluation |
| `--objective_weight` | `None` | Fixed weight vector, e.g. `0.6,0.2,0.2` |
| `--num_train_rl_epochs` | `50` | Override YAML RL epoch count |
| `--use_gpi` | `1` | `1` = use GPI, `0` = standard PI |
| `--n_preferences` | `128` | Preference samples per RL batch |
| `--overwrite_sim` | false | Regenerate user simulators |
| `--ablation` | `""` | Ablation variant (e.g. `no_rl`) |
| `--exp_name` | `""` | Experiment name for logging |
| `--log_dir` | `logs` | Directory for file logs |
| `--project_name` | `MODPL` | W&B project name |

#### DMORL-only (`run_dmorl.py`)

| Argument | Default | Description |
|---|---|---|
| `--n_basic_skills` | (from YAML) | Override number of basic skills |
| `--n_advanced_skills` | (from YAML) | Override number of advanced skills |
| `--n_skill_train_epochs` | (from YAML) | RL epochs per basic skill |
| `--n_advanced_train_epochs` | (from YAML) | Total advanced skill RL epochs |
| `--dynamic_weight_horizon` | (from YAML) | Re-query LLM every T turns |
| `--use_dynamic_weight` | (from YAML) | Enable LLM dynamic weighting |
| `--no_dynamic_weight` | — | Disable dynamic weighting (ablation) |
| `--use_hints` | (from YAML) | Use accumulated hints in weight query |
| `--run_curriculum` | (from YAML) | Run Phase 1a + 1b |
| `--no_curriculum` | — | Skip curriculum, go straight to Phase 2 |
| `--force_rediscover_skills` | false | Re-query LLM for skills even if saved |

---

## 9. Running Experiments

### Negotiation (CraigslistBargain)

```bash
# Full DMORL training
python run_dmorl.py \
    --scenario negotiation \
    --datasets craigslist_bargain \
    --models dmorl \
    --gen_models chatgpt \
    --metrics sr,deal_rate,sl_ratio,fairness,avg_turn \
    --loggers terminal,file \
    --n_basic_skills 5 \
    --n_advanced_skills 5 \
    --n_skill_train_epochs 10 \
    --dynamic_weight_horizon 3 \
    --use_dynamic_weight \
    --seed 42

# Ablation: no curriculum (Phase 2 only)
python run_dmorl.py \
    --scenario negotiation \
    --datasets craigslist_bargain \
    --models dmorl \
    --gen_models chatgpt \
    --metrics sr,deal_rate,sl_ratio,fairness \
    --loggers terminal \
    --no_curriculum \
    --seed 42

# Ablation: no dynamic weight
python run_dmorl.py \
    --scenario negotiation \
    --datasets craigslist_bargain \
    --models dmorl \
    --gen_models chatgpt \
    --metrics sr,deal_rate,sl_ratio,fairness \
    --loggers terminal \
    --no_dynamic_weight \
    --seed 42
```

### Recommendation (DuRecDial)

```bash
python run_dmorl.py \
    --scenario recommendation \
    --datasets durecdial \
    --domain movie \
    --models dmorl \
    --gen_models chatgpt \
    --metrics sr,user_reward,item_freq,avg_turn \
    --loggers terminal,file \
    --n_basic_skills 5 \
    --n_advanced_skills 5 \
    --use_dynamic_weight \
    --seed 42
```

### Emotional Support (ESConv)

```bash
python run_dmorl.py \
    --scenario emotional_support \
    --datasets es_conv \
    --models dmorl \
    --gen_models chatgpt \
    --metrics sr,user_reward,toxicity,avg_turn \
    --loggers terminal,file \
    --n_basic_skills 5 \
    --n_advanced_skills 5 \
    --use_dynamic_weight \
    --seed 42
```

---

## 10. PADPP Baseline

The original PADPP code is fully preserved. Run it with `run.py`:

```bash
# Negotiation
python run.py \
    --scenario negotiation \
    --datasets craigslist_bargain \
    --models padpp \
    --gen_models chatgpt \
    --metrics sr,deal_rate,sl_ratio,fairness,avg_turn \
    --loggers terminal \
    --seed 42

# Recommendation
python run.py \
    --scenario recommendation \
    --datasets durecdial \
    --models padpp \
    --gen_models chatgpt \
    --metrics sr,user_reward,item_freq,avg_turn \
    --loggers terminal \
    --seed 42
```

---

## 11. Citation

If you use this code, please cite both PADPP and our DMORL work:

```bibtex
@inproceedings{dao-liao-2025-one,
    title     = "One Planner To Guide Them All! Learning Adaptive Conversational Planners for Goal-oriented Dialogues",
    author    = "Dao, Huy Quang and Liao, Lizi",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    year      = "2025",
    url       = "https://aclanthology.org/2025.emnlp-main.1123/",
}

@inproceedings{tran2026dmorl,
    title     = "Dynamic Multi-Objective Reinforcement Learning for Dialogue Policy Planning",
    author    = "Tran, Thai and ...",
    booktitle = "Proceedings of the 2026 IEEE International Conference on Data Mining (ICDM)",
    year      = "2026",
}
```
