# DMORL – Dynamic Multi-Objective Reinforcement Learning for Dialogue

An improvement over PADPP that adds:
1. **LLM-guided Skill Discovery** – uses Qwen3-32B (DeepInfra) to propose semantically meaningful weight vectors
2. **Curriculum Training** – trains basic skills first, then advanced skills, before full PADPP RLT
3. **Dynamic Weighting** – LLM selects optimal objective weights every T turns at inference time
4. **Post-Dialogue Refinement** – LLM extracts tactical hints after each dialogue to improve future runs

## Architecture

```
PADPP (base)
├── dmorl/
│   ├── llm_controller.py   # DeepInfra client + SkillLibrary + DynamicWeightController + HintManager
│   ├── config.py           # DMORLConfig (extends PADPPConfig)
│   ├── model.py            # DMORLModel (extends PADPPModel, adds skill-library GPI)
│   ├── trainer.py          # DMORLTrainer (3-phase training)
│   └── pipeline.py         # DMORLPipeline (3 scenarios)
├── config/models/
│   ├── DMORL_NEG.yaml
│   ├── DMORL_REC.yaml
│   └── DMORL_ES.yaml
└── run_dmorl.py            # Entry point
```

## Training Phases

### Phase 1a – Basic Skills Curriculum
- LLM proposes N basic skills (e.g. "Active Listening", "Price Focus")
- Each skill has a weight vector on the objective simplex
- Agent trains for `n_skill_train_epochs` RL epochs per skill with fixed weight
- Creates stable, convergent policies for each skill

### Phase 1b – Advanced Skills
- LLM proposes M advanced skills building on the basic ones
- Agent trains with `p_skill=0.6` probability of sampling advanced skill weights
- GPI provides implicit teacher-policy regularisation

### Phase 2 – Full PADPP RLT
- Standard PADPP reinforcement learning with random weight sampling
- Generalises the policy across the full objective simplex

### Phase 3 – Post-Dialogue Refinement (inference-time)
- After each dialogue, LLM analyses history and generates tactical hints
- Hints persisted to JSON and fed to the Dynamic Weight Controller
- Dynamic Weight Controller queries LLM every T turns to update objective weights

## Usage

```bash
# Negotiation – full training
python run_dmorl.py \
    --scenario negotiation \
    --datasets craigslist_bargain \
    --models dmorl \
    --gen_models chatgpt \
    --metrics sr,deal_rate,sl_ratio,fairness,avg_turn \
    --loggers terminal,file \
    --n_basic_skills 5 \
    --n_advanced_skills 5 \
    --dynamic_weight_horizon 3

# Recommendation – test only (load checkpoint)
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
```

## Key Config Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_basic_skills` | 5 | Number of basic skills to discover via LLM |
| `n_advanced_skills` | 5 | Number of advanced skills |
| `n_skill_train_epochs` | 10 | RL epochs per basic skill |
| `n_advanced_train_epochs` | 10 | RL epochs for advanced skill phase |
| `use_dynamic_weight` | true | Enable LLM-based dynamic weighting at inference |
| `dynamic_weight_horizon` | 3 | Re-query LLM every T turns |
| `use_hints` | true | Feed accumulated hints to weight controller |
| `run_curriculum` | true | Enable Phase 1a + 1b |
| `force_rediscover_skills` | false | Re-run LLM skill discovery even if file exists |

## LLM API

Uses DeepInfra's OpenAI-compatible API with model `Qwen/Qwen3-32B`.
