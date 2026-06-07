# Guideline — How to Run Each Baseline (PPDPP / DPDP / PADPP / TRIP)

Tổng hợp đầy đủ command để chạy **PPDPP, DPDP, PADPP, TRIP** trên 3 nhóm dataset:
- `cb` — Craigslist Bargain (negotiation) + HMOD bargain scenarios
- `esc` — ESConv (emotional support)
- `cima` — CIMA (recommendation) / `durecdial` / `inspired`

Tất cả runner đều ghi kết quả ra `outputs/<framework>_eval_*/…/metrics.json` đúng schema để gộp bằng:

```bash
.venv/bin/python scripts/standardize_benchmark.py outputs/<dir1> outputs/<dir2> …
```

---

## 0. Chuẩn bị môi trường (chạy 1 lần trên server)

```bash
# 0.1 Tạo virtualenv (Python 3.10+)
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 0.2 Tải Roberta-large vào cache cục bộ (PPDPP/DPDP/TRIP đều dùng)
export HF_HOME=$PWD/cache/hf
python -c "from transformers import RobertaTokenizer, RobertaConfig; \
    RobertaTokenizer.from_pretrained('roberta-large', cache_dir='cache/hf'); \
    RobertaConfig.from_pretrained('roberta-large', cache_dir='cache/hf')"

# 0.3 Cấu hình FPT endpoint trong .env (đã có sẵn trong repo nếu copy đầy đủ)
cat > .env <<'EOF'
FPT_API_KEY=<your-fpt-key>
FPT_API_URL=https://mkp-api.fptcloud.com/v1
FPT_MODEL=Llama-3.3-70B-Instruct
POLICY_LLM_API_KEY=<your-fpt-key>
POLICY_LLM_BASE_URL=https://mkp-api.fptcloud.com/v1
POLICY_LLM_MODEL=Llama-3.3-70B-Instruct
EOF
```

> Mọi command bên dưới dùng `.venv/bin/python` để chắc chắn không dính Python hệ thống thiếu `fastchat`.

---

## 1. Dataset / scenario có sẵn

| Tên (CLI `--data_name`) | Bộ dữ liệu | YAML scenario HMOD (nếu có) |
|---|---|---|
| `cb` | Craigslist Bargain | [config/scenario/generated/hmod_bargain_train_scenarios.yaml](ICDM2026/config/scenario/generated/hmod_bargain_train_scenarios.yaml) / [hmod_bargain_test_scenarios.yaml](ICDM2026/config/scenario/generated/hmod_bargain_test_scenarios.yaml) |
| `esc` | ESConv | (dùng PPDPP default) |
| `cima` | CIMA tutoring | (dùng PPDPP default) |
| `durecdial` | DuRecDial 2.0 (PADPP only) | qua `--scenario recommendation --datasets durecdial` |
| `inspired` | INSPIRED (PADPP only) | qua `--scenario recommendation --datasets inspired` |

HMOD recommendation: [hmod_recommendation_test_scenarios.yaml](ICDM2026/config/scenario/generated/hmod_recommendation_test_scenarios.yaml) (dùng `--data_name cb` + path YAML này; xem mục PADPP cho biến thể recommendation).

---

## 2. PPDPP — Plug-and-Play Dialogue Policy Planner

Module: [PPDPP/run.py](ICDM2026/PPDPP/run.py)

### 2.1 Eval-only trên Craigslist Bargain (HMOD test scenarios, 250 cases, rule judge)

```bash
.venv/bin/python -m PPDPP.run \
  --data_name cb --system chatgpt --user chatgpt --critic chatgpt \
  --test_scenario_file config/scenario/generated/hmod_bargain_test_scenarios.yaml \
  --num_cases 250 --objective uniform --judge_model rule \
  --sft_dir sft --max_turn 8 --cache_dir cache/hf \
  --output_dir outputs/ppdpp_eval_bargain --do_eval
```

### 2.2 Eval với strict LLM judge (đắt hơn ~3-5×)

```bash
.venv/bin/python -m PPDPP.run \
  --data_name cb --system chatgpt --user chatgpt --critic chatgpt \
  --test_scenario_file config/scenario/generated/hmod_bargain_test_scenarios.yaml \
  --num_cases 250 --objective uniform --judge_model llm \
  --sft_dir sft --max_turn 8 --cache_dir cache/hf \
  --output_dir outputs/ppdpp_eval_strictllm --do_eval
```

### 2.3 Train + Eval (REINFORCE 10 epoch × 100 sample)

```bash
.venv/bin/python -m PPDPP.run \
  --data_name cb --system chatgpt --user chatgpt --critic chatgpt \
  --scenario_file config/scenario/generated/hmod_bargain_train_scenarios.yaml \
  --test_scenario_file config/scenario/generated/hmod_bargain_test_scenarios.yaml \
  --train_num_cases 200 --num_cases 50 \
  --objective uniform --judge_model rule \
  --sft_dir sft --max_turn 8 \
  --max_steps 10 --sample_times 100 --learning_rate 1e-6 --gamma 0.999 \
  --cache_dir cache/hf --output_dir outputs/ppdpp_train_bargain \
  --do_train --do_eval
```

### 2.4 ESConv (emotional support)

```bash
.venv/bin/python -m PPDPP.run \
  --data_name esc --system chatgpt --user chatgpt --critic chatgpt \
  --num_cases 200 --objective uniform --judge_model rule \
  --sft_dir sft --max_turn 8 --cache_dir cache/hf \
  --output_dir outputs/ppdpp_eval_esc --do_eval
```

### 2.5 CIMA

```bash
.venv/bin/python -m PPDPP.run \
  --data_name cima --system chatgpt --user chatgpt --critic chatgpt \
  --num_cases 200 --objective uniform --judge_model rule \
  --sft_dir sft --max_turn 8 --cache_dir cache/hf \
  --output_dir outputs/ppdpp_eval_cima --do_eval
```

---

## 3. DPDP — Dual-Process Dialogue Planner (PPDPP + MCTS slow-thinking)

Module: [dpdp_baseline/run.py](ICDM2026/dpdp_baseline/run.py)

Cờ DPDP riêng:
- `--planner_mode {policy|mcts|dual}` — `dual` = paper default
- `--mcts_eta 0.3` — ngưỡng entropy kích hoạt slow path
- `--mcts_top_k 2` — số nhánh roll-out
- `--mcts_cache 256` — LRU cache cho roll-out

### 3.1 Dual planner trên HMOD bargain (paper setting)

```bash
.venv/bin/python -m dpdp_baseline.run \
  --data_name cb --system chatgpt --user chatgpt --critic chatgpt \
  --test_scenario_file config/scenario/generated/hmod_bargain_test_scenarios.yaml \
  --num_cases 250 --objective uniform --judge_model rule \
  --planner_mode dual --mcts_eta 0.3 --mcts_top_k 2 --mcts_cache 256 \
  --max_turn 8 --cache_dir cache/hf \
  --output_dir outputs/dpdp_eval_bargain_dual --do_eval
```

### 3.2 Ablations: policy-only và mcts-only

```bash
# Chỉ fast policy (≈ PPDPP base)
.venv/bin/python -m dpdp_baseline.run … --planner_mode policy \
  --output_dir outputs/dpdp_eval_bargain_policy --do_eval

# Chỉ MCTS slow-think
.venv/bin/python -m dpdp_baseline.run … --planner_mode mcts \
  --mcts_top_k 3 --output_dir outputs/dpdp_eval_bargain_mcts --do_eval
```

### 3.3 ESConv / CIMA

Cùng template — đổi `--data_name esc` hoặc `--data_name cima` và bỏ `--test_scenario_file` để dùng split mặc định của PPDPP.

---

## 4. PADPP — Preference-Aware Dialogue Policy Planner (DMORL family)

Entry chính: [run.py](ICDM2026/run.py) (training/SFT) và 3 eval script chuyên dụng.

> PADPP dùng cấu trúc khác hẳn (Pipeline + Trainer + DataProcessor theo scenario). Truyền scenario qua `--scenario` thay vì `--data_name`.

### 4.1 Train PADPP (Phase 1 anchor + Phase 2 GPI) trên Craigslist Bargain

```bash
.venv/bin/python run_dmorl.py \
  --scenario negotiation --datasets craigslist_bargain \
  --models dmorl --gen_models fpt --model_type fpt \
  --metrics sr,deal_rate,sl_ratio,fairness,avg_turn \
  --loggers terminal --n_eval_episodes 50 \
  --n_rpadpp_epochs 30 --alpha_rpadpp 0.5 \
  --output_dir checkpoints/dmorl_phase1_neg
```

### 4.2 Eval PADPP/DMORL từ checkpoint (Table 2 negotiation)

```bash
.venv/bin/python eval_dmorl.py \
  --scenario negotiation --datasets craigslist_bargain \
  --models dmorl --gen_models fpt --model_type fpt \
  --metrics sr,deal_rate,sl_ratio,fairness,avg_turn \
  --loggers terminal \
  --checkpoint checkpoints/dmorl_phase1a.pth \
  --skills_file dmorl_skills_neg.json \
  --n_eval_episodes 30 \
  --output_dir eval_results/padpp_neg
```

### 4.3 Reproduce **PADPP Table 2** — chạy đủ 4 setting (uniform / gain / fair / deal)

Script đã có sẵn:

```bash
bash run_padpp_table2_negotiation.sh 30   # 30 = số episodes
```

Tương đương vòng for chạy `eval_llm_uniform.py --weight_setting {uniform,gain,fair,deal}` (xem [run_padpp_table2_negotiation.sh](ICDM2026/run_padpp_table2_negotiation.sh)).

### 4.4 PADPP trên Emotional Support (ESConv)

```bash
.venv/bin/python eval_llm_baseline.py \
  --scenario emotional_support --datasets es_conv \
  --models dmorl --gen_models fpt --model_type fpt \
  --metrics sr,avg_turn --loggers terminal \
  --n_eval_episodes 30 \
  --output_dir eval_results/padpp_esc
```

### 4.5 PADPP trên Recommendation (DuRecDial / INSPIRED)

```bash
# DuRecDial
.venv/bin/python eval_llm_baseline.py \
  --scenario recommendation --datasets durecdial --domain movie \
  --models dmorl --gen_models fpt --model_type fpt \
  --metrics sr,avg_turn --loggers terminal \
  --n_eval_episodes 30 \
  --output_dir eval_results/padpp_durecdial_movie

# INSPIRED
.venv/bin/python eval_llm_baseline.py \
  --scenario recommendation --datasets inspired --domain movie \
  --models dmorl --gen_models fpt --model_type fpt \
  --metrics sr,avg_turn --loggers terminal \
  --n_eval_episodes 30 \
  --output_dir eval_results/padpp_inspired_movie
```

---

## 5. TRIP — Theory-of-Mind + Population-Based User Simulation

Module: [trip_baseline/run.py](ICDM2026/trip_baseline/run.py) (xây trên hạ tầng PPDPP)

Cờ TRIP riêng:
- `--trip_use_uasp` — bật ToM prefix (User-Aware Strategy Planning)
- `--trip_use_pbtp` — bật pool 40 persona Big-Five × Decision-Making
- `--trip_population_size 40` — kích thước pool
- `--trip_disable_tom_llm` — dùng ToM heuristic offline (chỉ cho smoke test)
- `--trip_tom_cache 512` — LRU cache cho ToM
- `--trip_tom_max_tokens 128` — token cap cho ToM call
- `--trip_tom_model` / `--trip_tom_base_url` — override endpoint (mặc định lấy từ `.env`)

### 5.1 TRIP full (UASP + PBTP) trên HMOD bargain, dùng FPT cho ToM

```bash
.venv/bin/python -m trip_baseline.run \
  --data_name cb --system chatgpt --user chatgpt --critic chatgpt \
  --test_scenario_file config/scenario/generated/hmod_bargain_test_scenarios.yaml \
  --num_cases 250 --objective uniform --judge_model rule \
  --trip_use_uasp --trip_use_pbtp --trip_population_size 40 \
  --trip_tom_cache 512 --trip_tom_max_tokens 128 \
  --max_turn 8 --cache_dir cache/hf \
  --output_dir outputs/trip_eval_bargain_full --do_eval
```

### 5.2 Ablations TRIP

```bash
# UASP-only (bỏ persona pool)
.venv/bin/python -m trip_baseline.run … --trip_use_uasp \
  --output_dir outputs/trip_eval_uasp_only --do_eval

# PBTP-only (bỏ ToM prefix)
.venv/bin/python -m trip_baseline.run … --trip_use_pbtp --trip_population_size 40 \
  --output_dir outputs/trip_eval_pbtp_only --do_eval

# Smoke test offline (không gọi LLM cho ToM)
.venv/bin/python -m trip_baseline.run … --trip_use_uasp --trip_use_pbtp \
  --trip_disable_tom_llm --num_cases 3 \
  --output_dir outputs/trip_eval_smoke --do_eval
```

### 5.3 Override endpoint TRIP ToM mà không sửa `.env`

```bash
.venv/bin/python -m trip_baseline.run … \
  --trip_use_uasp --trip_tom_model anthropic/claude-sonnet-4-6 \
  --trip_tom_base_url https://api.deepinfra.com/v1/openai
```

### 5.4 TRIP + Train (REINFORCE giống PPDPP)

```bash
.venv/bin/python -m trip_baseline.run \
  --data_name cb --system chatgpt --user chatgpt --critic chatgpt \
  --scenario_file config/scenario/generated/hmod_bargain_train_scenarios.yaml \
  --test_scenario_file config/scenario/generated/hmod_bargain_test_scenarios.yaml \
  --train_num_cases 200 --num_cases 50 \
  --objective uniform --judge_model rule \
  --trip_use_uasp --trip_use_pbtp --trip_population_size 40 \
  --max_steps 10 --sample_times 100 --learning_rate 1e-6 --gamma 0.999 \
  --max_turn 8 --cache_dir cache/hf \
  --output_dir outputs/trip_train_bargain --do_train --do_eval
```

### 5.5 TRIP trên ESConv / CIMA

Đổi `--data_name esc` hoặc `--data_name cima` và `--trip_tom_task esc` / `--trip_tom_task cb` để chọn prompt ToM phù hợp. (Bộ persona ESConv tự động dùng heuristic vì paper chưa cung cấp resisting strategies riêng cho ESConv.)

---

## 6. Sanity check trước khi chạy dài

Mỗi runner đều có chế độ smoke 3 case (~1-2 phút):

```bash
# PPDPP smoke
.venv/bin/python -m PPDPP.run … --num_cases 3 --max_turn 4 --judge_model rule --do_eval

# DPDP smoke
.venv/bin/python -m dpdp_baseline.run … --num_cases 3 --max_turn 4 --judge_model rule \
  --planner_mode dual --do_eval

# TRIP smoke
.venv/bin/python -m trip_baseline.run … --num_cases 3 --max_turn 4 --judge_model rule \
  --trip_use_uasp --trip_use_pbtp --trip_disable_tom_llm --do_eval

# PADPP smoke (Phase-2 eval với 3 episode)
.venv/bin/python eval_dmorl.py … --n_eval_episodes 3
```

---

## 7. Tổng hợp kết quả cuối cùng

Sau khi mỗi script hoàn tất, gộp tất cả output về một bảng chuẩn hóa:

```bash
.venv/bin/python scripts/standardize_benchmark.py \
  outputs/ppdpp_eval_bargain \
  outputs/dpdp_eval_bargain_dual \
  outputs/dpdp_eval_bargain_policy \
  outputs/trip_eval_bargain_full \
  outputs/trip_eval_uasp_only \
  outputs/trip_eval_pbtp_only \
  eval_results/padpp_neg \
  --limit 0
```

Output sẽ in 1 bảng `framework | run_name | n | deal_rate | gsr | avg_turn | cvr | t2da | objective | judge` rồi tổng hợp trung bình mỗi framework. PPDPP / DPDP / TRIP / PADPP đều được nhận dạng tự động qua trường `model` trong `metrics.json`.

---

## 8. Recipe nhanh chạy đủ bộ trên server (HMOD bargain, 250 cases, rule judge)

```bash
set -e
export PYTHONUNBUFFERED=1
COMMON="--data_name cb --system chatgpt --user chatgpt --critic chatgpt \
  --test_scenario_file config/scenario/generated/hmod_bargain_test_scenarios.yaml \
  --num_cases 250 --objective uniform --judge_model rule \
  --max_turn 8 --cache_dir cache/hf --do_eval"

# 1) PPDPP
.venv/bin/python -m PPDPP.run $COMMON --sft_dir sft \
  --output_dir outputs/ppdpp_eval_bargain

# 2) DPDP dual
.venv/bin/python -m dpdp_baseline.run $COMMON \
  --planner_mode dual --mcts_eta 0.3 --mcts_top_k 2 --mcts_cache 256 \
  --output_dir outputs/dpdp_eval_bargain_dual

# 3) TRIP full
.venv/bin/python -m trip_baseline.run $COMMON \
  --trip_use_uasp --trip_use_pbtp --trip_population_size 40 \
  --trip_tom_cache 512 --trip_tom_max_tokens 128 \
  --output_dir outputs/trip_eval_bargain_full

# 4) PADPP (Table-2 reproduction)
bash run_padpp_table2_negotiation.sh 30

# 5) Gộp bảng
.venv/bin/python scripts/standardize_benchmark.py \
  outputs/ppdpp_eval_bargain outputs/dpdp_eval_bargain_dual \
  outputs/trip_eval_bargain_full eval_results/llm_uniform_*.json
```

---

## 9. Troubleshooting nhanh

| Triệu chứng | Nguyên nhân thường gặp | Fix |
|---|---|---|
| `OSError: storage_fast/...` | `--cache_dir` mặc định trỏ path nội bộ tác giả | Luôn thêm `--cache_dir cache/hf` |
| `ImportError: fastchat` | Đang gọi Python hệ thống | Dùng đúng `.venv/bin/python` |
| MPS OOM | Mac Apple Silicon, sequence dài | Runner đã có fallback tự động → 384/256/192/128 |
| `[trip-tom] fallback to heuristic` | Endpoint FPT lỗi / hết quota | Kiểm tra `.env` + xem dòng `[trip] ToM endpoint -> …` ở đầu run |
| `inferred framework: unknown` ở bước standardize | `metrics.json` thiếu trường `model` | Đảm bảo chạy đúng runner gốc, không tự sửa schema |
| PADPP báo missing `dmorl_skills_neg.json` | Chưa chạy Phase 1 | Chạy `run_dmorl.py` trước rồi point `--skills_file` đúng |
