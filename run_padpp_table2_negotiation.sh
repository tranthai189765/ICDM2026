#!/usr/bin/env bash
#
# Reproduce PADPP Table 2 (Negotiation / CraigslistBargain) row-by-row with
# the LLM-as-policy baseline. Each setting uses a different weight vector
# matching the paper convention:
#   uniform:  [1/3, 1/3, 1/3]
#   gain:     [1, 0, 0]
#   fair:     [0, 1, 0]
#   deal:     [0, 0, 1]
#
# Each run is independent (~20-30 min on FPT cloud, depending on n_episodes).
# Output JSONs are saved to eval_results/llm_uniform_<ts>.json.
#
# Usage:  bash run_padpp_table2_negotiation.sh   [N_EPISODES]
#   N_EPISODES defaults to 30.
#
set -e

N=${1:-30}

COMMON_FLAGS=(
    --scenario negotiation
    --datasets craigslist_bargain
    --models dmorl
    --gen_models fpt --model_type fpt
    --metrics sr,deal_rate,sl_ratio,fairness,avg_turn
    --loggers terminal
    --n_eval_episodes "$N"
    --cot
)

for SETTING in uniform gain fair deal; do
    echo
    echo "=============================================================="
    echo " Running setting=$SETTING   (n=$N episodes)"
    echo "=============================================================="
    python eval_llm_uniform.py \
        --weight_setting "$SETTING" \
        "${COMMON_FLAGS[@]}"
    echo
done

echo
echo "All 4 PADPP Table 2 (Negotiation) settings complete."
echo "Detailed results in eval_results/llm_uniform_*.json"
