#!/usr/bin/env bash
# run_combo.sh — pos_discount + EMA lambda combo: lambda in {0.5, 0.7} x
# {gsm8k, math500, omnimath, bigmath} = 8 runs. Sequential.
set -e; set -o pipefail
VENV="/root/aiim/venv"; EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_TOKEN:?HF_TOKEN not set}"; source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface/}"
export SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-300}"
cd "${EXP_DIR}"
for DS in gsm8k math500 omnimath bigmath; do
  for LAM in 0.5 0.7; do
    echo "=== [$(date -Is)] dataset=$DS pos_discount lam=$LAM steps=$SMOKE_MAX_STEPS starting ==="
    rm -rf "${EXP_DIR}/outputs_${DS}_pos_discount_lam${LAM}" \
           "${EXP_DIR}/unsloth_compiled_cache" "${EXP_DIR}/grpo_trainer_lora_model" 2>/dev/null || true
    python train.py --dataset "$DS" --method pos_discount --lam "$LAM" 2>&1 | tee "train_${DS}_posdisc_lam${LAM}.log"
    echo "=== [$(date -Is)] dataset=$DS pos_discount lam=$LAM DONE ==="
  done
done
echo "=== [$(date -Is)] exp_063 combo ALL 8 RUNS COMPLETE ==="
