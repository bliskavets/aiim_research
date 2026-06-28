#!/usr/bin/env bash
# run_overnight.sh — 3 datasets x 2 setups = 6 runs, sequential (native venv).
# dataset-outer so each sub-experiment completes early; per-run log train_<ds>_<m>.log.
set -e; set -o pipefail
VENV="/root/aiim/venv"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_TOKEN:?HF_TOKEN env var not set}"
source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface/}"
export SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-300}"
cd "${EXP_DIR}"
for DS in gsm8k math500 omnimath; do
  for M in grpo_grop gtpo_ema_flipped_fixed; do
    echo "=== [$(date -Is)] dataset=$DS method=$M steps=$SMOKE_MAX_STEPS starting ==="
    rm -rf "${EXP_DIR}/outputs_${DS}_${M}" "${EXP_DIR}/unsloth_compiled_cache" \
           "${EXP_DIR}/grpo_trainer_lora_model" 2>/dev/null || true
    python train.py --dataset "$DS" --method "$M" 2>&1 | tee "train_${DS}_${M}.log"
    echo "=== [$(date -Is)] dataset=$DS method=$M DONE ==="
  done
done
echo "=== [$(date -Is)] exp_061 ALL 6 RUNS COMPLETE ==="
