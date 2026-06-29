#!/usr/bin/env bash
# run_overnight.sh — exp_062: 3 datasets x 5 runnable methods = 15 runs, sequential.
# (grop@grpo and gtpo_ema_flipped_fixed are REUSED from exp_061 — not re-run.)
# dataset-outer so each dataset's full comparison completes progressively.
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
  for M in grpo sign_gate pos_discount raw_c ref_delta; do
    echo "=== [$(date -Is)] dataset=$DS method=$M steps=$SMOKE_MAX_STEPS starting ==="
    rm -rf "${EXP_DIR}/outputs_${DS}_${M}" "${EXP_DIR}/unsloth_compiled_cache" \
           "${EXP_DIR}/grpo_trainer_lora_model" 2>/dev/null || true
    python train.py --dataset "$DS" --method "$M" 2>&1 | tee "train_${DS}_${M}.log"
    echo "=== [$(date -Is)] dataset=$DS method=$M DONE ==="
  done
done
echo "=== [$(date -Is)] exp_062 ALL 15 RUNS COMPLETE ==="
