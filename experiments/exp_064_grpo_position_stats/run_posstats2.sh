#!/usr/bin/env bash
# run_posstats2.sh — posstats for math500 & omnimath (gsm8k+bigmath already done).
set -e; set -o pipefail
VENV="/root/aiim/venv"; EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_TOKEN:?HF_TOKEN not set}"; source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface/}"
export SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-300}"
cd "${EXP_DIR}"
for DS in math500 omnimath; do
  echo "=== [$(date -Is)] posstats dataset=$DS steps=$SMOKE_MAX_STEPS starting ==="
  rm -rf "${EXP_DIR}/outputs_${DS}_grpo_posstats" "${EXP_DIR}/unsloth_compiled_cache" \
         "${EXP_DIR}/grpo_trainer_lora_model" 2>/dev/null || true
  python train.py --dataset "$DS" --method grpo_posstats 2>&1 | tee "train_${DS}_grpo_posstats.log"
  echo "=== [$(date -Is)] posstats dataset=$DS DONE ==="
done
echo "=== [$(date -Is)] exp_064 posstats2 COMPLETE ==="
