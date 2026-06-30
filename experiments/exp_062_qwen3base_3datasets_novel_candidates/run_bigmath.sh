#!/usr/bin/env bash
# run_bigmath.sh — the 4 non-entropy candidates on Big-Math int-2000 (exp_058 setup),
# to overlay with GRPO / GROP@GRPO / gtpo_ema_flipped(FIXED) reused from exp_058.
set -e; set -o pipefail
VENV="/root/aiim/venv"; EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_TOKEN:?HF_TOKEN not set}"; source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface/}"
export SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-300}"
cd "${EXP_DIR}"
for M in sign_gate pos_discount raw_c ref_delta; do
  echo "=== [$(date -Is)] dataset=bigmath method=$M steps=$SMOKE_MAX_STEPS starting ==="
  rm -rf "${EXP_DIR}/outputs_bigmath_${M}" "${EXP_DIR}/unsloth_compiled_cache" \
         "${EXP_DIR}/grpo_trainer_lora_model" 2>/dev/null || true
  python train.py --dataset bigmath --method "$M" 2>&1 | tee "train_bigmath_${M}.log"
  echo "=== [$(date -Is)] dataset=bigmath method=$M DONE ==="
done
echo "=== [$(date -Is)] exp_062 bigmath candidates COMPLETE ==="
