#!/usr/bin/env bash
# run_058_diag.sh — DIAGNOSTIC re-run of bare gtpo_ema_flipped (identical math,
# same seed 3407 => reproduces the length-explosion trajectory) with heavy
# per-step JSONL logging to diag/. 420 steps to match the original bare run.
set -e; set -o pipefail
VENV="/root/aiim/venv"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_TOKEN:?HF_TOKEN env var not set}"
source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
export SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-420}"
cd "${EXP_DIR}"
M=gtpo_ema_flipped_diag
echo "=== [$(date -Is)] method=$M  steps=$SMOKE_MAX_STEPS  starting ==="
rm -rf "${EXP_DIR}/outputs_$M" "${EXP_DIR}/unsloth_compiled_cache" \
       "${EXP_DIR}/grpo_trainer_lora_model" \
       "${EXP_DIR}/diag/diag_gtpo_ema_flipped_gens.jsonl" \
       "${EXP_DIR}/diag/diag_gtpo_ema_flipped_shape.jsonl" 2>/dev/null || true
python train.py --method "$M" 2>&1 | tee "train_$M.log"
echo "=== [$(date -Is)] method=$M  DONE ==="
