#!/usr/bin/env bash
# run_058_grop.sh — gtpo_ema_flipped + Group Relative Overlong Punishment (GROP,
# Appendix D of arXiv:2508.04349). 300 steps to match the length-penalty family.
set -e; set -o pipefail
VENV="/root/aiim/venv"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_TOKEN:?HF_TOKEN env var not set}"
source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
export SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-300}"
cd "${EXP_DIR}"
M=gtpo_ema_flipped_grop
echo "=== [$(date -Is)] method=$M steps=$SMOKE_MAX_STEPS starting ==="
rm -rf "${EXP_DIR}/outputs_$M" "${EXP_DIR}/unsloth_compiled_cache" "${EXP_DIR}/grpo_trainer_lora_model" 2>/dev/null || true
python train.py --method "$M" 2>&1 | tee "train_$M.log"
echo "=== [$(date -Is)] method=$M DONE ==="
