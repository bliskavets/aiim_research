#!/usr/bin/env bash
# run_058_adaptlen_pm.sh — the 2 NEW per-polarity adaptive-length-penalty methods
# (own knee L_+/L_- within each group's O+/O- subgroup), one after another.
# #5 always-on, #6 gated by low-temp success (t=0, t2=0.5). Native venv. Does NOT
# touch any existing candidate's logs.
set -e; set -o pipefail
VENV="/root/aiim/venv"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_TOKEN:?HF_TOKEN env var not set}"
source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
export SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-300}"
cd "${EXP_DIR}"
for M in gtpo_ema_adaptlen_pm gtpo_ema_adaptlen_pm_gated; do
  echo "=== [$(date -Is)] method=$M  steps=$SMOKE_MAX_STEPS  starting ==="
  rm -rf "${EXP_DIR}/outputs_$M" "${EXP_DIR}/unsloth_compiled_cache" \
         "${EXP_DIR}/grpo_trainer_lora_model" 2>/dev/null || true
  python train.py --method "$M" 2>&1 | tee "train_$M.log"
  echo "=== [$(date -Is)] method=$M  DONE ==="
done
echo "=== [$(date -Is)] exp_058 per-polarity adaptive-length-penalty methods COMPLETE ==="
