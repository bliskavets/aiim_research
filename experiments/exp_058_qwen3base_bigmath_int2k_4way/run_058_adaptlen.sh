#!/usr/bin/env bash
# run_058_adaptlen.sh — the 2 NEW adaptive-length-penalty methods, one after
# another (native venv). Adaptive knee L = max((Lmin+Lmax)/2, Lmean) per group;
# penalty in [-0.5,0]; #3 always-on, #4 gated by low-temp success (t=0, t2=0.5).
# Does NOT touch any existing candidate's logs.
set -e; set -o pipefail
VENV="/root/aiim/venv"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_TOKEN:?HF_TOKEN env var not set}"
source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
export SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-300}"
cd "${EXP_DIR}"
for M in gtpo_ema_adaptlen gtpo_ema_adaptlen_gated; do
  echo "=== [$(date -Is)] method=$M  steps=$SMOKE_MAX_STEPS  starting ==="
  rm -rf "${EXP_DIR}/outputs_$M" "${EXP_DIR}/unsloth_compiled_cache" \
         "${EXP_DIR}/grpo_trainer_lora_model" 2>/dev/null || true
  python train.py --method "$M" 2>&1 | tee "train_$M.log"
  echo "=== [$(date -Is)] method=$M  DONE ==="
done
echo "=== [$(date -Is)] exp_058 adaptive-length-penalty methods COMPLETE ==="
