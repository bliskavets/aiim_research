#!/usr/bin/env bash
# run_058_lenpen.sh — the 2 NEW length-penalty methods (one after another),
# native venv. Does NOT touch the 4 existing candidates' logs.
set -e; set -o pipefail
VENV="/root/aiim/venv"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_TOKEN:?HF_TOKEN env var not set}"
source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
cd "${EXP_DIR}"
for M in gtpo_ema_lenpen gtpo_ema_lenpen_gated; do
  echo "=== [$(date -Is)] method=$M starting ==="
  rm -rf "${EXP_DIR}/outputs_$M" "${EXP_DIR}/unsloth_compiled_cache" "${EXP_DIR}/grpo_trainer_lora_model" 2>/dev/null || true
  python train.py --method "$M" 2>&1 | tee "train_$M.log"
done
echo "=== [$(date -Is)] exp_058 length-penalty methods COMPLETE ==="
