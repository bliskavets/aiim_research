#!/usr/bin/env bash
# run_060.sh — CONTROL: grpo vs grpo_s_entropy(beta2=0), sequential, native venv.
# Validates that the (fixed, injection-based) GRPO-S code reduces to ~GRPO when
# the entropy bonus is off. Early-stop once the curves clearly track or diverge.
set -e
set -o pipefail
VENV="/root/aiim/venv"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_TOKEN:?HF_TOKEN env var not set}"
source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1
cd "${EXP_DIR}"
for M in grpo_s_entropy; do
  echo "=== [$(date -Is)] method=$M — starting ==="
  rm -rf "${EXP_DIR}/outputs_$M" "${EXP_DIR}/unsloth_compiled_cache" "${EXP_DIR}/grpo_trainer_lora_model" 2>/dev/null || true
  python train.py --method "$M" 2>&1 | tee "train_$M.log"
done
echo "=== [$(date -Is)] exp_060 COMPLETE ==="
