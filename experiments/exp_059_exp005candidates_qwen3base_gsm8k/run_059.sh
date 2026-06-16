#!/usr/bin/env bash
# run_059.sh — exp_005 candidates + GRPO baseline on Qwen3-4B-Base / GSM8K.
# Sequential, native venv. Order: grpo (baseline) -> gtpo_conf -> grpo_s_conf.
set -e
set -o pipefail
VENV="/root/aiim/venv"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_TOKEN:?HF_TOKEN env var not set}"
source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1
cd "${EXP_DIR}"
for M in grpo gtpo_conf grpo_s_conf; do
  echo "=== [$(date -Is)] method=$M — starting ==="
  rm -rf "${EXP_DIR}/outputs_$M" "${EXP_DIR}/unsloth_compiled_cache" "${EXP_DIR}/grpo_trainer_lora_model" 2>/dev/null || true
  python train.py --method "$M" 2>&1 | tee "train_$M.log"
done
echo "=== [$(date -Is)] exp_059 COMPLETE ==="
