#!/usr/bin/env bash
# run_057_rest.sh — run the shaped candidates sequentially AFTER grpo was
# stopped early. Order (user-requested): gtpo_ema_flipped first, then
# grpo_s_entropy, then gtpo_conf. Preserves train_grpo.log / outputs_grpo.
# Native venv.
set -e
set -o pipefail
VENV="/root/aiim/venv"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_TOKEN:?HF_TOKEN env var not set}"

source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1

cd "${EXP_DIR}"
for M in gtpo_ema_flipped grpo_s_entropy gtpo_conf; do
  echo "=== [$(date -Is)] method=$M — wiping prior artefacts ==="
  rm -rf "${EXP_DIR}/outputs_$M" \
         "${EXP_DIR}/unsloth_compiled_cache" \
         "${EXP_DIR}/grpo_trainer_lora_model" 2>/dev/null || true
  echo "=== [$(date -Is)] method=$M — starting train ==="
  python train.py --method "$M" 2>&1 | tee "train_$M.log"
done
echo "=== [$(date -Is)] exp_057 shaped methods COMPLETE ==="
