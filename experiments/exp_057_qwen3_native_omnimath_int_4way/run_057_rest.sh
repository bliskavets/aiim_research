#!/usr/bin/env bash
# run_057_rest.sh — run the 3 shaped candidates sequentially with the FIXED
# (injection) shaping actually applied. grpo baseline is reused from its earlier
# @492 run (the fix does not touch the plain-GRPO path), so this preserves
# train_grpo.log / outputs_grpo and only (re)runs the shaped methods.
# Order: gtpo_conf (most reliable shaped variant) first, then grpo_s_entropy,
# then gtpo_ema_flipped. Native venv.
set -e
set -o pipefail
VENV="/root/aiim/venv"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_TOKEN:?HF_TOKEN env var not set}"

source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1

cd "${EXP_DIR}"
for M in gtpo_conf grpo_s_entropy gtpo_ema_flipped; do
  echo "=== [$(date -Is)] method=$M — wiping prior artefacts ==="
  rm -rf "${EXP_DIR}/outputs_$M" \
         "${EXP_DIR}/unsloth_compiled_cache" \
         "${EXP_DIR}/grpo_trainer_lora_model" 2>/dev/null || true
  echo "=== [$(date -Is)] method=$M — starting train ==="
  python train.py --method "$M" 2>&1 | tee "train_$M.log"
done
echo "=== [$(date -Is)] exp_057 shaped methods COMPLETE ==="
