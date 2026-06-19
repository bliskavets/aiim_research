#!/usr/bin/env bash
set -e; set -o pipefail
source /root/aiim/venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
export SMOKE_MAX_STEPS=420
cd "$(dirname "${BASH_SOURCE[0]}")"
for M in grpo gtpo_ema_flipped; do
  echo "=== [$(date -Is)] method=$M (420 steps) ==="
  rm -rf outputs_$M unsloth_compiled_cache grpo_trainer_lora_model 2>/dev/null || true
  python train.py --method "$M" 2>&1 | tee "train_$M.log"
done
echo "=== baselines DONE ==="
