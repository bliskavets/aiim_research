#!/usr/bin/env bash
# run_nucleus.sh — exp_068: nucleus_c (top-p k for C) on base FIXED lam=0.7 + pos_discount.
# top_p in {0.7,0.8,0.9,0.95} x 4 datasets = 16 runs, min_k=1, sampling stays 1.0.
# Baseline (pos_discount lam0.7 k5) reused from exp_066; GRPO reused from exp_063.
set -e; set -o pipefail
VENV="/root/aiim/venv"; EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_TOKEN:?HF_TOKEN not set}"; source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface/}"
export SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-300}"
cd "${EXP_DIR}"
for DS in gsm8k math500 bigmath omnimath; do
  for P in 0.7 0.8 0.9 0.95; do
    echo "=== [$(date -Is)] dataset=$DS nucleus_c top_p=$P starting ==="
    rm -rf outputs_* unsloth_compiled_cache grpo_trainer_lora_model 2>/dev/null || true
    python train.py --dataset "$DS" --method nucleus_c --top_p "$P" --min_k 1 2>&1 | tee "train_${DS}_nucleus_p${P}.log"
    echo "=== [$(date -Is)] dataset=$DS nucleus_c top_p=$P DONE ==="
  done
done
echo "=== [$(date -Is)] exp_068 nucleus ALL 16 RUNS COMPLETE ==="
