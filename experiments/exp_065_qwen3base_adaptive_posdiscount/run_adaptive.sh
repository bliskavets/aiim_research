#!/usr/bin/env bash
# run_adaptive.sh — exp_065 adaptive pos_discount shortlist (PC1,C1,P1,PC2) x 4 datasets
# = 16 runs. base = FIXED lam=0.7. dataset-outer (each dataset's sweep completes early).
set -e; set -o pipefail
VENV="/root/aiim/venv"; EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_TOKEN:?HF_TOKEN not set}"; source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface/}"
export SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-300}"
cd "${EXP_DIR}"
for DS in gsm8k math500 bigmath omnimath; do
  for M in adisc_pc1 adisc_c1 adisc_p1 adisc_pc2; do
    echo "=== [$(date -Is)] dataset=$DS method=$M steps=$SMOKE_MAX_STEPS starting ==="
    rm -rf "${EXP_DIR}/outputs_${DS}_${M}" "${EXP_DIR}/unsloth_compiled_cache" \
           "${EXP_DIR}/grpo_trainer_lora_model" 2>/dev/null || true
    python train.py --dataset "$DS" --method "$M" 2>&1 | tee "train_${DS}_${M}.log"
    echo "=== [$(date -Is)] dataset=$DS method=$M DONE ==="
  done
done
echo "=== [$(date -Is)] exp_065 adaptive ALL 16 RUNS COMPLETE ==="
