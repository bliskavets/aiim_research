#!/usr/bin/env bash
# run_topk_reordered.sh — remaining datasets in K-OUTER order: k=5 (all), k=10 (all),
# k=40 (all), over math500/bigmath/omnimath. (gsm8k already done separately.)
set -e; set -o pipefail
VENV="/root/aiim/venv"; EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_TOKEN:?HF_TOKEN not set}"; source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface/}"
export SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-300}"
cd "${EXP_DIR}"
for TK in 5 10 40; do
  for DS in math500 bigmath omnimath; do
    echo "=== [$(date -Is)] dataset=$DS top_k=$TK steps=$SMOKE_MAX_STEPS starting ==="
    rm -rf "${EXP_DIR}/outputs_${DS}_pos_discount_lam0.7_k${TK}" \
           "${EXP_DIR}/unsloth_compiled_cache" "${EXP_DIR}/grpo_trainer_lora_model" 2>/dev/null || true
    python train.py --dataset "$DS" --method pos_discount --lam 0.7 --top_k "$TK" 2>&1 | tee "train_${DS}_posdisc_lam0.7_k${TK}.log"
    echo "=== [$(date -Is)] dataset=$DS top_k=$TK DONE ==="
  done
  echo "=== [$(date -Is)] exp_066 reordered k=$TK ALL DATASETS DONE ==="
done
echo "=== [$(date -Is)] exp_066 reordered REMAINING COMPLETE ==="
