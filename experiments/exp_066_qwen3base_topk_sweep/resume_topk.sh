#!/usr/bin/env bash
# resume_topk.sh — finish the top_k sweep after the logprob-dump pause:
# omnimath k=10; k=40 (math500,bigmath,omnimath); then NEW k=3 & k=1 on all 4 datasets.
set -e; set -o pipefail
VENV="/root/aiim/venv"; EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_TOKEN:?HF_TOKEN not set}"; source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface/}"
export SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-300}"
cd "${EXP_DIR}"
run() {  # dataset top_k
  local DS=$1 TK=$2
  echo "=== [$(date -Is)] dataset=$DS top_k=$TK starting ==="
  rm -rf "outputs_${DS}_pos_discount_lam0.7_k${TK}" unsloth_compiled_cache grpo_trainer_lora_model 2>/dev/null || true
  python train.py --dataset "$DS" --method pos_discount --lam 0.7 --top_k "$TK" 2>&1 | tee "train_${DS}_posdisc_lam0.7_k${TK}.log"
  echo "=== [$(date -Is)] dataset=$DS top_k=$TK DONE ==="
}
run omnimath 10
for DS in math500 bigmath omnimath; do run $DS 40; done
for TK in 3 1; do for DS in gsm8k math500 bigmath omnimath; do run $DS $TK; done; done
echo "=== [$(date -Is)] exp_066 resume+k1k3 COMPLETE ==="
