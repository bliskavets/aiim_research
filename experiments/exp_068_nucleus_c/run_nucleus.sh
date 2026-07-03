#!/usr/bin/env bash
# run_nucleus.sh — exp_068. Per dataset: baseline pos_discount(lam0.5,k=5) + nucleus_c
# at top_p in {0.7,0.8,0.9,0.95} (min_k=1, sampling stays 1.0). dataset-outer. 20 runs.
set -e; set -o pipefail
VENV="/root/aiim/venv"; EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_TOKEN:?HF_TOKEN not set}"; source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface/}"
export SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-300}"
cd "${EXP_DIR}"
runlog() { rm -rf outputs_* unsloth_compiled_cache grpo_trainer_lora_model 2>/dev/null || true; }
for DS in gsm8k math500 bigmath omnimath; do
  echo "=== [$(date -Is)] dataset=$DS method=posdisc_l0.5_k5 starting ==="
  runlog; python train.py --dataset "$DS" --method pos_discount --lam 0.5 --top_k 5 2>&1 | tee "train_${DS}_posdisc_lam0.5_k5.log"
  echo "=== [$(date -Is)] dataset=$DS method=posdisc_l0.5_k5 DONE ==="
  for P in 0.7 0.8 0.9 0.95; do
    echo "=== [$(date -Is)] dataset=$DS method=nucleus_c top_p=$P starting ==="
    runlog; python train.py --dataset "$DS" --method nucleus_c --top_p "$P" --min_k 1 2>&1 | tee "train_${DS}_nucleus_p${P}.log"
    echo "=== [$(date -Is)] dataset=$DS method=nucleus_c top_p=$P DONE ==="
  done
done
echo "=== [$(date -Is)] exp_068 nucleus ALL 20 RUNS COMPLETE ==="
