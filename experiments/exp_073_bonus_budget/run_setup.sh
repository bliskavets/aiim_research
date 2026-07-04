#!/usr/bin/env bash
# exp_071 — zero-variance gate on top of the current best (posdisc λ0.7 k5).
# omnimath first (the target dataset: ~40-50% zero-variance groups), then regression checks.
set -u
cd "$(dirname "$0")"
export HF_TOKEN=$(cat /workspace/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token 2>/dev/null)
PY=/root/aiim/venv/bin/python
METHOD=flipped_budget
for DS in gsm8k math500 bigmath omnimath; do
  echo "=== $METHOD | $DS ==="
  "$PY" train.py --dataset "$DS" --method "$METHOD" 2>&1 | tee "train_${DS}_${METHOD}.log"
done
echo "ALL DONE $METHOD"
