#!/usr/bin/env bash
# exp_078_gspo — gspo baseline + gspo_shaped (our per-token shaping on top), 4 datasets each.
set -u
cd "$(dirname "$0")"
export HF_TOKEN=$(cat /workspace/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token 2>/dev/null)
PY=/root/aiim/venv/bin/python
for METHOD in gspo gspo_shaped; do
  for DS in gsm8k math500 bigmath omnimath; do
    echo "=== $METHOD | $DS ==="
    "$PY" train.py --dataset "$DS" --method "$METHOD" 2>&1 | tee "train_${DS}_${METHOD}.log"
  done
done
echo "ALL DONE gspo"
