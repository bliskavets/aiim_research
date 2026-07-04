#!/usr/bin/env bash
# exp_076 — DAPO baseline (Clip-Higher eps=0.2/0.28 + token-level dapo loss + overlong masking)
# on all 4 datasets. Plain GRPOTrainer + DAPO config knobs.
set -u
cd "$(dirname "$0")"
export HF_TOKEN=$(cat /workspace/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token 2>/dev/null)
PY=/root/aiim/venv/bin/python
METHOD=dapo
for DS in gsm8k math500 bigmath omnimath; do
  echo "=== $METHOD | $DS ==="
  "$PY" train.py --dataset "$DS" --method "$METHOD" 2>&1 | tee "train_${DS}_${METHOD}.log"
done
echo "ALL DONE $METHOD"
