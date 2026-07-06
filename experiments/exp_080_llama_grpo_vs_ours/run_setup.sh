#!/usr/bin/env bash
# exp_080 — Llama-3.2-3B-Instruct: GRPO baseline vs Ours (best per-token shaping:
# gtpo_ema_flipped FIXED + pos_discount, λ0.7, k5), all 4 datasets, same hyperparameters
# as the Qwen3-4B-Base study.
set -u
cd "$(dirname "$0")"
export HF_TOKEN=$(cat /workspace/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token 2>/dev/null)
PY=/root/aiim/venv/bin/python
for METHOD in grpo ours; do
  for DS in gsm8k math500 bigmath omnimath; do
    echo "=== $METHOD | $DS ==="
    "$PY" train.py --dataset "$DS" --method "$METHOD" 2>&1 | tee "train_${DS}_${METHOD}.log"
  done
done
echo "ALL DONE exp_080 llama"
