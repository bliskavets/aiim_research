#!/usr/bin/env bash
# exp_083 — Llama-3.2-3B-Instruct (exp_050 harness): GRPO vs gtpo_ema_flipped (ORIGINAL,
# exp_050 pre-FIX) vs gtpo_ema_flipped_fixed (group-visible FIX). bigmath (exp_050 dataset).
# Same shaping config for both flavours (lam0.9, top_k=20) to isolate the FIX. 500 steps.
set -u
cd "$(dirname "$0")"
export HF_TOKEN=$(cat /workspace/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token 2>/dev/null)
PY=/root/aiim/venv/bin/python
for METHOD in grpo gtpo_ema_flipped gtpo_ema_flipped_fixed; do
  echo "=== $METHOD | bigmath ==="
  "$PY" train.py --dataset bigmath --method "$METHOD" 2>&1 | tee "train_bigmath_${METHOD}.log"
done
echo "ALL DONE exp_083"
