#!/usr/bin/env bash
# exp_077 — our best per-token shaping (pos_discount FIXED, λ0.7, k5) applied ON TOP of DAPO
# (same Clip-Higher + token-level loss + overlong masking config as exp_076). 4 datasets.
set -u
cd "$(dirname "$0")"
export HF_TOKEN=$(cat /workspace/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token 2>/dev/null)
PY=/root/aiim/venv/bin/python
# pull DAPO baseline logs from exp_076 for the comparison plot
for DS in gsm8k math500 bigmath omnimath; do
  cp -f "../exp_076_dapo/train_${DS}_dapo.log" "train_${DS}_dapo.log" 2>/dev/null || true
done
METHOD=dapo_shaped
for DS in gsm8k math500 bigmath omnimath; do
  echo "=== $METHOD | $DS ==="
  "$PY" train.py --dataset "$DS" --method "$METHOD" 2>&1 | tee "train_${DS}_${METHOD}.log"
done
echo "ALL DONE $METHOD"
