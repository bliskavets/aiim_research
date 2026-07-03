#!/usr/bin/env bash
# exp_069 — rank-based adaptive-k C: k = clamp(rank_of_sampled_token, 1, 5).
# Single config (cap=5, min_k=1, lam=0.7, pos_discount) across 4 datasets.
set -u
cd "$(dirname "$0")"
export HF_TOKEN=$(cat /workspace/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token 2>/dev/null)
PY=/root/aiim/venv/bin/python
for DS in gsm8k math500 bigmath omnimath; do
  echo "=== rank_c | $DS ==="
  "$PY" train.py --dataset "$DS" --method rank_c \
    2>&1 | tee "train_${DS}_rank_c.log"
done
echo "ALL DONE"
