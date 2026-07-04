#!/usr/bin/env bash
# exp_070 — rank-FLOOR C: k = max(rank_of_sampled_token, 5) = clamp(rank, 5, 256).
# argmax-sampled tokens (~83%) get k=5 (stable value from exp_066); k grows only
# when the model sampled from the tail. Single config across 4 datasets.
set -u
cd "$(dirname "$0")"
export HF_TOKEN=$(cat /workspace/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token 2>/dev/null)
PY=/root/aiim/venv/bin/python
for DS in gsm8k math500 bigmath omnimath; do
  echo "=== rank_floor_c | $DS ==="
  "$PY" train.py --dataset "$DS" --method rank_floor_c \
    2>&1 | tee "train_${DS}_rank_floor_c.log"
done
echo "ALL DONE"
