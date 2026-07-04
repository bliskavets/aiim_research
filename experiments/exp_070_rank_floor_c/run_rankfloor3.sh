#!/usr/bin/env bash
# exp_070b — rank-FLOOR-3 C: k = max(rank_of_sampled_token, 3) = clamp(rank, 3, 256).
# Floor at exp_066's sweet-spot k=3. Single config across 4 datasets.
set -u
cd "$(dirname "$0")"
export HF_TOKEN=$(cat /workspace/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token 2>/dev/null)
PY=/root/aiim/venv/bin/python
for DS in gsm8k math500 bigmath omnimath; do
  echo "=== rank_floor3_c | $DS ==="
  "$PY" train.py --dataset "$DS" --method rank_floor3_c \
    2>&1 | tee "train_${DS}_rank_floor3_c.log"
done
echo "ALL DONE"
