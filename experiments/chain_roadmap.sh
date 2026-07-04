#!/usr/bin/env bash
# Queue for the exp_071–075 roadmap setups (analysis/exp055-070_deep_analysis.md §5).
# Waits for any running train.py (exp_070 omnimath floor3) to finish, then runs the
# five setups sequentially, 4 datasets each (20 runs total).
set -u
cd "$(dirname "$0")"

echo "[chain] waiting for GPU (current train.py to exit)..."
while pgrep -f "train.py --dataset" >/dev/null 2>&1; do sleep 60; done
echo "[chain] GPU free, starting roadmap queue"

for D in exp_071_zero_variance_gate exp_072_branch_entropy exp_073_bonus_budget \
         exp_074_surprisal_credit exp_075_final_combo; do
  echo "[chain] ===== $D ====="
  bash "$D/run_setup.sh"
done
echo "[chain] ROADMAP QUEUE DONE"
