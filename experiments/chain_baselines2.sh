#!/usr/bin/env bash
# exp_078 (GSPO) + exp_079 (Dr.GRPO), each baseline+shaped ×4 datasets.
# Waits for the DAPO chain to finish (marker), then runs sequentially.
set -u
cd "$(dirname "$0")"
echo "[b2-chain] waiting for DAPO chain (marker in chain_dapo.console.log)..."
while ! grep -q "DAPO QUEUE DONE" chain_dapo.console.log 2>/dev/null; do sleep 120; done
while pgrep -f "train.py --dataset" >/dev/null 2>&1; do sleep 60; done
echo "[b2-chain] GPU free — starting GSPO/Dr.GRPO queue"
for D in exp_078_gspo exp_079_drgrpo; do
  echo "[b2-chain] ===== $D ====="
  bash "$D/run_setup.sh"
done
echo "[b2-chain] BASELINES2 QUEUE DONE"
