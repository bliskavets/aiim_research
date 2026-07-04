#!/usr/bin/env bash
# exp_076 (DAPO baseline) + exp_077 (our shaping on DAPO), queued AFTER the roadmap chain.
# Waits for the roadmap chain's completion marker so it never grabs the GPU mid-queue.
set -u
cd "$(dirname "$0")"
echo "[dapo-chain] waiting for roadmap chain to finish (marker in chain_roadmap.console.log)..."
while ! grep -q "ROADMAP QUEUE DONE" chain_roadmap.console.log 2>/dev/null; do sleep 120; done
# extra guard: ensure no train.py is mid-run
while pgrep -f "train.py --dataset" >/dev/null 2>&1; do sleep 60; done
echo "[dapo-chain] roadmap done, GPU free — starting DAPO queue"
for D in exp_076_dapo exp_077_dapo_shaped; do
  echo "[dapo-chain] ===== $D ====="
  bash "$D/run_setup.sh"
done
echo "[dapo-chain] DAPO QUEUE DONE"
