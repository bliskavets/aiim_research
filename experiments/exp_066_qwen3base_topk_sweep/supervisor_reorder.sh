#!/usr/bin/env bash
# supervisor_reorder.sh — wait for gsm8k k=40 to finish, stop the old dataset-outer
# chain, then launch the K-outer reordered runner for the remaining datasets.
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$EXP_DIR"
echo "[sup] waiting for gsm8k top_k=40 DONE..."
while ! grep -q "dataset=gsm8k top_k=40 DONE" chain_topk.console.log 2>/dev/null; do
  if ! pgrep -f "run_topk.sh|chain_topk.sh" >/dev/null 2>&1; then echo "[sup] old chain gone before gsm8k k40 done!"; break; fi
  sleep 60
done
echo "[sup] gsm8k k40 finished -> stopping old dataset-outer chain"
pkill -f "run_topk.sh" 2>/dev/null || true
pkill -f "chain_topk.sh" 2>/dev/null || true
pkill -9 -f "train.py --dataset" 2>/dev/null || true
sleep 25
export HF_TOKEN="$(cat /workspace/.cache/huggingface/token)"
export HF_HOME=/workspace/.cache/huggingface/
echo "[sup] launching reordered (k-outer) runner"
bash run_topk_reordered.sh
