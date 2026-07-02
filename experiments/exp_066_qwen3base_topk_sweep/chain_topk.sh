#!/usr/bin/env bash
# chain_topk.sh — wait for exp_065's 16-run batch to finish, smoke one top_k config,
# then launch the 12-run top_k sweep. No set -e (handle failures).
VENV="/root/aiim/venv"; EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HOME=/workspace/.cache/huggingface/
export HF_TOKEN="$(cat /workspace/.cache/huggingface/token)"
cd "${EXP_DIR}"
E65="/root/aiim/aiim_research/experiments/exp_065_qwen3base_adaptive_posdiscount/chain_adaptive.console.log"
echo "[chain_topk] waiting for exp_065 batch to finish..."
while ! grep -q "adaptive ALL 16 RUNS COMPLETE" "$E65" 2>/dev/null; do
  if ! pgrep -f "chain_adaptive.sh|run_adaptive.sh" >/dev/null 2>&1 && ! grep -q "ALL 16 RUNS COMPLETE" "$E65" 2>/dev/null; then
    echo "[chain_topk] exp_065 gone; checking if complete..."; grep -q "ALL 16 RUNS COMPLETE" "$E65" 2>/dev/null || echo "[chain_topk] WARN exp_065 not complete; proceeding anyway"; break
  fi
  sleep 180
done
echo "[chain_topk] smoking top_k=5 (2 steps)"
rm -rf outputs_gsm8k_pos_discount_lam0.7_k5 unsloth_compiled_cache grpo_trainer_lora_model 2>/dev/null
SMOKE_MAX_STEPS=2 python train.py --dataset gsm8k --method pos_discount --lam 0.7 --top_k 5 > smoke_k5.log 2>&1
rc=$?
if [ $rc -eq 0 ] && grep -q "top_k override" smoke_k5.log && grep -q "pos_discount/used_group_shaped" smoke_k5.log; then
  rm -rf outputs_gsm8k_pos_discount_lam0.7_k5 unsloth_compiled_cache grpo_trainer_lora_model 2>/dev/null
  echo "[chain_topk] smoke OK -> launching 12-run sweep"
  SMOKE_MAX_STEPS=300 bash run_topk.sh
else
  echo "[chain_topk] ABORTED: smoke failed (rc=$rc)"
fi
