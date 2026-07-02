#!/usr/bin/env bash
# supervisor_dump.sh — pause exp_066 after current run, run GRPO logprob dump
# (gsm8k+bigmath, 100 steps) + viewer, then resume exp_066 (k10/k40 + k3/k1).
EXP67="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP66="/root/aiim/aiim_research/experiments/exp_066_qwen3base_topk_sweep"
VENV="/root/aiim/venv"; source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HOME=/workspace/.cache/huggingface/
export HF_TOKEN="$(cat /workspace/.cache/huggingface/token)"
echo "[dump-sup] waiting for exp_066 bigmath k=10 to finish..."
while ! grep -q "dataset=bigmath top_k=10 DONE" "$EXP66/supervisor_reorder.console.log" 2>/dev/null; do
  if ! pgrep -f "run_topk_reordered.sh" >/dev/null 2>&1; then echo "[dump-sup] exp_066 runner gone"; break; fi
  sleep 60
done
echo "[dump-sup] pausing exp_066"
pkill -f "supervisor_reorder.sh" 2>/dev/null || true
pkill -f "run_topk_reordered.sh" 2>/dev/null || true
pkill -9 -f "train.py --dataset" 2>/dev/null || true
sleep 25
cd "$EXP67"
for DS in gsm8k bigmath; do
  echo "[dump-sup] === logprob dump $DS (100 steps) ==="
  rm -rf "outputs_${DS}_grpo_lpdump" unsloth_compiled_cache grpo_trainer_lora_model "diag/lpdump_${DS}" 2>/dev/null || true
  SMOKE_MAX_STEPS=100 python train.py --dataset "$DS" --method grpo_lpdump 2>&1 | tee "dump_${DS}.log"
  echo "[dump-sup] === dump $DS DONE ==="
  python view_logprob_coverage.py "$DS" 2>&1 | tee "coverage_${DS}.txt" || true
done
echo "[dump-sup] === DUMPS COMPLETE -> resuming exp_066 ==="
cd "$EXP66"; SMOKE_MAX_STEPS=300 bash resume_topk.sh
