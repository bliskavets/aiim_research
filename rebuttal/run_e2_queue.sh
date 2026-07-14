#!/usr/bin/env bash
# E2 job queue (Self-Refine + Reflexion on MATH-500 and IFEval, budget-matched to SAGE).
# Waits for the main queue (run_aaai_queue.sh) to finish so the two do not contend for
# the single GPU, then runs the four E2 jobs back-to-back.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /root/aiim/.venv/bin/activate
source /root/aiim/.env.session

echo "[e2] waiting for main queue to finish..."
until grep -q "\[queue\] ALL DONE" logs/aaai_queue.log 2>/dev/null; do sleep 60; done
echo "[e2] main queue done; starting E2 $(date '+%F %T')"

COMMON="--ip localhost --port 9090 --model-name Qwen/Qwen3-8B-FP8 --batch-size 16 --budget 21 --seed 42"

run() {  # name, output-dir, command...
  local name="$1"; local out="$2"; shift 2
  if [[ -d "$out" && -n "$(ls -A "$out" 2>/dev/null)" ]]; then
    echo "[e2] SKIP $name (exists)"; return
  fi
  echo "[e2] START $name $(date '+%F %T')"
  if "$@" >"logs/queue_${name}.log" 2>&1; then
    echo "[e2] DONE  $name $(date '+%F %T')"
  else
    echo "[e2] FAIL  $name (see logs/queue_${name}.log)"
  fi
}

for mode in self_refine reflexion; do
  run "e2_${mode}_math_s42" "logs/e2_${mode}_math_s42" \
    python experiments/e2_self_refine/run_math500.py $COMMON --mode $mode \
      --num-samples 500 --output-path "logs/e2_${mode}_math_s42"
  run "e2_${mode}_ifeval_s42" "logs/e2_${mode}_ifeval_s42" \
    python experiments/e2_self_refine/run_ifeval.py $COMMON --mode $mode \
      --num-samples 541 --output-path "logs/e2_${mode}_ifeval_s42"
done
echo "[e2] ALL DONE $(date '+%F %T')"
