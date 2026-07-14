#!/usr/bin/env bash
# Sequential GPU job queue for AAAI-2027 strengthening (single H200, one vLLM server).
# Not -e: a single job failure must not abort the rest of the queue.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /root/aiim/.venv/bin/activate
source /root/aiim/.env.session

MODEL=Qwen/Qwen3-8B-FP8
COMMON="--ip localhost --port 9090 --model-name $MODEL --batch-size 16"
MATH_JUDGE="--judge-prompt configs/math500_judge_prompt.txt --judge-config configs/math500_judge_config.json"
IFEVAL_JUDGE="--judge-prompt configs/ifeval_judge_prompt.txt --judge-config configs/ifeval_judge_config.json"
SAGE="--num-optimization-epochs 2 --number-of-gens-per-epoch 7"

run() {  # name, output-dir, command...
  local name="$1"; local out="$2"; shift 2
  if [[ -d "$out" && -n "$(ls -A "$out" 2>/dev/null)" ]]; then
    echo "[queue] SKIP $name (exists: $out)"; return
  fi
  echo "[queue] START $name $(date '+%F %T')"
  if "$@" >"logs/queue_${name}.log" 2>&1; then
    echo "[queue] DONE  $name $(date '+%F %T')"
  else
    echo "[queue] FAIL  $name (see logs/queue_${name}.log)"
  fi
}

# Multi-seed for the stochastic method (MATH baseline is greedy/seed-invariant at N=500,
# so its CI comes from bootstrapping over problems, not extra seeds).
for S in 7 123; do
  run "math_sage_s$S" "logs/sage_math_full_s$S" \
    python experiments/b2_mmin_ablation/run_mmin_sweep.py $COMMON --num-samples 500 --seed $S \
      --m-min-values 1 $MATH_JUDGE $SAGE --output-path "logs/sage_math_full_s$S"

  run "mmlu_base_s$S" "logs/c3_mmlu_baseline_s$S" \
    python experiments/c3_mmlu_pro/run_mmlu_pro_baseline.py $COMMON --num-samples 500 --seed $S \
      --output-path "logs/c3_mmlu_baseline_s$S"

  run "mmlu_sage_s$S" "logs/c3_mmlu_sage_s$S" \
    python experiments/c3_mmlu_pro/run_mmlu_pro_sage.py $COMMON --num-samples 500 --seed $S \
      $SAGE --output-path "logs/c3_mmlu_sage_s$S"
done

# IFEval / E4 (verifiable, judge-free scoring), seed 42
run "ifeval_baseline" "logs/a3_ifeval_baseline" \
  python experiments/a3_ifeval/run_ifeval_baseline.py $COMMON --num-samples 541 --seed 42 \
    --output-path "logs/a3_ifeval_baseline"
run "ifeval_bon" "logs/a3_ifeval_bon" \
  python experiments/a3_ifeval/run_ifeval_bon.py $COMMON --num-samples 541 --seed 42 --n-candidates 7 \
    $IFEVAL_JUDGE --output-path "logs/a3_ifeval_bon"
run "ifeval_sage" "logs/a3_ifeval_sage" \
  python experiments/a3_ifeval/run_ifeval_sage.py $COMMON --num-samples 541 --seed 42 \
    $IFEVAL_JUDGE $SAGE --output-path "logs/a3_ifeval_sage"

echo "[queue] ALL DONE $(date '+%F %T')"
