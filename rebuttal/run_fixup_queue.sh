#!/usr/bin/env bash
# Re-run B4 (Qwen3-1.7B) and C1 (Qwen3-32B) which failed in the first final-queue pass
# due to argument-name mismatches (b4 uses --n-candidates not --number-of-gens-per-epoch;
# c1 uses --judge-prompt-math500 not --judge-prompt). Correct args below.
# Runs after the RM requeue so the three chained queues never contend for the GPU.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /root/aiim/.venv/bin/activate
source /root/aiim/.env.session
export PYTHONUNBUFFERED=1
PORT=9090
MJ="--judge-prompt configs/math500_judge_prompt.txt --judge-config configs/math500_judge_config.json"
IJ="--judge-prompt configs/ifeval_judge_prompt.txt --judge-config configs/ifeval_judge_config.json"

log() { echo "[fix] $1 $(date '+%F %T')"; }
kill_server() { fuser -k ${PORT}/tcp 2>/dev/null || true; for i in $(seq 1 30); do curl -sf http://localhost:${PORT}/health >/dev/null 2>&1 || return 0; sleep 2; done; }
start_server() {
  kill_server; sleep 5; : > logs/vllm_server.log
  nohup vllm serve "$1" --host 0.0.0.0 --port ${PORT} --tensor-parallel-size 1 \
      --max-model-len "$3" --gpu-memory-utilization "$2" >> logs/vllm_server.log 2>&1 &
  log "SERVER starting $1 util=$2"
  for i in $(seq 1 240); do
    curl -sf http://localhost:${PORT}/health >/dev/null 2>&1 && { log "SERVER ready $1"; return 0; }
    grep -qE "EngineCore failed|Engine core initialization failed" logs/vllm_server.log && { log "SERVER FAILED $1"; return 1; }
    sleep 5
  done
  log "SERVER TIMEOUT $1"; return 1
}
run() { local name="$1" out="$2"; shift 2; if [[ -d "$out" && -n "$(ls -A "$out" 2>/dev/null)" ]]; then log "SKIP $name"; return; fi; log "START $name"; if "$@" > "logs/queue_${name}.log" 2>&1; then log "DONE  $name"; else log "FAIL  $name"; fi; }

log "waiting for RM requeue ALL DONE"
until grep -q "\[rmq\] ALL DONE" logs/rm_queue.log 2>/dev/null; do sleep 60; done
log "RM requeue done; starting fixup"

# B4: Qwen3-1.7B
if start_server "Qwen/Qwen3-1.7B-FP8" 0.85 16384; then SMALL="Qwen/Qwen3-1.7B-FP8";
else start_server "Qwen/Qwen3-1.7B" 0.85 16384 && SMALL="Qwen/Qwen3-1.7B" || SMALL=""; fi
if [[ -n "$SMALL" ]]; then
  run "b4_math" "logs/b4_small_math500" \
    python experiments/b4_small_model/run_small_model_math500.py \
      --ip localhost --port ${PORT} --model-name "$SMALL" --seed 42 --batch-size 16 \
      --num-samples 500 --num-optimization-epochs 2 --n-candidates 7 ${MJ} \
      --output-path logs/b4_small_math500
  run "b4_ifeval" "logs/b4_small_ifeval" \
    python experiments/b4_small_model/run_small_model_ifeval.py \
      --ip localhost --port ${PORT} --model-name "$SMALL" --seed 42 --batch-size 16 \
      --num-samples 541 --num-optimization-epochs 2 --n-candidates 7 ${IJ} \
      --output-path logs/b4_small_ifeval
fi

# C1: Qwen3-32B
if start_server "Qwen/Qwen3-32B-FP8" 0.90 32768; then
  run "c1_32b_math" "logs/c1_qwen32b_math500" \
    python experiments/c1_qwen32b/run_qwen32b.py \
      --ip localhost --port ${PORT} --model-name Qwen/Qwen3-32B-FP8 --seed 42 --batch-size 16 \
      --benchmark math500 --num-samples 500 --num-optimization-epochs 2 --number-of-gens-per-epoch 7 \
      --judge-prompt-math500 configs/math500_judge_prompt.txt \
      --judge-config-math500 configs/math500_judge_config.json \
      --output-path logs/c1_qwen32b_math500
fi

log "ALL DONE"
