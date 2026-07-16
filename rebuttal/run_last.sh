#!/usr/bin/env bash
# Final remaining jobs, clean single queue (GPU fully freed, zombie reaped):
#   E9  : BoN + Skywork-Reward-V2-Qwen3-8B on 8B @0.55 (RM already downloaded)
#   b4_math : Qwen3-1.7B @0.40 32768 (eval now tolerant of a single flaky request)
#   C1  : Qwen3-32B-FP8 @0.85 (waits for the background download)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /root/aiim/.venv/bin/activate
source /root/aiim/.env.session
export PYTHONUNBUFFERED=1
PORT=9090
MJ="--judge-prompt configs/math500_judge_prompt.txt --judge-config configs/math500_judge_config.json"
log() { echo "[last] $1 $(date '+%F %T')"; }
kill_server() { pkill -9 -f "vllm serve" 2>/dev/null || true; pkill -9 -f "EngineCore" 2>/dev/null || true; for i in $(seq 1 30); do curl -sf http://localhost:${PORT}/health >/dev/null 2>&1 || { sleep 4; return 0; }; sleep 2; done; }
start_server() {
  kill_server; : > logs/vllm_server.log
  nohup vllm serve "$1" --host 0.0.0.0 --port ${PORT} --tensor-parallel-size 1 --max-model-len "$3" --gpu-memory-utilization "$2" >> logs/vllm_server.log 2>&1 &
  log "SERVER starting $1 util=$2"
  for i in $(seq 1 360); do
    if curl -sf http://localhost:${PORT}/health >/dev/null 2>&1; then
      got=$(curl -sS http://localhost:${PORT}/v1/models --max-time 5 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "")
      [[ "$got" == "$1" ]] && { log "SERVER ready $1"; return 0; }
    fi
    grep -qE "EngineCore failed|Engine core initialization failed|OutOfMemory|ValueError" logs/vllm_server.log && { log "SERVER FAILED $1"; return 1; }
    sleep 5
  done
  log "SERVER TIMEOUT $1"; return 1
}
run() { local name="$1" out="$2"; shift 2; if [[ -d "$out" && -n "$(ls -A "$out" 2>/dev/null)" ]]; then log "SKIP $name"; return; fi; log "START $name"; if "$@" > "logs/queue_${name}.log" 2>&1; then log "DONE  $name"; else log "FAIL  $name"; fi; }

# E9: 8B + Skywork V2 RM
start_server "Qwen/Qwen3-8B-FP8" 0.55 32768 || exit 1
run "e9_rm_math" "logs/c2_updated_rm_math500" python experiments/c2_updated_rm/run_updated_rm.py \
  --ip localhost --port ${PORT} --model-name Qwen/Qwen3-8B-FP8 --seed 42 --batch-size 8 \
  --benchmark math500 --method bon_rm --num-samples 500 --n-candidates 7 --rm-device cuda:0 \
  --output-path logs/c2_updated_rm_math500
run "e9_rm_ifeval" "logs/c2_updated_rm_ifeval" python experiments/c2_updated_rm/run_updated_rm.py \
  --ip localhost --port ${PORT} --model-name Qwen/Qwen3-8B-FP8 --seed 42 --batch-size 8 \
  --benchmark ifeval --method bon_rm --num-samples 541 --n-candidates 7 --rm-device cuda:0 \
  --output-path logs/c2_updated_rm_ifeval

# b4_math: 1.7B
rm -rf logs/b4_small_math500
if start_server "Qwen/Qwen3-1.7B-FP8" 0.40 32768; then SMALL="Qwen/Qwen3-1.7B-FP8";
else start_server "Qwen/Qwen3-1.7B" 0.40 32768 && SMALL="Qwen/Qwen3-1.7B" || SMALL=""; fi
[[ -n "$SMALL" ]] && run "b4_math" "logs/b4_small_math500" python experiments/b4_small_model/run_small_model_math500.py \
  --ip localhost --port ${PORT} --model-name "$SMALL" --seed 42 --batch-size 16 \
  --num-samples 500 --num-optimization-epochs 2 --n-candidates 7 ${MJ} --output-path logs/b4_small_math500

# C1: 32B (wait for download)
log "waiting for 32B download"
for i in $(seq 1 300); do grep -q "DL32_DONE_0" /root/aiim/external/qwen32b_dl.log 2>/dev/null && { log "32B download done"; break; }; sleep 60; done
if start_server "Qwen/Qwen3-32B-FP8" 0.85 32768; then
  run "c1_32b_math" "logs/c1_qwen32b_math500" python experiments/c1_qwen32b/run_qwen32b.py \
    --ip localhost --port ${PORT} --model-name Qwen/Qwen3-32B-FP8 --seed 42 --batch-size 16 \
    --benchmark math500 --num-samples 500 --num-optimization-epochs 2 --number-of-gens-per-epoch 7 \
    --judge-prompt-math500 configs/math500_judge_prompt.txt --judge-config-math500 configs/math500_judge_config.json \
    --output-path logs/c1_qwen32b_math500
fi
log "ALL DONE"
