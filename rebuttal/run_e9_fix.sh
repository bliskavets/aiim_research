#!/usr/bin/env bash
# E9 (BoN + Skywork RM) re-run: run_updated_rm.py has no --reward-model (RM is hardcoded
# to Skywork-Reward-Qwen-2.5-7B-v0.2); correct flags are --method bon_rm --benchmark ...
# --rm-device. Needs the 8B server at reduced memory so the RM fits. Waits for the main
# remaining queue (B4/C1) to finish so it does not fight for the GPU.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /root/aiim/.venv/bin/activate
source /root/aiim/.env.session
export PYTHONUNBUFFERED=1
PORT=9090
log() { echo "[e9] $1 $(date '+%F %T')"; }
kill_server() { pkill -9 -f "vllm serve" 2>/dev/null || true; pkill -9 -f "EngineCore" 2>/dev/null || true; for i in $(seq 1 30); do curl -sf http://localhost:${PORT}/health >/dev/null 2>&1 || { sleep 3; return 0; }; sleep 2; done; }
start_server() {
  kill_server; : > logs/vllm_server.log
  nohup vllm serve "$1" --host 0.0.0.0 --port ${PORT} --tensor-parallel-size 1 --max-model-len "$3" --gpu-memory-utilization "$2" >> logs/vllm_server.log 2>&1 &
  log "SERVER starting $1 util=$2"
  for i in $(seq 1 300); do
    if curl -sf http://localhost:${PORT}/health >/dev/null 2>&1; then
      got=$(curl -sS http://localhost:${PORT}/v1/models --max-time 5 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "")
      [[ "$got" == "$1" ]] && { log "SERVER ready $1"; return 0; }
    fi
    grep -qE "EngineCore failed|Engine core initialization failed|OutOfMemory" logs/vllm_server.log && { log "SERVER FAILED $1"; return 1; }
    sleep 5
  done
  log "SERVER TIMEOUT $1"; return 1
}
run() { local name="$1" out="$2"; shift 2; if [[ -d "$out" && -n "$(ls -A "$out" 2>/dev/null)" ]]; then log "SKIP $name"; return; fi; log "START $name"; if "$@" > "logs/queue_${name}.log" 2>&1; then log "DONE  $name"; else log "FAIL  $name"; fi; }

log "waiting for remaining queue ALL DONE"
until grep -q "\[rem\] ALL DONE" logs/remaining_queue.log 2>/dev/null; do sleep 60; done
log "remaining done; starting E9"

start_server "Qwen/Qwen3-8B-FP8" 0.70 32768 || exit 1
run "e9_rm_math" "logs/c2_updated_rm_math500" python experiments/c2_updated_rm/run_updated_rm.py \
  --ip localhost --port ${PORT} --model-name Qwen/Qwen3-8B-FP8 --seed 42 --batch-size 8 \
  --benchmark math500 --method bon_rm --num-samples 500 --n-candidates 7 --rm-device cuda:0 \
  --output-path logs/c2_updated_rm_math500
run "e9_rm_ifeval" "logs/c2_updated_rm_ifeval" python experiments/c2_updated_rm/run_updated_rm.py \
  --ip localhost --port ${PORT} --model-name Qwen/Qwen3-8B-FP8 --seed 42 --batch-size 8 \
  --benchmark ifeval --method bon_rm --num-samples 541 --n-candidates 7 --rm-device cuda:0 \
  --output-path logs/c2_updated_rm_ifeval
log "ALL DONE"
