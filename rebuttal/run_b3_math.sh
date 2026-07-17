#!/usr/bin/env bash
# B3 aspect sensitivity, MATH-only (generic + task_specific; default already done).
# Writes "[b3] ALL DONE" so run_spo.sh (waiting on it) proceeds.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /root/aiim/.venv/bin/activate; source /root/aiim/.env.session
export PYTHONUNBUFFERED=1; PORT=9090
log(){ echo "[b3] $1 $(date '+%F %T')"; }
kill_server(){ pkill -9 -f "vllm serve" 2>/dev/null||true; pkill -9 -f "EngineCore" 2>/dev/null||true; for i in $(seq 1 30); do curl -sf http://localhost:${PORT}/health>/dev/null 2>&1||{ sleep 4; return 0; }; sleep 2; done; }
start_server(){ kill_server; : > logs/vllm_server.log; nohup vllm serve "$1" --host 0.0.0.0 --port ${PORT} --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization "$2" >> logs/vllm_server.log 2>&1 & log "SERVER starting $1"; for i in $(seq 1 300); do if curl -sf http://localhost:${PORT}/health>/dev/null 2>&1; then got=$(curl -sS http://localhost:${PORT}/v1/models --max-time 5 2>/dev/null|python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null||echo ""); [[ "$got" == "$1" ]]&&{ log "SERVER ready"; return 0; }; fi; sleep 5; done; return 1; }
run(){ local n="$1" o="$2"; shift 2; if [[ -d "$o" && -n "$(ls -A "$o" 2>/dev/null)" ]]; then log "SKIP $n"; return; fi; log "START $n"; if "$@">"logs/queue_${n}.log" 2>&1; then log "DONE  $n"; else log "FAIL  $n"; fi; }
start_server "Qwen/Qwen3-8B-FP8" 0.80 || exit 1
for aspect in generic task_specific; do
  run "b3_${aspect}_math500" "logs/b3_${aspect}_math500" \
    python experiments/b3_aspect_sensitivity/run_aspect_ablation.py \
      --ip localhost --port ${PORT} --model-name Qwen/Qwen3-8B-FP8 --seed 42 --batch-size 16 \
      --aspect-config ${aspect} --benchmark math500 --num-samples 500 \
      --num-optimization-epochs 2 --number-of-gens-per-epoch 7 --output-path logs/b3_${aspect}_math500
done
log "ALL DONE"
