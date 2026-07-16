#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /root/aiim/.venv/bin/activate; source /root/aiim/.env.session
export PYTHONUNBUFFERED=1; PORT=9090
MJ="--judge-prompt configs/math500_judge_prompt.txt --judge-config configs/math500_judge_config.json"
log(){ echo "[b4r] $1 $(date '+%F %T')"; }
kill_server(){ pkill -9 -f "vllm serve" 2>/dev/null||true; pkill -9 -f "EngineCore" 2>/dev/null||true; for i in $(seq 1 30); do curl -sf http://localhost:${PORT}/health>/dev/null 2>&1||{ sleep 4; return 0; }; sleep 2; done; }
start_server(){ kill_server; : > logs/vllm_server.log; nohup vllm serve "$1" --host 0.0.0.0 --port ${PORT} --tensor-parallel-size 1 --max-model-len "$3" --gpu-memory-utilization "$2" >> logs/vllm_server.log 2>&1 & log "SERVER starting $1"; for i in $(seq 1 300); do if curl -sf http://localhost:${PORT}/health>/dev/null 2>&1; then got=$(curl -sS http://localhost:${PORT}/v1/models --max-time 5 2>/dev/null|python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null||echo ""); [[ "$got" == "$1" ]]&&{ log "SERVER ready $1"; return 0; }; fi; grep -qE "EngineCore failed|Engine core init|OutOfMemory|ValueError" logs/vllm_server.log&&{ log "SERVER FAILED"; return 1; }; sleep 5; done; return 1; }
run(){ local n="$1" o="$2"; shift 2; if [[ -d "$o" && -n "$(ls -A "$o" 2>/dev/null)" ]]; then log "SKIP $n"; return; fi; log "START $n"; if "$@">"logs/queue_${n}.log" 2>&1; then log "DONE  $n"; else log "FAIL  $n"; fi; }
log "waiting for run_last ALL DONE"
until grep -q "\[last\] ALL DONE" logs/last_queue.log 2>/dev/null; do sleep 60; done
rm -rf logs/b4_small_math500
if start_server "Qwen/Qwen3-1.7B-FP8" 0.40 32768; then SMALL="Qwen/Qwen3-1.7B-FP8"; else start_server "Qwen/Qwen3-1.7B" 0.40 32768 && SMALL="Qwen/Qwen3-1.7B" || SMALL=""; fi
[[ -n "$SMALL" ]] && run "b4_math" "logs/b4_small_math500" python experiments/b4_small_model/run_small_model_math500.py --ip localhost --port ${PORT} --model-name "$SMALL" --seed 42 --batch-size 16 --num-samples 500 --num-optimization-epochs 2 --n-candidates 7 ${MJ} --output-path logs/b4_small_math500
log "ALL DONE"
