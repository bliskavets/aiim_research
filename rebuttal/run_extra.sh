#!/usr/bin/env bash
# Extra experiments (24h window): thinking-mode (R4-W1) + XSTest (safety completeness).
# All on Qwen3-8B. One server for all four jobs.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /root/aiim/.venv/bin/activate; source /root/aiim/.env.session
export PYTHONUNBUFFERED=1; PORT=9090
log(){ echo "[extra] $1 $(date '+%F %T')"; }
kill_server(){ pkill -9 -f "vllm serve" 2>/dev/null||true; pkill -9 -f "EngineCore" 2>/dev/null||true; for i in $(seq 1 30); do curl -sf http://localhost:${PORT}/health>/dev/null 2>&1||{ sleep 4; return 0; }; sleep 2; done; }
start_server(){ kill_server; : > logs/vllm_server.log; nohup vllm serve "$1" --host 0.0.0.0 --port ${PORT} --tensor-parallel-size 1 --max-model-len "$3" --gpu-memory-utilization "$2" >> logs/vllm_server.log 2>&1 & log "SERVER starting $1 util=$2"; for i in $(seq 1 300); do if curl -sf http://localhost:${PORT}/health>/dev/null 2>&1; then got=$(curl -sS http://localhost:${PORT}/v1/models --max-time 5 2>/dev/null|python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null||echo ""); [[ "$got" == "$1" ]]&&{ log "SERVER ready $1"; return 0; }; fi; grep -qE "EngineCore failed|Engine core init|OutOfMemory|ValueError" logs/vllm_server.log&&{ log "SERVER FAILED"; return 1; }; sleep 5; done; return 1; }
run(){ local n="$1" o="$2"; shift 2; if [[ -d "$o" && -n "$(ls -A "$o" 2>/dev/null)" ]]; then log "SKIP $n"; return; fi; log "START $n"; if "$@">"logs/queue_${n}.log" 2>&1; then log "DONE  $n"; else log "FAIL  $n"; fi; }

start_server "Qwen/Qwen3-8B-FP8" 0.75 32768 || exit 1

run "think_baseline" "logs/think_baseline_math" python experiments/thinking/run_math500_thinking_baseline.py \
  --ip localhost --port ${PORT} --model-name Qwen/Qwen3-8B-FP8 --num-samples 500 --seed 42 \
  --batch-size 24 --max-tokens 24576 --output-path logs/think_baseline_math

run "think_sage" "logs/think_sage_math" python experiments/thinking/run_math500_thinking_sage.py \
  --ip localhost --port ${PORT} --model-name Qwen/Qwen3-8B-FP8 --num-samples 150 --seed 42 \
  --batch-size 12 --output-path logs/think_sage_math

run "xstest_baseline" "logs/xstest_baseline" python experiments/thinking/run_xstest.py \
  --ip localhost --port ${PORT} --model-name Qwen/Qwen3-8B-FP8 --method baseline --num-samples -1 --seed 42 \
  --batch-size 24 --judge-model "openai/gpt-4.1" --output-path logs/xstest_baseline

run "xstest_sage" "logs/xstest_sage" python experiments/thinking/run_xstest.py \
  --ip localhost --port ${PORT} --model-name Qwen/Qwen3-8B-FP8 --method sage --num-samples -1 --seed 42 \
  --batch-size 12 --judge-model "openai/gpt-4.1" --output-path logs/xstest_sage
log "ALL DONE"
