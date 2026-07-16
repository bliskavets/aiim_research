#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /root/aiim/.venv/bin/activate; source /root/aiim/.env.session
export PYTHONUNBUFFERED=1; PORT=9090
MJ="--judge-prompt configs/math500_judge_prompt.txt --judge-config configs/math500_judge_config.json"
log(){ echo "[tail] $1 $(date '+%F %T')"; }
kill_server(){ pkill -9 -f "vllm serve" 2>/dev/null||true; pkill -9 -f "EngineCore" 2>/dev/null||true; for i in $(seq 1 30); do curl -sf http://localhost:${PORT}/health>/dev/null 2>&1||{ sleep 4; return 0; }; sleep 2; done; }
start_server(){ kill_server; : > logs/vllm_server.log; nohup vllm serve "$1" --host 0.0.0.0 --port ${PORT} --tensor-parallel-size 1 --max-model-len "$3" --gpu-memory-utilization "$2" >> logs/vllm_server.log 2>&1 & log "SERVER starting $1 util=$2"; for i in $(seq 1 360); do if curl -sf http://localhost:${PORT}/health>/dev/null 2>&1; then got=$(curl -sS http://localhost:${PORT}/v1/models --max-time 5 2>/dev/null|python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null||echo ""); [[ "$got" == "$1" ]]&&{ log "SERVER ready $1"; return 0; }; fi; grep -qE "EngineCore failed|Engine core init|OutOfMemory|ValueError" logs/vllm_server.log&&{ log "FAILED $1"; return 1; }; sleep 5; done; return 1; }
run(){ local n="$1" o="$2"; shift 2; if [[ -d "$o" && -n "$(ls -A "$o" 2>/dev/null)" ]]; then log "SKIP $n"; return; fi; log "START $n"; if "$@">"logs/queue_${n}.log" 2>&1; then log "DONE  $n"; else log "FAIL  $n"; fi; }

# 1) 32B baseline (valuable, ~1h)
if start_server "Qwen/Qwen3-32B-FP8" 0.55 32768; then
  run "c1_32b_baseline" "logs/c1_qwen32b_baseline" python experiments/c1_qwen32b/run_qwen32b.py \
    --ip localhost --port ${PORT} --model-name Qwen/Qwen3-32B-FP8 --seed 42 --batch-size 16 \
    --benchmark math500 --method baseline --num-samples 500 \
    --judge-prompt-math500 configs/math500_judge_prompt.txt --judge-config-math500 configs/math500_judge_config.json \
    --output-path logs/c1_qwen32b_baseline
fi

# 2) b4_math (1.7B, best-effort, serial)
rm -rf logs/b4_small_math500
if start_server "Qwen/Qwen3-1.7B-FP8" 0.30 32768; then SMALL="Qwen/Qwen3-1.7B-FP8"; else start_server "Qwen/Qwen3-1.7B" 0.30 32768 && SMALL="Qwen/Qwen3-1.7B" || SMALL=""; fi
[[ -n "$SMALL" ]] && run "b4_math" "logs/b4_small_math500" python experiments/b4_small_model/run_small_model_math500.py \
  --ip localhost --port ${PORT} --model-name "$SMALL" --seed 42 --batch-size 16 \
  --num-samples 500 --num-optimization-epochs 2 --n-candidates 7 ${MJ} --output-path logs/b4_small_math500
log "ALL DONE"
