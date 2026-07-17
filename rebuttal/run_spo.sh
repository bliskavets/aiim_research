#!/usr/bin/env bash
# SPO fair re-run (R3-Q1): Self-supervised prompt optimization with Qwen3-8B as the
# optimizer/evaluator/executor (NOT GPT-4.1), then evaluate the optimized prompt on
# MATH-500. Waits for B3 to finish first.
set -uo pipefail
REB="/root/aiim/aiim_research/rebuttal"; SPO="/root/aiim/external/SPO"; PORT=9090
source /root/aiim/.venv/bin/activate; source /root/aiim/.env.session
export PYTHONUNBUFFERED=1
log(){ echo "[spo] $1 $(date '+%F %T')"; }
kill_server(){ pkill -9 -f "vllm serve" 2>/dev/null||true; pkill -9 -f "EngineCore" 2>/dev/null||true; for i in $(seq 1 30); do curl -sf http://localhost:${PORT}/health>/dev/null 2>&1||{ sleep 4; return 0; }; sleep 2; done; }
start_server(){ kill_server; : > $REB/logs/vllm_server.log; nohup vllm serve "$1" --host 0.0.0.0 --port ${PORT} --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization "$2" >> $REB/logs/vllm_server.log 2>&1 & log "SERVER starting $1"; for i in $(seq 1 300); do if curl -sf http://localhost:${PORT}/health>/dev/null 2>&1; then got=$(curl -sS http://localhost:${PORT}/v1/models --max-time 5 2>/dev/null|python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null||echo ""); [[ "$got" == "$1" ]]&&{ log "SERVER ready"; return 0; }; fi; sleep 5; done; return 1; }

log "waiting for B3 ALL DONE"
until grep -q "\[b3\] ALL DONE" $REB/logs/b3_queue.log 2>/dev/null; do sleep 60; done
start_server "Qwen/Qwen3-8B-FP8" 0.80 || exit 1

# 1) SPO optimize with Qwen3-8B (fair)
log "SPO optimize (Qwen3-8B, max-rounds 6)"
cd $SPO
rm -rf workspace_full
python optimize.py --name Math --template Math.yaml \
  --opt-model "Qwen/Qwen3-8B-FP8" --eval-model "Qwen/Qwen3-8B-FP8" --exec-model "Qwen/Qwen3-8B-FP8" \
  --max-rounds 6 --workspace workspace_full > $REB/logs/queue_spo_optimize.log 2>&1 && log "optimize DONE" || log "optimize FAIL"

# 2) find best round prompt
BEST=$(grep -oE "Best Performing Round: [0-9]+" $REB/logs/queue_spo_optimize.log | grep -oE "[0-9]+" | tail -1)
[ -z "$BEST" ] && BEST=$(ls -d workspace_full/Math/prompts/round_*/ 2>/dev/null | sed 's#.*round_##;s#/##' | sort -n | tail -1)
PF="$SPO/workspace_full/Math/prompts/round_${BEST}/prompt.txt"
log "best round=$BEST prompt=$PF"

# 3) eval optimized prompt on MATH-500
cd $REB
if [ -f "$PF" ]; then
  python experiments/thinking/eval_spo_prompt.py --ip localhost --port ${PORT} \
    --model-name Qwen/Qwen3-8B-FP8 --prompt-file "$PF" --num-samples 500 --seed 42 \
    --batch-size 16 --output-path logs/spo_math_eval > logs/queue_spo_eval.log 2>&1 && log "eval DONE" || log "eval FAIL"
  cp "$PF" results_aaai2027/spo_optimized_prompt.txt 2>/dev/null
fi
log "ALL DONE"
