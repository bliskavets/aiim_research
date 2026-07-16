#!/usr/bin/env bash
# Consolidated re-run of everything that failed due to the broken (fuser-based)
# server restart: TPO x2 + scoring, E9 Skywork x2 (8B @0.70 so the RM fits),
# B4 x2 (1.7B), C1 (32B). Fixed kill_server uses pkill (fuser is not installed here)
# and verifies the SERVED model matches what was requested (the original bug was a
# health check passing against a stale server).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /root/aiim/.venv/bin/activate
source /root/aiim/.env.session
export PYTHONUNBUFFERED=1
PORT=9090
TPO_DIR=/root/aiim/external/TPO
MJ="--judge-prompt configs/math500_judge_prompt.txt --judge-config configs/math500_judge_config.json"
IJ="--judge-prompt configs/ifeval_judge_prompt.txt --judge-config configs/ifeval_judge_config.json"

log() { echo "[rem] $1 $(date '+%F %T')"; }

kill_server() {
  pkill -9 -f "vllm serve" 2>/dev/null || true
  pkill -9 -f "EngineCore" 2>/dev/null || true
  for i in $(seq 1 30); do
    curl -sf http://localhost:${PORT}/health >/dev/null 2>&1 || { sleep 3; return 0; }
    sleep 2
  done
}

start_server() {  # model, util, maxlen
  local model="$1" util="$2" maxlen="$3"
  kill_server
  : > logs/vllm_server.log
  nohup vllm serve "$model" --host 0.0.0.0 --port ${PORT} --tensor-parallel-size 1 \
      --max-model-len "$maxlen" --gpu-memory-utilization "$util" >> logs/vllm_server.log 2>&1 &
  log "SERVER starting $model util=$util"
  for i in $(seq 1 300); do
    if curl -sf http://localhost:${PORT}/health >/dev/null 2>&1; then
      local got
      got=$(curl -sS http://localhost:${PORT}/v1/models --max-time 5 2>/dev/null \
            | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "")
      if [[ "$got" == "$model" ]]; then log "SERVER ready $model"; return 0; fi
      log "SERVER health up but wrong model ($got != $model); waiting"
    fi
    grep -qE "EngineCore failed|Engine core initialization failed|OutOfMemory" logs/vllm_server.log && { log "SERVER FAILED $model"; return 1; }
    sleep 5
  done
  log "SERVER TIMEOUT $model"; return 1
}

run() { local name="$1" out="$2"; shift 2; if [[ -d "$out" && -n "$(ls -A "$out" 2>/dev/null)" ]]; then log "SKIP $name"; return; fi
  log "START $name"; if "$@" > "logs/queue_${name}.log" 2>&1; then log "DONE  $name"; else log "FAIL  $name"; fi; }

# ---- Phase A: 8B @0.70 (room for reward model) : TPO + E9 ----
start_server "Qwen/Qwen3-8B-FP8" 0.70 32768 || exit 1
export PYTHONPATH="${TPO_DIR}/textgrad-main:${PYTHONPATH:-}"
mkdir -p "${TPO_DIR}/results_aaai"

run "tpo_mmlu" "${TPO_DIR}/results_aaai/done_mmlu" bash -c "cd ${TPO_DIR} && python run.py \
  --data_path data_aaai/mmlu_pro_stem_s42.json --output_path results_aaai --ip localhost --port ${PORT} \
  --server_model server-Qwen/Qwen3-8B-FP8 --reward_model sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --tpo_mode tpo --sample_size 5 --max_iterations 2 --seed 42 --max_tokens_response 2048 --max_tokens_all 16384 --num_threads 16 \
  && mkdir -p results_aaai/done_mmlu && touch results_aaai/done_mmlu/ok"
run "tpo_ifeval" "${TPO_DIR}/results_aaai/done_ifeval" bash -c "cd ${TPO_DIR} && python run.py \
  --data_path data_aaai/ifeval_s42.json --output_path results_aaai --ip localhost --port ${PORT} \
  --server_model server-Qwen/Qwen3-8B-FP8 --reward_model sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --tpo_mode tpo --sample_size 5 --max_iterations 2 --seed 42 --max_tokens_response 2048 --max_tokens_all 16384 --num_threads 16 \
  && mkdir -p results_aaai/done_ifeval && touch results_aaai/done_ifeval/ok"
for t in mmlu ifeval; do
  if [[ "$t" == "mmlu" ]]; then p=mmlu_pro_stem_s42; else p=ifeval_s42; fi
  f=$(ls -t ${TPO_DIR}/results_aaai/${p}*seed42.json 2>/dev/null | head -1)
  [[ -n "${f:-}" ]] && python ${TPO_DIR}/score_tpo.py --task $t --tpo-output "$f" \
      --prompts ${TPO_DIR}/data_aaai/${p}.json --gold ${TPO_DIR}/data_aaai/${p}_gold.json \
      > logs/queue_tpo_${t}_score.log 2>&1 && log "TPO $t scored: $(grep accuracy logs/queue_tpo_${t}_score.log | tail -1)"
done

run "e9_rm_math" "logs/c2_updated_rm_math500" python experiments/c2_updated_rm/run_updated_rm.py \
  --ip localhost --port ${PORT} --model-name Qwen/Qwen3-8B-FP8 --seed 42 --batch-size 8 \
  --benchmark math500 --reward-model Skywork/Skywork-Reward-Qwen-2.5-7B-v0.2 \
  --num-samples 500 --n-candidates 7 --output-path logs/c2_updated_rm_math500
run "e9_rm_ifeval" "logs/c2_updated_rm_ifeval" python experiments/c2_updated_rm/run_updated_rm.py \
  --ip localhost --port ${PORT} --model-name Qwen/Qwen3-8B-FP8 --seed 42 --batch-size 8 \
  --benchmark ifeval --reward-model Skywork/Skywork-Reward-Qwen-2.5-7B-v0.2 \
  --num-samples 541 --n-candidates 7 --output-path logs/c2_updated_rm_ifeval

# ---- Phase B: 1.7B : B4 ----
if start_server "Qwen/Qwen3-1.7B-FP8" 0.85 16384; then SMALL="Qwen/Qwen3-1.7B-FP8";
else start_server "Qwen/Qwen3-1.7B" 0.85 16384 && SMALL="Qwen/Qwen3-1.7B" || SMALL=""; fi
if [[ -n "$SMALL" ]]; then
  run "b4_math" "logs/b4_small_math500" python experiments/b4_small_model/run_small_model_math500.py \
    --ip localhost --port ${PORT} --model-name "$SMALL" --seed 42 --batch-size 16 \
    --num-samples 500 --num-optimization-epochs 2 --n-candidates 7 ${MJ} --output-path logs/b4_small_math500
  run "b4_ifeval" "logs/b4_small_ifeval" python experiments/b4_small_model/run_small_model_ifeval.py \
    --ip localhost --port ${PORT} --model-name "$SMALL" --seed 42 --batch-size 16 \
    --num-samples 541 --num-optimization-epochs 2 --n-candidates 7 ${IJ} --output-path logs/b4_small_ifeval
fi

# ---- Phase C: 32B : C1 ----
if start_server "Qwen/Qwen3-32B-FP8" 0.90 32768; then
  run "c1_32b_math" "logs/c1_qwen32b_math500" python experiments/c1_qwen32b/run_qwen32b.py \
    --ip localhost --port ${PORT} --model-name Qwen/Qwen3-32B-FP8 --seed 42 --batch-size 16 \
    --benchmark math500 --num-samples 500 --num-optimization-epochs 2 --number-of-gens-per-epoch 7 \
    --judge-prompt-math500 configs/math500_judge_prompt.txt --judge-config-math500 configs/math500_judge_config.json \
    --output-path logs/c1_qwen32b_math500
fi

# ---- restore default 8B server for any further interactive work ----
start_server "Qwen/Qwen3-8B-FP8" 0.90 32768 || true
log "ALL DONE"
