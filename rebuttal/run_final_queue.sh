#!/usr/bin/env bash
# Final 48h GPU queue: TPO -> E9 (Skywork RM) -> B2 (m_min) -> B4 (1.7B) -> C1 (32B) -> A2-full.
# Waits for the E2 queue to finish first. Handles vLLM server restarts between phases.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /root/aiim/.venv/bin/activate
source /root/aiim/.env.session
export PYTHONUNBUFFERED=1

PORT=9090
MATH_JUDGE="--judge-prompt configs/math500_judge_prompt.txt --judge-config configs/math500_judge_config.json"
IFEVAL_JUDGE="--judge-prompt configs/ifeval_judge_prompt.txt --judge-config configs/ifeval_judge_config.json"
SAGE="--num-optimization-epochs 2 --number-of-gens-per-epoch 7"

log() { echo "[final] $1 $(date '+%F %T')"; }

kill_server() {
  fuser -k ${PORT}/tcp 2>/dev/null || true
  for i in $(seq 1 30); do
    curl -sf http://localhost:${PORT}/health >/dev/null 2>&1 || return 0
    sleep 2
  done
  return 0
}

start_server() {  # model, gpu_util, max_len
  local model="$1" util="$2" maxlen="$3"
  kill_server
  sleep 5
  : > logs/vllm_server.log
  nohup vllm serve "$model" --host 0.0.0.0 --port ${PORT} --tensor-parallel-size 1 \
      --max-model-len "$maxlen" --gpu-memory-utilization "$util" >> logs/vllm_server.log 2>&1 &
  log "SERVER starting $model util=$util"
  for i in $(seq 1 240); do  # up to 20 min (first-time weight download for 1.7B/32B)
    curl -sf http://localhost:${PORT}/health >/dev/null 2>&1 && { log "SERVER ready $model"; return 0; }
    if grep -qE "EngineCore failed|Engine core initialization failed" logs/vllm_server.log; then
      log "SERVER FAILED $model"; return 1
    fi
    sleep 5
  done
  log "SERVER TIMEOUT $model"; return 1
}

run() {  # name, outdir, cmd...
  local name="$1" out="$2"; shift 2
  if [[ -d "$out" && -n "$(ls -A "$out" 2>/dev/null)" ]]; then log "SKIP $name"; return; fi
  log "START $name"
  if "$@" > "logs/queue_${name}.log" 2>&1; then log "DONE  $name"; else log "FAIL  $name (logs/queue_${name}.log)"; fi
}

# ---- wait for E2 queue ----
log "waiting for E2 queue"
until grep -q "\[e2\] ALL DONE" logs/e2_queue.log 2>/dev/null; do sleep 60; done
log "E2 done; starting final queue"

# ---- Phase 1: TPO (needs RM alongside -> reduced server memory) ----
start_server "Qwen/Qwen3-8B-FP8" 0.70 32768 || exit 1
TPO_DIR=/root/aiim/external/TPO
export PYTHONPATH="${TPO_DIR}/textgrad-main:${PYTHONPATH:-}"
mkdir -p "${TPO_DIR}/results_aaai"

run "tpo_mmlu" "${TPO_DIR}/results_aaai/done_mmlu" \
  bash -c "cd ${TPO_DIR} && python run.py \
    --data_path data_aaai/mmlu_pro_stem_s42.json \
    --output_path results_aaai \
    --ip localhost --port ${PORT} \
    --server_model server-Qwen/Qwen3-8B-FP8 \
    --reward_model sfairXC/FsfairX-LLaMA3-RM-v0.1 \
    --tpo_mode tpo --sample_size 5 --max_iterations 2 --seed 42 \
    --max_tokens_response 2048 --max_tokens_all 16384 --num_threads 16 \
  && mkdir -p results_aaai/done_mmlu && touch results_aaai/done_mmlu/ok"

run "tpo_ifeval" "${TPO_DIR}/results_aaai/done_ifeval" \
  bash -c "cd ${TPO_DIR} && python run.py \
    --data_path data_aaai/ifeval_s42.json \
    --output_path results_aaai \
    --ip localhost --port ${PORT} \
    --server_model server-Qwen/Qwen3-8B-FP8 \
    --reward_model sfairXC/FsfairX-LLaMA3-RM-v0.1 \
    --tpo_mode tpo --sample_size 5 --max_iterations 2 --seed 42 \
    --max_tokens_response 2048 --max_tokens_all 16384 --num_threads 16 \
  && mkdir -p results_aaai/done_ifeval && touch results_aaai/done_ifeval/ok"

# score TPO
for t in mmlu ifeval; do
  f=$(ls -t ${TPO_DIR}/results_aaai/*${t}*seed42.json 2>/dev/null | grep -v done | head -1)
  if [[ -n "${f:-}" ]]; then
    if [[ "$t" == "mmlu" ]]; then p=mmlu_pro_stem_s42; else p=ifeval_s42; fi
    python ${TPO_DIR}/score_tpo.py --task $t --tpo-output "$f" \
      --prompts ${TPO_DIR}/data_aaai/${p}.json --gold ${TPO_DIR}/data_aaai/${p}_gold.json \
      > logs/queue_tpo_${t}_score.log 2>&1
    log "TPO $t scored: $(grep accuracy logs/queue_tpo_${t}_score.log | tail -1)"
  fi
done

# ---- Phase 2: E9 Skywork RM baseline (RM loads in-process; server stays at 0.70) ----
run "e9_rm_math" "logs/c2_updated_rm_math500" \
  python experiments/c2_updated_rm/run_updated_rm.py \
    --ip localhost --port ${PORT} --model-name Qwen/Qwen3-8B-FP8 --seed 42 --batch-size 8 \
    --benchmark math500 --reward-model Skywork/Skywork-Reward-Qwen-2.5-7B-v0.2 \
    --num-samples 500 --n-candidates 7 --output-path logs/c2_updated_rm_math500

run "e9_rm_ifeval" "logs/c2_updated_rm_ifeval" \
  python experiments/c2_updated_rm/run_updated_rm.py \
    --ip localhost --port ${PORT} --model-name Qwen/Qwen3-8B-FP8 --seed 42 --batch-size 8 \
    --benchmark ifeval --reward-model Skywork/Skywork-Reward-Qwen-2.5-7B-v0.2 \
    --num-samples 541 --n-candidates 7 --output-path logs/c2_updated_rm_ifeval

# ---- Phase 3: B2 m_min sweep (pure server; bump memory back) ----
start_server "Qwen/Qwen3-8B-FP8" 0.90 32768 || exit 1
run "b2_mmin_sweep" "logs/b2_mmin_248" \
  python experiments/b2_mmin_ablation/run_mmin_sweep.py \
    --ip localhost --port ${PORT} --model-name Qwen/Qwen3-8B-FP8 --seed 42 --batch-size 16 \
    --num-samples 500 --m-min-values "2,4,8" ${MATH_JUDGE} ${SAGE} \
    --output-path logs/b2_mmin_248

# ---- Phase 4: B4 small model 1.7B ----
if start_server "Qwen/Qwen3-1.7B-FP8" 0.85 16384; then SMALL="Qwen/Qwen3-1.7B-FP8";
else start_server "Qwen/Qwen3-1.7B" 0.85 16384 && SMALL="Qwen/Qwen3-1.7B" || SMALL=""; fi
if [[ -n "$SMALL" ]]; then
  run "b4_math" "logs/b4_small_math500" \
    python experiments/b4_small_model/run_small_model_math500.py \
      --ip localhost --port ${PORT} --model-name "$SMALL" --seed 42 --batch-size 16 \
      --num-samples 500 ${MATH_JUDGE} ${SAGE} --output-path logs/b4_small_math500
  run "b4_ifeval" "logs/b4_small_ifeval" \
    python experiments/b4_small_model/run_small_model_ifeval.py \
      --ip localhost --port ${PORT} --model-name "$SMALL" --seed 42 --batch-size 16 \
      --num-samples 541 ${IFEVAL_JUDGE} ${SAGE} --output-path logs/b4_small_ifeval
fi

# ---- Phase 5: C1 Qwen3-32B ----
if start_server "Qwen/Qwen3-32B-FP8" 0.90 32768; then
  run "c1_32b_math" "logs/c1_qwen32b_math500" \
    python experiments/c1_qwen32b/run_qwen32b.py \
      --ip localhost --port ${PORT} --model-name Qwen/Qwen3-32B-FP8 --seed 42 --batch-size 16 \
      --benchmark math500 --num-samples 500 ${MATH_JUDGE} ${SAGE} \
      --output-path logs/c1_qwen32b_math500
fi

# ---- Phase 6: A2 full latency (8B back) ----
start_server "Qwen/Qwen3-8B-FP8" 0.90 32768 || exit 1
run "a2_full" "logs/a2_latency_full" \
  python experiments/a2_latency_table/run_latency_comparison.py \
    --ip localhost --port ${PORT} --model-name Qwen/Qwen3-8B-FP8 --seed 42 \
    --num-samples 100 --n-candidates 7 --sage-epochs 2 ${MATH_JUDGE} \
    --output-path logs/a2_latency_full

log "ALL DONE"
