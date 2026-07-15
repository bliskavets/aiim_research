#!/usr/bin/env bash
# Re-run of the RM-dependent jobs (TPO x2, E9 Skywork x2) that failed in the first
# final-queue pass because `accelerate` was missing. accelerate is now installed.
# Waits for run_final_queue.sh to finish, then restarts the server at reduced memory
# (so the reward model fits alongside) and runs the four jobs + TPO scoring.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /root/aiim/.venv/bin/activate
source /root/aiim/.env.session
export PYTHONUNBUFFERED=1
PORT=9090
TPO_DIR=/root/aiim/external/TPO

log() { echo "[rmq] $1 $(date '+%F %T')"; }
kill_server() { fuser -k ${PORT}/tcp 2>/dev/null || true; for i in $(seq 1 30); do curl -sf http://localhost:${PORT}/health >/dev/null 2>&1 || return 0; sleep 2; done; }
start_server() {
  kill_server; sleep 5; : > logs/vllm_server.log
  nohup vllm serve "$1" --host 0.0.0.0 --port ${PORT} --tensor-parallel-size 1 \
      --max-model-len "$3" --gpu-memory-utilization "$2" >> logs/vllm_server.log 2>&1 &
  log "SERVER starting $1 util=$2"
  for i in $(seq 1 240); do
    curl -sf http://localhost:${PORT}/health >/dev/null 2>&1 && { log "SERVER ready"; return 0; }
    grep -qE "EngineCore failed|Engine core initialization failed" logs/vllm_server.log && { log "SERVER FAILED"; return 1; }
    sleep 5
  done
  log "SERVER TIMEOUT"; return 1
}
run() { local name="$1" out="$2"; shift 2; if [[ -d "$out" && -n "$(ls -A "$out" 2>/dev/null)" ]]; then log "SKIP $name"; return; fi; log "START $name"; if "$@" > "logs/queue_${name}.log" 2>&1; then log "DONE  $name"; else log "FAIL  $name"; fi; }

log "waiting for final queue ALL DONE"
until grep -q "\[final\] ALL DONE" logs/final_queue.log 2>/dev/null; do sleep 60; done
log "final queue done; starting RM requeue"

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

log "ALL DONE"
