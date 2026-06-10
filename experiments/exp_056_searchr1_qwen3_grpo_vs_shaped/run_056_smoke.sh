#!/usr/bin/env bash
# run_056_smoke.sh — short GPU smoke test of the Search-R1 pipeline.
# Single method (grpo), stub retriever, 30 steps. Validates that:
#   - SearchR1GRPOTrainer integrates with TRL+vLLM end-to-end
#   - multi-turn rollouts produce non-empty completions
#   - reward_em runs over the rollouts
#   - gradient step actually updates the model (KL > 0)
# Output goes to train_grpo_smoke.log so it doesn't clobber a real run.

set -e
set -o pipefail

REPO_ROOT="/mnt/data/aiim_research"
EXP_NAME="exp_056_searchr1_qwen3_grpo_vs_shaped"
EXP_DIR="${REPO_ROOT}/experiments/${EXP_NAME}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var not set}"

echo "=== [$(date -Is)] Smoke-test pre-flight ==="
sudo rm -rf \
  "${EXP_DIR}/unsloth_compiled_cache" \
  "${EXP_DIR}/grpo_trainer_lora_model" \
  "${EXP_DIR}/outputs_grpo_smoke" \
  "${EXP_DIR}/train_grpo_smoke.log" 2>/dev/null || true

docker run --rm --gpus all \
  --entrypoint /bin/bash \
  --user root \
  --network=host \
  -v /mnt/data:/mnt/data \
  -v "${EXP_DIR}:/workspace/${EXP_NAME}" \
  -e "HF_TOKEN=${HF_TOKEN}" \
  -e "SMOKE_MAX_STEPS=30" \
  -e "SMOKE_SUBSET=64" \
  -e "SMOKE_NUM_GEN=4" \
  unsloth/unsloth -c "
    set -e
    set -o pipefail
    cd /workspace/${EXP_NAME}
    source /opt/venv/bin/activate
    uv pip install -r requirements.txt --quiet
    uv pip install --no-deps --quiet unsloth==2026.3.7 unsloth_zoo
    uv pip install --quiet requests
    echo \"[smoke] env: SMOKE_MAX_STEPS=\$SMOKE_MAX_STEPS SUBSET=\$SMOKE_SUBSET NUM_GEN=\$SMOKE_NUM_GEN\"
    python -u train.py --method grpo --retriever stub 2>&1 | tee train_grpo_smoke.log
  " 2>&1 | tee "${EXP_DIR}/run_056_smoke.out"

echo ""
echo "=== [$(date -Is)] Smoke COMPLETE ==="
echo "Inspect: ${EXP_DIR}/train_grpo_smoke.log"
