#!/usr/bin/env bash
# run_058.sh — exp_059: Qwen3-4B-Instruct on hard Big-Math,
# grpo + 3 FIXED shaped candidates (tag-masked) on Big-Math int-2000.
#
# Runs in the project's unsloth/unsloth docker (our host convention), 4 methods
# sequentially in one container. No retrieval / no network needed (plain math).
# Wipes ONLY this experiment's own previous artefacts.

set -e
set -o pipefail

REPO_ROOT="/mnt/data/aiim_research"
EXP_NAME="exp_059_qwen3instruct_bigmath_hard_4way"
EXP_DIR="${REPO_ROOT}/experiments/${EXP_NAME}"
: "${HF_TOKEN:?HF_TOKEN env var not set}"
CONF_MICRO_BS="${CONF_MICRO_BS:-1}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-1000}"   # train.py reads as max_steps
SMOKE_SUBSET="${SMOKE_SUBSET:-2000}"
METHODS="${METHODS:-grpo grpo_s_entropy gtpo_conf gtpo_ema_flipped}"

echo "=== [$(date -Is)] Pre-flight: wiping exp_059's own previous artefacts (logs preserved; each method's tee overwrites its own) ==="
sudo rm -rf \
  "${EXP_DIR}/unsloth_compiled_cache" \
  "${EXP_DIR}/grpo_trainer_lora_model" \
  "${EXP_DIR}"/outputs_* 2>/dev/null || true

echo "=== [$(date -Is)] Launching exp_059 methods=[${METHODS}] (Qwen3-4B-BASE, max_steps=${SMOKE_MAX_STEPS}, conf_micro_bs=${CONF_MICRO_BS}) ==="
docker run --rm --gpus all \
  --entrypoint /bin/bash \
  --user root \
  -v /mnt/data:/mnt/data \
  -v "${EXP_DIR}:/workspace/${EXP_NAME}" \
  -e "HF_TOKEN=${HF_TOKEN}" \
  -e "CONF_MICRO_BS=${CONF_MICRO_BS}" \
  -e "SMOKE_MAX_STEPS=${SMOKE_MAX_STEPS}" \
  -e "SMOKE_SUBSET=${SMOKE_SUBSET}" \
  -e "PYTORCH_ALLOC_CONF=expandable_segments:True" \
  unsloth/unsloth -c "
    set -e
    set -o pipefail
    cd /workspace/${EXP_NAME}
    source /opt/venv/bin/activate
    uv pip install -r requirements.txt --quiet
    uv pip install --no-deps --quiet unsloth==2026.3.7 unsloth_zoo
    python -c 'import unsloth, trl, torch; print(\"unsloth\", unsloth.__version__, \"trl\", trl.__version__, \"torch\", torch.__version__)'
    for M in ${METHODS}; do
      echo \"=== [\$(date -Is)] method=\$M — wiping prior in-container artefacts ===\"
      rm -rf /workspace/${EXP_NAME}/outputs_\$M \
             /workspace/${EXP_NAME}/unsloth_compiled_cache \
             /workspace/${EXP_NAME}/grpo_trainer_lora_model 2>/dev/null || true
      echo \"=== [\$(date -Is)] method=\$M — starting train ===\"
      python train.py --method \$M 2>&1 | tee train_\$M.log
    done
  " 2>&1 | tee "${EXP_DIR}/run_058.out"

echo "=== [$(date -Is)] Restoring ownership ==="
sudo chown -R mle:mle "${EXP_DIR}" 2>&1 || true
echo "=== [$(date -Is)] exp_059 COMPLETE ==="
