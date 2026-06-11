#!/usr/bin/env bash
# run_056_resume_shaped.sh — resume exp_056 for the two PER-TOKEN shaped methods
# only: gtpo_ema_flipped then gtpo_conf (in that order).
#
# Why a separate script: the original run_056.sh ran all 4 methods and wiped
# all train_*.log on pre-flight. grpo (1000 steps) and grpo_s_entropy (OOM @785)
# are already done — we must NOT touch their logs. This script preserves them
# and only cleans the two shaped methods' own outputs.
#
# Both shaped trainers now chunk the per-token-confidence second forward over
# the batch dim (CONF_MICRO_BS, default 2) so the full-vocab fp32 logits tensor
# no longer OOMs the backward. Tag-mask over Search-R1 + Qwen3 tags is active
# for both (format_tag_patterns wired in train.py).
#
# Retrieval server `searchr1_retrieval` must already be UP on :8123 (do NOT
# restart it — index reload is slow).

set -e
set -o pipefail

REPO_ROOT="/mnt/data/aiim_research"
EXP_NAME="exp_056_searchr1_qwen3_grpo_vs_shaped"
EXP_DIR="${REPO_ROOT}/experiments/${EXP_NAME}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var not set}"
RETRIEVER_MODE="${RETRIEVER_MODE:-http}"
RETRIEVAL_URL="${RETRIEVAL_URL:-http://127.0.0.1:8123/retrieve}"
CONF_MICRO_BS="${CONF_MICRO_BS:-2}"

echo "=== [$(date -Is)] Resume exp_056 shaped methods (retriever=${RETRIEVER_MODE}, url=${RETRIEVAL_URL}, conf_micro_bs=${CONF_MICRO_BS}) ==="
echo "    Preserving train_grpo.log and train_grpo_s_entropy.log."

docker run --rm --gpus all \
  --entrypoint /bin/bash \
  --user root \
  --network=host \
  -v /mnt/data:/mnt/data \
  -v "${EXP_DIR}:/workspace/${EXP_NAME}" \
  -e "HF_TOKEN=${HF_TOKEN}" \
  -e "RETRIEVAL_URL=${RETRIEVAL_URL}" \
  -e "CONF_MICRO_BS=${CONF_MICRO_BS}" \
  -e "PYTORCH_ALLOC_CONF=expandable_segments:True" \
  unsloth/unsloth -c "
    set -e
    set -o pipefail
    cd /workspace/${EXP_NAME}
    echo '[setup] Activating base /opt/venv...'
    source /opt/venv/bin/activate
    echo '[setup] Overlay deps...'
    uv pip install -r requirements.txt --quiet
    uv pip install --no-deps --quiet unsloth==2026.3.7 unsloth_zoo
    uv pip install --quiet requests
    python -c 'import unsloth, trl, torch; print(\"unsloth\", unsloth.__version__, \"trl\", trl.__version__, \"torch\", torch.__version__)'

    for M in gtpo_ema_flipped gtpo_conf; do
      echo \"=== [\$(date -Is)] method=\$M — wiping prior in-container artefacts (own only) ===\"
      rm -rf /workspace/${EXP_NAME}/outputs_\$M \
             /workspace/${EXP_NAME}/unsloth_compiled_cache \
             /workspace/${EXP_NAME}/grpo_trainer_lora_model 2>/dev/null || true
      echo \"=== [\$(date -Is)] method=\$M — starting train ===\"
      python train.py --method \$M --retriever ${RETRIEVER_MODE} 2>&1 | tee train_\$M.log
    done
  " 2>&1 | tee "${EXP_DIR}/run_056_resume_shaped.out"

echo ""
echo "=== [$(date -Is)] Restoring ownership to mle:mle ==="
sudo chown -R mle:mle "${EXP_DIR}" 2>&1 || true
echo "=== [$(date -Is)] exp_056 shaped methods COMPLETE ==="
