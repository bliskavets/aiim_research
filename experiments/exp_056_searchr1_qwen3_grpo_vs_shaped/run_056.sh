#!/usr/bin/env bash
# run_056.sh — Search-R1 with 4 shaping candidates on Qwen3-4B.
# Default uses StubRetriever (mock docs). For a real run, first launch the
# retrieval server (see retrieval/README.md), then pass --retriever=http via
# the inner python call.

set -e
set -o pipefail

REPO_ROOT="/mnt/data/aiim_research"
EXP_NAME="exp_056_searchr1_qwen3_grpo_vs_shaped"
EXP_DIR="${REPO_ROOT}/experiments/${EXP_NAME}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var not set}"
RETRIEVER_MODE="${RETRIEVER_MODE:-stub}"      # stub | http
RETRIEVAL_URL="${RETRIEVAL_URL:-http://127.0.0.1:8000/retrieve}"

echo "=== [$(date -Is)] Pre-flight: wiping exp_056's own previous artefacts ==="
sudo rm -rf \
  "${EXP_DIR}/unsloth_compiled_cache" \
  "${EXP_DIR}/grpo_trainer_lora_model" \
  "${EXP_DIR}"/outputs_* \
  "${EXP_DIR}/run_056.out" \
  "${EXP_DIR}"/train_*.log 2>/dev/null || true

echo ""
echo "=== [$(date -Is)] Launching exp_056 (4 methods sequential, retriever=${RETRIEVER_MODE}) ==="
docker run --rm --gpus all \
  --entrypoint /bin/bash \
  --user root \
  --network=host \
  -v /mnt/data:/mnt/data \
  -v "${EXP_DIR}:/workspace/${EXP_NAME}" \
  -e "HF_TOKEN=${HF_TOKEN}" \
  -e "RETRIEVAL_URL=${RETRIEVAL_URL}" \
  -e "PYTORCH_ALLOC_CONF=expandable_segments:True" \
  unsloth/unsloth -c "
    set -e
    set -o pipefail
    cd /workspace/${EXP_NAME}
    echo '[setup] Activating base /opt/venv...'
    source /opt/venv/bin/activate
    echo '[setup] Overlay: numpy<2.3 + unsloth+unsloth_zoo (no-deps) + requests...'
    uv pip install -r requirements.txt --quiet
    uv pip install --no-deps --quiet unsloth==2026.3.7 unsloth_zoo
    uv pip install --quiet requests
    echo '[versions]'
    python -c 'import unsloth, trl, torch, numpy, vllm, transformers; print(\"unsloth\", unsloth.__version__, \"trl\", trl.__version__, \"torch\", torch.__version__, \"numpy\", numpy.__version__, \"vllm\", vllm.__version__, \"transformers\", transformers.__version__)'

    for M in grpo grpo_s_entropy gtpo_conf gtpo_ema_flipped; do
      echo \"=== [\$(date -Is)] method=\$M — wiping prior in-container artefacts ===\"
      rm -rf /workspace/${EXP_NAME}/outputs_\$M \
             /workspace/${EXP_NAME}/unsloth_compiled_cache \
             /workspace/${EXP_NAME}/grpo_trainer_lora_model 2>/dev/null || true
      echo \"=== [\$(date -Is)] method=\$M — starting train ===\"
      python train.py --method \$M --retriever ${RETRIEVER_MODE} 2>&1 | tee train_\$M.log
    done
  " 2>&1 | tee "${EXP_DIR}/run_056.out"

echo ""
echo "=== [$(date -Is)] Restoring ownership to mle:mle ==="
sudo chown -R mle:mle "${EXP_DIR}" 2>&1 || true

echo ""
echo "=== [$(date -Is)] exp_056 COMPLETE ==="
