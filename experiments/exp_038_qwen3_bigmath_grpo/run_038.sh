#!/usr/bin/env bash
# run_038.sh — Qwen3-4B GRPO baseline on Big-Math int-2000 (bs=4, gens=8, 1000 steps).

set -e

REPO_ROOT="/mnt/data/aiim_research"
EXP_NAME="exp_038_qwen3_bigmath_grpo"
EXP_DIR="${REPO_ROOT}/experiments/${EXP_NAME}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var not set}"

echo "=== Clearing Docker caches in long-running containers ==="
for cid in $(docker ps -q 2>/dev/null); do
  docker exec "$cid" bash -c "rm -rf ~/.cache/huggingface/hub ~/.cache/torch 2>/dev/null || true" 2>/dev/null && \
    echo "  cleared cache in $cid" || true
done

echo ""
echo "=== [$(date -Is)] Launching train.py ==="
docker run --rm --gpus all \
  --entrypoint /bin/bash \
  --user root \
  -v /mnt/data:/mnt/data \
  -v "${EXP_DIR}:/workspace/${EXP_NAME}" \
  -e "HF_TOKEN=${HF_TOKEN}" \
  unsloth/unsloth -c "
    set -e
    cd /workspace/${EXP_NAME}
    pip install --no-deps --quiet unsloth==2026.3.7 unsloth_zoo
    python -c 'import unsloth, trl, torch, vllm, transformers; print(\"unsloth\", unsloth.__version__, \"trl\", trl.__version__, \"torch\", torch.__version__, \"vllm\", vllm.__version__, \"transformers\", transformers.__version__)'
    python train.py
  " 2>&1 | tee "${EXP_DIR}/train.log"

echo ""
echo "=== [$(date -Is)] Restoring ownership ==="
sudo chown -R mle:mle "${EXP_DIR}" 2>&1 || true

echo "=== [$(date -Is)] exp_038 COMPLETE ==="
