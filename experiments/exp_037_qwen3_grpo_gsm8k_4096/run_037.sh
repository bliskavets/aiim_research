#!/usr/bin/env bash
# run_037.sh — GRPO baseline Qwen3-4B on GSM8K (max_seq=4096).

set -e

REPO_ROOT="/mnt/data/aiim_research"
EXP_NAME="exp_037_qwen3_grpo_gsm8k_4096"
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
    uv venv /tmp/venv_${EXP_NAME} --system-site-packages --quiet
    source /tmp/venv_${EXP_NAME}/bin/activate
    uv pip install 'numpy<2.3' --quiet
    uv pip install --no-deps --quiet unsloth==2026.3.7 unsloth_zoo
    python -c 'import unsloth, trl, torch, numpy, vllm, transformers; print(\"unsloth\", unsloth.__version__, \"trl\", trl.__version__, \"torch\", torch.__version__, \"numpy\", numpy.__version__, \"vllm\", vllm.__version__, \"transformers\", transformers.__version__)'
    python train.py
  " 2>&1 | tee "${EXP_DIR}/train.log"

echo ""
echo "=== [$(date -Is)] Restoring ownership ==="
sudo chown -R mle:mle "${EXP_DIR}" 2>&1 || true

echo "=== [$(date -Is)] exp_037 COMPLETE ==="
