#!/usr/bin/env bash
# run_025.sh — launch exp_025 (pure-proof GTPO-EMA on GSM8K).
# Same runtime pattern as exp_024: --entrypoint /bin/bash to bypass
# supervisord, uv venv with --system-site-packages, unsloth+zoo via --no-deps
# to avoid torch 2.10 / transformers 5.3 dragged in by full resolution,
# --user root for mount write access, chown back to mle:mle at end.

set -e

REPO_ROOT="/mnt/data/aiim_research"
EXP_NAME="exp_025_pure_proof_gtpo_ema"
EXP_DIR="${REPO_ROOT}/experiments/${EXP_NAME}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var not set}"

echo "=== Clearing Docker caches in long-running containers ==="
for cid in $(docker ps -q 2>/dev/null); do
  docker exec "$cid" bash -c "rm -rf ~/.cache/huggingface/hub ~/.cache/torch 2>/dev/null || true" 2>/dev/null && \
    echo "  cleared cache in $cid" || true
done

run_one () {
  local script="$1"
  local log="$2"
  echo ""
  echo "=== [$(date -Is)] Launching ${script} → ${log} ==="
  docker run --rm --gpus all \
    --entrypoint /bin/bash \
    --user root \
    -v /mnt/data:/mnt/data \
    -v "${EXP_DIR}:/workspace/${EXP_NAME}" \
    -e "HF_TOKEN=${HF_TOKEN}" \
    unsloth/unsloth -c "
      set -e
      cd /workspace/${EXP_NAME}
      echo '[setup] Creating venv with uv (system-site-packages)...'
      uv venv /tmp/venv_${EXP_NAME} --system-site-packages --quiet
      source /tmp/venv_${EXP_NAME}/bin/activate
      echo '[setup] Overlay: numpy<2.3 + unsloth+unsloth_zoo (no-deps)...'
      uv pip install -r requirements.txt --quiet
      uv pip install --no-deps --quiet unsloth==2026.3.7 unsloth_zoo
      echo '[versions]'
      python -c 'import unsloth, trl, torch, numpy, vllm, transformers; print(\"unsloth\", unsloth.__version__, \"trl\", trl.__version__, \"torch\", torch.__version__, \"numpy\", numpy.__version__, \"vllm\", vllm.__version__, \"transformers\", transformers.__version__)'
      echo '[run] Starting ${script}...'
      python ${script}
    " 2>&1 | tee "${EXP_DIR}/${log}"
  echo "=== [$(date -Is)] Finished ${script} ==="
}

run_one train_gtpo_ema_proof.py train_gtpo_ema_proof.log

echo ""
echo "=== [$(date -Is)] Restoring ownership to mle:mle ==="
sudo chown -R mle:mle "${EXP_DIR}" 2>&1 || true

echo ""
echo "=== [$(date -Is)] exp_025 COMPLETE ==="
