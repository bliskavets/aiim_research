#!/usr/bin/env bash
# run_053_resume_ema.sh — resume exp_053 after grpo finished but gtpo_ema_flipped
# crashed with a transient NVML init error (host docker GPU passthrough glitch
# between python processes). grpo log is preserved; only gtpo_ema_flipped reruns.

set -e

REPO_ROOT="/mnt/data/aiim_research"
EXP_NAME="exp_053_qwen3_bigmath_xhard_grpo_vs_ema"
EXP_DIR="${REPO_ROOT}/experiments/${EXP_NAME}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var not set}"

echo "=== [$(date -Is)] Pre-flight: wiping ONLY gtpo_ema_flipped artefacts (keep grpo intact) ==="
sudo rm -rf \
  "${EXP_DIR}/unsloth_compiled_cache" \
  "${EXP_DIR}/grpo_trainer_lora_model" \
  "${EXP_DIR}/outputs_gtpo_ema_flipped" \
  "${EXP_DIR}/train_gtpo_ema_flipped.log" 2>/dev/null || true

echo ""
echo "=== [$(date -Is)] Launching exp_053 gtpo_ema_flipped only ==="
docker run --rm --gpus all \
  --entrypoint /bin/bash \
  --user root \
  -v /mnt/data:/mnt/data \
  -v "${EXP_DIR}:/workspace/${EXP_NAME}" \
  -e "HF_TOKEN=${HF_TOKEN}" \
  unsloth/unsloth -c "
    set -e
    cd /workspace/${EXP_NAME}
    echo '[setup] Activating base /opt/venv...'
    source /opt/venv/bin/activate
    echo '[setup] Overlay: numpy<2.3 + unsloth+unsloth_zoo (no-deps)...'
    uv pip install -r requirements.txt --quiet
    uv pip install --no-deps --quiet unsloth==2026.3.7 unsloth_zoo
    echo '[versions]'
    python -c 'import unsloth, trl, torch, numpy, vllm, transformers; print(\"unsloth\", unsloth.__version__, \"trl\", trl.__version__, \"torch\", torch.__version__, \"numpy\", numpy.__version__, \"vllm\", vllm.__version__, \"transformers\", transformers.__version__)'

    M=gtpo_ema_flipped
    echo \"=== [\$(date -Is)] method=\$M — starting train ===\"
    python train.py --method \$M 2>&1 | tee train_\$M.log
  " 2>&1 | tee "${EXP_DIR}/run_053_resume.out"

echo ""
echo "=== [$(date -Is)] Restoring ownership to mle:mle ==="
sudo chown -R mle:mle "${EXP_DIR}" 2>&1 || true

echo ""
echo "=== [$(date -Is)] exp_053 gtpo_ema_flipped COMPLETE ==="
