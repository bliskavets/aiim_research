#!/usr/bin/env bash
# run_049.sh — GRPO + 3 candidates on Big-Math int-2000, identical hyperparameters.
# Runs all four methods sequentially inside ONE unsloth container.
#
# Per-run hygiene: before each method, this experiment's OWN previous outputs
# (checkpoints, unsloth compiled cache, saved lora, prior log) are wiped so
# each method starts from a clean slate. We never touch other containers'
# caches — only this experiment's local artefacts under ${EXP_DIR}.

set -e

REPO_ROOT="/mnt/data/aiim_research"
EXP_NAME="exp_049_bigmath_int2k_candidates"
EXP_DIR="${REPO_ROOT}/experiments/${EXP_NAME}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var not set}"

echo "=== [$(date -Is)] Pre-flight: wiping exp_049's own previous artefacts ==="
sudo rm -rf \
  "${EXP_DIR}/unsloth_compiled_cache" \
  "${EXP_DIR}/grpo_trainer_lora_model" \
  "${EXP_DIR}"/outputs_* \
  "${EXP_DIR}/run_049.out" \
  "${EXP_DIR}"/train_*.log 2>/dev/null || true

echo ""
echo "=== [$(date -Is)] Launching exp_049 (4 methods, sequential) ==="
docker run --rm --gpus all \
  --entrypoint /bin/bash \
  --user root \
  -v /mnt/data:/mnt/data \
  -v "${EXP_DIR}:/workspace/${EXP_NAME}" \
  -e "HF_TOKEN=${HF_TOKEN}" \
  unsloth/unsloth -c "
    set -e
    cd /workspace/${EXP_NAME}
    echo '[setup] Activating base /opt/venv (uv venv --system-site-packages misses /opt/venv on current unsloth image)...'
    source /opt/venv/bin/activate
    echo '[setup] Overlay: numpy<2.3 + unsloth+unsloth_zoo (no-deps)...'
    uv pip install -r requirements.txt --quiet
    uv pip install --no-deps --quiet unsloth==2026.3.7 unsloth_zoo
    echo '[versions]'
    python -c 'import unsloth, trl, torch, numpy, vllm, transformers; print(\"unsloth\", unsloth.__version__, \"trl\", trl.__version__, \"torch\", torch.__version__, \"numpy\", numpy.__version__, \"vllm\", vllm.__version__, \"transformers\", transformers.__version__)'

    for M in grpo grpo_s_entropy gtpo_conf gtpo_ema_flipped; do
      echo \"=== [\$(date -Is)] method=\$M — wiping prior in-container artefacts ===\"
      rm -rf /workspace/${EXP_NAME}/outputs_\$M \
             /workspace/${EXP_NAME}/unsloth_compiled_cache \
             /workspace/${EXP_NAME}/grpo_trainer_lora_model 2>/dev/null || true
      echo \"=== [\$(date -Is)] method=\$M — starting train ===\"
      python train.py --method \$M 2>&1 | tee train_\$M.log
    done
  " 2>&1 | tee "${EXP_DIR}/run_049.out"

echo ""
echo "=== [$(date -Is)] Restoring ownership to mle:mle ==="
sudo chown -R mle:mle "${EXP_DIR}" 2>&1 || true

echo ""
echo "=== [$(date -Is)] exp_049 COMPLETE ==="
