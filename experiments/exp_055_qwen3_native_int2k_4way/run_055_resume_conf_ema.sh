#!/usr/bin/env bash
# run_055_resume_conf_ema.sh — gtpo_conf + gtpo_ema_flipped only.
# exp_055 grpo and grpo_s_entropy were both stopped early (~738 steps each)
# after both saturated at reward L50 ~+5.8 on the easy Big-Math subset.
# train_grpo.log and train_grpo_s_entropy.log preserved.

set -e
set -o pipefail

REPO_ROOT="/mnt/data/aiim_research"
EXP_NAME="exp_055_qwen3_native_int2k_4way"
EXP_DIR="${REPO_ROOT}/experiments/${EXP_NAME}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var not set}"

echo "=== [$(date -Is)] Pre-flight: wiping ONLY gtpo_conf + gtpo_ema_flipped artefacts ==="
sudo rm -rf \
  "${EXP_DIR}/unsloth_compiled_cache" \
  "${EXP_DIR}/grpo_trainer_lora_model" \
  "${EXP_DIR}/outputs_gtpo_conf" \
  "${EXP_DIR}/outputs_gtpo_ema_flipped" \
  "${EXP_DIR}/train_gtpo_conf.log" \
  "${EXP_DIR}/train_gtpo_ema_flipped.log" 2>/dev/null || true

echo ""
echo "=== [$(date -Is)] Launching exp_055 (gtpo_conf + gtpo_ema_flipped, sequential) ==="
docker run --rm --gpus all \
  --entrypoint /bin/bash \
  --user root \
  -v /mnt/data:/mnt/data \
  -v "${EXP_DIR}:/workspace/${EXP_NAME}" \
  -e "HF_TOKEN=${HF_TOKEN}" \
  unsloth/unsloth -c "
    set -e
    set -o pipefail
    cd /workspace/${EXP_NAME}
    echo '[setup] Activating base /opt/venv...'
    source /opt/venv/bin/activate
    echo '[setup] Overlay: numpy<2.3 + unsloth+unsloth_zoo (no-deps)...'
    uv pip install -r requirements.txt --quiet
    uv pip install --no-deps --quiet unsloth==2026.3.7 unsloth_zoo
    echo '[versions]'
    python -c 'import unsloth, trl, torch, numpy, vllm, transformers; print(\"unsloth\", unsloth.__version__, \"trl\", trl.__version__, \"torch\", torch.__version__, \"numpy\", numpy.__version__, \"vllm\", vllm.__version__, \"transformers\", transformers.__version__)'

    for M in gtpo_conf gtpo_ema_flipped; do
      echo \"=== [\$(date -Is)] method=\$M — wiping prior in-container artefacts ===\"
      rm -rf /workspace/${EXP_NAME}/outputs_\$M \
             /workspace/${EXP_NAME}/unsloth_compiled_cache \
             /workspace/${EXP_NAME}/grpo_trainer_lora_model 2>/dev/null || true
      echo \"=== [\$(date -Is)] method=\$M — starting train ===\"
      python train.py --method \$M 2>&1 | tee train_\$M.log
    done
  " 2>&1 | tee "${EXP_DIR}/run_055_resume_conf_ema.out"

echo ""
echo "=== [$(date -Is)] Restoring ownership to mle:mle ==="
sudo chown -R mle:mle "${EXP_DIR}" 2>&1 || true

echo ""
echo "=== [$(date -Is)] exp_055 gtpo_conf + gtpo_ema_flipped COMPLETE ==="
