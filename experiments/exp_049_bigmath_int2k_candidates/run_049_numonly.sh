#!/usr/bin/env bash
# run_049_numonly.sh — same 4 methods as run_049.sh, but with --rewards numonly.
# Only reward_answer_numeric is used (no format_exact / format_approximate /
# answer_exact). Tests how each shaping method behaves when there is no
# tag-format pressure and the only reward signal is "did the model emit the
# correct number anywhere in the completion".

set -e

REPO_ROOT="/mnt/data/aiim_research"
EXP_NAME="exp_049_bigmath_int2k_candidates"
EXP_DIR="${REPO_ROOT}/experiments/${EXP_NAME}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var not set}"

echo "=== [$(date -Is)] Pre-flight: wiping numonly artefacts from previous runs ==="
sudo rm -rf \
  "${EXP_DIR}/unsloth_compiled_cache" \
  "${EXP_DIR}/grpo_trainer_lora_model" \
  "${EXP_DIR}"/outputs_*_numonly \
  "${EXP_DIR}/run_049_numonly.out" \
  "${EXP_DIR}"/train_*_numonly.log 2>/dev/null || true

echo ""
echo "=== [$(date -Is)] Launching exp_049 numonly (4 methods, sequential) ==="
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
      echo \"=== [\$(date -Is)] method=\$M [numonly] — wiping prior in-container artefacts ===\"
      rm -rf /workspace/${EXP_NAME}/outputs_\${M}_numonly \
             /workspace/${EXP_NAME}/unsloth_compiled_cache \
             /workspace/${EXP_NAME}/grpo_trainer_lora_model 2>/dev/null || true
      echo \"=== [\$(date -Is)] method=\$M [numonly] — starting train ===\"
      python train.py --method \$M --rewards numonly 2>&1 | tee train_\${M}_numonly.log
    done
  " 2>&1 | tee "${EXP_DIR}/run_049_numonly.out"

echo ""
echo "=== [$(date -Is)] Restoring ownership to mle:mle ==="
sudo chown -R mle:mle "${EXP_DIR}" 2>&1 || true

echo ""
echo "=== [$(date -Is)] exp_049 numonly COMPLETE ==="
