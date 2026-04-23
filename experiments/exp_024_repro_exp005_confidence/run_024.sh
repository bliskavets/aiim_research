#!/usr/bin/env bash
# run_024.sh — launch exp_024 (reproduction of exp_005).
#
# Follows skills/new_experiment.md pattern (uv venv + uv pip install -r
# requirements.txt) but with two deviations forced by the current image:
#   1. --entrypoint /bin/bash — the default /usr/local/bin/entrypoint.sh
#      in unsloth/unsloth:latest ends by handing control to supervisord
#      (jupyter/ollama/sshd) and never runs the command passed via CMD.
#   2. requirements.txt pins numpy<2.3 — the base image ships numpy 2.4.1,
#      which is incompatible with numba (required by vllm's ngram proposer).

set -e

REPO_ROOT="/mnt/data/aiim_research"
EXP_NAME="exp_024_repro_exp005_confidence"
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
  # --user root: the image's default uid (1001) cannot write to the
  # host-owned mount (mle:mle, uid 1000). After the run we chown the
  # experiment directory back to mle:mle.
  docker run --rm --gpus all \
    --entrypoint /bin/bash \
    --user root \
    -v /mnt/data:/mnt/data \
    -v "${EXP_DIR}:/workspace/${EXP_NAME}" \
    -e "HF_TOKEN=${HF_TOKEN}" \
    unsloth/unsloth -c "
      set -e
      cd /workspace/${EXP_NAME}
      echo '[setup] Creating venv with uv (system site-packages so base vllm/torch remain visible)...'
      uv venv /tmp/venv_${EXP_NAME} --system-site-packages --quiet
      source /tmp/venv_${EXP_NAME}/bin/activate
      echo '[setup] Overlay: numpy<2.3 + unsloth+unsloth_zoo (no-deps, keep base torch/vllm/transformers)...'
      uv pip install -r requirements.txt --quiet
      uv pip install --no-deps --quiet unsloth==2026.3.7 unsloth_zoo
      echo '[versions]'
      python -c 'import unsloth, trl, torch, numpy, vllm; print(\"unsloth\", unsloth.__version__, \"trl\", trl.__version__, \"torch\", torch.__version__, \"numpy\", numpy.__version__, \"vllm\", vllm.__version__)'
      echo '[run] Starting ${script}...'
      python ${script}
    " 2>&1 | tee "${EXP_DIR}/${log}"
  echo "=== [$(date -Is)] Finished ${script} ==="
}

run_one train_gtpo_conf.py   train_gtpo_conf.log
run_one train_grpo_s_conf.py train_grpo_s_conf.log

echo ""
echo "=== [$(date -Is)] Restoring ownership to mle:mle ==="
sudo chown -R mle:mle "${EXP_DIR}" 2>&1 || true

echo ""
echo "=== [$(date -Is)] exp_024 COMPLETE ==="
