#!/usr/bin/env bash
# run_experiment.sh — launch one experiment in Docker
# Usage: bash skills/run_experiment.sh exp_NNN_name [train_script.py]

set -e

EXP_NAME="${1:?Usage: $0 exp_NNN_name [train_script.py]}"
TRAIN_SCRIPT="${2:-train.py}"
REPO_ROOT="/mnt/data/aiim_research"
EXP_DIR="${REPO_ROOT}/experiments/${EXP_NAME}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var not set}"

if [ ! -d "$EXP_DIR" ]; then
  echo "ERROR: $EXP_DIR not found"
  exit 1
fi

echo "=== Clearing Docker caches ==="
for cid in $(docker ps -q 2>/dev/null); do
  docker exec "$cid" bash -c "rm -rf ~/.cache/huggingface/hub ~/.cache/torch 2>/dev/null || true" && \
    echo "  Cleared cache in container $cid"
done

echo ""
echo "=== Launching: $EXP_NAME / $TRAIN_SCRIPT ==="
docker run --rm --gpus all \
  -v /mnt/data:/mnt/data \
  -v "${EXP_DIR}:/workspace/${EXP_NAME}" \
  -e "HF_TOKEN=${HF_TOKEN}" \
  unsloth/unsloth bash -c "
    set -e
    cd /workspace/${EXP_NAME}
    echo '[setup] Creating venv with uv...'
    uv venv /tmp/venv_${EXP_NAME} --quiet
    source /tmp/venv_${EXP_NAME}/bin/activate
    echo '[setup] Installing requirements...'
    uv pip install -r requirements.txt --quiet
    echo '[run] Starting training...'
    python ${TRAIN_SCRIPT} 2>&1 | tee train.log
  "
