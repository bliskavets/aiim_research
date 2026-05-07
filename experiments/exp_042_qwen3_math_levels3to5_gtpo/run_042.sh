#!/usr/bin/env bash
set -e
EXP_NAME="exp_042_qwen3_math_levels3to5_gtpo"
EXP_DIR="/mnt/data/aiim_research/experiments/${EXP_NAME}"
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
    python train.py
  " 2>&1 | tee "${EXP_DIR}/train.log"
