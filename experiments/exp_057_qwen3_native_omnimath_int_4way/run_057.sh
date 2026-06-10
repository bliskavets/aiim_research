#!/usr/bin/env bash
# run_057.sh — exp_057: Qwen3-4B GRPO + 3 shaped candidates (tag-masked) on the
# Omni-MATH integer-answer subset (1971 problems), Qwen3 native format. Same
# model/methods/hyperparameters as exp_055 — only the dataset changes.
#
# NOTE: this machine is itself an unprivileged docker container with the GPU
# passed through (no docker-in-docker, no /mnt/data). So unlike exp_055's
# run_055.sh, we run NATIVELY in the prebuilt uv venv — no docker wrapper.
# Stack (validated): unsloth 2026.3.7, trl 0.23.1, torch 2.9.1+cu128,
# vllm 0.16.0, transformers 4.57.6, numpy 2.2.6 on NVIDIA H200 143GB.

set -e
set -o pipefail   # so an OOM in python is not swallowed by `tee`

VENV="/root/aiim/venv"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_TOKEN:?HF_TOKEN env var not set}"

echo "=== [$(date -Is)] Pre-flight: wiping exp_057's own previous artefacts ==="
rm -rf \
  "${EXP_DIR}/unsloth_compiled_cache" \
  "${EXP_DIR}/grpo_trainer_lora_model" \
  "${EXP_DIR}"/outputs_* \
  "${EXP_DIR}"/train_*.log 2>/dev/null || true

source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1

echo "[versions]"
python -c 'import unsloth, trl, torch, numpy, vllm, transformers; print("unsloth", unsloth.__version__, "trl", trl.__version__, "torch", torch.__version__, "numpy", numpy.__version__, "vllm", vllm.__version__, "transformers", transformers.__version__)'

cd "${EXP_DIR}"
for M in grpo grpo_s_entropy gtpo_conf gtpo_ema_flipped; do
  echo "=== [$(date -Is)] method=$M — wiping prior in-place artefacts ==="
  rm -rf "${EXP_DIR}/outputs_$M" \
         "${EXP_DIR}/unsloth_compiled_cache" \
         "${EXP_DIR}/grpo_trainer_lora_model" 2>/dev/null || true
  echo "=== [$(date -Is)] method=$M — starting train ==="
  python train.py --method "$M" 2>&1 | tee "train_$M.log"
done

echo ""
echo "=== [$(date -Is)] exp_057 COMPLETE ==="
