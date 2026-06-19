#!/usr/bin/env bash
# run_058_lenpen_Lsweep.sh — L-sweep for the two length-penalty methods.
#   L in {3096, 2048, 1536} (this order) x {gtpo_ema_lenpen, gtpo_ema_lenpen_gated}
# Sequential (native venv). Each run logs to train_<method>_L<L>.log so all six
# curves survive for the comparison plot. Does NOT touch the 4 base candidates'
# logs nor the L=1024 baseline logs.
set -e; set -o pipefail
VENV="/root/aiim/venv"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_TOKEN:?HF_TOKEN env var not set}"
source "${VENV}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
export SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-300}"
cd "${EXP_DIR}"

for L in 3096 2048 1536; do
  for M in gtpo_ema_lenpen gtpo_ema_lenpen_gated; do
    echo "=== [$(date -Is)] method=$M  L=$L  steps=$SMOKE_MAX_STEPS  starting ==="
    rm -rf "${EXP_DIR}/outputs_$M" "${EXP_DIR}/unsloth_compiled_cache" \
           "${EXP_DIR}/grpo_trainer_lora_model" 2>/dev/null || true
    LENGTH_L="$L" python train.py --method "$M" 2>&1 | tee "train_${M}_L${L}.log"
    echo "=== [$(date -Is)] method=$M  L=$L  DONE ==="
  done
done
echo "=== [$(date -Is)] exp_058 L-sweep COMPLETE ==="
