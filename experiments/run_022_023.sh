#!/usr/bin/env bash
# run_022_023.sh — Sequential launcher for exp_022 → exp_023
# Usage: HF_TOKEN=<token> bash run_022_023.sh

set -euo pipefail

EXPERIMENTS=(
    "exp_022_bigmath_llama_gtpo_binary"
    "exp_023_bigmath_llama_gtpo_ema_binary"
)

BASE_DIR="/mnt/data/aiim_research/experiments"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not set."
    exit 1
fi

echo "======================================================================"
echo "Sequential run: exp_022 → exp_023"
echo "Started: $(date)"
echo "======================================================================"

for EXP in "${EXPERIMENTS[@]}"; do
    EXP_DIR="$BASE_DIR/$EXP"
    LOG_FILE="$EXP_DIR/train.log"

    echo ""
    echo "STARTING: $EXP at $(date)"
    echo "Log: $LOG_FILE"

    if [[ ! -f "$EXP_DIR/train.py" ]]; then
        echo "ERROR: $EXP_DIR/train.py not found — skipping"
        continue
    fi

    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true

    HF_TOKEN="$HF_TOKEN" python "$EXP_DIR/train.py" 2>&1 | tee "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "ERROR: $EXP exited with code $EXIT_CODE — stopping chain"
        exit $EXIT_CODE
    fi

    echo "FINISHED: $EXP at $(date)"
    sleep 10
done

echo ""
echo "======================================================================"
echo "exp_022 + exp_023 COMPLETE at $(date)"
echo "======================================================================"
