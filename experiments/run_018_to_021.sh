#!/usr/bin/env bash
# run_018_to_021.sh
# Sequential launcher for exp_018 → exp_019 → exp_020 → exp_021
# Each experiment runs to completion before the next starts.
# Usage: HF_TOKEN=<token> bash run_018_to_021.sh

set -euo pipefail

EXPERIMENTS=(
    "exp_018_bigmath_llama_gtpo_ema"
    "exp_019_bigmath_llama_grpos_entropy"
    "exp_020_bigmath_llama_gtpo_entropy"
    "exp_021_bigmath_llama_gtpo_conf"
)

BASE_DIR="/mnt/data/aiim_research/experiments"
LOG_DIR="/mnt/data/logs"
mkdir -p "$LOG_DIR"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not set. Run: HF_TOKEN=<token> bash run_018_to_021.sh"
    exit 1
fi

echo "======================================================================"
echo "Sequential run: exp_018 → exp_021"
echo "Started: $(date)"
echo "======================================================================"

for EXP in "${EXPERIMENTS[@]}"; do
    EXP_DIR="$BASE_DIR/$EXP"
    LOG_FILE="$EXP_DIR/train.log"

    echo ""
    echo "======================================================================"
    echo "STARTING: $EXP"
    echo "Time: $(date)"
    echo "Log: $LOG_FILE"
    echo "======================================================================"

    if [[ ! -f "$EXP_DIR/train.py" ]]; then
        echo "ERROR: $EXP_DIR/train.py not found — skipping"
        continue
    fi

    # GPU memory check before each experiment
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true

    # Run training, tee to log
    HF_TOKEN="$HF_TOKEN" python "$EXP_DIR/train.py" 2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}
    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "ERROR: $EXP exited with code $EXIT_CODE — stopping chain"
        exit $EXIT_CODE
    fi

    echo "FINISHED: $EXP at $(date)"

    # Brief GPU cool-down between experiments
    sleep 10
done

echo ""
echo "======================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "Finished: $(date)"
echo "Experiments: ${EXPERIMENTS[*]}"
echo "======================================================================"
