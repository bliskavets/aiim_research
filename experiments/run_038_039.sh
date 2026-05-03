#!/usr/bin/env bash
# run_038_039.sh — sequential launcher: exp_038 (GRPO baseline) → exp_039 (GTPO-EMA-flipped).
# Both on Qwen3-4B, Big-Math int-2000, bs=4, gens=8, 1000 steps, max_seq=4096.

set -e

HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var not set}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "=== [$(date -Is)] Starting exp_038 ==="
echo "=========================================="
bash "${SCRIPT_DIR}/exp_038_qwen3_bigmath_grpo/run_038.sh"

echo ""
echo "=========================================="
echo "=== [$(date -Is)] Starting exp_039 ==="
echo "=========================================="
bash "${SCRIPT_DIR}/exp_039_qwen3_bigmath_pure_proof_gtpo_ema/run_039.sh"

echo ""
echo "=========================================="
echo "=== [$(date -Is)] ALL DONE (038 + 039) ==="
echo "=========================================="
