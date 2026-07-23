#!/usr/bin/env bash
# Launch an OpenAI-compatible vLLM server for SAGE.
# Usage: bash scripts/serve_vllm.sh [MODEL] [PORT] [MAX_LEN] [GPU_UTIL]
set -euo pipefail
MODEL="${1:-Qwen/Qwen3-8B-FP8}"
PORT="${2:-9090}"
MAX_LEN="${3:-32768}"
GPU_UTIL="${4:-0.85}"

# SAGE reads token log-probabilities from the /v1/completions endpoint, so the
# server must expose it (vLLM does by default). A large context lets the judge
# read long candidates without truncation.
exec vllm serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL"
