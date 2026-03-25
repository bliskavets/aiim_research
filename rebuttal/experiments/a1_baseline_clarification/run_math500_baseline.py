#!/usr/bin/env python3
"""A1 — Baseline clarification: single-generation on MATH-500.

Runs Qwen3-8B (non-thinking mode by default) on MATH-500 without any SAGE
refinement to establish the baseline accuracy numbers cited in the paper.

Usage:
    python run_math500_baseline.py \
        --ip localhost --port 9090 \
        --model-name Qwen/Qwen3-8B-FP8 \
        --num-samples 500 \
        --output-path /path/to/outputs
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure rebuttal root is importable
_REBUTTAL_ROOT = Path(__file__).resolve().parents[2]
if str(_REBUTTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_REBUTTAL_ROOT))

from core.vllm_client import get_engine
from core.math500_eval import Math500Eval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MATH-500 single-generation baseline")
    p.add_argument("--ip", type=str, default="localhost")
    p.add_argument("--port", type=int, default=9090)
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B-FP8")
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument("--continue-from", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=8,
                   help="Number of problems to evaluate concurrently")
    p.add_argument(
        "--thinking-mode",
        action="store_true",
        default=False,
        help="Use Qwen3 thinking mode (do NOT append /nothink). "
             "Defaults to non-thinking mode.",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    base_url = f"http://{args.ip}:{args.port}/v1"

    engine_type = "no_think" if args.thinking_mode else "aug"
    mode_label = "thinking" if args.thinking_mode else "non-thinking"
    print(f"[A1] Running MATH-500 baseline — model: {args.model_name}, mode: {mode_label}")

    engine = get_engine(
        base_url=base_url,
        model=args.model_name,
        timeout=180.0,
        type=engine_type,
    )

    GEN_PARAMS = {
        "n": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": args.seed,
        "max_tokens": 4096,
    }

    async def solver(prompt: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        texts = await engine.agenerate(prompt, **GEN_PARAMS)
        elapsed = time.perf_counter() - t0
        output_text = texts[0] if isinstance(texts, list) else texts
        return {
            "output": output_text,
            "time_s": elapsed,
            "model": args.model_name,
            "thinking_mode": args.thinking_mode,
        }

    evaluator = Math500Eval(
        number_of_samples_to_test=args.num_samples,
        seed=args.seed,
        output_path=args.output_path,
        continue_from_file=args.continue_from,
        max_workers=args.batch_size,
    )

    wall_start = time.perf_counter()
    summary = await evaluator.run_async(solver=solver)
    wall_elapsed = time.perf_counter() - wall_start

    print(f"\n[A1] Results ({mode_label}):")
    print(f"  Accuracy       : {summary['accuracy']:.4f}")
    print(f"  Correct        : {summary['num_correct']} / {summary['num_samples']}")
    print(f"  Wall time      : {wall_elapsed:.1f}s ({wall_elapsed / max(summary['num_samples'], 1):.2f}s/problem)")
    print(f"  Output file    : {summary['output_file']}")


if __name__ == "__main__":
    asyncio.run(main())
