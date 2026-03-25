#!/usr/bin/env python3
"""A3 — IFEval single-generation baseline.

Usage:
    python run_ifeval_baseline.py \
        --ip localhost --port 9090 \
        --model-name Qwen/Qwen3-8B-FP8 \
        --num-samples 541 \
        --output-path /path/to/outputs
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict

_REBUTTAL_ROOT = Path(__file__).resolve().parents[2]
if str(_REBUTTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_REBUTTAL_ROOT))

from core.vllm_client import get_engine
from core.ifeval_eval import IFEval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IFEval single-generation baseline")
    p.add_argument("--ip", type=str, default="localhost")
    p.add_argument("--port", type=int, default=9090)
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B-FP8")
    p.add_argument("--num-samples", type=int, default=541)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument("--continue-from", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    base_url = f"http://{args.ip}:{args.port}/v1"

    print(f"[A3-baseline] Running IFEval baseline — model: {args.model_name}")

    engine = get_engine(base_url=base_url, model=args.model_name, timeout=180.0, type="aug")

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
        }

    evaluator = IFEval(
        number_of_samples_to_test=args.num_samples,
        seed=args.seed,
        output_path=args.output_path,
        continue_from_file=args.continue_from,
        batch_size=args.batch_size,
    )

    wall_start = time.perf_counter()
    summary = await evaluator.run_async(solver=solver)
    wall_elapsed = time.perf_counter() - wall_start

    print(f"\n[A3-baseline] Final results:")
    print(f"  Prompt accuracy     : {summary['prompt_accuracy']:.4f}")
    print(f"  Instruction accuracy: {summary['instruction_accuracy']:.4f}")
    print(f"  Prompt correct      : {summary['prompt_correct']} / {summary['prompt_total']}")
    print(f"  Instruction correct : {summary['instruction_correct']} / {summary['instruction_total']}")
    print(f"  Wall time           : {wall_elapsed:.1f}s")
    print(f"  Output file         : {summary['output_file']}")


if __name__ == "__main__":
    asyncio.run(main())
