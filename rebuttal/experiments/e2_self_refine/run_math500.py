#!/usr/bin/env python3
"""E2 - Self-Refine / Reflexion on MATH-500 (matched budget vs SAGE).

Usage:
    python run_math500.py --mode self_refine --num-samples 500 --seed 42 \
        --budget 21 --output-path logs/e2_selfrefine_math_s42
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

from core.math500_eval import Math500Eval
from experiments.e2_self_refine.solver import refine_query


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-Refine/Reflexion on MATH-500")
    p.add_argument("--ip", type=str, default="localhost")
    p.add_argument("--port", type=int, default=9090)
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B-FP8")
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument("--continue-from", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--mode", type=str, default="self_refine",
                   choices=["self_refine", "reflexion"])
    p.add_argument("--budget", type=int, default=21,
                   help="Total solution generations (SAGE default: n_gens*(1+epochs)=21)")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    base_url = f"http://{args.ip}:{args.port}/v1"
    print(f"[E2-{args.mode}] MATH-500 model={args.model_name} budget={args.budget}")

    async def solver(prompt: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        result = await refine_query(
            prompt, base_url=base_url, model_name=args.model_name,
            mode=args.mode, budget=args.budget,
        )
        result["time_s"] = time.perf_counter() - t0
        return result

    evaluator = Math500Eval(
        number_of_samples_to_test=args.num_samples,
        seed=args.seed,
        output_path=args.output_path,
        continue_from_file=args.continue_from,
        max_workers=args.batch_size,
    )
    summary = await evaluator.run_async(solver=solver)
    print(f"\n[E2-{args.mode}] MATH-500 accuracy: {summary['accuracy']:.4f} "
          f"({summary['num_correct']}/{summary['num_samples']})")
    print(f"[E2-{args.mode}] Output file: {summary['output_file']}")


if __name__ == "__main__":
    asyncio.run(main())
