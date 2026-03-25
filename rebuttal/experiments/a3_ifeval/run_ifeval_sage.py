#!/usr/bin/env python3
"""A3 — IFEval evaluation with SAGE.

Runs the SAGE refinement pipeline on IFEval prompts and scores with the official
instruction_following_eval verifiers for final accuracy.

Usage:
    python run_ifeval_sage.py \
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

from core.ifeval_eval import IFEval
from experiments.sage.solver import process_query, load_prompt, load_configurations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IFEval SAGE evaluation")
    p.add_argument("--ip", type=str, default="localhost")
    p.add_argument("--port", type=int, default=9090)
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B-FP8")
    p.add_argument("--num-samples", type=int, default=541)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument("--continue-from", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-optimization-epochs", type=int, default=2)
    p.add_argument("--number-of-gens-per-epoch", type=int, default=7)
    p.add_argument("--m-min", type=int, default=1,
                   help="Minimum group size for best/worst group formation")
    p.add_argument(
        "--judge-prompt",
        type=str,
        default=str(_REBUTTAL_ROOT / "configs" / "ifeval_judge_prompt.txt"),
    )
    p.add_argument(
        "--judge-config",
        type=str,
        default=str(_REBUTTAL_ROOT / "configs" / "ifeval_judge_config.json"),
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    base_url = f"http://{args.ip}:{args.port}/v1"

    judge_prompt_template = load_prompt(args.judge_prompt)
    judge_configurations = load_configurations(args.judge_config)

    print(f"[A3-SAGE] Running IFEval with SAGE — model: {args.model_name}")

    async def solver(prompt: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        result = await process_query(
            prompt,
            judge_prompt_templates=[judge_prompt_template],
            judge_configurations=[judge_configurations],
            num_optimization_epochs=args.num_optimization_epochs,
            base_url=base_url,
            model_name=args.model_name,
            number_of_gens_per_epoch=args.number_of_gens_per_epoch,
            m_min=args.m_min,
        )
        result["time_s"] = time.perf_counter() - t0
        return result

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

    print(f"\n[A3-SAGE] Final results:")
    print(f"  Prompt accuracy     : {summary['prompt_accuracy']:.4f}")
    print(f"  Instruction accuracy: {summary['instruction_accuracy']:.4f}")
    print(f"  Prompt correct      : {summary['prompt_correct']} / {summary['prompt_total']}")
    print(f"  Instruction correct : {summary['instruction_correct']} / {summary['instruction_total']}")
    print(f"  Wall time           : {wall_elapsed:.1f}s")
    print(f"  Output file         : {summary['output_file']}")


if __name__ == "__main__":
    asyncio.run(main())
