#!/usr/bin/env python3
"""B2 — m_min ablation: sweep minimum group size on MATH-500.

Runs SAGE with different m_min values and reports accuracy for each.
This ablates the sensitivity of SAGE to the minimum required group size
for best/worst contrastive group formation.

Usage:
    python run_mmin_sweep.py \
        --ip localhost --port 9090 \
        --model-name Qwen/Qwen3-8B-FP8 \
        --num-samples 500 \
        --m-min-values 1,2,4,6,8 \
        --output-path /path/to/outputs
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_REBUTTAL_ROOT = Path(__file__).resolve().parents[2]
if str(_REBUTTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_REBUTTAL_ROOT))

from core.math500_eval import Math500Eval
from core.helpers import ensure_output_dir
from experiments.sage.solver import process_query, load_prompt, load_configurations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAGE m_min ablation sweep on MATH-500")
    p.add_argument("--ip", type=str, default="localhost")
    p.add_argument("--port", type=int, default=9090)
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B-FP8")
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument("--continue-from", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-optimization-epochs", type=int, default=2)
    p.add_argument("--number-of-gens-per-epoch", type=int, default=7)
    p.add_argument(
        "--m-min-values",
        type=str,
        default="1,2,4,6,8",
        help="Comma-separated list of m_min values to sweep",
    )
    p.add_argument(
        "--judge-prompt",
        type=str,
        default=str(_REBUTTAL_ROOT / "configs" / "math500_judge_prompt.txt"),
    )
    p.add_argument(
        "--judge-config",
        type=str,
        default=str(_REBUTTAL_ROOT / "configs" / "math500_judge_config.json"),
    )
    return p.parse_args()


async def run_single_mmin(
    m_min: int,
    args: argparse.Namespace,
    judge_prompt_template: str,
    judge_configurations: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    base_url = f"http://{args.ip}:{args.port}/v1"
    mmin_out_dir = out_dir / f"mmin_{m_min}"
    mmin_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[B2] Running m_min={m_min}")

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
            m_min=m_min,
        )
        result["time_s"] = time.perf_counter() - t0
        result["m_min"] = m_min
        return result

    evaluator = Math500Eval(
        number_of_samples_to_test=args.num_samples,
        seed=args.seed,
        output_path=str(mmin_out_dir),
        continue_from_file=args.continue_from,
        max_workers=args.batch_size,
    )

    t0 = time.perf_counter()
    summary = await evaluator.run_async(solver=solver)
    summary["m_min"] = m_min
    summary["wall_time_s"] = time.perf_counter() - t0

    print(f"[B2] m_min={m_min} accuracy={summary['accuracy']:.4f} "
          f"({summary['num_correct']}/{summary['num_samples']}) "
          f"in {summary['wall_time_s']:.1f}s")
    return summary


async def main() -> None:
    args = parse_args()
    m_min_values = [int(x.strip()) for x in args.m_min_values.split(",") if x.strip()]
    out_dir = ensure_output_dir(args.output_path)

    judge_prompt_template = load_prompt(args.judge_prompt)
    judge_configurations = load_configurations(args.judge_config)

    print(f"[B2] m_min ablation sweep: {m_min_values}")
    print(f"[B2] Model: {args.model_name}, Samples: {args.num_samples}, Seed: {args.seed}")

    all_results: List[Dict[str, Any]] = []
    for m_min in m_min_values:
        result = await run_single_mmin(m_min, args, judge_prompt_template, judge_configurations, out_dir)
        all_results.append(result)

    # Print summary table
    print("\n" + "=" * 55)
    print(f"{'m_min':>6} {'Accuracy':>10} {'Correct':>10} {'Time (s)':>10}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['m_min']:>6} {r['accuracy']:>10.4f} {r['num_correct']:>4}/{r['num_samples']:<4} {r['wall_time_s']:>10.1f}")
    print("=" * 55)

    summary_path = out_dir / "mmin_sweep_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[B2] Summary saved to: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
