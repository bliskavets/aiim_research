#!/usr/bin/env python3
"""C1 — Qwen3-32B: SAGE + baseline on MATH-500 and IFEval.

Tests scalability of SAGE to a 32B parameter model.
Supports tensor-parallel via the vLLM server (just point to the served endpoint).

Usage:
    # Serve Qwen3-32B with tensor parallel first:
    #   vllm serve Qwen/Qwen3-32B --tensor-parallel-size 4 --port 9090

    python run_qwen32b.py \
        --ip localhost --port 9090 \
        --model-name Qwen/Qwen3-32B \
        --benchmark math500 \
        --method sage \
        --num-samples 500 \
        --output-path /path/to/outputs
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

_REBUTTAL_ROOT = Path(__file__).resolve().parents[2]
if str(_REBUTTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_REBUTTAL_ROOT))

from core.vllm_client import get_engine
from core.math500_eval import Math500Eval
from core.ifeval_eval import IFEval
from core.helpers import ensure_output_dir
from experiments.sage.solver import process_query, load_prompt, load_configurations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen3-32B SAGE + baseline on MATH-500 and IFEval")
    p.add_argument("--ip", type=str, default="localhost")
    p.add_argument("--port", type=int, default=9090)
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-32B")
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument("--continue-from", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=2,
                   help="Concurrency (lower for large models to avoid OOM)")
    p.add_argument("--num-optimization-epochs", type=int, default=2)
    p.add_argument("--number-of-gens-per-epoch", type=int, default=7)
    p.add_argument(
        "--benchmark",
        type=str,
        choices=["math500", "ifeval"],
        default="math500",
    )
    p.add_argument(
        "--method",
        type=str,
        choices=["baseline", "sage"],
        default="sage",
    )
    p.add_argument(
        "--judge-prompt-math500",
        type=str,
        default=str(_REBUTTAL_ROOT / "configs" / "math500_judge_prompt.txt"),
    )
    p.add_argument(
        "--judge-config-math500",
        type=str,
        default=str(_REBUTTAL_ROOT / "configs" / "math500_judge_config.json"),
    )
    p.add_argument(
        "--judge-prompt-ifeval",
        type=str,
        default=str(_REBUTTAL_ROOT / "configs" / "ifeval_judge_prompt.txt"),
    )
    p.add_argument(
        "--judge-config-ifeval",
        type=str,
        default=str(_REBUTTAL_ROOT / "configs" / "ifeval_judge_config.json"),
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    base_url = f"http://{args.ip}:{args.port}/v1"
    out_dir = ensure_output_dir(args.output_path)

    if args.benchmark == "math500":
        judge_prompt_template = load_prompt(args.judge_prompt_math500)
        judge_configurations = load_configurations(args.judge_config_math500)
    else:
        judge_prompt_template = load_prompt(args.judge_prompt_ifeval)
        judge_configurations = load_configurations(args.judge_config_ifeval)

    print(f"[C1] model={args.model_name}, benchmark={args.benchmark}, method={args.method}")

    GEN_PARAMS = {"n": 1, "temperature": 0.0, "top_p": 1.0, "seed": args.seed, "max_tokens": 4096}
    engine = get_engine(base_url=base_url, model=args.model_name, timeout=600.0, type="aug")

    if args.method == "baseline":
        async def solver(prompt: str) -> Dict[str, Any]:
            t0 = time.perf_counter()
            texts = await engine.agenerate(prompt, **GEN_PARAMS)
            return {
                "output": texts[0] if isinstance(texts, list) else texts,
                "time_s": time.perf_counter() - t0,
            }
    else:
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
            )
            result["time_s"] = time.perf_counter() - t0
            return result

    wall_start = time.perf_counter()
    if args.benchmark == "math500":
        evaluator = Math500Eval(
            number_of_samples_to_test=args.num_samples,
            seed=args.seed,
            output_path=str(out_dir),
            continue_from_file=args.continue_from,
            max_workers=args.batch_size,
        )
        summary = await evaluator.run_async(solver=solver)
        print(f"\n[C1] MATH-500 accuracy: {summary['accuracy']:.4f} "
              f"({summary['num_correct']}/{summary['num_samples']})")
    else:
        evaluator = IFEval(
            number_of_samples_to_test=args.num_samples,
            seed=args.seed,
            output_path=str(out_dir),
            continue_from_file=args.continue_from,
            batch_size=args.batch_size,
        )
        summary = await evaluator.run_async(solver=solver)
        print(f"\n[C1] IFEval prompt accuracy: {summary['prompt_accuracy']:.4f}")
        print(f"[C1] IFEval instruction accuracy: {summary['instruction_accuracy']:.4f}")

    summary["method"] = args.method
    summary["benchmark"] = args.benchmark
    summary["model"] = args.model_name
    summary["wall_time_s"] = time.perf_counter() - wall_start

    result_path = out_dir / f"qwen32b_{args.benchmark}_{args.method}.json"
    result_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[C1] Results saved to: {result_path}")


if __name__ == "__main__":
    asyncio.run(main())
