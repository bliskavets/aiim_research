#!/usr/bin/env python3
"""B3 — Aspect sensitivity ablation.

Compares three aspect configurations for the SAGE judge on MATH-500 and IFEval:
  1. default      — multi-aspect (as used in the paper)
  2. generic      — single aspect: "overall quality and correctness"
  3. task_specific — task-tailored aspects

Usage:
    python run_aspect_ablation.py \
        --ip localhost --port 9090 \
        --model-name Qwen/Qwen3-8B-FP8 \
        --aspect-config default \
        --benchmark math500 \
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
from typing import Any, Dict, List

_REBUTTAL_ROOT = Path(__file__).resolve().parents[2]
if str(_REBUTTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_REBUTTAL_ROOT))

from core.math500_eval import Math500Eval
from core.ifeval_eval import IFEval
from core.helpers import ensure_output_dir
from experiments.sage.solver import process_query, load_prompt, load_configurations


# ============================================================
# Aspect configurations
# ============================================================

MATH500_JUDGE_CONFIGS = {
    "default": {
        "judge_prompt": str(_REBUTTAL_ROOT / "configs" / "math500_judge_prompt.txt"),
        "judge_config": str(_REBUTTAL_ROOT / "configs" / "math500_judge_config.json"),
    },
    "generic": {
        "judge_prompt_text": (
            "You are evaluating the overall quality and correctness of a response.\n\n"
            "Problem:\n{prompt}\n\nResponse:\n{answer}\n\n"
            "Evaluate the overall quality and correctness.\n"
            "<verdict>correct</verdict> or <verdict>incorrect</verdict>."
        ),
        "judge_config_dict": {
            "overall_quality": {
                "pos_labels": ["correct"],
                "neg_labels": ["incorrect"],
                "labels": ["correct", "incorrect"],
                "opening_tag": "<verdict>",
                "closing_tag": "</verdict>",
                "weight": 1.0,
            }
        },
    },
    "task_specific": {
        "judge_prompt_text": (
            "You are a mathematics expert. Evaluate this solution across three dimensions:\n"
            "1. Logical coherence of the reasoning\n"
            "2. Calculation accuracy\n"
            "3. Step completeness\n\n"
            "Problem:\n{prompt}\n\nSolution:\n{answer}\n\n"
            "Think step by step, then give your verdict.\n"
            "<verdict>correct</verdict> if all dimensions are satisfied, "
            "<verdict>incorrect</verdict> otherwise."
        ),
        "judge_config_dict": {
            "math_quality": {
                "pos_labels": ["correct"],
                "neg_labels": ["incorrect"],
                "labels": ["correct", "incorrect"],
                "opening_tag": "<verdict>",
                "closing_tag": "</verdict>",
                "weight": 1.0,
            }
        },
    },
}

IFEVAL_JUDGE_CONFIGS = {
    "default": {
        "judge_prompt": str(_REBUTTAL_ROOT / "configs" / "ifeval_judge_prompt.txt"),
        "judge_config": str(_REBUTTAL_ROOT / "configs" / "ifeval_judge_config.json"),
    },
    "generic": {
        "judge_prompt_text": (
            "You are evaluating the overall quality of a response.\n\n"
            "Instruction:\n{prompt}\n\nResponse:\n{answer}\n\n"
            "Does the response satisfy the instruction?\n"
            "<verdict>yes</verdict> or <verdict>no</verdict>."
        ),
        "judge_config_dict": {
            "overall_quality": {
                "pos_labels": ["yes"],
                "neg_labels": ["no"],
                "labels": ["yes", "no"],
                "opening_tag": "<verdict>",
                "closing_tag": "</verdict>",
                "weight": 1.0,
            }
        },
    },
    "task_specific": {
        "judge_prompt_text": (
            "You are evaluating instruction following. Check:\n"
            "1. Format requirements (structure, length, style)\n"
            "2. Content requirements (topics, constraints, keywords)\n"
            "3. Tone and phrasing requirements\n\n"
            "Instruction:\n{prompt}\n\nResponse:\n{answer}\n\n"
            "Analyze each requirement, then give your verdict.\n"
            "<verdict>yes</verdict> if all requirements are met, <verdict>no</verdict> otherwise."
        ),
        "judge_config_dict": {
            "instruction_quality": {
                "pos_labels": ["yes"],
                "neg_labels": ["no"],
                "labels": ["yes", "no"],
                "opening_tag": "<verdict>",
                "closing_tag": "</verdict>",
                "weight": 1.0,
            }
        },
    },
}


def resolve_judge_config(aspect_config: str, benchmark: str):
    """Return (judge_prompt_template_str, judge_configurations_dict) for the given config."""
    configs = MATH500_JUDGE_CONFIGS if benchmark == "math500" else IFEVAL_JUDGE_CONFIGS
    cfg = configs[aspect_config]

    if "judge_prompt" in cfg:
        judge_prompt_template = load_prompt(cfg["judge_prompt"])
    else:
        judge_prompt_template = cfg["judge_prompt_text"]

    if "judge_config" in cfg:
        judge_configurations = load_configurations(cfg["judge_config"])
    else:
        judge_configurations = cfg["judge_config_dict"]

    return judge_prompt_template, judge_configurations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAGE aspect sensitivity ablation")
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
        "--aspect-config",
        type=str,
        choices=["default", "generic", "task_specific"],
        default="default",
        help="Aspect configuration to use for the judge",
    )
    p.add_argument(
        "--benchmark",
        type=str,
        choices=["math500", "ifeval"],
        default="math500",
    )
    return p.parse_args()


async def run_math500(args: argparse.Namespace, judge_prompt_template: str, judge_configurations: Dict, out_dir: Path) -> Dict:
    base_url = f"http://{args.ip}:{args.port}/v1"

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

    evaluator = Math500Eval(
        number_of_samples_to_test=args.num_samples,
        seed=args.seed,
        output_path=str(out_dir),
        continue_from_file=args.continue_from,
        max_workers=args.batch_size,
    )
    return await evaluator.run_async(solver=solver)


async def run_ifeval(args: argparse.Namespace, judge_prompt_template: str, judge_configurations: Dict, out_dir: Path) -> Dict:
    base_url = f"http://{args.ip}:{args.port}/v1"

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

    evaluator = IFEval(
        number_of_samples_to_test=args.num_samples,
        seed=args.seed,
        output_path=str(out_dir),
        continue_from_file=args.continue_from,
        batch_size=args.batch_size,
    )
    return await evaluator.run_async(solver=solver)


async def main() -> None:
    args = parse_args()
    out_dir = ensure_output_dir(args.output_path) / f"{args.benchmark}_{args.aspect_config}"
    out_dir.mkdir(parents=True, exist_ok=True)

    judge_prompt_template, judge_configurations = resolve_judge_config(args.aspect_config, args.benchmark)

    print(f"[B3] Aspect config: {args.aspect_config}, Benchmark: {args.benchmark}")
    print(f"[B3] Model: {args.model_name}, Samples: {args.num_samples}")

    t0 = time.perf_counter()
    if args.benchmark == "math500":
        summary = await run_math500(args, judge_prompt_template, judge_configurations, out_dir)
        print(f"\n[B3] MATH-500 accuracy ({args.aspect_config}): {summary['accuracy']:.4f}")
        summary["aspect_config"] = args.aspect_config
        summary["benchmark"] = "math500"
    else:
        summary = await run_ifeval(args, judge_prompt_template, judge_configurations, out_dir)
        print(f"\n[B3] IFEval prompt accuracy ({args.aspect_config}): {summary['prompt_accuracy']:.4f}")
        print(f"[B3] IFEval instruction accuracy ({args.aspect_config}): {summary['instruction_accuracy']:.4f}")
        summary["aspect_config"] = args.aspect_config
        summary["benchmark"] = "ifeval"
    summary["wall_time_s"] = time.perf_counter() - t0

    result_path = out_dir / "aspect_ablation_result.json"
    result_path.write_text(json.dumps(summary, indent=2))
    print(f"[B3] Results saved to: {result_path}")


if __name__ == "__main__":
    asyncio.run(main())
