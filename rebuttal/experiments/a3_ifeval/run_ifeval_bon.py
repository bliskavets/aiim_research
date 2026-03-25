#!/usr/bin/env python3
"""A3 — IFEval Best-of-N with SAGE judge.

Generates N candidates for each IFEval prompt and selects the best one
using the SAGE self-judge. Final accuracy is measured by the official
instruction_following_eval verifiers.

Usage:
    python run_ifeval_bon.py \
        --ip localhost --port 9090 \
        --model-name Qwen/Qwen3-8B-FP8 \
        --num-samples 541 \
        --n-candidates 7 \
        --output-path /path/to/outputs
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_REBUTTAL_ROOT = Path(__file__).resolve().parents[2]
if str(_REBUTTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_REBUTTAL_ROOT))

from core.vllm_client import get_engine
from core.ifeval_eval import IFEval
from experiments.sage.solver import (
    load_prompt,
    load_configurations,
    score_answers_async,
    score_and_parse,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IFEval Best-of-N with SAGE judge")
    p.add_argument("--ip", type=str, default="localhost")
    p.add_argument("--port", type=int, default=9090)
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B-FP8")
    p.add_argument("--num-samples", type=int, default=541)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument("--continue-from", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-candidates", type=int, default=7,
                   help="Number of candidates to generate per prompt")
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

    print(f"[A3-BoN] Running IFEval BoN (N={args.n_candidates}) — model: {args.model_name}")

    gen_engine = get_engine(base_url=base_url, model=args.model_name, timeout=300.0, type="aug")
    judge_engine = get_engine(base_url=base_url, model=args.model_name, timeout=300.0, type="aug")

    GEN_PARAMS = {
        "n": args.n_candidates,
        "temperature": 0.8,
        "top_p": 0.95,
        "seed": args.seed,
        "max_tokens": 4096,
    }
    JUDGE_PARAMS = {
        "n": 1,
        "temperature": 0.1,
        "top_p": 0.95,
        "seed": 7,
        "max_tokens": 512,
    }

    async def solver(prompt: str) -> Dict[str, Any]:
        t0 = time.perf_counter()

        texts = await gen_engine.agenerate(prompt, **GEN_PARAMS)
        if isinstance(texts, str):
            texts = [texts]

        candidate_answers: List[Dict[str, Any]] = [{"answer": t} for t in texts]
        await score_answers_async(
            judge_engine=judge_engine,
            judge_prompt_template=judge_prompt_template,
            instruction=prompt,
            answers=candidate_answers,
            gen_params_judge=JUDGE_PARAMS,
        )
        score_and_parse(candidate_answers, judge_configurations, [judge_prompt_template])

        best = max(candidate_answers, key=lambda a: a.get("final_llm_judge_score", float("-inf")))
        elapsed = time.perf_counter() - t0
        return {
            "output": best["answer"],
            "time_s": elapsed,
            "n_candidates": len(texts),
            "best_score": best.get("final_llm_judge_score"),
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

    print(f"\n[A3-BoN] Final results:")
    print(f"  Prompt accuracy     : {summary['prompt_accuracy']:.4f}")
    print(f"  Instruction accuracy: {summary['instruction_accuracy']:.4f}")
    print(f"  Prompt correct      : {summary['prompt_correct']} / {summary['prompt_total']}")
    print(f"  Instruction correct : {summary['instruction_correct']} / {summary['instruction_total']}")
    print(f"  Wall time           : {wall_elapsed:.1f}s")
    print(f"  Output file         : {summary['output_file']}")


if __name__ == "__main__":
    asyncio.run(main())
