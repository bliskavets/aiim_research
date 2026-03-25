#!/usr/bin/env python3
"""B4 — Small model evaluation on IFEval (SAGE + baseline + BoN).

Same methodology as run_small_model_math500.py but targeting IFEval.
Tracks log-probability margin variance as a noise proxy.

Usage:
    python run_small_model_ifeval.py \
        --ip localhost --port 9090 \
        --model-name Qwen/Qwen3-1.7B \
        --num-samples 541 \
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

import numpy as np

_REBUTTAL_ROOT = Path(__file__).resolve().parents[2]
if str(_REBUTTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_REBUTTAL_ROOT))

from core.vllm_client import get_engine
from core.ifeval_eval import IFEval
from core.helpers import ensure_output_dir
from experiments.sage.solver import (
    process_query,
    load_prompt,
    load_configurations,
    score_answers_async,
    score_and_parse,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Small model IFEval: SAGE + baseline + BoN")
    p.add_argument("--ip", type=str, default="localhost")
    p.add_argument("--port", type=int, default=9090)
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--num-samples", type=int, default=541)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument("--continue-from", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-optimization-epochs", type=int, default=2)
    p.add_argument("--n-candidates", type=int, default=7)
    p.add_argument(
        "--methods",
        type=str,
        default="baseline,bon,sage",
        help="Comma-separated methods to run: baseline,bon,sage",
    )
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


def compute_logprob_margin_variance(all_answers: List[Dict[str, Any]]) -> float:
    scores_by_epoch: Dict[int, List[float]] = {}
    for ans in all_answers:
        epoch = ans.get("epoch", -1)
        score = ans.get("final_llm_judge_score", float("nan"))
        scores_by_epoch.setdefault(epoch, []).append(score)
    variances: List[float] = []
    for scores in scores_by_epoch.values():
        if len(scores) > 1:
            variances.append(float(np.var(scores)))
    return float(np.mean(variances)) if variances else 0.0


async def main() -> None:
    args = parse_args()
    base_url = f"http://{args.ip}:{args.port}/v1"
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    judge_prompt_template = load_prompt(args.judge_prompt)
    judge_configurations = load_configurations(args.judge_config)

    print(f"[B4] Small model IFEval — model: {args.model_name}, methods: {methods}")

    engine = get_engine(base_url=base_url, model=args.model_name, timeout=300, type="aug")
    judge_engine = get_engine(base_url=base_url, model=args.model_name, timeout=300, type="aug")

    GEN_PARAMS = {"n": 1, "temperature": 0.0, "top_p": 1.0, "seed": args.seed, "max_tokens": 4096}
    JUDGE_PARAMS = {"n": 1, "temperature": 0.1, "top_p": 0.95, "seed": 7, "max_tokens": 512}

    summaries: Dict[str, Any] = {}

    if "baseline" in methods:
        print("\n--- Baseline ---")

        async def baseline_solver(prompt: str) -> Dict[str, Any]:
            t0 = time.perf_counter()
            texts = await engine.agenerate(prompt, **GEN_PARAMS)
            elapsed = time.perf_counter() - t0
            return {"output": texts[0] if isinstance(texts, list) else texts, "time_s": elapsed}

        evaluator = IFEval(
            number_of_samples_to_test=args.num_samples,
            seed=args.seed,
            output_path=str(ensure_output_dir(args.output_path) / "baseline"),
            continue_from_file=args.continue_from,
            batch_size=args.batch_size,
        )
        summaries["baseline"] = await evaluator.run_async(solver=baseline_solver)

    if "bon" in methods:
        print("\n--- BoN ---")

        async def bon_solver(prompt: str) -> Dict[str, Any]:
            t0 = time.perf_counter()
            params = {**GEN_PARAMS, "n": args.n_candidates}
            texts = await engine.agenerate(prompt, **params)
            if isinstance(texts, str):
                texts = [texts]
            candidate_answers = [{"answer": t} for t in texts]
            await score_answers_async(
                judge_engine=judge_engine,
                judge_prompt_template=judge_prompt_template,
                instruction=prompt,
                answers=candidate_answers,
                gen_params_judge=JUDGE_PARAMS,
            )
            score_and_parse(candidate_answers, judge_configurations, [judge_prompt_template])
            best = max(candidate_answers, key=lambda a: a.get("final_llm_judge_score", float("-inf")))
            return {"output": best["answer"], "time_s": time.perf_counter() - t0}

        evaluator = IFEval(
            number_of_samples_to_test=args.num_samples,
            seed=args.seed,
            output_path=str(ensure_output_dir(args.output_path) / "bon"),
            continue_from_file=args.continue_from,
            batch_size=args.batch_size,
        )
        summaries["bon"] = await evaluator.run_async(solver=bon_solver)

    if "sage" in methods:
        print("\n--- SAGE ---")

        logprob_variances: List[float] = []

        async def sage_solver(prompt: str) -> Dict[str, Any]:
            t0 = time.perf_counter()
            result = await process_query(
                prompt,
                judge_prompt_templates=[judge_prompt_template],
                judge_configurations=[judge_configurations],
                num_optimization_epochs=args.num_optimization_epochs,
                base_url=base_url,
                model_name=args.model_name,
                number_of_gens_per_epoch=args.n_candidates,
            )
            elapsed = time.perf_counter() - t0
            var = compute_logprob_margin_variance(result.get("all_answers", []))
            logprob_variances.append(var)
            result["time_s"] = elapsed
            result["logprob_margin_variance"] = var
            return result

        evaluator = IFEval(
            number_of_samples_to_test=args.num_samples,
            seed=args.seed,
            output_path=str(ensure_output_dir(args.output_path) / "sage"),
            continue_from_file=args.continue_from,
            batch_size=args.batch_size,
        )
        sage_summary = await evaluator.run_async(solver=sage_solver)
        if logprob_variances:
            sage_summary["mean_logprob_margin_variance"] = float(np.mean(logprob_variances))
        summaries["sage"] = sage_summary

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"{'Method':<12} {'Prompt Acc':>12} {'Instr Acc':>12}")
    print("-" * 70)
    for method, s in summaries.items():
        if "prompt_accuracy" in s:
            p_acc = s["prompt_accuracy"]
            i_acc = s["instruction_accuracy"]
        else:
            p_acc = i_acc = float("nan")
        noise = ""
        if method == "sage" and "mean_logprob_margin_variance" in s:
            noise = f"  noise_var={s['mean_logprob_margin_variance']:.4f}"
        print(f"{method:<12} {p_acc:>12.4f} {i_acc:>12.4f}{noise}")
    print("=" * 70)

    out_dir = ensure_output_dir(args.output_path)
    summary_path = out_dir / "small_model_ifeval_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, default=str))
    print(f"\n[B4] Summary saved to: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
