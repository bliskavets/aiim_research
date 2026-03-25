#!/usr/bin/env python3
"""B4 — Small model evaluation on MATH-500 (SAGE + baseline + BoN).

Runs three methods with a small model (default: Qwen3-1.7B) and additionally
tracks log-probability margin variance across epochs as a noise proxy.

Usage:
    python run_small_model_math500.py \
        --ip localhost --port 9090 \
        --model-name Qwen/Qwen3-1.7B \
        --num-samples 500 \
        --output-path /path/to/outputs
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_REBUTTAL_ROOT = Path(__file__).resolve().parents[2]
if str(_REBUTTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_REBUTTAL_ROOT))

from core.vllm_client import get_engine
from core.math500_eval import Math500Eval
from core.helpers import ensure_output_dir, build_prompt, equations_are_equal_new
from experiments.sage.solver import (
    process_query,
    load_prompt,
    load_configurations,
    score_answers_async,
    score_and_parse,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Small model MATH-500: SAGE + baseline + BoN")
    p.add_argument("--ip", type=str, default="localhost")
    p.add_argument("--port", type=int, default=9090)
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--num-samples", type=int, default=500)
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
        default=str(_REBUTTAL_ROOT / "configs" / "math500_judge_prompt.txt"),
    )
    p.add_argument(
        "--judge-config",
        type=str,
        default=str(_REBUTTAL_ROOT / "configs" / "math500_judge_config.json"),
    )
    return p.parse_args()


def compute_logprob_margin_variance(all_answers: List[Dict[str, Any]]) -> float:
    """Compute variance of final_llm_judge_score across epochs as noise proxy."""
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


async def run_baseline(engine, problems, answers, gen_params, out_dir: Path) -> List[dict]:
    records = []
    correct = 0
    for idx, (problem, gt_answer) in enumerate(zip(problems, answers)):
        prompt = build_prompt(problem, None)
        t0 = time.perf_counter()
        texts = await engine.agenerate(prompt, **gen_params)
        elapsed = time.perf_counter() - t0
        output_text = texts[0] if isinstance(texts, list) else texts
        is_correct = equations_are_equal_new(prompt, gt_answer, output_text) == 1
        correct += int(is_correct)
        record = {
            "method": "baseline",
            "index": idx,
            "is_correct": is_correct,
            "time_s": elapsed,
            "running_accuracy": correct / (idx + 1),
        }
        records.append(record)
        print(f"[baseline] {idx+1}/{len(problems)} correct={is_correct} acc={correct/(idx+1):.3f} time={elapsed:.1f}s")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    return records


async def run_bon(engine, judge_engine, problems, answers, gen_params, judge_prompt_template, judge_configurations, n, out_dir: Path) -> List[dict]:
    records = []
    correct = 0
    JUDGE_PARAMS = {"n": 1, "temperature": 0.1, "top_p": 0.95, "seed": 7, "max_tokens": 512}
    for idx, (problem, gt_answer) in enumerate(zip(problems, answers)):
        prompt = build_prompt(problem, None)
        params = {**gen_params, "n": n}
        t0 = time.perf_counter()
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
        elapsed = time.perf_counter() - t0
        best = max(candidate_answers, key=lambda a: a.get("final_llm_judge_score", float("-inf")))
        is_correct = equations_are_equal_new(prompt, gt_answer, best["answer"]) == 1
        correct += int(is_correct)
        records.append({"method": "bon", "index": idx, "is_correct": is_correct, "time_s": elapsed, "running_accuracy": correct / (idx + 1)})
        print(f"[bon] {idx+1}/{len(problems)} correct={is_correct} acc={correct/(idx+1):.3f} time={elapsed:.1f}s")
    (out_dir / "bon_records.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return records


async def run_sage(problems, answers, base_url, model_name, judge_prompt_template, judge_configurations, n_candidates, num_epochs, out_dir: Path) -> List[dict]:
    records = []
    correct = 0
    for idx, (problem, gt_answer) in enumerate(zip(problems, answers)):
        prompt = build_prompt(problem, None)
        t0 = time.perf_counter()
        result = await process_query(
            prompt,
            judge_prompt_templates=[judge_prompt_template],
            judge_configurations=[judge_configurations],
            num_optimization_epochs=num_epochs,
            base_url=base_url,
            model_name=model_name,
            number_of_gens_per_epoch=n_candidates,
        )
        elapsed = time.perf_counter() - t0
        output_text = result.get("output", "")
        is_correct = equations_are_equal_new(prompt, gt_answer, output_text) == 1
        correct += int(is_correct)
        logprob_var = compute_logprob_margin_variance(result.get("all_answers", []))
        records.append({
            "method": "sage",
            "index": idx,
            "is_correct": is_correct,
            "time_s": elapsed,
            "running_accuracy": correct / (idx + 1),
            "logprob_margin_variance": logprob_var,
        })
        print(f"[sage] {idx+1}/{len(problems)} correct={is_correct} acc={correct/(idx+1):.3f} time={elapsed:.1f}s noise_var={logprob_var:.4f}")
    (out_dir / "sage_records.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return records


async def main() -> None:
    args = parse_args()
    base_url = f"http://{args.ip}:{args.port}/v1"
    out_dir = ensure_output_dir(args.output_path)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    judge_prompt_template = load_prompt(args.judge_prompt)
    judge_configurations = load_configurations(args.judge_config)

    print(f"[B4] Small model MATH-500 — model: {args.model_name}, methods: {methods}")

    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500")
    split = "test" if "test" in ds else list(ds.keys())[0]
    all_problems = list(ds[split]["problem"])
    all_answers = list(ds[split]["answer"])

    indices = list(range(len(all_problems)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[: args.num_samples]
    problems = [all_problems[i] for i in indices]
    answers = [all_answers[i] for i in indices]

    GEN_PARAMS = {"n": 1, "temperature": 0.0, "top_p": 1.0, "seed": args.seed, "max_tokens": 4096}
    engine = get_engine(base_url=base_url, model=args.model_name, timeout=300, type="aug")
    judge_engine = get_engine(base_url=base_url, model=args.model_name, timeout=300, type="aug")

    all_results: Dict[str, List[dict]] = {}
    if "baseline" in methods:
        all_results["baseline"] = await run_baseline(engine, problems, answers, GEN_PARAMS, out_dir)
    if "bon" in methods:
        all_results["bon"] = await run_bon(engine, judge_engine, problems, answers, GEN_PARAMS, judge_prompt_template, judge_configurations, args.n_candidates, out_dir)
    if "sage" in methods:
        all_results["sage"] = await run_sage(problems, answers, base_url, args.model_name, judge_prompt_template, judge_configurations, args.n_candidates, args.num_optimization_epochs, out_dir)

    print("\n" + "=" * 60)
    print(f"{'Method':<12} {'Accuracy':>10} {'Mean time (s)':>15}")
    print("-" * 60)
    for method, records in all_results.items():
        if records:
            acc = sum(r["is_correct"] for r in records) / len(records)
            mean_t = sum(r["time_s"] for r in records) / len(records)
            noise = ""
            if method == "sage":
                mean_var = sum(r.get("logprob_margin_variance", 0) for r in records) / len(records)
                noise = f"  noise_var={mean_var:.4f}"
            print(f"{method:<12} {acc:>10.4f} {mean_t:>15.2f}{noise}")
    print("=" * 60)

    summary = {m: {"accuracy": sum(r["is_correct"] for r in recs) / len(recs), "mean_time_s": sum(r["time_s"] for r in recs) / len(recs), "n": len(recs)} for m, recs in all_results.items() if recs}
    (out_dir / "small_model_math500_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
