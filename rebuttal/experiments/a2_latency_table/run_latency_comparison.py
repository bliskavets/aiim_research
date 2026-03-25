#!/usr/bin/env python3
"""A2 — Latency table: compare SAGE, BoN (SAGE judge), BoN-random, and baseline.

Runs all four methods on 100 MATH-500 problems and reports a table:
    method | accuracy | mean_time_s | mean_tokens_generated

Usage:
    python run_latency_comparison.py \
        --ip localhost --port 9090 \
        --model-name Qwen/Qwen3-8B-FP8 \
        --num-samples 100 \
        --n-candidates 7 \
        --sage-epochs 2 \
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

_REBUTTAL_ROOT = Path(__file__).resolve().parents[2]
if str(_REBUTTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_REBUTTAL_ROOT))

from core.vllm_client import get_engine
from core.helpers import append_jsonl, ensure_output_dir, equations_are_equal_new, build_prompt
from experiments.sage.solver import (
    process_query,
    score_answers_async,
    score_and_parse,
    load_prompt,
    load_configurations,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MATH-500 latency comparison")
    p.add_argument("--ip", type=str, default="localhost")
    p.add_argument("--port", type=int, default=9090)
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B-FP8")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument("--n-candidates", type=int, default=7,
                   help="N for BoN; also used as number_of_gens_per_epoch for SAGE")
    p.add_argument("--sage-epochs", type=int, default=2)
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


def count_tokens(texts: List[str], tokens_per_char: float = 0.25) -> int:
    """Estimate token count from character length (rough proxy)."""
    return int(sum(len(t) for t in texts) * tokens_per_char)


async def run_baseline(engine, problems, answers, gen_params, out_dir: Path) -> List[dict]:
    records = []
    for idx, (problem, gt_answer) in enumerate(zip(problems, answers)):
        prompt = build_prompt(problem, None)
        t0 = time.perf_counter()
        texts = await engine.agenerate(prompt, **gen_params)
        elapsed = time.perf_counter() - t0
        output_text = texts[0] if isinstance(texts, list) else texts
        is_correct = equations_are_equal_new(prompt, gt_answer, output_text) == 1
        records.append({
            "method": "baseline",
            "index": idx,
            "is_correct": is_correct,
            "time_s": elapsed,
            "tokens_est": count_tokens([output_text]),
        })
        print(f"[baseline] {idx + 1}/{len(problems)} correct={is_correct} time={elapsed:.1f}s")
    append_jsonl(records, out_dir / "latency_baseline.jsonl")
    return records


async def run_bon_random(engine, problems, answers, gen_params, n: int, out_dir: Path) -> List[dict]:
    records = []
    for idx, (problem, gt_answer) in enumerate(zip(problems, answers)):
        prompt = build_prompt(problem, None)
        params = {**gen_params, "n": n}
        t0 = time.perf_counter()
        texts = await engine.agenerate(prompt, **params)
        elapsed = time.perf_counter() - t0
        if isinstance(texts, str):
            texts = [texts]
        chosen = random.choice(texts)
        is_correct = equations_are_equal_new(prompt, gt_answer, chosen) == 1
        records.append({
            "method": "bon_random",
            "index": idx,
            "is_correct": is_correct,
            "time_s": elapsed,
            "tokens_est": count_tokens(texts),
        })
        print(f"[bon_random] {idx + 1}/{len(problems)} correct={is_correct} time={elapsed:.1f}s")
    append_jsonl(records, out_dir / "latency_bon_random.jsonl")
    return records


async def run_bon_sage_judge(
    engine, judge_engine, problems, answers, gen_params, n: int,
    judge_prompt_template: str, configurations: Dict[str, Any], out_dir: Path
) -> List[dict]:
    records = []
    for idx, (problem, gt_answer) in enumerate(zip(problems, answers)):
        prompt = build_prompt(problem, None)
        params = {**gen_params, "n": n}
        t0 = time.perf_counter()
        texts = await engine.agenerate(prompt, **params)
        if isinstance(texts, str):
            texts = [texts]

        candidate_answers = [{"answer": t} for t in texts]
        judge_params = {"n": 1, "temperature": 0.1, "top_p": 0.95, "seed": 7, "max_tokens": 512}
        await score_answers_async(
            judge_engine=judge_engine,
            judge_prompt_template=judge_prompt_template,
            instruction=prompt,
            answers=candidate_answers,
            gen_params_judge=judge_params,
        )
        score_and_parse(candidate_answers, configurations, [judge_prompt_template])
        elapsed = time.perf_counter() - t0

        best = max(candidate_answers, key=lambda a: a.get("final_llm_judge_score", -1e10))
        is_correct = equations_are_equal_new(prompt, gt_answer, best["answer"]) == 1
        records.append({
            "method": "bon_sage_judge",
            "index": idx,
            "is_correct": is_correct,
            "time_s": elapsed,
            "tokens_est": count_tokens(texts),
        })
        print(f"[bon_sage_judge] {idx + 1}/{len(problems)} correct={is_correct} time={elapsed:.1f}s")
    append_jsonl(records, out_dir / "latency_bon_sage_judge.jsonl")
    return records


async def run_sage(
    problems, answers, judge_prompt_template, judge_configurations,
    base_url, model_name, n_candidates, sage_epochs, out_dir: Path
) -> List[dict]:
    records = []
    for idx, (problem, gt_answer) in enumerate(zip(problems, answers)):
        prompt = build_prompt(problem, None)
        t0 = time.perf_counter()
        result = await process_query(
            prompt,
            judge_prompt_templates=[judge_prompt_template],
            judge_configurations=[judge_configurations],
            num_optimization_epochs=sage_epochs,
            base_url=base_url,
            model_name=model_name,
            number_of_gens_per_epoch=n_candidates,
        )
        elapsed = time.perf_counter() - t0
        output_text = result.get("output", "")
        is_correct = equations_are_equal_new(prompt, gt_answer, output_text) == 1
        all_answers_texts = [a.get("answer", "") for a in result.get("all_answers", [])]
        records.append({
            "method": "sage",
            "index": idx,
            "is_correct": is_correct,
            "time_s": elapsed,
            "tokens_est": count_tokens(all_answers_texts),
        })
        print(f"[sage] {idx + 1}/{len(problems)} correct={is_correct} time={elapsed:.1f}s")
    append_jsonl(records, out_dir / "latency_sage.jsonl")
    return records


def print_table(all_records: Dict[str, List[dict]]) -> None:
    print("\n" + "=" * 70)
    print(f"{'Method':<20} {'Accuracy':>10} {'Mean time (s)':>15} {'Mean tokens':>12}")
    print("-" * 70)
    for method, records in all_records.items():
        if not records:
            continue
        accuracy = sum(r["is_correct"] for r in records) / len(records)
        mean_time = sum(r["time_s"] for r in records) / len(records)
        mean_tokens = sum(r["tokens_est"] for r in records) / len(records)
        print(f"{method:<20} {accuracy:>10.4f} {mean_time:>15.2f} {mean_tokens:>12.0f}")
    print("=" * 70)


async def main() -> None:
    args = parse_args()
    base_url = f"http://{args.ip}:{args.port}/v1"
    out_dir = ensure_output_dir(args.output_path)

    judge_prompt_template = load_prompt(args.judge_prompt)
    judge_configurations = load_configurations(args.judge_config)

    # Load MATH-500
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

    print(f"[A2] Comparing methods on {len(problems)} MATH-500 problems")

    GEN_PARAMS = {
        "n": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": args.seed,
        "max_tokens": 4096,
    }

    engine = get_engine(base_url=base_url, model=args.model_name, timeout=300, type="aug")
    judge_engine = get_engine(base_url=base_url, model=args.model_name, timeout=300, type="aug")

    all_records: Dict[str, List[dict]] = {}

    print("\n--- Baseline ---")
    all_records["baseline"] = await run_baseline(engine, problems, answers, GEN_PARAMS, out_dir)

    print("\n--- BoN (random) ---")
    all_records["bon_random"] = await run_bon_random(engine, problems, answers, GEN_PARAMS, args.n_candidates, out_dir)

    print("\n--- BoN (SAGE judge) ---")
    all_records["bon_sage_judge"] = await run_bon_sage_judge(
        engine, judge_engine, problems, answers, GEN_PARAMS, args.n_candidates,
        judge_prompt_template, judge_configurations, out_dir
    )

    print("\n--- SAGE ---")
    all_records["sage"] = await run_sage(
        problems, answers, judge_prompt_template, judge_configurations,
        base_url, args.model_name, args.n_candidates, args.sage_epochs, out_dir
    )

    print_table(all_records)

    summary_path = out_dir / "latency_summary.json"
    summary = {
        method: {
            "accuracy": sum(r["is_correct"] for r in recs) / len(recs),
            "mean_time_s": sum(r["time_s"] for r in recs) / len(recs),
            "mean_tokens": sum(r["tokens_est"] for r in recs) / len(recs),
            "n": len(recs),
        }
        for method, recs in all_records.items() if recs
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[A2] Summary saved to: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
