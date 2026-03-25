#!/usr/bin/env python3
"""C3 — MMLU-Pro STEM: single-generation baseline.

Usage:
    python run_mmlu_pro_baseline.py \
        --ip localhost --port 9090 \
        --model-name Qwen/Qwen3-8B-FP8 \
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
from typing import Any, Dict, List, Optional

_REBUTTAL_ROOT = Path(__file__).resolve().parents[2]
if str(_REBUTTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_REBUTTAL_ROOT))

from core.vllm_client import get_engine
from core.helpers import ensure_output_dir, append_jsonl

# Re-use shared utilities from the SAGE script
from experiments.c3_mmlu_pro.run_mmlu_pro_sage import (
    load_mmlu_pro_stem,
    format_mmlu_pro_prompt,
    extract_answer_letter,
    get_correct_letter,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MMLU-Pro STEM single-generation baseline")
    p.add_argument("--ip", type=str, default="localhost")
    p.add_argument("--port", type=int, default=9090)
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B-FP8")
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument("--continue-from", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    base_url = f"http://{args.ip}:{args.port}/v1"
    out_dir = ensure_output_dir(args.output_path)
    output_file = out_dir / "mmlu_pro_baseline.jsonl"

    rows = load_mmlu_pro_stem(max_samples=args.num_samples, seed=args.seed)

    done_indices: set[int] = set()
    if args.continue_from:
        with open(args.continue_from, "r") as f:
            for line in f:
                rec = json.loads(line)
                done_indices.add(rec.get("index", -1))
        print(f"[C3-baseline] Resuming: {len(done_indices)} already done")

    print(f"[C3-baseline] Running baseline on {len(rows)} MMLU-Pro STEM problems")

    engine = get_engine(base_url=base_url, model=args.model_name, timeout=180.0, type="aug")
    GEN_PARAMS = {
        "n": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": args.seed,
        "max_tokens": 4096,
    }

    correct = 0
    total = 0

    # Process in batches for concurrency
    pending = [(i, row) for i, row in enumerate(rows) if i not in done_indices]

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        prompts = [format_mmlu_pro_prompt(row) for _, row in batch]

        t0 = time.perf_counter()
        tasks = [engine.agenerate(prompt, **GEN_PARAMS) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        elapsed_batch = time.perf_counter() - t0

        for (i, row), prompt, texts in zip(batch, prompts, results):
            output_text = texts[0] if isinstance(texts, list) else texts
            gt_letter = get_correct_letter(row)
            predicted_letter = extract_answer_letter(output_text)
            is_correct = (
                predicted_letter is not None
                and gt_letter is not None
                and predicted_letter == gt_letter
            )
            total += 1
            correct += int(is_correct)
            log_record = {
                "index": i,
                "question": row.get("question", ""),
                "subject": row.get("category", ""),
                "prompt": prompt,
                "gt_letter": gt_letter,
                "predicted_letter": predicted_letter,
                "output": output_text,
                "is_correct": is_correct,
                "time_s": elapsed_batch / len(batch),
                "running_accuracy": correct / total,
            }
            append_jsonl(log_record, output_file)

        print(f"[C3-baseline] {total}/{len(pending)} acc={correct/total:.3f} batch_time={elapsed_batch:.1f}s")

    wall_elapsed = time.perf_counter()
    print(f"\n[C3-baseline] MMLU-Pro STEM baseline accuracy: {correct/total:.4f} ({correct}/{total})")
    print(f"[C3-baseline] Output file: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
