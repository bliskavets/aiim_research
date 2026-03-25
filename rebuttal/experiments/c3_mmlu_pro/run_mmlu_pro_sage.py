#!/usr/bin/env python3
"""C3 — MMLU-Pro STEM: SAGE evaluation.

Loads the MMLU-Pro STEM subset (science/math/engineering/biology/chemistry/physics)
from TIGER-Lab/MMLU-Pro, presents each question in multiple-choice format, and
runs SAGE for selection. Evaluates accuracy by matching the output letter.

Usage:
    python run_mmlu_pro_sage.py \
        --ip localhost --port 9090 \
        --model-name Qwen/Qwen3-8B-FP8 \
        --num-samples 500 \
        --output-path /path/to/outputs
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_REBUTTAL_ROOT = Path(__file__).resolve().parents[2]
if str(_REBUTTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_REBUTTAL_ROOT))

from core.helpers import ensure_output_dir, append_jsonl
from experiments.sage.solver import process_query


STEM_SUBJECTS = {
    "math", "physics", "chemistry", "biology",
    "engineering", "computer science",
}

OPTION_LETTERS = list("ABCDEFGHIJ")

MMLU_PRO_PROMPT_TEMPLATE = (
    "Answer the following multiple choice question. "
    "Think step by step, then state your answer as a single letter.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "After your reasoning, write your final answer as: <answer>X</answer> "
    "where X is the correct letter."
)

MMLU_PRO_JUDGE_PROMPT = (
    "You are evaluating whether a response correctly answers a multiple-choice question.\n\n"
    "Question and options:\n{prompt}\n\n"
    "Response:\n{answer}\n\n"
    "Does the response select the correct answer? "
    "Check if the final answer letter matches what would be correct.\n"
    "<verdict>correct</verdict> if the answer is correct, <verdict>incorrect</verdict> otherwise."
)

MMLU_PRO_JUDGE_CONFIG = {
    "mmlu_correctness": {
        "pos_labels": ["correct"],
        "neg_labels": ["incorrect"],
        "labels": ["correct", "incorrect"],
        "opening_tag": "<verdict>",
        "closing_tag": "</verdict>",
        "weight": 1.0,
    }
}


def load_mmlu_pro_stem(max_samples: Optional[int] = None, seed: int = 42) -> List[Dict[str, Any]]:
    """Load MMLU-Pro STEM subset from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    split = "test" if "test" in ds else list(ds.keys())[0]

    rows = []
    for item in ds[split]:
        subject = str(item.get("category", "")).lower()
        if any(s in subject for s in STEM_SUBJECTS):
            rows.append(dict(item))

    random.Random(seed).shuffle(rows)
    if max_samples is not None:
        rows = rows[:max_samples]
    print(f"[C3] Loaded {len(rows)} MMLU-Pro STEM examples")
    return rows


def format_mmlu_pro_prompt(row: Dict[str, Any]) -> str:
    """Format a MMLU-Pro row as a multiple-choice prompt."""
    options = row.get("options", [])
    option_lines = "\n".join(
        f"{OPTION_LETTERS[i]}) {opt}" for i, opt in enumerate(options)
    )
    return MMLU_PRO_PROMPT_TEMPLATE.format(
        question=row["question"],
        options=option_lines,
    )


def extract_answer_letter(text: str) -> Optional[str]:
    """Extract the answer letter from <answer>X</answer> tag or last standalone letter."""
    m = re.search(r"<answer>\s*([A-J])\s*</answer>", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: find the last standalone capital letter A-J
    matches = re.findall(r"\b([A-J])\b", text)
    if matches:
        return matches[-1].upper()
    return None


def get_correct_letter(row: Dict[str, Any]) -> Optional[str]:
    """Get the correct answer letter from the row."""
    answer_idx = row.get("answer_index")
    if answer_idx is not None and 0 <= answer_idx < len(OPTION_LETTERS):
        return OPTION_LETTERS[answer_idx]
    answer = row.get("answer", "")
    if answer and answer[0].upper() in OPTION_LETTERS:
        return answer[0].upper()
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MMLU-Pro STEM SAGE evaluation")
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
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    base_url = f"http://{args.ip}:{args.port}/v1"
    out_dir = ensure_output_dir(args.output_path)
    output_file = out_dir / "mmlu_pro_sage.jsonl"

    rows = load_mmlu_pro_stem(max_samples=args.num_samples, seed=args.seed)

    # Handle continue-from
    done_indices: set[int] = set()
    if args.continue_from:
        with open(args.continue_from, "r") as f:
            for line in f:
                rec = json.loads(line)
                done_indices.add(rec.get("index", -1))
        print(f"[C3] Resuming: {len(done_indices)} already done")

    correct = 0
    total = 0
    print(f"[C3] Running SAGE on {len(rows)} MMLU-Pro STEM problems")

    for i, row in enumerate(rows):
        if i in done_indices:
            continue

        prompt = format_mmlu_pro_prompt(row)
        gt_letter = get_correct_letter(row)

        t0 = time.perf_counter()
        result = await process_query(
            prompt,
            judge_prompt_templates=[MMLU_PRO_JUDGE_PROMPT],
            judge_configurations=[MMLU_PRO_JUDGE_CONFIG],
            num_optimization_epochs=args.num_optimization_epochs,
            base_url=base_url,
            model_name=args.model_name,
            number_of_gens_per_epoch=args.number_of_gens_per_epoch,
        )
        elapsed = time.perf_counter() - t0

        output_text = result.get("output", "")
        predicted_letter = extract_answer_letter(output_text)
        is_correct = (predicted_letter is not None and gt_letter is not None and predicted_letter == gt_letter)
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
            "time_s": elapsed,
            "running_accuracy": correct / total,
        }
        append_jsonl(log_record, output_file)
        print(f"[C3] {total}/{len(rows)} correct={is_correct} pred={predicted_letter} gt={gt_letter} "
              f"acc={correct/total:.3f} time={elapsed:.1f}s")

    print(f"\n[C3] MMLU-Pro STEM SAGE accuracy: {correct/total:.4f} ({correct}/{total})")
    print(f"[C3] Output file: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
