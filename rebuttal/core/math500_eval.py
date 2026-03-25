"""MATH-500 evaluation harness."""
from __future__ import annotations

import asyncio
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Optional
import datetime as _dt
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from core.helpers import ensure_output_dir, build_prompt, append_jsonl, equations_are_equal_new


@dataclass
class EvalConfig:
    number_of_samples_to_test: int
    seed: int
    prompt_template: Optional[Any]
    output_dir: Path
    continue_from_file: Optional[str] = None
    max_workers: Optional[int] = None


class Math500Eval:
    """Evaluate a solver on MATH-500 problems.

    The solver must accept a prompt string and return a dict with at least "output".
    """

    def __init__(
        self,
        number_of_samples_to_test: int,
        seed: int,
        prompt_template: Optional[Any] = None,
        output_path: Optional[str] = None,
        continue_from_file: Optional[str] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        self.config = EvalConfig(
            number_of_samples_to_test=number_of_samples_to_test,
            seed=seed,
            prompt_template=prompt_template,
            output_dir=ensure_output_dir(output_path),
            continue_from_file=continue_from_file,
            max_workers=max_workers,
        )
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.config.output_dir / f"math500_eval_{timestamp}.jsonl"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_dataset(self):
        ds = load_dataset("HuggingFaceH4/MATH-500")
        split = "test" if "test" in ds else list(ds.keys())[0]
        return list(ds[split]["problem"]), list(ds[split]["answer"])

    def _sample_indices(self, total: int) -> list[int]:
        indices = list(range(total))
        random.Random(self.config.seed).shuffle(indices)
        return indices[: self.config.number_of_samples_to_test]

    def _load_continue_file(self, indices: list[int]) -> tuple[list[int], list[dict]]:
        """Remove already-done indices and return records from the continue file."""
        done_indices: list[int] = []
        done_records: list[dict] = []
        if not self.config.continue_from_file:
            return done_indices, done_records
        with open(self.config.continue_from_file, "r") as f:
            for line in f:
                rec = json.loads(line)
                idx = rec["index"]
                if idx in indices:
                    indices.remove(idx)
                    done_indices.append(idx)
                    done_records.append(rec)
        if done_records:
            print(f"Reusing {len(done_records)} results from {self.config.continue_from_file}")
            append_jsonl(done_records, self.output_file)
        return done_indices, done_records

    # ------------------------------------------------------------------
    # Single-problem evaluation
    # ------------------------------------------------------------------

    def _evaluate_single(
        self, idx: int, problem: str, gt_answer: str, solver: Callable[[str], dict]
    ) -> dict:
        prompt = build_prompt(problem, self.config.prompt_template)
        try:
            result = solver(prompt) or {}
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
            raise

        output_text = str(result.get("output", ""))
        is_correct = equations_are_equal_new(prompt, gt_answer, output_text) == 1

        return {
            "index": int(idx),
            "problem": problem,
            "prompt": prompt,
            "gt_answer": gt_answer,
            "output": output_text,
            "is_correct": bool(is_correct),
            **{k: v for k, v in result.items() if k != "output"},
        }

    async def _evaluate_single_async(
        self, idx: int, problem: str, gt_answer: str, solver: Callable
    ) -> dict:
        prompt = build_prompt(problem, self.config.prompt_template)
        try:
            result = (await solver(prompt)) or {}
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
            raise

        output_text = str(result.get("output", ""))
        is_correct = equations_are_equal_new(prompt, gt_answer, output_text) == 1

        return {
            "index": int(idx),
            "problem": problem,
            "prompt": prompt,
            "gt_answer": gt_answer,
            "output": output_text,
            "is_correct": bool(is_correct),
            **{k: v for k, v in result.items() if k != "output"},
        }

    # ------------------------------------------------------------------
    # Synchronous (threaded) evaluation
    # ------------------------------------------------------------------

    def run(self, solver: Callable[[str], dict]) -> dict:
        """Run evaluation with a synchronous solver using a thread pool."""
        problems, answers = self._load_dataset()
        indices = self._sample_indices(len(problems))
        self._load_continue_file(indices)

        correct = 0
        total = 0
        stats_lock = threading.Lock()
        file_lock = threading.Lock()
        max_workers = self.config.max_workers or 8

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self._evaluate_single, idx, problems[idx], answers[idx], solver
                ): idx
                for idx in indices
            }
            with tqdm(total=len(indices), desc="Evaluating MATH-500") as pbar:
                for future in as_completed(future_to_idx):
                    log_record = future.result()
                    with stats_lock:
                        total += 1
                        correct += int(log_record["is_correct"])
                        log_record["running_accuracy"] = correct / total
                    with file_lock:
                        append_jsonl(log_record, self.output_file)
                    pbar.set_postfix({"accuracy": f"{log_record['running_accuracy']:.3f}", "correct": f"{correct}/{total}"})
                    pbar.update(1)

        return {
            "num_samples": total,
            "num_correct": correct,
            "accuracy": (correct / total) if total else 0.0,
            "output_file": str(self.output_file),
        }

    # ------------------------------------------------------------------
    # Async evaluation
    # ------------------------------------------------------------------

    async def run_async(self, solver: Callable) -> dict:
        """Run evaluation with an async solver using batched concurrency."""
        problems, answers = self._load_dataset()
        indices = self._sample_indices(len(problems))
        self._load_continue_file(indices)

        correct = 0
        total = 0
        batch_size = self.config.max_workers or 8

        with tqdm(total=len(indices), desc="Evaluating MATH-500") as pbar:
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                tasks = [
                    self._evaluate_single_async(idx, problems[idx], answers[idx], solver)
                    for idx in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=False)
                for log_record in results:
                    total += 1
                    correct += int(log_record["is_correct"])
                    log_record["running_accuracy"] = correct / total
                    append_jsonl(log_record, self.output_file)
                    pbar.set_postfix({"accuracy": f"{log_record['running_accuracy']:.3f}", "correct": f"{correct}/{total}"})
                    pbar.update(1)

        return {
            "num_samples": total,
            "num_correct": correct,
            "accuracy": (correct / total) if total else 0.0,
            "output_file": str(self.output_file),
        }
