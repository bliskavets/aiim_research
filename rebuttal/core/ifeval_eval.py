"""IFEval evaluation harness using the official instruction_following_eval verifiers."""
from __future__ import annotations

import asyncio
import json
import random
import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm

from core.helpers import ensure_output_dir, append_jsonl


@dataclass
class IFEvalConfig:
    number_of_samples_to_test: int
    seed: int
    output_dir: Path
    continue_from_file: Optional[str] = None
    batch_size: int = 8


@dataclass
class IFEvalMetrics:
    """Aggregated IFEval accuracy metrics."""
    prompt_total: int = 0
    prompt_correct: int = 0
    instruction_total: int = 0
    instruction_correct: int = 0

    @property
    def prompt_accuracy(self) -> float:
        return self.prompt_correct / self.prompt_total if self.prompt_total else 0.0

    @property
    def instruction_accuracy(self) -> float:
        return self.instruction_correct / self.instruction_total if self.instruction_total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_total": self.prompt_total,
            "prompt_correct": self.prompt_correct,
            "prompt_accuracy": self.prompt_accuracy,
            "instruction_total": self.instruction_total,
            "instruction_correct": self.instruction_correct,
            "instruction_accuracy": self.instruction_accuracy,
        }


def _check_instructions(
    prompt: str,
    response: str,
    instruction_id_list: List[str],
    kwargs_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Use the instruction_following_eval package to check each instruction.

    Returns a dict with per-instruction results and overall prompt-level pass/fail.
    """
    try:
        from instruction_following_eval import instructions_registry  # type: ignore
    except ImportError:
        raise ImportError(
            "Please install the instruction_following_eval package: "
            "pip install instruction-following-eval"
        )

    per_instruction: List[Dict[str, Any]] = []
    all_followed = True

    for instr_id, kwargs in zip(instruction_id_list, kwargs_list):
        try:
            instr_cls = instructions_registry.INSTRUCTION_DICT.get(instr_id)
            if instr_cls is None:
                # Unknown instruction type — mark as not followed
                per_instruction.append({"instruction_id": instr_id, "followed": False, "error": "unknown_instruction_id"})
                all_followed = False
                continue
            instr_obj = instr_cls(instr_id)
            instr_obj.build_description(**kwargs)
            followed = bool(instr_obj.check_following(response))
        except Exception as e:
            followed = False
            per_instruction.append({"instruction_id": instr_id, "followed": followed, "error": f"{type(e).__name__}: {e}"})
            all_followed = False
            continue
        per_instruction.append({"instruction_id": instr_id, "followed": followed})
        if not followed:
            all_followed = False

    return {
        "prompt_followed": all_followed,
        "per_instruction": per_instruction,
    }


class IFEval:
    """Evaluate a solver on the IFEval benchmark.

    Uses the official instruction_following_eval verifiers for scoring.
    Reports prompt-level accuracy (all instructions followed) and
    instruction-level accuracy (fraction of individual instructions followed).

    The solver must accept a prompt string and return a dict with at least "output".
    """

    def __init__(
        self,
        number_of_samples_to_test: int,
        seed: int,
        output_path: Optional[str] = None,
        continue_from_file: Optional[str] = None,
        batch_size: int = 8,
    ) -> None:
        self.config = IFEvalConfig(
            number_of_samples_to_test=number_of_samples_to_test,
            seed=seed,
            output_dir=ensure_output_dir(output_path),
            continue_from_file=continue_from_file,
            batch_size=batch_size,
        )
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.config.output_dir / f"ifeval_{timestamp}.jsonl"

    # ------------------------------------------------------------------
    # Dataset loading and index selection
    # ------------------------------------------------------------------

    def _load_dataset(self) -> List[Dict[str, Any]]:
        ds = load_dataset("google/IFEval")
        split = "train" if "train" in ds else list(ds.keys())[0]
        rows = []
        for item in ds[split]:
            rows.append({
                "key": item["key"],
                "prompt": item["prompt"],
                "instruction_id_list": list(item["instruction_id_list"]),
                "kwargs": [dict(kw) for kw in item["kwargs"]],
            })
        return rows

    def _sample_indices(self, total: int) -> List[int]:
        indices = list(range(total))
        random.Random(self.config.seed).shuffle(indices)
        return indices[: self.config.number_of_samples_to_test]

    def _load_continue_file(self, rows: List[Dict], indices: List[int]) -> tuple[List[int], List[dict]]:
        done_keys: set[str] = set()
        done_records: List[dict] = []
        if not self.config.continue_from_file:
            return indices, done_records
        with open(self.config.continue_from_file, "r") as f:
            for line in f:
                rec = json.loads(line)
                done_keys.add(str(rec.get("key", "")))
                done_records.append(rec)

        remaining_indices = [i for i in indices if str(rows[i]["key"]) not in done_keys]
        if done_records:
            print(f"Reusing {len(done_records)} results from {self.config.continue_from_file}")
            append_jsonl(done_records, self.output_file)
        return remaining_indices, done_records

    # ------------------------------------------------------------------
    # Single-example evaluation
    # ------------------------------------------------------------------

    def _score_record(self, row: Dict[str, Any], output_text: str) -> Dict[str, Any]:
        check = _check_instructions(
            prompt=row["prompt"],
            response=output_text,
            instruction_id_list=row["instruction_id_list"],
            kwargs_list=row["kwargs"],
        )
        return check

    async def _evaluate_single_async(self, idx: int, row: Dict[str, Any], solver: Callable) -> dict:
        try:
            result = (await solver(row["prompt"])) or {}
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}

        output_text = str(result.get("output", ""))
        check = self._score_record(row, output_text)

        return {
            "index": idx,
            "key": row["key"],
            "prompt": row["prompt"],
            "instruction_id_list": row["instruction_id_list"],
            "output": output_text,
            "prompt_followed": check["prompt_followed"],
            "per_instruction": check["per_instruction"],
            **{k: v for k, v in result.items() if k not in ("output",)},
        }

    def _evaluate_single(self, idx: int, row: Dict[str, Any], solver: Callable) -> dict:
        try:
            result = solver(row["prompt"]) or {}
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}

        output_text = str(result.get("output", ""))
        check = self._score_record(row, output_text)

        return {
            "index": idx,
            "key": row["key"],
            "prompt": row["prompt"],
            "instruction_id_list": row["instruction_id_list"],
            "output": output_text,
            "prompt_followed": check["prompt_followed"],
            "per_instruction": check["per_instruction"],
            **{k: v for k, v in result.items() if k not in ("output",)},
        }

    # ------------------------------------------------------------------
    # Metrics aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _update_metrics(metrics: IFEvalMetrics, log_record: dict) -> None:
        metrics.prompt_total += 1
        if log_record.get("prompt_followed"):
            metrics.prompt_correct += 1
        for instr_result in log_record.get("per_instruction", []):
            metrics.instruction_total += 1
            if instr_result.get("followed"):
                metrics.instruction_correct += 1

    @staticmethod
    def _metrics_from_records(records: List[dict]) -> IFEvalMetrics:
        m = IFEvalMetrics()
        for rec in records:
            IFEval._update_metrics(m, rec)
        return m

    # ------------------------------------------------------------------
    # Async evaluation
    # ------------------------------------------------------------------

    async def run_async(self, solver: Callable) -> dict:
        """Run IFEval with an async solver using batched concurrency."""
        rows = self._load_dataset()
        indices = self._sample_indices(len(rows))
        indices, done_records = self._load_continue_file(rows, indices)

        metrics = self._metrics_from_records(done_records)
        batch_size = self.config.batch_size

        with tqdm(total=len(indices), desc="Evaluating IFEval") as pbar:
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                tasks = [self._evaluate_single_async(idx, rows[idx], solver) for idx in batch]
                results = await asyncio.gather(*tasks, return_exceptions=False)
                for log_record in results:
                    self._update_metrics(metrics, log_record)
                    log_record["running_prompt_accuracy"] = metrics.prompt_accuracy
                    log_record["running_instruction_accuracy"] = metrics.instruction_accuracy
                    append_jsonl(log_record, self.output_file)
                    pbar.set_postfix({
                        "prompt_acc": f"{metrics.prompt_accuracy:.3f}",
                        "instr_acc": f"{metrics.instruction_accuracy:.3f}",
                    })
                    pbar.update(1)

        summary = {
            **metrics.to_dict(),
            "output_file": str(self.output_file),
        }
        print(f"\nIFEval results: prompt_accuracy={metrics.prompt_accuracy:.4f}, "
              f"instruction_accuracy={metrics.instruction_accuracy:.4f}")
        return summary

    # ------------------------------------------------------------------
    # Synchronous evaluation
    # ------------------------------------------------------------------

    def run(self, solver: Callable) -> dict:
        """Run IFEval with a synchronous solver."""
        rows = self._load_dataset()
        indices = self._sample_indices(len(rows))
        indices, done_records = self._load_continue_file(rows, indices)

        metrics = self._metrics_from_records(done_records)

        with tqdm(total=len(indices), desc="Evaluating IFEval") as pbar:
            for idx in indices:
                log_record = self._evaluate_single(idx, rows[idx], solver)
                self._update_metrics(metrics, log_record)
                log_record["running_prompt_accuracy"] = metrics.prompt_accuracy
                log_record["running_instruction_accuracy"] = metrics.instruction_accuracy
                append_jsonl(log_record, self.output_file)
                pbar.set_postfix({
                    "prompt_acc": f"{metrics.prompt_accuracy:.3f}",
                    "instr_acc": f"{metrics.instruction_accuracy:.3f}",
                })
                pbar.update(1)

        summary = {
            **metrics.to_dict(),
            "output_file": str(self.output_file),
        }
        print(f"\nIFEval results: prompt_accuracy={metrics.prompt_accuracy:.4f}, "
              f"instruction_accuracy={metrics.instruction_accuracy:.4f}")
        return summary
