"""AlpacaEval evaluation harness."""
from __future__ import annotations

import asyncio
import json
import random
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from datasets import load_dataset

from core.helpers import ensure_output_dir, append_jsonl


@dataclass
class AlpacaEvalConfig:
    number_of_samples_to_test: int
    seed: int
    output_dir: Path
    run_evaluator: bool
    evaluator_name: str
    continue_from_file: Optional[str] = None
    specific_instructs_to_process: Optional[Union[List[str], str]] = None
    model_outputs_save_freq: int = 0


class AlpacaEval:
    """Evaluate a solver on AlpacaEval prompts and optionally run alpaca_eval evaluator.

    The solver must accept a prompt string and return a dict with at least "output".
    """

    def __init__(
        self,
        number_of_samples_to_test: int,
        seed: int,
        output_path: Optional[str] = None,
        run_evaluator: bool = False,
        evaluator_name: str = "alpaca_eval_gpt4_turbo_fn",
        continue_from_file: Optional[str] = None,
        specific_instructs_to_process: Optional[Union[List[str], str]] = None,
        model_outputs_save_freq: int = 0,
    ) -> None:
        self.config = AlpacaEvalConfig(
            number_of_samples_to_test=number_of_samples_to_test,
            seed=seed,
            output_dir=ensure_output_dir(output_path),
            run_evaluator=run_evaluator,
            evaluator_name=evaluator_name,
            continue_from_file=continue_from_file,
            specific_instructs_to_process=specific_instructs_to_process,
            model_outputs_save_freq=model_outputs_save_freq,
        )
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.outputs_jsonl = self.config.output_dir / f"alpaca_eval_records_{timestamp}.jsonl"
        self.model_outputs_json = self.config.output_dir / f"alpaca_model_outputs_{timestamp}.json"

    def _get_specific_instruction_set(self) -> Optional[Set[str]]:
        source = self.config.specific_instructs_to_process
        if source is None:
            return None
        if isinstance(source, list):
            return set(str(x) for x in source)
        path = Path(str(source))
        if path.exists() and path.is_file():
            result: Set[str] = set()
            try:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict) and "instruction" in obj:
                                result.add(str(obj["instruction"]))
                        except Exception:
                            continue
            except Exception:
                return set()
            return result
        return {str(source)}

    # ------------------------------------------------------------------
    # Async evaluation
    # ------------------------------------------------------------------

    async def run_eval_async(self, solver: Callable[[str], dict], batch_size: int = 10) -> dict:
        """Run evaluation on AlpacaEval prompts asynchronously."""
        alpaca = load_dataset("tatsu-lab/alpaca_eval")
        split_name = "eval" if "eval" in alpaca else list(alpaca.keys())[0]
        features = alpaca[split_name].features
        instructions: List[str] = (
            list(alpaca[split_name]["instruction"])
            if "instruction" in features
            else list(alpaca[split_name]["inputs"])
        )

        allowed_instructions = self._get_specific_instruction_set()
        if allowed_instructions is not None:
            instructions = [i for i in instructions if i in allowed_instructions]

        model_outputs: List[Dict[str, str]] = []
        already_done: Set[str] = set()
        if self.config.continue_from_file:
            with open(self.config.continue_from_file, "r") as f:
                for line in f:
                    rec = json.loads(line)
                    model_outputs.append({"instruction": rec["instruction"], "output": rec["output"], "generator": "model"})
                    already_done.add(str(rec.get("instruction", "")))

        remaining_indices = [i for i, instr in enumerate(instructions) if instr not in already_done]
        random.Random(self.config.seed).shuffle(remaining_indices)
        desired = self.config.number_of_samples_to_test
        remaining_to_run = max(0, desired - len(model_outputs))
        indices = remaining_indices[:remaining_to_run]

        print(f"Resuming: {len(model_outputs)} done. Running {len(indices)} more.")

        total = 0
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_instructions = [instructions[idx] for idx in batch_indices]
            batch_results = await asyncio.gather(*[solver(instr) for instr in batch_instructions])
            for j, result in enumerate(batch_results):
                output_text = str(result.get("output", ""))
                model_outputs.append({"instruction": batch_instructions[j], "output": output_text, "generator": "model"})
                log_record = {
                    "index": len(model_outputs),
                    "instruction": batch_instructions[j],
                    "prompt": batch_instructions[j],
                    "output": output_text,
                    **{k: v for k, v in result.items() if k != "output"},
                }
                total += 1
                print(f"[{total}/{len(indices)}] Logged output")
                append_jsonl(log_record, self.outputs_jsonl)
                if self.config.model_outputs_save_freq > 0 and (len(model_outputs) % self.config.model_outputs_save_freq == 0):
                    self.model_outputs_json.write_text(json.dumps(model_outputs, ensure_ascii=False, indent=2), encoding="utf-8")

        self.model_outputs_json.write_text(json.dumps(model_outputs, ensure_ascii=False, indent=2), encoding="utf-8")

        summary: Dict[str, Any] = {
            "num_samples": total,
            "records_file": str(self.outputs_jsonl),
            "model_outputs_file": str(self.model_outputs_json),
        }
        if self.config.run_evaluator:
            try:
                from alpaca_eval import evaluate
                res = evaluate(
                    model_outputs=model_outputs,
                    name="model",
                    annotators_config=self.config.evaluator_name,
                    is_return_instead_of_print=True,
                )
                summary["alpaca_eval_results"] = res
            except Exception as e:
                summary["alpaca_eval_error"] = f"{type(e).__name__}: {e}"
        return summary

    # ------------------------------------------------------------------
    # Synchronous evaluation
    # ------------------------------------------------------------------

    def run(self, solver: Callable[[str], dict]) -> dict:
        """Run evaluation synchronously."""
        alpaca = load_dataset("tatsu-lab/alpaca_eval")
        split_name = "eval" if "eval" in alpaca else list(alpaca.keys())[0]
        features = alpaca[split_name].features
        instructions: List[str] = (
            list(alpaca[split_name]["instruction"])
            if "instruction" in features
            else list(alpaca[split_name]["inputs"])
        )

        allowed_instructions = self._get_specific_instruction_set()
        if allowed_instructions is not None:
            instructions = [i for i in instructions if i in allowed_instructions]

        N = len(instructions)
        indices = list(range(N))
        random.Random(self.config.seed).shuffle(indices)
        indices = indices[: self.config.number_of_samples_to_test]

        model_outputs: List[Dict[str, str]] = []
        total = 0

        for i, idx in enumerate(indices):
            instruction = instructions[idx]
            try:
                result = solver(instruction) or {}
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
                raise

            output_text = str(result.get("output", ""))
            log_record = {
                "index": int(idx),
                "instruction": instruction,
                "prompt": instruction,
                "output": output_text,
                **{k: v for k, v in result.items() if k != "output"},
            }
            total += 1
            print(f"[{i + 1}/{len(indices)}] Logged output")
            append_jsonl(log_record, self.outputs_jsonl)
            model_outputs.append({"instruction": instruction, "output": output_text, "generator": "model"})
            if self.config.model_outputs_save_freq > 0 and (len(model_outputs) % self.config.model_outputs_save_freq == 0):
                self.model_outputs_json.write_text(json.dumps(model_outputs, ensure_ascii=False, indent=2), encoding="utf-8")

        self.model_outputs_json.write_text(json.dumps(model_outputs, ensure_ascii=False, indent=2), encoding="utf-8")

        summary: Dict[str, Any] = {
            "num_samples": total,
            "records_file": str(self.outputs_jsonl),
            "model_outputs_file": str(self.model_outputs_json),
        }
        if self.config.run_evaluator:
            try:
                from alpaca_eval import evaluate
                res = evaluate(
                    model_outputs=model_outputs,
                    name="model",
                    annotators_config=self.config.evaluator_name,
                    is_return_instead_of_print=True,
                )
                summary["alpaca_eval_results"] = res
            except Exception as e:
                summary["alpaca_eval_error"] = f"{type(e).__name__}: {e}"
        return summary
