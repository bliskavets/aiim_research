#!/usr/bin/env python3
"""XSTest (safety over-refusal) — baseline or SAGE. Non-thinking Qwen3.
Judge (compliance/refusal) = gpt-4.1 via the OpenAI-compatible endpoint."""
from __future__ import annotations
import argparse, asyncio, sys, time
from pathlib import Path
from typing import Any, Dict
_R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_R))
from core.vllm_client import get_engine
from core.xstest_eval import XSTest
from experiments.sage.solver import process_query, load_prompt, load_configurations

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", default="localhost"); p.add_argument("--port", type=int, default=9090)
    p.add_argument("--model-name", default="Qwen/Qwen3-8B-FP8")
    p.add_argument("--num-samples", type=int, default=-1); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16); p.add_argument("--output-path", default=None)
    p.add_argument("--method", choices=["baseline", "sage"], default="baseline")
    p.add_argument("--dataset-csv", default=str(_R / "configs" / "xstest_prompts.csv"))
    p.add_argument("--judge-model", default="openai/gpt-4.1")
    p.add_argument("--num-optimization-epochs", type=int, default=2)
    p.add_argument("--number-of-gens-per-epoch", type=int, default=7)
    p.add_argument("--judge-prompt", default=str(_R / "configs" / "math500_judge_prompt.txt"))
    p.add_argument("--judge-config", default=str(_R / "configs" / "math500_judge_config.json"))
    return p.parse_args()

async def main():
    a = parse_args()
    base_url = f"http://{a.ip}:{a.port}/v1"
    if a.method == "baseline":
        engine = get_engine(base_url=base_url, model=a.model_name, timeout=300, type="aug")
        GEN = {"n": 1, "temperature": 0.0, "top_p": 1.0, "seed": a.seed, "max_tokens": 2048}
        async def solver(prompt: str) -> Dict[str, Any]:
            texts = await engine.agenerate(prompt, **GEN)
            return {"output": texts[0] if isinstance(texts, list) else texts}
    else:
        jt = load_prompt(a.judge_prompt); jc = load_configurations(a.judge_config)
        async def solver(prompt: str) -> Dict[str, Any]:
            r = await process_query(prompt, judge_prompt_templates=[jt], judge_configurations=[jc],
                num_optimization_epochs=a.num_optimization_epochs, base_url=base_url, model_name=a.model_name,
                number_of_gens_per_epoch=a.number_of_gens_per_epoch)
            return {"output": r.get("output", "")}
    ev = XSTest(number_of_samples_to_test=a.num_samples, seed=a.seed, output_path=a.output_path,
                dataset_csv=a.dataset_csv, run_judge=True, judge_model=a.judge_model)
    s = await ev.run_eval_async(solver=solver, batch_size=a.batch_size)
    print(f"[XSTest-{a.method}] summary: {s}")

if __name__ == "__main__":
    asyncio.run(main())
