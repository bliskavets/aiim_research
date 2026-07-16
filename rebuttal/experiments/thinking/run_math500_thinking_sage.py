#!/usr/bin/env python3
"""SAGE on top of Qwen3 thinking mode (MATH-500). Tests whether SAGE adds accuracy
over the strong reasoning baseline (R4-W1)."""
from __future__ import annotations
import argparse, asyncio, sys, time
from pathlib import Path
from typing import Any, Dict
_R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_R))
from core.math500_eval import Math500Eval
import experiments.sage.solver as sage_solver
from experiments.sage.solver import process_query, load_prompt, load_configurations

# Thinking generations are long; raise the SAGE generation budget so candidate
# reasoning is not truncated before the boxed answer (epoch gens use these globals).
sage_solver.DEFAULT_GEN_PARAMS["max_tokens"] = 16384

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", default="localhost"); p.add_argument("--port", type=int, default=9090)
    p.add_argument("--model-name", default="Qwen/Qwen3-8B-FP8")
    p.add_argument("--num-samples", type=int, default=200); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=12); p.add_argument("--output-path", default=None)
    p.add_argument("--num-optimization-epochs", type=int, default=2)
    p.add_argument("--number-of-gens-per-epoch", type=int, default=7)
    p.add_argument("--judge-prompt", default=str(_R / "configs" / "math500_judge_prompt.txt"))
    p.add_argument("--judge-config", default=str(_R / "configs" / "math500_judge_config.json"))
    return p.parse_args()

async def main():
    a = parse_args()
    base_url = f"http://{a.ip}:{a.port}/v1"
    jt = load_prompt(a.judge_prompt); jc = load_configurations(a.judge_config)
    async def solver(prompt: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        r = await process_query(prompt, judge_prompt_templates=[jt], judge_configurations=[jc],
            num_optimization_epochs=a.num_optimization_epochs, base_url=base_url, model_name=a.model_name,
            number_of_gens_per_epoch=a.number_of_gens_per_epoch, engine_type="think")
        r["time_s"] = time.perf_counter() - t0
        return r
    ev = Math500Eval(number_of_samples_to_test=a.num_samples, seed=a.seed, output_path=a.output_path, max_workers=a.batch_size)
    s = await ev.run_async(solver=solver)
    print(f"[THINK-SAGE] MATH-500 accuracy: {s['accuracy']:.4f} ({s['num_correct']}/{s['num_samples']})")

if __name__ == "__main__":
    asyncio.run(main())
