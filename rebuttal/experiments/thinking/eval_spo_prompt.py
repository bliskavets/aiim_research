#!/usr/bin/env python3
"""Evaluate an SPO-optimized prompt on MATH-500 with non-thinking Qwen3-8B."""
from __future__ import annotations
import argparse, asyncio, sys, time
from pathlib import Path
from typing import Any, Dict
_R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_R))
from core.vllm_client import get_engine
from core.math500_eval import Math500Eval

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", default="localhost"); p.add_argument("--port", type=int, default=9090)
    p.add_argument("--model-name", default="Qwen/Qwen3-8B-FP8")
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--num-samples", type=int, default=500); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16); p.add_argument("--output-path", default=None)
    return p.parse_args()

async def main():
    a = parse_args()
    opt = Path(a.prompt_file).read_text().strip()
    print(f"[SPO-eval] optimized prompt ({len(opt)} chars): {opt[:200]}...")
    engine = get_engine(base_url=f"http://{a.ip}:{a.port}/v1", model=a.model_name, timeout=300, type="aug")
    GEN = {"n": 1, "temperature": 0.0, "top_p": 1.0, "seed": a.seed, "max_tokens": 4096}
    async def solver(problem_prompt: str) -> Dict[str, Any]:
        # problem_prompt already has the default boxed suffix from Math500Eval.build_prompt;
        # prepend the SPO-optimized instruction.
        full = f"{opt}\n\n{problem_prompt}"
        texts = await engine.agenerate(full, **GEN)
        return {"output": texts[0] if isinstance(texts, list) else texts}
    ev = Math500Eval(number_of_samples_to_test=a.num_samples, seed=a.seed, output_path=a.output_path, max_workers=a.batch_size)
    s = await ev.run_async(solver=solver)
    print(f"[SPO-eval] MATH-500 accuracy: {s['accuracy']:.4f} ({s['num_correct']}/{s['num_samples']})")

if __name__ == "__main__":
    asyncio.run(main())
