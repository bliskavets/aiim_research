#!/usr/bin/env python3
"""Thinking-mode MATH-500 baseline (Qwen3 reasoning). Addresses R4-W1: does the
tech-report reasoning score hold, and how does it compare to SAGE non-thinking."""
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
    p.add_argument("--num-samples", type=int, default=500); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16); p.add_argument("--output-path", default=None)
    p.add_argument("--max-tokens", type=int, default=16384)
    return p.parse_args()

async def main():
    a = parse_args()
    engine = get_engine(base_url=f"http://{a.ip}:{a.port}/v1", model=a.model_name, timeout=600, type="think")
    GEN = {"n": 1, "temperature": 0.6, "top_p": 0.95, "seed": a.seed, "max_tokens": a.max_tokens}
    async def solver(prompt: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        texts = await engine.agenerate(prompt, **GEN)
        out = texts[0] if isinstance(texts, list) else texts
        return {"output": out, "time_s": time.perf_counter() - t0}
    ev = Math500Eval(number_of_samples_to_test=a.num_samples, seed=a.seed, output_path=a.output_path, max_workers=a.batch_size)
    s = await ev.run_async(solver=solver)
    print(f"[THINK-baseline] MATH-500 accuracy: {s['accuracy']:.4f} ({s['num_correct']}/{s['num_samples']})")

if __name__ == "__main__":
    asyncio.run(main())
