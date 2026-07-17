#!/usr/bin/env python3
"""AlpacaEval generation (baseline or SAGE) on the local alpaca_eval.json instructions.
Saves outputs + char/token length for the verbosity analysis. Judging is separate."""
from __future__ import annotations
import argparse, asyncio, json, random, sys, time
from pathlib import Path
from typing import Any, Dict
_R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_R))
from core.vllm_client import get_engine
from experiments.sage.solver import process_query, load_prompt, load_configurations

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", default="localhost"); p.add_argument("--port", type=int, default=9090)
    p.add_argument("--model-name", default="Qwen/Qwen3-8B-FP8")
    p.add_argument("--num-samples", type=int, default=200); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--method", choices=["baseline", "sage"], default="baseline")
    p.add_argument("--data", default=str(_R / "configs" / "alpaca_eval.json"))
    p.add_argument("--output", required=True)
    p.add_argument("--judge-prompt", default=str(_R / "configs" / "ifeval_judge_prompt.txt"))
    p.add_argument("--judge-config", default=str(_R / "configs" / "ifeval_judge_config.json"))
    return p.parse_args()

async def main():
    a = parse_args()
    base_url = f"http://{a.ip}:{a.port}/v1"
    data = json.load(open(a.data))
    idx = list(range(len(data))); random.Random(a.seed).shuffle(idx); idx = idx[:a.num_samples]
    items = [data[i] for i in idx]
    if a.method == "baseline":
        engine = get_engine(base_url=base_url, model=a.model_name, timeout=300, type="aug")
        GEN = {"n": 1, "temperature": 0.7, "top_p": 0.95, "seed": a.seed, "max_tokens": 2048}
        async def solve(instr):
            t = await engine.agenerate(instr, **GEN)
            return t[0] if isinstance(t, list) else t
    else:
        jt = load_prompt(a.judge_prompt); jc = load_configurations(a.judge_config)
        async def solve(instr):
            r = await process_query(instr, judge_prompt_templates=[jt], judge_configurations=[jc],
                num_optimization_epochs=2, base_url=base_url, model_name=a.model_name, number_of_gens_per_epoch=7)
            return r.get("output", "")
    out = []
    for s in range(0, len(items), a.batch_size):
        batch = items[s:s+a.batch_size]
        res = await asyncio.gather(*[solve(it["instruction"]) for it in batch])
        for it, o in zip(batch, res):
            out.append({"instruction": it["instruction"], "output": o, "reference": it["output"],
                        "out_chars": len(o), "ref_chars": len(it["output"])})
        print(f"[alpaca-{a.method}] {len(out)}/{len(items)}")
    json.dump(out, open(a.output, "w"), ensure_ascii=False, indent=1)
    import statistics
    print(f"[alpaca-{a.method}] done. median out_chars={statistics.median([x['out_chars'] for x in out]):.0f} "
          f"ref_chars={statistics.median([x['ref_chars'] for x in out]):.0f}")

if __name__ == "__main__":
    asyncio.run(main())
