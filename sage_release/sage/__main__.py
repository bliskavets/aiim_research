"""Command-line entry point:  python -m sage --preset math500 --input prompts.jsonl

Reads a JSONL file of ``{"prompt": ...}`` records, runs SAGE with the given
benchmark preset against a running vLLM server, and writes ``{"prompt", "output"}``
records to the output file.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from sage.config import load_preset, PRESETS
from sage.solver import process_query


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="sage", description="Run SAGE test-time alignment.")
    p.add_argument("--preset", required=True, choices=sorted(PRESETS),
                   help="benchmark judge recipe / best settings")
    p.add_argument("--input", required=True, help="JSONL file with a 'prompt' field per line")
    p.add_argument("--output", required=True, help="destination JSONL file")
    p.add_argument("--model-name", default="Qwen/Qwen3-8B-FP8")
    p.add_argument("--base-url", default="http://localhost:9090/v1")
    p.add_argument("--engine-type", default=None, choices=[None, "aug", "think", "no_think"],
                   help="override the preset engine (use 'think' for reasoning-mode runs)")
    p.add_argument("--concurrency", type=int, default=8, help="max concurrent problems")
    return p.parse_args(argv)


async def _run(args) -> None:
    overrides = {"engine_type": args.engine_type} if args.engine_type else {}
    kwargs = load_preset(args.preset, **overrides)

    records = [json.loads(line) for line in open(args.input) if line.strip()]
    sem = asyncio.Semaphore(args.concurrency)
    out = [None] * len(records)

    async def _one(i, rec):
        async with sem:
            try:
                res = await process_query(
                    rec["prompt"], model_name=args.model_name, base_url=args.base_url, **kwargs
                )
                out[i] = {"prompt": rec["prompt"], "output": res["output"]}
            except Exception as exc:  # keep going; record the failure
                out[i] = {"prompt": rec["prompt"], "output": "", "error": repr(exc)}
            print(f"[sage] {sum(x is not None for x in out)}/{len(records)} done", file=sys.stderr)

    await asyncio.gather(*[_one(i, r) for i, r in enumerate(records)])
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[sage] wrote {len(out)} records -> {args.output}", file=sys.stderr)


def main(argv=None) -> None:
    asyncio.run(_run(_parse_args(argv)))


if __name__ == "__main__":
    main()
