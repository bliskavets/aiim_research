"""E1: closed-book contamination baseline for FinOpsBench-v2.

For every v2 environment, the model receives the original system prompt
(scenario + tool signatures + question) but NO callable tools, and is asked
to answer directly. If FinQA memorization provided an answer pathway, the
model could recall the gold value without any tool calls. Keeping the tool
signatures in the prompt makes the test conservative: the model sees strictly
more information than a plain closed-book setup, so the measured accuracy is
an upper bound on what contamination can deliver.

Usage:
    export OPENROUTER_API_KEY=...
    python run_closed_book.py --model openai/gpt-5-mini --reasoning_effort low
    python run_closed_book.py --model qwen/qwen3-30b-a3b
    python run_closed_book.py --model openai/gpt-4.1 --sample 300

Results go to results/<model-slug>.jsonl (one line per item, resumable).
"""

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

from openai import AsyncOpenAI

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

INSTRUCTION = (
    "The tools listed above are unavailable in this session. "
    "Answer the question using only the scenario text above and your own knowledge. "
    "If the exact value cannot be determined, give your single best estimate. "
    "Reply with the final answer only: a plain number (or yes/no), no other text."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_root", type=Path, default=Path("/tmp/FinOpsBench"),
                   help="Path to a FinOpsBench checkout (uses v2/finqa_agents and v2/compare_outputs.py)")
    p.add_argument("--model", required=True, help="OpenRouter model id, e.g. openai/gpt-5-mini")
    p.add_argument("--reasoning_effort", default=None, choices=[None, "low", "medium", "high"],
                   help="Reasoning effort for thinking models (kept low for cost control)")
    p.add_argument("--max_tokens", type=int, default=3000)
    p.add_argument("--sample", type=int, default=None, help="Random subsample size (seed 13)")
    p.add_argument("--limit", type=int, default=None, help="Process at most N items (pilot runs)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out_dir", type=Path, default=Path(__file__).parent / "results")
    return p.parse_args()


def load_items(root: Path) -> list[dict]:
    items = []
    for d in sorted((root / "v2" / "finqa_agents").glob("agent_*")):
        prompt_f = d / "agent_system_prompt.txt"
        gold_f = d / "initial_solution.txt"
        if prompt_f.is_file() and gold_f.is_file():
            items.append({
                "agent_id": d.name,
                "system_prompt": prompt_f.read_text(),
                "gold": gold_f.read_text().strip(),
            })
    return items


async def ask(client: AsyncOpenAI, args: argparse.Namespace, item: dict, sem: asyncio.Semaphore) -> dict:
    extra_body = {}
    if args.reasoning_effort:
        extra_body["reasoning"] = {"effort": args.reasoning_effort}
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": item["system_prompt"]},
                        {"role": "user", "content": INSTRUCTION},
                    ],
                    max_tokens=args.max_tokens,
                    extra_body=extra_body,
                )
                usage = resp.usage.model_dump() if resp.usage else {}
                return {
                    "agent_id": item["agent_id"],
                    "gold": item["gold"],
                    "prediction": (resp.choices[0].message.content or "").strip(),
                    "cost": usage.get("cost"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                }
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return {"agent_id": item["agent_id"], "gold": item["gold"], "error": str(e)}
                await asyncio.sleep(5 * (attempt + 1))


async def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.benchmark_root / "v2"))
    from compare_outputs import compare_answers  # noqa: E402  (benchmark scoring, unchanged)

    items = load_items(args.benchmark_root)
    print(f"{len(items)} items with prompt+gold")
    if args.sample:
        items = random.Random(13).sample(items, args.sample)
    if args.limit:
        items = items[: args.limit]

    out_path = args.out_dir / (args.model.replace("/", "_") + ".jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["agent_id"] for line in out_path.open() if line.strip()}
        print(f"resuming: {len(done)} already done")
    todo = [it for it in items if it["agent_id"] not in done]

    client = AsyncOpenAI(base_url=OPENROUTER_API_BASE, api_key=os.environ["OPENROUTER_API_KEY"])
    sem = asyncio.Semaphore(args.concurrency)

    n_done = n_pass = 0
    total_cost = 0.0
    with out_path.open("a") as f_out:
        tasks = [asyncio.create_task(ask(client, args, it, sem)) for it in todo]
        for fut in asyncio.as_completed(tasks):
            rec = await fut
            if "error" not in rec:
                rec["passed"] = bool(compare_answers(rec["prediction"], rec["gold"]))
                n_pass += rec["passed"]
                total_cost += rec.get("cost") or 0.0
            f_out.write(json.dumps(rec) + "\n")
            f_out.flush()
            n_done += 1
            if n_done % 50 == 0 or n_done == len(todo):
                print(f"{n_done}/{len(todo)} acc-so-far={n_pass / max(n_done, 1):.3f} cost=${total_cost:.3f}")

    print(f"finished {len(todo)} items, run cost ${total_cost:.4f} -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
