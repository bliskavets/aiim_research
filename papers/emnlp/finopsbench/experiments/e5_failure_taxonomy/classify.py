"""Classify each failing trace into one failure category via an LLM.

Reads failures.jsonl, writes classified.jsonl (resumable). Cost is taken from
OpenRouter's per-request usage.cost.

Usage:
    export OPENROUTER_API_KEY=...
    python classify.py --model openai/gpt-4.1-mini [--limit N]
"""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

from openai import AsyncOpenAI

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

CATEGORIES = {
    "wrong_tool_selection": "used the wrong tool or a distractor tool / queried the wrong table or entity",
    "malformed_arguments": "called the right tool but with wrong arguments / wrong SQL predicate, join, filter or threshold",
    "incomplete_retrieval": "did not retrieve all the rows/values the answer needs (missed data, stopped early)",
    "calculation_error": "retrieved the right data but aggregated/computed it wrong (arithmetic, sum, ratio)",
    "financial_misunderstanding": "misread the financial concept, definition, or business rule the question asks for",
    "format_unit_error": "reached the right underlying value but wrong unit / sign / rounding / output format",
    "round_limit_exhaustion": "ran out of steps or never produced a usable final answer",
    "other": "none of the above",
}

PROMPT = """You are analysing why a financial-analysis agent got a task WRONG.
Assign exactly ONE category that best explains the primary cause of failure.

Categories:
{cats}

# Question
{query}

# Expected (gold) answer
{expected}

# Agent's final answer
{final}

# Agent's tool-call trace
{trace}
{judge}
Respond with a JSON object only:
{{"category": "<one key from the list>", "rationale": "<one short sentence>"}}"""


def key(r):
    return hashlib.md5(f"{r['version']}|{r['model']}|{r['query']}".encode()).hexdigest()


async def classify_one(client, model, r, sem):
    judge = f"\n# Independent judge's reasoning for why it is wrong\n{r['judge_reasoning']}\n" if r.get("judge_reasoning") else ""
    prompt = PROMPT.format(
        cats="\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items()),
        query=r["query"], expected=r["expected"], final=r["final_answer"],
        trace=r["trace"][:4000], judge=judge,
    )
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}],
                    max_tokens=300, response_format={"type": "json_object"},
                )
                usage = resp.usage.model_dump() if resp.usage else {}
                obj = json.loads(resp.choices[0].message.content or "{}")
                cat = obj.get("category", "other")
                if cat not in CATEGORIES:
                    cat = "other"
                return {**{k: r[k] for k in ("version", "model", "query")},
                        "category": cat, "rationale": obj.get("rationale", ""),
                        "cost": usage.get("cost")}
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return {**{k: r[k] for k in ("version", "model", "query")},
                            "category": "other", "rationale": f"[error] {e}", "cost": 0}
                await asyncio.sleep(4 * (attempt + 1))


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/gpt-4.1-mini")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--infile", type=Path, default=Path(__file__).parent / "failures.jsonl")
    p.add_argument("--outfile", type=Path, default=Path(__file__).parent / "classified.jsonl")
    args = p.parse_args()

    rows = [json.loads(l) for l in args.infile.open() if l.strip()]
    done = {}
    if args.outfile.exists():
        for l in args.outfile.open():
            if l.strip():
                d = json.loads(l)
                done[hashlib.md5(f"{d['version']}|{d['model']}|{d['query']}".encode()).hexdigest()] = True
    todo = [r for r in rows if key(r) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(rows)} failures, {len(done)} already classified, {len(todo)} to do")

    client = AsyncOpenAI(base_url=OPENROUTER_API_BASE, api_key=os.environ["OPENROUTER_API_KEY"])
    sem = asyncio.Semaphore(args.concurrency)
    cost = 0.0
    with args.outfile.open("a") as f:
        tasks = [asyncio.create_task(classify_one(client, args.model, r, sem)) for r in todo]
        for i, fut in enumerate(asyncio.as_completed(tasks), 1):
            rec = await fut
            cost += rec.get("cost") or 0.0
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} classified, cost ${cost:.3f}")
    print(f"done, run cost ${cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
