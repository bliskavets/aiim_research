"""Information-access ladder for FinOpsBench-v2 (Experiment 1).

Same 200 items, same model, three information-access modes:
  (a) question_only : just the question, no data, no tools  -> parametric/memorised
  (b) full_context  : original FinQA pre-text+table+post-text + question, no tools
                      (this is the static-FinQA condition = the reviewer's "reading" upper bound)
  (c) agentic       : tools only (run separately via the SA runner / E4)

This file runs the two no-tool rungs (a) and (b). Scoring uses the benchmark's
own compare_answers against our gold (initial_solution.txt). FinQA context is
recovered from train.json via the positional agent_N -> train[N] mapping.

Usage:
    export OPENROUTER_API_KEY=...
    python run_context.py --mode full_context --model openai/gpt-4.1-mini [--limit 10]
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI

API = "https://openrouter.ai/api/v1"
HERE = Path(__file__).parent

INSTR_QO = ("Answer the following financial question using only your own knowledge. "
            "No data is provided. Give your single best estimate. "
            "Reply with the final answer only: a plain number (or yes/no).")
INSTR_FC = ("Answer the question using ONLY the company disclosure provided above "
            "(narrative + table). Think step by step, then end with a line "
            "'Final answer: <value>' where the value is a plain number (or yes/no).")


def finqa_context(sample):
    pre = "\n".join(sample.get("pre_text", []))
    post = "\n".join(sample.get("post_text", []))
    t = sample.get("table") or []
    tbl = ""
    if t:
        rows = ["| " + " | ".join(map(str, t[0])) + " |", "|" + "---|" * len(t[0])]
        rows += ["| " + " | ".join(map(str, r)) + " |" for r in t[1:]]
        tbl = "\n".join(rows)
    return f"{pre}\n\n{tbl}\n\n{post}".strip()


async def ask(client, model, item, sem, max_tokens):
    async with sem:
        for a in range(3):
            try:
                r = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": item["prompt"]}], max_tokens=max_tokens)
                u = r.usage.model_dump() if r.usage else {}
                return item, (r.choices[0].message.content or "").strip(), u.get("cost") or 0.0
            except Exception as e:  # noqa: BLE001
                if a == 2:
                    return item, f"[error] {e}", 0.0
                await asyncio.sleep(4 * (a + 1))


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_root", type=Path, default=Path("/tmp/FinOpsBench"))
    p.add_argument("--finqa_train", type=Path, default=Path("/tmp/finqa_train.json"))
    p.add_argument("--mode", required=True, choices=["question_only", "full_context", "finqa_canonical"])
    p.add_argument("--model", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max_tokens", type=int, default=2500)
    args = p.parse_args()

    sys.path.insert(0, str(args.benchmark_root / "v2"))
    from compare_outputs import compare_answers

    subset = json.loads((HERE / "subset_200.json").read_text())
    train = json.loads(args.finqa_train.read_text())
    root = args.benchmark_root / "v2" / "finqa_agents"

    items = []
    for aid in subset:
        n = int(aid.split("_")[-1])
        d = root / aid
        gold = (d / "initial_solution.txt").read_text().strip()
        sample = train[n]
        q = sample["qa"]["question"]
        if args.mode == "question_only":
            prompt = f"{INSTR_QO}\n\nQuestion: {q}"
        elif args.mode == "finqa_canonical":
            # the native FinQA input: gold-retrieved supporting facts (qa.model_input)
            facts = "\n".join(f"- {txt}" for _id, txt in sample["qa"].get("model_input", []))
            prompt = (f"Based on the following financial facts, answer the question.\n\n"
                      f"Facts:\n{facts}\n\n{INSTR_FC}\n\nQuestion: {q}")
        else:
            prompt = f"Company disclosure:\n{finqa_context(sample)}\n\n{INSTR_FC}\n\nQuestion: {q}"
        items.append({"agent_id": aid, "gold": gold, "prompt": prompt})
    if args.limit:
        items = items[: args.limit]

    out = HERE / "results" / f"{args.mode}_{args.model.replace('/', '_')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["agent_id"] for l in out.open()} if out.exists() else set()
    todo = [it for it in items if it["agent_id"] not in done]

    client = AsyncOpenAI(base_url=API, api_key=os.environ["OPENROUTER_API_KEY"])
    sem = asyncio.Semaphore(args.concurrency)
    cost = 0.0
    with out.open("a") as f:
        tasks = [asyncio.create_task(ask(client, args.model, it, sem, args.max_tokens)) for it in todo]
        for fut in asyncio.as_completed(tasks):
            it, pred, c = await fut
            cost += c
            ok = bool(compare_answers(pred, it["gold"]))
            f.write(json.dumps({"agent_id": it["agent_id"], "gold": it["gold"], "prediction": pred,
                                "passed": ok, "cost": c}) + "\n")
    # tally over the whole file
    rows = [json.loads(l) for l in out.open()]
    acc = 100 * sum(r["passed"] for r in rows) / len(rows)
    print(f"{args.mode} / {args.model}: n={len(rows)} acc={acc:.1f}% run_cost=${cost:.3f} -> {out.name}")


if __name__ == "__main__":
    asyncio.run(main())
