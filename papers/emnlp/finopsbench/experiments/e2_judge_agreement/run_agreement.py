"""E2: LLM-judge vs deterministic numeric matching on FinOpsBench-v1.

Takes the released v1 pool (query, expected_output, reference agent trace),
treats the trace's final message as the candidate answer, and scores it two
ways on the numeric subset:

1. deterministically -- extract the last number from both answers and compare
   with the same tolerance rule used by FinOpsBench-v2 scoring;
2. with the LLM judge -- the *verbatim* EVALUATE_RESULT_PROMPT from the v1
   evaluation harness, run through OpenRouter.

Reports percentage agreement, Cohen's kappa, and the confusion matrix.
A high agreement means the LLM judge behaves as a calibrated comparator
rather than an extra source of evaluation uncertainty.

Usage:
    export OPENROUTER_API_KEY=...
    python run_agreement.py --judge_model openai/o4-mini --sample 500
"""

import argparse
import asyncio
import gzip
import json
import os
import random
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

# Verbatim from finopsbench_v1/evaluation/run_eval.py (RichPromptTemplate body),
# with {{...}} placeholders filled by str.format below.
JUDGE_PROMPT = """You are a financial analysis expert. You need to check if the agent's answer is correct.
You are given a query, the correct answer, and the agent's answer.
You need to evaluate the agent's answer and report if it is correct or not.

# Query
{query}

# Correct answer
{expected_output}

# Agent's answer
{agent_answer}

# Output format
```json
{{
    "reasoning": "... your reasoning here ...",
    "correct": true | false,
}}
```
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_root", type=Path, default=Path("/tmp/FinOpsBench"))
    p.add_argument("--judge_model", default="openai/o4-mini",
                   help="OpenRouter id of the judge (paper default: o4-mini)")
    p.add_argument("--sample", type=int, default=500, help="Numeric-subset sample size (seed 13)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out_dir", type=Path, default=Path(__file__).parent / "results")
    return p.parse_args()


def final_answer(item: dict) -> str | None:
    dialog = item.get("agent_dialog") or []
    for msg in reversed(dialog):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return None


def parse_judge_verdict(text: str) -> bool | None:
    m = re.search(r'"correct"\s*:\s*(true|false)', text, re.IGNORECASE)
    return m.group(1).lower() == "true" if m else None


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float:
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


async def judge_one(client: AsyncOpenAI, model: str, rec: dict, sem: asyncio.Semaphore) -> dict:
    prompt = JUDGE_PROMPT.format(
        query=rec["query"], expected_output=rec["gold"], agent_answer=rec["answer"]
    )
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                )
                usage = resp.usage.model_dump() if resp.usage else {}
                verdict = parse_judge_verdict(resp.choices[0].message.content or "")
                return {**rec, "judge_correct": verdict, "cost": usage.get("cost")}
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return {**rec, "judge_correct": None, "error": str(e)}
                await asyncio.sleep(5 * (attempt + 1))


async def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.benchmark_root / "v2"))
    from compare_outputs import extract_number_from_answer, numbers  # noqa: E402

    def _to_value(match: tuple) -> float:
        num, _, percent = match
        value = float(num.replace(",", ""))
        return value / 100 if percent.endswith("%") else value

    n_total = 0

    pool = args.benchmark_root / "v1" / "data" / "finopsbench_v1_pool.jsonl.gz"
    records = []
    for line in gzip.open(pool, "rt"):
        item = json.loads(line)
        answer = final_answer(item)
        gold = item.get("expected_output")
        if not answer or not gold:
            continue
        gold_numbers = numbers.findall(gold)
        n_total += 1
        if len(gold_numbers) != 1:
            continue  # scalar-numeric subset: deterministic scoring is well-defined
        gold_value, gold_precision = extract_number_from_answer(gold)
        tolerance = 10 ** (-gold_precision) * 0.6
        matched = any(
            abs(_to_value(m) - gold_value) <= tolerance for m in numbers.findall(answer)
        )
        records.append({
            "query": item["query"],
            "gold": gold,
            "answer": answer,
            "numeric_match": matched,
        })
    print(f"scalar-numeric-gold items: {len(records)}/{n_total} "
          f"(expected_output contains exactly one number; deterministic rule: "
          f"some number in the answer matches it within the v2 tolerance)")

    sample = random.Random(13).sample(records, min(args.sample, len(records)))

    out_path = args.out_dir / f"agreement_scalar_{args.judge_model.replace('/', '_')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["query"] for l in out_path.open() if l.strip()}
        print(f"resuming: {len(done)} already judged")
    todo = [r for r in sample if r["query"] not in done]

    client = AsyncOpenAI(base_url=OPENROUTER_API_BASE, api_key=os.environ["OPENROUTER_API_KEY"])
    sem = asyncio.Semaphore(args.concurrency)
    total_cost = 0.0
    with out_path.open("a") as f_out:
        tasks = [asyncio.create_task(judge_one(client, args.judge_model, r, sem)) for r in todo]
        for i, fut in enumerate(asyncio.as_completed(tasks), 1):
            rec = await fut
            total_cost += rec.get("cost") or 0.0
            f_out.write(json.dumps(rec) + "\n")
            f_out.flush()
            if i % 50 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} judged, cost ${total_cost:.3f}")

    # ---- report ----
    pairs = []
    for line in out_path.open():
        rec = json.loads(line)
        if rec.get("judge_correct") is not None:
            pairs.append((rec["numeric_match"], rec["judge_correct"]))
    n = len(pairs)
    agree = sum(a == b for a, b in pairs)
    both_t = sum(a and b for a, b in pairs)
    both_f = sum((not a) and (not b) for a, b in pairs)
    num_only = sum(a and not b for a, b in pairs)
    judge_only = sum(b and not a for a, b in pairs)
    summary = {
        "judge_model": args.judge_model,
        "n": n,
        "scalar_numeric_gold_items_in_pool": len(records),
        "agreement": round(agree / n, 4),
        "cohen_kappa": round(cohen_kappa(pairs), 4),
        "confusion": {
            "both_correct": both_t,
            "both_incorrect": both_f,
            "numeric_only": num_only,
            "judge_only": judge_only,
        },
        "judge_run_cost_usd": round(total_cost, 4),
    }
    print(json.dumps(summary, indent=2))
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
