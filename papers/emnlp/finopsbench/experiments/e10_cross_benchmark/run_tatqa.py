"""Cross-benchmark control: run the SAME model on an external open competitor
(TAT-QA, static financial table+text QA) and compare with FinOpsBench-v2.

TAT-QA is one of the reading-comprehension finance benchmarks the paper contrasts
against. We evaluate its *arithmetic* questions (numeric, comparable to ours) in
the standard static setting (table + paragraphs in the prompt, no tools) and score
numerically with the benchmark's own percent-robust comparator. The point: the same
model reads a static finance benchmark well, whereas FinOpsBench requires tool use
(its closed-book accuracy collapses) — a capability static competitors do not test.

Usage:
    export OPENROUTER_API_KEY=...
    python run_tatqa.py --model openai/gpt-4.1-mini --n 150
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI

API = "https://openrouter.ai/api/v1"
HERE = Path(__file__).parent
sys.path.insert(0, "/tmp/FinOpsBench/v2")
from compare_outputs import compare_answers, extract_number_from_answer  # noqa: E402


def robust(pred, gold):
    if compare_answers(pred, gold):
        return True
    pv, _ = extract_number_from_answer(pred)
    gv, gp = extract_number_from_answer(gold)
    if pv == "no answer" or gv == "no answer":
        return False
    tol = max(10 ** (-gp) * 0.6, abs(gv) * 0.01)
    return abs(pv - gv) <= tol or abs(pv / 100 - gv) <= tol or abs(pv - gv * 100) <= max(tol * 100, 0.6)


def table_md(t):
    rows = t["table"] if isinstance(t, dict) else t
    return "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)


def load(path, n):
    docs = json.loads(Path(path).read_text())
    items = []
    for doc in docs:
        paras = "\n".join(p["text"] if isinstance(p, dict) else str(p) for p in doc.get("paragraphs", []))
        tbl = table_md(doc["table"])
        for q in doc["questions"]:
            if q.get("answer_type") != "arithmetic":
                continue
            ans = q["answer"]
            gold = str(ans[0] if isinstance(ans, list) else ans)
            scale = q.get("scale") or ""
            items.append({"uid": q["uid"], "question": q["question"], "gold": gold,
                          "scale": scale, "table": tbl, "paras": paras})
    import random
    return random.Random(13).sample(items, min(n, len(items)))


async def ask(client, model, it, sem):
    prompt = (f"Financial disclosure:\n{it['paras']}\n\nTable:\n{it['table']}\n\n"
              f"Question: {it['question']}\n"
              f"Think step by step, then end with 'Final answer: <value>' as a plain number"
              + (f" (in {it['scale']})." if it['scale'] else "."))
    async with sem:
        for a in range(3):
            try:
                r = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}], max_tokens=2500)
                u = r.usage.model_dump() if r.usage else {}
                return it, (r.choices[0].message.content or "").strip(), u.get("cost") or 0.0
            except Exception as e:  # noqa: BLE001
                if a == 2:
                    return it, f"[error] {e}", 0.0
                await asyncio.sleep(4 * (a + 1))


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tatqa", default="/tmp/tatqa_dev.json")
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--concurrency", type=int, default=10)
    args = p.parse_args()
    items = load(args.tatqa, args.n)
    out = HERE / "results" / f"tatqa_{args.model.replace('/', '_')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=API, api_key=os.environ["OPENROUTER_API_KEY"])
    sem = asyncio.Semaphore(args.concurrency)
    cost = 0.0
    with out.open("w") as f:
        tasks = [asyncio.create_task(ask(client, args.model, it, sem)) for it in items]
        for fut in asyncio.as_completed(tasks):
            it, pred, c = await fut
            cost += c
            ok = robust(pred, it["gold"])
            f.write(json.dumps({"uid": it["uid"], "gold": it["gold"], "scale": it["scale"],
                                "prediction": pred, "passed": ok, "cost": c}) + "\n")
    rows = [json.loads(l) for l in out.open()]
    acc = 100 * sum(r["passed"] for r in rows) / len(rows)
    print(f"TAT-QA (arithmetic, static reading) / {args.model}: n={len(rows)} acc={acc:.1f}% cost=${cost:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
