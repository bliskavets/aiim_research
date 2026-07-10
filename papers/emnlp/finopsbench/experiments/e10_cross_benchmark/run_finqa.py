"""E12 extension: run the SAME model on FinQA in the static reading setting.

Mirrors run_tatqa.py. FinQA questions are answered from the provided table +
pre/post text (no tools), scored with the v2 percent-robust comparator. Paired
with FinOpsBench-v2 closed-book vs agentic, this shows static finance QA is largely
solved by reading while FinOpsBench requires tool use.

Usage:
    export OPENROUTER_API_KEY=...
    python run_finqa.py --model openai/gpt-4.1-mini --finqa /tmp/finqa_train.json --n 200
"""
import argparse, asyncio, json, os, sys
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


def table_md(rows):
    return "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)


def load(path, n):
    docs = json.loads(Path(path).read_text())
    items = []
    for d in docs:
        qa = d.get("qa") or {}
        q = qa.get("question")
        if not q:
            continue
        gold = qa.get("exe_ans")
        gold = str(gold) if gold is not None else str(qa.get("answer", ""))
        pre = " ".join(d.get("pre_text") or [])
        post = " ".join(d.get("post_text") or [])
        items.append({"uid": d.get("id"), "question": q, "gold": gold,
                      "table": table_md(d.get("table") or []),
                      "text": (pre + "\n" + post).strip()})
    import random
    return random.Random(13).sample(items, min(n, len(items)))


async def ask(client, model, it, sem):
    prompt = (f"Financial disclosure:\n{it['text']}\n\nTable:\n{it['table']}\n\n"
              f"Question: {it['question']}\n"
              "Think step by step, then end with 'Final answer: <value>' as a plain number.")
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
    p.add_argument("--finqa", default="/tmp/finqa_train.json")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=10)
    args = p.parse_args()
    items = load(args.finqa, args.n)
    out = HERE / "results" / f"finqa_{args.model.replace('/', '_')}.jsonl"
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
            f.write(json.dumps({"uid": it["uid"], "gold": it["gold"],
                                "prediction": pred, "passed": ok, "cost": c}) + "\n")
    rows = [json.loads(l) for l in out.open()]
    acc = 100 * sum(r["passed"] for r in rows) / len(rows)
    print(f"FinQA (static reading) / {args.model}: n={len(rows)} acc={acc:.1f}% cost=${cost:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
