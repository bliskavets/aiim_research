"""Judge-swap robustness: does the evaluation verdict depend on the judge model?

We take the 170 human-labelled v1 items (92 numeric-vs-judge disagreements +
78 random agreement-stratum), and re-score each (query, gold, answer) triple with
several judge models from different vendors, using the paper's exact judge prompt.
We then measure how each judge agrees with the human label and with the others.
If the verdict is a property of the answer, not the model, agreement stays high
and the choice of judge model does not matter.

Original judge (o4-mini) verdicts are already stored as `judge_correct`.

Usage:
    export OPENROUTER_API_KEY=...
    python judge_swap.py --models anthropic/claude-sonnet-4.5 google/gemini-2.5-flash deepseek/deepseek-chat
"""
import argparse, asyncio, json, os, re
from pathlib import Path
from openai import AsyncOpenAI

HERE = Path(__file__).parent
API = "https://openrouter.ai/api/v1"

PROMPT = """You are a financial analysis expert. You need to check if the agent's answer is correct.
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
    "correct": true | false
}}
```
"""


def load_items():
    items, seen = [], set()
    files = [
        HERE / "results" / "disagreements_for_human_annotation.jsonl",
        HERE.parent / "e3_human_eval" / "data" / "sample_v1_judge.jsonl",
    ]
    for f in files:
        for line in f.read_text().splitlines():
            d = json.loads(line)
            hl = d.get("human_label")
            if hl not in (True, False):
                continue
            q = d["query"]
            if q in seen:
                continue
            seen.add(q)
            items.append({"query": q, "gold": d["gold"], "answer": d["answer"],
                          "human": bool(hl), "o4-mini": bool(d["judge_correct"])})
    return items


def parse_verdict(text):
    m = re.search(r'"correct"\s*:\s*(true|false)', text, re.I)
    if m:
        return m.group(1).lower() == "true"
    # fallback: last standalone true/false
    m = re.findall(r'\b(true|false)\b', text, re.I)
    return (m[-1].lower() == "true") if m else None


async def judge(client, model, it, sem):
    p = PROMPT.format(query=it["query"], expected_output=it["gold"], agent_answer=it["answer"])
    async with sem:
        for a in range(3):
            try:
                r = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": p}], max_tokens=1200)
                return parse_verdict(r.choices[0].message.content or ""), (r.usage.model_dump().get("cost") or 0.0 if r.usage else 0.0)
            except Exception as e:  # noqa: BLE001
                if a == 2:
                    return None, 0.0
                await asyncio.sleep(4 * (a + 1))


def kappa(a, b):
    """Cohen's kappa for two boolean lists (paired, drop None)."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if not n:
        return None
    po = sum(x == y for x, y in pairs) / n
    pa = sum(x for x, _ in pairs) / n
    pb = sum(y for _, y in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe != 1 else 1.0


def fleiss(cols):
    """Fleiss kappa across R judges (list of boolean lists), drop items with any None."""
    rows = list(zip(*cols))
    rows = [r for r in rows if all(v is not None for v in r)]
    n, R = len(rows), len(cols)
    if not n:
        return None
    p_yes = sum(sum(r) for r in rows) / (n * R)
    pe = p_yes ** 2 + (1 - p_yes) ** 2
    Pi = [ (sum(r) ** 2 + (R - sum(r)) ** 2 - R) / (R * (R - 1)) for r in rows ]
    pbar = sum(Pi) / n
    return round((pbar - pe) / (1 - pe), 3) if pe != 1 else 1.0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    items = load_items()
    print(f"loaded {len(items)} human-labelled items")
    client = AsyncOpenAI(base_url=API, api_key=os.environ["OPENROUTER_API_KEY"])
    sem = asyncio.Semaphore(a.concurrency)
    cost = 0.0
    for model in a.models:
        res = await asyncio.gather(*[judge(client, model, it, sem) for it in items])
        for it, (v, c) in zip(items, res):
            it[model] = v
            cost += c
        got = sum(1 for it in items if it[model] is not None)
        print(f"  {model}: {got}/{len(items)} verdicts")
    judges = ["o4-mini"] + a.models
    human = [it["human"] for it in items]
    summary = {"n": len(items), "cost": round(cost, 3), "per_judge": {}, "inter_judge_fleiss": None}
    for jm in judges:
        col = [it.get(jm) for it in items]
        agree = [x == h for x, h in zip(col, human) if x is not None]
        summary["per_judge"][jm] = {"agreement_with_human_pct": round(100 * sum(agree) / len(agree), 1),
                                    "cohen_kappa_vs_human": kappa(col, human)}
    summary["inter_judge_fleiss"] = fleiss([[it.get(jm) for it in items] for jm in judges])
    # pairwise agreement among judges (mean)
    (HERE / "results" / "judge_swap.json").write_text(json.dumps(summary, indent=2))
    (HERE / "results" / "judge_swap_items.jsonl").write_text(
        "\n".join(json.dumps(it) for it in items))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
