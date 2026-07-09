"""Measure the real $/tokens/runtime to construct ONE FinOpsBench-v1 example.

Replays the actual v1 pipeline LLM calls through OpenRouter with the SAME models
the paper used (query/schema/data/trace = gpt-4.1-mini; SQL-repair/answer-check =
o4-mini; panel = claude-sonnet-4 + o4-mini + o3-mini). Prompts are copied verbatim
from v1/finopsbench_v1/pipeline/*.py. Records per-stage tokens, cost (usage.cost)
and wall-time; the query-generation call yields 20 queries so its cost is amortised.

Usage:
    export OPENROUTER_API_KEY=...
    python measure_v1.py --n 1
"""

import argparse
import json
import os
import re
import sqlite3
import time
from pathlib import Path

from openai import OpenAI

API = "https://openrouter.ai/api/v1"
GEN = "openai/gpt-4.1-mini"
REASON = "openai/o4-mini"
PANEL = ["anthropic/claude-sonnet-4", "openai/o4-mini", "openai/o3-mini"]

SEED_QUERY = "What is the total outstanding balance on all overdue invoices by supplier category?"

P_QUERY = """## Task
You are a financial expert in business fincances and audit. Study the sample query users gave to
a financial AI system and output 20 new queries a different user in a different situation could
give to the same AI system. Do not output the original query.

## Query
{question}

## Output format
Output JSON like this:
[
  {{"user_role": "...", "query": "..."}}, ...
]"""

P_SCHEMA = """# Task
You are a QA expert working on AI agents in finances. You are given a query asked to that agent.
You job is to create a database schema that is
- Plausible for a real company
- Sufficient for answering that query
- Succinct

# Input query
{query}

# Output format
Output your reasoning followed by a single Markdown-fenced SQL block that contains SQL statements for creating the database. Optimize for sqlite3"""

P_DATA = """<task>
You are a QA expert working on AI agents in finances. You are given a query asked to that agent and a database schema that should be queried.
Create the data that should be stored in this database that:
- is sufficient to answer the query
- is looking natural
- contains some distractor data that can guide the agent in a wrong direction
Also, create the expected answer to that query given the created data.
Today's date is 2025-07-24.
</task>

<query>
{query}
</query>

<database_schema>
{schema}
</database_schema>

<output_format_description>
# Reasoning
...
# Data
```sql
... a single SQL code block that inserts the data ...
```
# Expected output
...
</output_format_description>"""

P_JUDGE = """# Task
You are a senior QA engineer working on AI agents for finance. Judge the test case and check:
- data natural, trace reasonable, trace sound, reasoning grounded, answer addresses the query.

# Test case
## Query
{query}
## SQL schema
{schema}
## Data
{data}
## Agent trace
{trace}

# Output format
Reply with JSON: {{"data_is_natural":1|0,"trace_is_reasonable":1|0,"trace_is_sound":1|0,"reasoning_is_grounded":1|0,"answer_is_sound":1|0,"feedback":"..."}}"""

P_CHECK = """# Task
You are a financial expert. You have a user's query and two answers. The first is definitely correct.
Judge whether the second answer is correct.

## User query
{query}
## Answer 1
{expected}
## Answer 2
{got}

# Output format
```json
{{"reasoning":"...","is_correct":true|false}}
```"""

SYS_AGENT = """You are a financial analysis assistant for the {role}.
You have access to a tool `execute_sql` that runs SQL against the company database.
Today's date is 2025-07-24.
The database schema is:
{schema}"""

SQL_TOOL = [{"type": "function", "function": {
    "name": "execute_sql",
    "description": "Execute a SQL query against the company financial database and return JSON results.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]


def sql_block(text):
    m = re.search(r"```sql\s*(.*?)```", text, re.S)
    return m.group(1).strip() if m else text


def call(client, model, messages, tools=None, max_tokens=6000):
    t0 = time.time()
    kw = dict(model=model, messages=messages, max_tokens=max_tokens)
    if tools:
        kw["tools"] = tools
    resp = client.chat.completions.create(**kw)
    dt = time.time() - t0
    u = resp.usage.model_dump() if resp.usage else {}
    return resp, {"cost": u.get("cost") or 0.0, "in": u.get("prompt_tokens"),
                  "out": u.get("completion_tokens"), "s": round(dt, 1)}


def run_sql(conn, q):
    try:
        cur = conn.cursor()
        cur.execute(q)
        if cur.description:
            cols = [d[0] for d in cur.description]
            return json.dumps({"columns": cols, "rows": cur.fetchall()[:50]})
        conn.commit()
        return json.dumps({"rows_affected": cur.rowcount})
    except Exception as e:
        return json.dumps({"error": str(e)})


def measure_one(client):
    stages = []

    def rec(name, model, meta, note=""):
        stages.append({"stage": name, "model": model, **meta, "note": note})
        print(f"  {name:26} {model:26} in={meta['in']} out={meta['out']} ${meta['cost']:.4f} {meta['s']}s {note}")

    # 1 query gen (amortised /20)
    r, m = call(client, GEN, [{"role": "user", "content": P_QUERY.format(question=SEED_QUERY)}], max_tokens=4000)
    rec("1_query_gen (/20)", GEN, {**m, "cost": m["cost"] / 20}, "cost shown per-example (1 call -> 20 queries)")
    try:
        query = json.loads(re.search(r"\[.*\]", r.choices[0].message.content, re.S).group(0))[0]["query"]
    except Exception:
        query = "Which suppliers have overdue invoices that have not been approved for payment yet?"

    # 2 schema
    r, m = call(client, GEN, [{"role": "user", "content": P_SCHEMA.format(query=query)}]); rec("2_schema_gen", GEN, m)
    schema = sql_block(r.choices[0].message.content)

    # 3 data
    r, m = call(client, GEN, [{"role": "user", "content": P_DATA.format(query=query, schema=schema)}]); rec("3_data_gen", GEN, m)
    data_txt = r.choices[0].message.content
    data_sql = sql_block(data_txt)
    expected = data_txt.split("Expected output")[-1][:400].strip()

    # 4 build DB (+ optional o4-mini repair)
    conn = sqlite3.connect(":memory:")
    build_ok = True
    try:
        conn.executescript(schema); conn.executescript(data_sql); conn.commit()
    except Exception as e:
        build_ok = False
        r, m = call(client, REASON,
                    [{"role": "user", "content": f"Fix this failing sqlite script. Error: {e}\n\nSCHEMA:\n{schema}\n\nDATA:\n{data_sql}\n\nReturn corrected full SQL in a ```sql block."}])
        rec("4_sql_repair", REASON, m, "(triggered: build failed)")
        try:
            fixed = sql_block(r.choices[0].message.content)
            conn = sqlite3.connect(":memory:"); conn.executescript(fixed); conn.commit(); build_ok = True
        except Exception:
            pass
    if build_ok:
        rec("4_sql_validation", "(sqlite, local)", {"cost": 0.0, "in": 0, "out": 0, "s": 0.0}, "executed, no repair needed")

    # 5 agent trace (native tool-calling loop, up to 6 rounds)
    msgs = [{"role": "system", "content": SYS_AGENT.format(role="Financial Controller", schema=schema)},
            {"role": "user", "content": query}]
    trace_cost = trace_in = trace_out = 0
    trace_s = 0.0
    final = ""
    for _ in range(6):
        r, m = call(client, GEN, msgs, tools=SQL_TOOL, max_tokens=3000)
        trace_cost += m["cost"]; trace_in += m["in"] or 0; trace_out += m["out"] or 0; trace_s += m["s"]
        msg = r.choices[0].message
        tcs = msg.tool_calls or []
        msgs.append({"role": "assistant", "content": msg.content or "", "tool_calls": [tc.model_dump() for tc in tcs] if tcs else None})
        if not tcs:
            final = msg.content or ""
            break
        for tc in tcs:
            try:
                q = json.loads(tc.function.arguments).get("query", "")
            except Exception:
                q = ""
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": run_sql(conn, q) if build_ok else "{}"})
    rec("5_agent_trace (loop)", GEN, {"cost": trace_cost, "in": trace_in, "out": trace_out, "s": round(trace_s, 1)}, "multi-round tool-calling")
    trace_text = json.dumps(msgs)[:6000]

    # 6 panel (3 judges)
    for jm in PANEL:
        r, m = call(client, jm, [{"role": "user", "content": P_JUDGE.format(query=query, schema=schema, data=data_sql[:3000], trace=trace_text)}])
        rec(f"6_judge:{jm.split('/')[-1]}", jm, m)

    # final answer-consistency check
    r, m = call(client, REASON, [{"role": "user", "content": P_CHECK.format(query=query, expected=expected, got=final)}])
    rec("final_answer_check", REASON, m)

    tot = sum(s["cost"] for s in stages)
    sec = sum(s["s"] for s in stages)
    print(f"  => total ${tot:.4f}, {sec:.0f}s\n")
    return {"stages": stages, "total_cost": round(tot, 4), "total_seconds": round(sec, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "v1_cost.json")
    args = p.parse_args()
    client = OpenAI(base_url=API, api_key=os.environ["OPENROUTER_API_KEY"])
    runs = []
    for i in range(args.n):
        print(f"--- v1 example {i+1}/{args.n} ---")
        runs.append(measure_one(client))
    avg_c = sum(r["total_cost"] for r in runs) / len(runs)
    avg_s = sum(r["total_seconds"] for r in runs) / len(runs)
    out = {"n_measured": len(runs), "avg_cost_per_example": round(avg_c, 4),
           "avg_seconds_per_example": round(avg_s, 1), "runs": runs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"AVG per v1 example: ${avg_c:.4f}, {avg_s:.0f}s (n={len(runs)}) -> {args.out}")


if __name__ == "__main__":
    main()
