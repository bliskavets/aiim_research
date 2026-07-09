"""Collect failing agent traces from both benchmark versions into one file.

v1 (primary): the archived evaluated runs (eval_sample_evaluated_*.jsonl) — clean
    because each carries the model's own run (evaluation.agent_dialog) plus the
    LLM judge verdict (evaluation.correct) and the judge's reasoning.
v2 (cross-version contrast): the E4 smolagents runs (DeepSeek-V3, Claude-4.5),
    scored by execution against the gold answer.

For every failure we also compute deterministic process metrics from the trace
(tool-call count, SQL/tool errors, turns, round-exhaustion). Output: failures.jsonl.

Usage:
    python extract_failures.py --v1_dir /tmp/ushmax/exp/expand-10k \
        --e4_dir ../e4_new_models/results --cap 150
"""

import argparse
import glob
import json
import random
from pathlib import Path

V1_EVALUATED = {
    "GPT-5": "eval_sample_evaluated_gpt_5.jsonl",
    "o4-mini": "eval_sample_evaluated_o4_mini.jsonl",
    "GPT-4.1": "eval_sample_evaluated_gpt_4.1.jsonl",
    "GPT-4.1-mini": "eval_sample_evaluated_gpt-4.1_mini.jsonl",
}
V2_RUNS = {
    "Claude-Sonnet-4.5": "anthropic_claude-sonnet-4.5",
    "DeepSeek-V3": "deepseek_deepseek-chat-v3-0324",
}
MAX_ROUNDS_V1, MAX_STEPS_V2 = 6, 10


def clip(s, n=600):
    s = str(s or "")
    return s if len(s) <= n else s[:n] + "…"


def v1_trace(dialog):
    """Compact a v1 SQL-agent dialog + deterministic metrics."""
    calls, tool_out, n_err, final = [], [], 0, ""
    for m in dialog:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                q = (tc.get("tool_kwargs") or {}).get("query", "")
                calls.append(q)
            if m.get("content"):
                final = m["content"]
        elif m.get("role") == "tool":
            c = m.get("content") or ""
            tool_out.append(c)
            if '"error"' in c.lower() or "error executing" in c.lower() or c.strip().lower().startswith("error"):
                n_err += 1
    lines = []
    for i, q in enumerate(calls):
        lines.append(f"SQL[{i+1}]: {clip(q, 300)}")
        if i < len(tool_out):
            lines.append(f"  -> {clip(tool_out[i], 200)}")
    return {
        "trace": "\n".join(lines),
        "final_answer": clip(final, 500),
        "n_tool_calls": len(calls),
        "n_sql_errors": n_err,
        "round_exhausted": len(calls) >= MAX_ROUNDS_V1 and not final,
    }


def v2_trace(verbose):
    steps = verbose.get("messages", [])[1:] if isinstance(verbose, dict) else []
    lines, tools, n_err, n_steps = [], [], 0, 0
    for st in steps:
        if not isinstance(st, dict) or "step_number" not in st:
            continue
        n_steps += 1
        for tc in st.get("tool_calls") or []:
            fn = (tc.get("function") or {}).get("name")
            args = (tc.get("function") or {}).get("arguments")
            if fn:
                tools.append(fn)
                lines.append(f"call[{n_steps}]: {fn}({clip(args, 120)})")
        if st.get("error"):
            n_err += 1
            lines.append(f"  ERROR: {clip(st['error'], 150)}")
        elif st.get("observations") is not None:
            lines.append(f"  -> {clip(st['observations'], 150)}")
    return {
        "trace": "\n".join(lines),
        "final_answer": clip(verbose.get("output"), 300),
        "tools_called": tools,
        "n_tool_calls": len(tools),
        "n_steps": n_steps,
        "n_tool_errors": n_err,
        "round_exhausted": n_steps >= MAX_STEPS_V2,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v1_dir", type=Path, default=Path("/tmp/ushmax/exp/expand-10k"))
    p.add_argument("--e4_dir", type=Path, default=Path(__file__).parent.parent / "e4_new_models" / "results")
    p.add_argument("--benchmark_root", type=Path, default=Path("/tmp/FinOpsBench"))
    p.add_argument("--cap", type=int, default=150, help="Max failures per model kept for classification (seed 13)")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "failures.jsonl")
    args = p.parse_args()

    rng = random.Random(13)
    out = []
    totals = {}

    # ---- v1 ----
    for model, fname in V1_EVALUATED.items():
        path = args.v1_dir / fname
        if not path.exists():
            print(f"skip {model}: {path} missing")
            continue
        fails = []
        for line in path.open():
            r = json.loads(line)
            ev = r.get("evaluation")
            if ev and ev.get("correct") is False and ev.get("agent_dialog"):
                t = v1_trace(ev["agent_dialog"])
                fails.append({
                    "version": "v1", "model": model, "query": r["query"],
                    "expected": clip(r.get("expected_output"), 400),
                    "judge_reasoning": clip(ev.get("reasoning"), 400), **t,
                })
        totals[model] = len(fails)
        if len(fails) > args.cap:
            fails = rng.sample(fails, args.cap)
        out.extend(fails)

    # ---- v2 ----
    agents_root = args.benchmark_root / "v2" / "finqa_agents"
    for model, sub in V2_RUNS.items():
        scored = args.e4_dir / f"{sub}.json"
        if not scored.exists():
            print(f"skip {model}: {scored} missing")
            continue
        results = {r["agent_id"]: r for r in json.load(scored.open())}
        fails = []
        for aid, r in results.items():
            if r.get("passed"):
                continue
            vpath = args.e4_dir / sub / f"{aid}_verbose.json"
            if not vpath.exists():
                continue
            try:
                verbose = json.load(vpath.open())
            except Exception:
                continue
            t = v2_trace(verbose)
            q = ""
            sp = agents_root / aid / "agent_system_prompt.txt"
            if sp.exists():
                import re
                m = re.search(r"(?is)Question\s*[-\s]*\n(.+?)(?:\nGuidelines|\Z)", sp.read_text())
                q = clip(m.group(1).strip() if m else "", 300)
            fails.append({
                "version": "v2", "model": model, "query": q,
                "expected": r.get("ground_truth_answer"),
                "judge_reasoning": "", **t,
            })
        totals[model] = len(fails)
        if len(fails) > args.cap:
            fails = rng.sample(fails, args.cap)
        out.extend(fails)

    with args.out.open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("total failures per model:", json.dumps(totals, indent=2))
    print(f"kept {len(out)} (cap {args.cap}/model) -> {args.out.name}")


if __name__ == "__main__":
    main()
