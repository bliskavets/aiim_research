"""Exploratory: run a finance-specialized LLM (served locally via vLLM, OpenAI-
compatible) through the benchmark's real v2 agentic harness (SA_openrouter.py),
to see whether a finance-domain model can act as a tool-using agent and what it
scores. Internal only; not referenced in the rebuttal unless the result helps.

Per item: copy SA_openrouter.py into the agent dir, run it there (so `tool_proxy`
resolves to that env's tools), pointing smolagents at the local vLLM endpoint.
Score the final answer against initial_solution.txt with the benchmark comparator.

Usage:
    python run_finance_local.py --model_name finance --api_base http://localhost:8000/v1 --n 25
"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path("/tmp/FinOpsBench/v2")
AGENTS = ROOT / "finqa_agents"
SA = Path(__file__).parent / "SA_openrouter.py"
E4PY = "/tmp/e4venv/bin/python"
sys.path.insert(0, str(ROOT))
from compare_outputs import compare_answers  # noqa: E402


def run_one(agent_dir, model_name, api_base, timeout):
    dst = agent_dir / "SA_openrouter.py"
    shutil.copy(SA, dst)
    out = agent_dir / "_fin_out.txt"
    vout = agent_dir / "_fin_verbose.json"
    cmd = [E4PY, "SA_openrouter.py",
           "--system_prompt_file", "agent_system_prompt.txt",
           "--output", str(out), "--output_verbose", str(vout),
           "--model_name", model_name, "--api_key", "dummy", "--api_base", api_base]
    try:
        subprocess.run(cmd, cwd=agent_dir, timeout=timeout, capture_output=True)
    except subprocess.TimeoutExpired:
        return None, False, "timeout"
    if not out.is_file():
        return None, False, "no-output"
    pred = out.read_text().strip()
    tool_called = False
    if vout.is_file():
        try:
            msgs = json.loads(vout.read_text()).get("messages", [])
            tool_called = any("tool" in json.dumps(m).lower() and ("tool_call" in json.dumps(m).lower() or m.get("role") == "tool") for m in msgs)
        except Exception:
            pass
    return pred, tool_called, "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--api_base", default="http://localhost:8000/v1")
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--timeout", type=int, default=180)
    a = p.parse_args()
    subset = json.loads((Path(__file__).parent.parent / "e8_access_ladder" / "subset_200.json").read_text())
    ids = subset[: a.n] if isinstance(subset, list) else list(subset)[: a.n]
    rows = []
    for i, aid in enumerate(ids):
        d = AGENTS / (aid if str(aid).startswith("agent_") else f"agent_{aid}")
        gold_f = d / "initial_solution.txt"
        if not (d / "agent_system_prompt.txt").is_file() or not gold_f.is_file():
            continue
        gold = gold_f.read_text().strip()
        pred, tool_called, status = run_one(d, a.model_name, a.api_base, a.timeout)
        ok = bool(pred) and compare_answers(pred, gold)
        rows.append({"agent": d.name, "status": status, "tool_called": tool_called,
                     "passed": ok, "pred": (pred or "")[:200], "gold": gold})
        print(f"[{i+1}/{len(ids)}] {d.name}: status={status} tool_called={tool_called} passed={ok}", flush=True)
    n = len(rows) or 1
    summary = {
        "model": a.model_name, "n": len(rows),
        "tool_call_rate": round(100 * sum(r["tool_called"] for r in rows) / n, 1),
        "accuracy": round(100 * sum(r["passed"] for r in rows) / n, 1),
        "n_ok_status": sum(r["status"] == "ok" for r in rows),
    }
    outp = Path(__file__).parent / "results" / "finance_local.json"
    outp.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
