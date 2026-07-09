"""Run every v2 reference plan in its own environment and merge a traced,
value-annotated copy into the validity sample.

Each plan is actually executed (via _plan_trace_runner.py, cwd = the agent
directory) so the annotated variable values are real, not guessed. Adds these
fields to data/sample_v2_validity.jsonl (preserving human_label and all others):
  annotated_plan, plan_run_ok, plan_computed_answer, plan_error,
  plan_answer_matches_gold

Usage:
    python trace_plans.py [--python /tmp/e4venv/bin/python] [--timeout 90]
"""

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
RUNNER = HERE / "_plan_trace_runner.py"
SAMPLE = HERE / "data" / "sample_v2_validity.jsonl"


def norm_num(s):
    s = str(s).strip().lower().replace("$", "").replace(",", "").replace("%", "").replace("(", "-").replace(")", "")
    m = re.findall(r"-?\d+\.?\d*", s)
    try:
        return float(m[-1]) if m else None
    except ValueError:
        return None


def trace_one(agent_dir: Path, python: str, timeout: int) -> dict:
    plan = agent_dir / "correct_plan_augmented.py"
    if not plan.is_file():
        return {"plan_run_ok": False, "plan_error": "no correct_plan_augmented.py"}
    out_json = Path(tempfile.mktemp(suffix=".json"))
    try:
        proc = subprocess.run(
            [python, str(RUNNER), "correct_plan_augmented.py", str(out_json)],
            cwd=agent_dir, capture_output=True, text=True, timeout=timeout,
        )
        if not out_json.exists():
            return {"plan_run_ok": False, "plan_error": (proc.stderr or "runner produced no output")[-400:]}
        r = json.loads(out_json.read_text())
        return {
            "annotated_plan": r.get("annotated_source"),
            "plan_run_ok": r.get("ok", False),
            "plan_computed_answer": r.get("computed_answer"),
            "plan_error": r.get("error"),
            "n_vars_traced": r.get("n_vars_traced", 0),
        }
    except subprocess.TimeoutExpired:
        return {"plan_run_ok": False, "plan_error": f"timeout after {timeout}s"}
    finally:
        out_json.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_root", type=Path, default=Path("/tmp/FinOpsBench"))
    p.add_argument("--python", default="/tmp/e4venv/bin/python")
    p.add_argument("--timeout", type=int, default=90)
    args = p.parse_args()

    rows = [json.loads(l) for l in SAMPLE.open() if l.strip()]
    agents_root = args.benchmark_root / "v2" / "finqa_agents"
    ok = matched = 0
    for i, row in enumerate(rows, 1):
        res = trace_one(agents_root / row["agent_id"], args.python, args.timeout)
        row.update(res)
        if res.get("plan_run_ok"):
            ok += 1
            g, c = norm_num(row.get("gold", "")), norm_num(res.get("plan_computed_answer") or "")
            row["plan_answer_matches_gold"] = (g is not None and c is not None and abs(g - c) < 0.15)
            matched += bool(row["plan_answer_matches_gold"])
        else:
            row["plan_answer_matches_gold"] = False
        if i % 20 == 0 or i == len(rows):
            print(f"{i}/{len(rows)} traced | ran_ok={ok} | computed==gold={matched}")

    with SAMPLE.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"done: {ok}/{len(rows)} plans executed, {matched} computed answers match gold -> {SAMPLE.name}")


if __name__ == "__main__":
    main()
