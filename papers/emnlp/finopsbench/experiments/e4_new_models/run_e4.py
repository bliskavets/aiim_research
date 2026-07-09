"""E4: evaluate additional model families (Claude, DeepSeek) on FinOpsBench-v2.

Uses the benchmark's own smolagents runner (agent_runners/SA.py) unchanged —
the same harness as the paper — orchestrated from here so that no experiment
code lives in the benchmark repository.

Cost control: OpenRouter's credits endpoint is snapshotted before/after the
run and every ``--cost_check_every`` items; the run aborts if the projected
total exceeds ``--budget_usd``. Per-run spend is appended to results/costs.json.

Usage:
    export OPENROUTER_API_KEY=...
    python run_e4.py --model anthropic/claude-sonnet-4 --sample 200 --budget_usd 10
    python run_e4.py --model deepseek/deepseek-chat-v3-0324 --budget_usd 6
"""

import argparse
import concurrent.futures
import json
import os
import random
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_root", type=Path, default=Path("/tmp/FinOpsBench"))
    p.add_argument("--model", required=True, help="OpenRouter model id, e.g. anthropic/claude-sonnet-4")
    p.add_argument("--runner", default="SA", help="Runner from the benchmark's agent_runners/")
    p.add_argument("--python", default=sys.executable, help="Python with smolagents+mlflow installed")
    p.add_argument("--sample", type=int, default=None, help="Random subsample size (seed 13)")
    p.add_argument("--subset_file", type=Path, default=None, help="JSON list of agent_ids to restrict to")
    p.add_argument("--limit", type=int, default=None, help="Cap items this invocation (pilots)")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--budget_usd", type=float, required=True, help="Hard cap for this run")
    p.add_argument("--cost_check_every", type=int, default=25)
    p.add_argument("--timeout", type=int, default=600, help="Per-item subprocess timeout, seconds")
    p.add_argument("--out_dir", type=Path, default=Path(__file__).parent / "results")
    return p.parse_args()


def credits_used() -> float:
    req = urllib.request.Request(
        OPENROUTER_API_BASE + "/credits",
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)["data"]
    return float(data["total_usage"])


def run_agent(agent_dir: Path, runner_src: Path, out_dir: Path, args: argparse.Namespace) -> dict:
    output = (out_dir / (agent_dir.name + ".txt")).absolute()
    verbose = (out_dir / (agent_dir.name + "_verbose.json")).absolute()
    shutil.copy(runner_src, agent_dir / "runner.py")
    cmd = [
        args.python, "runner.py",
        "--system_prompt_file", "agent_system_prompt.txt",
        "--output", str(output),
        "--output_verbose", str(verbose),
        "--model_name", args.model,
        "--api_key", os.environ["OPENROUTER_API_KEY"],
        "--api_base", OPENROUTER_API_BASE,
    ]
    try:
        proc = subprocess.run(cmd, cwd=agent_dir, capture_output=True, text=True, timeout=args.timeout)
        ok = proc.returncode == 0 and output.exists()
        return {"agent_id": agent_dir.name, "ok": ok,
                "stderr_tail": proc.stderr[-400:] if not ok else None}
    except subprocess.TimeoutExpired:
        return {"agent_id": agent_dir.name, "ok": False, "stderr_tail": "timeout"}


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.benchmark_root / "v2"))
    from compare_outputs import compare_answers  # noqa: E402

    if args.runner.endswith(".py"):
        runner_src = Path(__file__).parent / args.runner
    else:
        runner_src = args.benchmark_root / "v2" / "agent_runners" / (args.runner + ".py")
    agents_root = args.benchmark_root / "v2" / "finqa_agents"
    dirs = [d for d in sorted(agents_root.glob("agent_*"))
            if (d / "agent_system_prompt.txt").is_file()
            and (d / "initial_solution.txt").is_file()
            and (d / "tool_proxy.py").is_file()]
    if args.subset_file:
        import json as _json
        keep = set(_json.loads(args.subset_file.read_text()))
        dirs = [d for d in dirs if d.name in keep]
    print(f"{len(dirs)} runnable environments")
    if args.sample:
        dirs = random.Random(13).sample(dirs, args.sample)

    slug = args.model.replace("/", "_")
    out_dir = args.out_dir / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = [d for d in dirs if not (out_dir / (d.name + ".txt")).exists()]
    if args.limit:
        dirs = dirs[: args.limit]
    print(f"{len(dirs)} to run (rest already done)")

    usage_start = credits_used()
    print(f"credits used before run: ${usage_start:.3f}")
    spent = 0.0
    stop = False
    done = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pending = {pool.submit(run_agent, d, runner_src, out_dir, args): d for d in dirs}
        for fut in concurrent.futures.as_completed(pending):
            rec = fut.result()
            done += 1
            if not rec["ok"]:
                print(f"FAIL {rec['agent_id']}: {rec['stderr_tail']}")
            if done % args.cost_check_every == 0 or done == len(dirs):
                spent = credits_used() - usage_start
                per_item = spent / done
                projected = per_item * len(dirs)
                print(f"{done}/{len(dirs)} spent=${spent:.3f} "
                      f"(${per_item:.4f}/item, projected ${projected:.2f})")
                if spent > args.budget_usd or (done >= 10 and projected > args.budget_usd * 1.15):
                    print(f"BUDGET STOP: spent ${spent:.2f}, projected ${projected:.2f} "
                          f"> cap ${args.budget_usd}")
                    for f in pending:
                        f.cancel()
                    stop = True
                    break

    spent = credits_used() - usage_start

    # ---- score everything present in out_dir ----
    results = []
    for f in sorted(out_dir.glob("agent_*.txt")):
        agent_id = f.stem
        gold = (agents_root / agent_id / "initial_solution.txt").read_text().strip()
        pred = f.read_text().strip()
        results.append({"agent_id": agent_id, "ground_truth_answer": gold,
                        "agent_output": pred, "passed": bool(compare_answers(pred, gold))})
    acc = sum(r["passed"] for r in results) / len(results) * 100 if results else 0.0
    (args.out_dir / f"{slug}.json").write_text(json.dumps(results, indent=1))

    costs_f = args.out_dir / "costs.json"
    costs = json.loads(costs_f.read_text()) if costs_f.exists() else []
    costs.append({"model": args.model, "items_run_now": done, "items_scored_total": len(results),
                  "accuracy": round(acc, 1), "spent_usd": round(spent, 4),
                  "budget_usd": args.budget_usd, "stopped_by_budget": stop})
    costs_f.write_text(json.dumps(costs, indent=1))

    print(f"\n=== {args.model}: n={len(results)} accuracy={acc:.1f}% "
          f"spend this run=${spent:.3f} (stopped_by_budget={stop}) ===")


if __name__ == "__main__":
    main()
