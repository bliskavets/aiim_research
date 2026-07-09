"""Measure the real $/tokens/runtime to construct ONE FinOpsBench-v2 example.

Replays the actual 10-stage generator prompt (prompts.prompt_template, exactly as
create_dataset_v3.make_long_prompt builds it) for a real FinQA item, using an
already-built agent_* directory as the cumulative past-stage context so each
stage's prompt has realistic size. Calls the real generator model (o3) through
OpenRouter and records per-stage input/output tokens, cost (usage.cost) and
wall-time. This faithfully measures the LLM cost that dominates construction.

Usage:
    export OPENROUTER_API_KEY=...
    python measure_v2.py --agent_id agent_338 --model openai/o3 --n 1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
STEP_FILES = ["initial_solution.py", "synthetic_db_generator.py", "tools.py",
              "correct_plan.py", "synthetic_db_generator_augmented.py", "tools_augmented.py",
              "correct_plan_augmented.py", "agent_system_prompt.txt", "agent.py", "test_agent.py"]


def build_stage_prompt(pt, sample, step, agent_dir, extra_agent_py):
    def tbl(s):
        t = s.get("table") or []
        if not t:
            return ""
        rows = ["| " + " | ".join(map(str, t[0])) + " |", "|" + "---|" * len(t[0])]
        rows += ["| " + " | ".join(map(str, r)) + " |" for r in t[1:]]
        return "\n".join(rows)
    past = []
    for i, sf in enumerate(STEP_FILES[: step - 1]):
        body = (agent_dir / sf).read_text() if (agent_dir / sf).exists() else ""
        past.append(f"Step {i+1}. {sf}:\n```python\n{body}\n```")
    add = ""
    if step in (9, 10):
        add = f"Additional agent file:\n```python\n{extra_agent_py}\n```\n"
    return pt.format(
        pre_text="\n".join(sample.get("pre_text", [])),
        table=tbl(sample),
        post_text="\n".join(sample.get("post_text", [])),
        question=sample["qa"]["question"], answer=sample["qa"]["answer"],
        current_step=step, past_information="\n".join(past),
        additional_agent_file_content=add,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_root", type=Path, default=Path("/tmp/FinOpsBench"))
    p.add_argument("--finqa_train", type=Path, default=Path("/tmp/finqa_train.json"))
    p.add_argument("--pipeline_dir", type=Path, default=Path("/tmp/FinOpsBench/v2/pipeline"))
    p.add_argument("--agent_id", default="agent_338", help="Existing dir used as cumulative context")
    p.add_argument("--model", default="openai/o3")
    p.add_argument("--n", type=int, default=1, help="How many examples to measure (averaged)")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "v2_cost.json")
    args = p.parse_args()

    sys.path.insert(0, str(args.pipeline_dir))
    from prompts import prompt_template

    train = json.loads(args.finqa_train.read_text())
    client = OpenAI(base_url=OPENROUTER_API_BASE, api_key=os.environ["OPENROUTER_API_KEY"])
    extra_agent_py = (args.benchmark_root / "v2" / "finqa_agents" / "agent_0" / "agent.py").read_text()

    # pick n example dirs that exist and map to a train item
    root = args.benchmark_root / "v2" / "finqa_agents"
    ids = [args.agent_id] + [d.name for d in sorted(root.glob("agent_*")) if d.name != args.agent_id]
    measured = []
    for aid in ids:
        if len(measured) >= args.n:
            break
        agent_dir = root / aid
        n_idx = int(aid.split("_")[-1])
        if not agent_dir.exists() or n_idx >= len(train):
            continue
        sample = train[n_idx]
        stages = []
        for step in range(1, 11):
            prompt = build_stage_prompt(prompt_template, sample, step, agent_dir, extra_agent_py)
            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=args.model, messages=[{"role": "user", "content": prompt}],
                    max_tokens=8000,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  stage {step} error: {e}")
                stages.append({"step": step, "error": str(e)[:120]})
                continue
            dt = time.time() - t0
            u = resp.usage.model_dump() if resp.usage else {}
            stages.append({"step": step, "file": STEP_FILES[step - 1],
                           "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                           "cost": u.get("cost"), "seconds": round(dt, 1)})
            print(f"  [{aid}] stage {step:2d} {STEP_FILES[step-1]:34} "
                  f"in={u.get('prompt_tokens')} out={u.get('completion_tokens')} "
                  f"${u.get('cost'):.4f} {dt:.1f}s")
        tot_cost = sum(s.get("cost") or 0 for s in stages)
        tot_time = sum(s.get("seconds") or 0 for s in stages)
        measured.append({"agent_id": aid, "stages": stages,
                         "total_cost": round(tot_cost, 4), "total_seconds": round(tot_time, 1)})
        print(f"  => {aid}: total ${tot_cost:.4f}, {tot_time:.0f}s\n")

    avg_cost = sum(m["total_cost"] for m in measured) / len(measured)
    avg_time = sum(m["total_seconds"] for m in measured) / len(measured)
    out = {"model": args.model, "n_measured": len(measured),
           "avg_cost_per_example": round(avg_cost, 4), "avg_seconds_per_example": round(avg_time, 1),
           "runs": measured}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"AVG per v2 example: ${avg_cost:.4f}, {avg_time:.0f}s  (n={len(measured)}) -> {args.out}")


if __name__ == "__main__":
    main()
