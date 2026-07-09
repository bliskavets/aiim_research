"""Build the two E3 human-annotation sets, reusing the E2 labels.

Set A (v1 judge accuracy): the non-contested scalar cases (judge == numeric).
Combined with the 92 already-labelled contested cases from E2, this gives an
unbiased, stratified estimate of judge accuracy over the whole scalar subset.

Set B (v2 validity): a random sample of v2 environments for holistic validity
labelling (question + gold answer + reference plan + tool list).

Usage: python build_samples.py [--benchmark_root /tmp/FinOpsBench]
       [--n_v1 80] [--n_v2 100]
"""

import argparse
import json
import random
import re
from pathlib import Path

E2_SCALAR = Path(__file__).parent.parent / "e2_judge_agreement" / "results" / "agreement_scalar_openai_o4-mini.jsonl"


def build_v1_judge(n: int, out: Path) -> int:
    rows = [json.loads(l) for l in E2_SCALAR.open() if l.strip()]
    rows = [r for r in rows if r.get("judge_correct") is not None]
    noncont = [r for r in rows if r["numeric_match"] == r["judge_correct"]]
    sample = random.Random(13).sample(noncont, min(n, len(noncont)))
    with out.open("w") as f:
        for r in sample:
            f.write(json.dumps({
                "task": "v1_answer",
                "query": r["query"],
                "gold": r["gold"],
                "answer": r["answer"],
                "numeric_match": r["numeric_match"],
                "judge_correct": r["judge_correct"],
                "human_label": None,
            }, ensure_ascii=False) + "\n")
    return len(sample)


def extract_question(system_prompt: str) -> str:
    m = re.search(r"(?im)^\s*Question\s*\n[-\s]*\n?(.+?)(?:\n\s*Guidelines|\Z)", system_prompt, re.S)
    if m:
        return m.group(1).strip()
    return system_prompt[-600:].strip()


def build_v2_validity(n: int, root: Path, out: Path) -> int:
    dirs = [d for d in sorted((root / "v2" / "finqa_agents").glob("agent_*"))
            if (d / "agent_system_prompt.txt").is_file() and (d / "initial_solution.txt").is_file()]
    sample = random.Random(13).sample(dirs, min(n, len(dirs)))
    with out.open("w") as f:
        for d in sample:
            sp = (d / "agent_system_prompt.txt").read_text()
            plan_f = d / "correct_plan_augmented.py"
            tools_f = d / "tools_augmented.py"
            tool_names = re.findall(r"^def ([a-z_][a-z0-9_]*)", tools_f.read_text(), re.M) if tools_f.is_file() else []
            f.write(json.dumps({
                "task": "v2_validity",
                "agent_id": d.name,
                "question": extract_question(sp),
                "gold": (d / "initial_solution.txt").read_text().strip(),
                "tool_names": tool_names,
                "reference_plan": plan_f.read_text() if plan_f.is_file() else "",
                "human_label": None,
            }, ensure_ascii=False) + "\n")
    return len(sample)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_root", type=Path, default=Path("/tmp/FinOpsBench"))
    p.add_argument("--n_v1", type=int, default=80)
    p.add_argument("--n_v2", type=int, default=100)
    args = p.parse_args()
    data = Path(__file__).parent / "data"
    data.mkdir(parents=True, exist_ok=True)
    a = build_v1_judge(args.n_v1, data / "sample_v1_judge.jsonl")
    b = build_v2_validity(args.n_v2, args.benchmark_root, data / "sample_v2_validity.jsonl")
    print(f"v1 judge-accuracy (non-contested) sample: {a} -> data/sample_v1_judge.jsonl")
    print(f"v2 validity sample: {b} -> data/sample_v2_validity.jsonl")


if __name__ == "__main__":
    main()
