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


def _labelled_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for l in path.open() if l.strip() and json.loads(l).get("human_label") is not None)


def _guard(path: Path, force: bool) -> bool:
    """Return True if it is safe to (over)write *path*."""
    k = _labelled_count(path)
    if k and not force:
        print(f"SKIP {path.name}: {k} human labels already present (pass --force to overwrite)")
        return False
    return True


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


def finqa_table_md(sample: dict) -> str:
    table = sample.get("table") or []
    if not table:
        return ""
    lines = ["| " + " | ".join(str(c) for c in table[0]) + " |",
             "| " + " | ".join("---" for _ in table[0]) + " |"]
    for row in table[1:]:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def norm_num(s: str):
    s = str(s).strip().lower().replace("$", "").replace(",", "").replace("%", "").replace("(", "-").replace(")", "")
    m = re.findall(r"-?\d+\.?\d*", s)
    try:
        return float(m[-1]) if m else None
    except ValueError:
        return None


def build_v2_validity(n: int, root: Path, out: Path, finqa_train: Path) -> int:
    train = json.loads(finqa_train.read_text()) if finqa_train.is_file() else []
    dirs = [d for d in sorted((root / "v2" / "finqa_agents").glob("agent_*"))
            if (d / "agent_system_prompt.txt").is_file() and (d / "initial_solution.txt").is_file()]
    sample = random.Random(13).sample(dirs, min(n, len(dirs)))
    with out.open("w") as f:
        for d in sample:
            sp = (d / "agent_system_prompt.txt").read_text()
            plan_f = d / "correct_plan_augmented.py"
            tools_f = d / "tools_augmented.py"
            tool_src = tools_f.read_text() if tools_f.is_file() else ""
            tool_names = re.findall(r"^def ([a-z_][a-z0-9_]*)", tool_src, re.M)
            dbgen_f = d / "synthetic_db_generator_augmented.py"
            if not dbgen_f.is_file():
                dbgen_f = d / "synthetic_db_generator.py"
            db_generator_src = dbgen_f.read_text() if dbgen_f.is_file() else ""
            gold = (d / "initial_solution.txt").read_text().strip()

            # original FinQA item: agent_<N> is built positionally from train[N]
            n_idx = int(d.name.split("_")[-1])
            finqa = {}
            if 0 <= n_idx < len(train):
                s = train[n_idx]
                fq_ans = str(s["qa"].get("answer", ""))
                g, a = norm_num(gold), norm_num(fq_ans)
                finqa = {
                    "finqa_id": s.get("id"),
                    "finqa_question": s["qa"]["question"],
                    "finqa_answer": fq_ans,
                    "finqa_pre_text": "\n".join(s.get("pre_text", [])),
                    "finqa_post_text": "\n".join(s.get("post_text", [])),
                    "finqa_table_md": finqa_table_md(s),
                    "finqa_answer_matches_gold": (g is not None and a is not None and abs(g - a) < 0.15),
                }

            f.write(json.dumps({
                "task": "v2_validity",
                "agent_id": d.name,
                "question": extract_question(sp),
                "gold": gold,
                "tool_names": tool_names,
                "tool_source": tool_src,
                "db_generator_source": db_generator_src,
                "reference_plan": plan_f.read_text() if plan_f.is_file() else "",
                **finqa,
                "human_label": None,
            }, ensure_ascii=False) + "\n")
    return len(sample)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_root", type=Path, default=Path("/tmp/FinOpsBench"))
    p.add_argument("--n_v1", type=int, default=80)
    p.add_argument("--n_v2", type=int, default=100)
    p.add_argument("--finqa_train", type=Path, default=Path("/tmp/finqa_train.json"),
                   help="FinQA dataset/train.json (used only at build time to attach the original item)")
    p.add_argument("--force", action="store_true", help="Overwrite sample files even if they already carry human labels")
    args = p.parse_args()
    data = Path(__file__).parent / "data"
    data.mkdir(parents=True, exist_ok=True)
    v1_out, v2_out = data / "sample_v1_judge.jsonl", data / "sample_v2_validity.jsonl"
    if _guard(v1_out, args.force):
        print(f"v1 judge-accuracy (non-contested) sample: {build_v1_judge(args.n_v1, v1_out)} -> {v1_out.name}")
    if _guard(v2_out, args.force):
        print(f"v2 validity sample: {build_v2_validity(args.n_v2, args.benchmark_root, v2_out, args.finqa_train)} -> {v2_out.name}")


if __name__ == "__main__":
    main()
