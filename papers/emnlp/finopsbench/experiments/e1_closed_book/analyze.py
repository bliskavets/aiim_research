"""Summarise E1 closed-book results and compare with the paper's agentic numbers.

Usage: python analyze.py [--benchmark_root /tmp/FinOpsBench]
"""

import argparse
import json
from pathlib import Path

# Agentic accuracy on FinOpsBench-v2 from the submission (Table 2)
PAPER_AGENTIC_V2 = {
    "openai_gpt-5-mini": 67.5,
    "openai_gpt-4.1": 60.6,
    "qwen_qwen3-30b-a3b": 53.0,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_root", type=Path, default=Path("/tmp/FinOpsBench"))
    args = p.parse_args()
    results_dir = Path(__file__).parent / "results"

    summary = {}
    print(f"{'model':30} {'n':>5} {'closed-book':>12} {'agentic(paper)':>15} {'delta':>7} {'cost':>8}")
    for f in sorted(results_dir.glob("*.jsonl")):
        recs = [json.loads(line) for line in f.open()]
        scored = [r for r in recs if "passed" in r]
        if not scored:
            continue
        acc = sum(r["passed"] for r in scored) / len(scored) * 100
        cost = sum(r.get("cost") or 0 for r in scored)
        slug = f.stem
        agentic = PAPER_AGENTIC_V2.get(slug)
        delta = f"{acc - agentic:+.1f}" if agentic else "n/a"
        print(f"{slug:30} {len(scored):5d} {acc:11.1f}% {agentic or float('nan'):14.1f}% {delta:>7} ${cost:7.3f}")
        summary[slug] = {
            "n": len(scored),
            "closed_book_acc": round(acc, 1),
            "agentic_acc_paper": agentic,
            "cost_usd": round(cost, 4),
        }

    # Per-item association with agentic success (gpt-4.1 subset released in the repo)
    agentic_f = args.benchmark_root / "v2" / "results" / "gpt-4.1.json"
    closed_f = results_dir / "openai_gpt-4.1.jsonl"
    if agentic_f.exists() and closed_f.exists():
        agentic = {r["agent_id"]: r["passed"] for r in json.load(agentic_f.open())}
        closed = {json.loads(l)["agent_id"]: json.loads(l).get("passed")
                  for l in closed_f.open() if "passed" in l}
        common = [aid for aid in agentic if aid in closed]
        if common:
            a = [agentic[i] for i in common]
            c = [closed[i] for i in common]
            n11 = sum(x and y for x, y in zip(a, c))
            n10 = sum(x and not y for x, y in zip(a, c))
            n01 = sum((not x) and y for x, y in zip(a, c))
            n00 = sum((not x) and (not y) for x, y in zip(a, c))
            print(f"\ngpt-4.1 per-item overlap (n={len(common)}): "
                  f"agentic+closed={n11}, agentic-only={n10}, closed-only={n01}, neither={n00}")
            summary["gpt41_overlap"] = {"n": len(common), "both": n11, "agentic_only": n10,
                                        "closed_only": n01, "neither": n00}

    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwritten to {results_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
