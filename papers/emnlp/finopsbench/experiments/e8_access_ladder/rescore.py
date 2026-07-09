"""Percent-robust re-scoring for the access ladder.

The benchmark's compare_answers treats a trailing '%' as /100. A CoT model that
prints "Final answer: 52.32" for a gold of "52.32%" is correct but scored wrong
by exactly 100x. In the real agentic runs the reference-plan-style output keeps
the '%' sign, so this mismatch does not arise; here we make scoring robust to
the percent-scaling ambiguity so every ladder rung is credited consistently.

Usage: python rescore.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/FinOpsBench/v2")
from compare_outputs import compare_answers, extract_number_from_answer  # noqa: E402

HERE = Path(__file__).parent


def robust_match(pred: str, gold: str) -> bool:
    if compare_answers(pred, gold):
        return True
    pv, _ = extract_number_from_answer(pred)
    gv, gp = extract_number_from_answer(gold)
    if pv == "no answer" or gv == "no answer":
        return False
    tol = 10 ** (-gp) * 0.6
    # accept a pure percent-scaling difference (either direction)
    return abs(pv / 100 - gv) <= tol or abs(pv - gv * 100) <= max(tol * 100, 0.6)


def score_file(path: Path):
    rows = [json.loads(l) for l in path.open()]
    base = sum(r["passed"] for r in rows)
    robust = sum(robust_match(r["prediction"], r["gold"]) for r in rows)
    return len(rows), base, robust


def main():
    print(f"{'file':52} {'n':>4} {'raw%':>6} {'robust%':>8}")
    summary = {}
    for f in sorted((HERE / "results").glob("*.jsonl")):
        n, base, rob = score_file(f)
        print(f"{f.name:52} {n:>4} {100*base/n:>5.1f}% {100*rob/n:>7.1f}%")
        summary[f.stem] = {"n": n, "raw_acc": round(100 * base / n, 1), "robust_acc": round(100 * rob / n, 1)}
    (HERE / "results" / "rescored.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
