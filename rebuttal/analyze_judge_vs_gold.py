#!/usr/bin/env python3
"""E3b + E8: judge-vs-gold cross-check and calibration from logged SAGE candidates.

Uses the already-logged MATH-500 SAGE runs (all 21 candidates per problem carry
final_llm_judge_score). Grades each problem's UNIQUE boxed answers against gold with
the same o3 equivalence judge (disk-cached), then reports:

  E3b (reward-hacking probe): oracle accuracy (a correct candidate exists),
  judge-selected accuracy, and the oracle gap: how often the judge picks a wrong
  answer although a correct candidate exists. A self-preference-blind judge would
  show a large gap; a correctness-tracking judge a small one.

  E8 (calibration): per-candidate correctness vs sigmoid(margin) confidence:
  ECE (15 bins), Brier score, selection AUC, and a reliability table.

Usage: python analyze_judge_vs_gold.py --run logs/sage_math_full_s42/mmin_1 [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.helpers import equations_are_equal_new, parse_answer  # noqa: E402


def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dir with math500_eval_*.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="output json path")
    args = ap.parse_args()

    f = sorted(glob.glob(f"{args.run}/math500_eval_*.jsonl"))[-1]
    print(f"[e3b] reading {f}")

    per_problem = []
    n_graded_calls = 0
    with open(f) as fin:
        for li, line in enumerate(fin):
            if args.limit and li >= args.limit:
                break
            r = json.loads(line)
            prompt, gt = r["prompt"], r["gt_answer"]
            cands = r.get("all_answers", [])
            # unique boxed answers among candidates
            uniq: dict = {}
            for c in cands:
                txt = c.get("answer", "")
                if "boxed{" not in txt:
                    key = None
                else:
                    key = parse_answer(txt)
                sc = c.get("final_llm_judge_score")
                uniq.setdefault(key, []).append(sc)
            # grade unique keys (None = unparseable -> wrong)
            graded: dict = {}
            for key in uniq:
                if key is None:
                    graded[key] = 0
                    continue
                graded[key] = equations_are_equal_new(prompt, gt, "\\boxed{" + key + "}")
                n_graded_calls += 1
            # per-candidate records
            cand_rec = []
            for c in cands:
                txt = c.get("answer", "")
                key = parse_answer(txt) if "boxed{" in txt else None
                cand_rec.append({
                    "correct": int(graded.get(key, 0)),
                    "score": c.get("final_llm_judge_score"),
                })
            oracle = int(any(cr["correct"] for cr in cand_rec))
            selected_correct = int(bool(r.get("is_correct")))
            per_problem.append({
                "index": r.get("index"),
                "oracle": oracle,
                "selected_correct": selected_correct,
                "candidates": cand_rec,
            })
            if (li + 1) % 50 == 0:
                print(f"[e3b] {li+1} problems, {n_graded_calls} judge calls so far")

    n = len(per_problem)
    oracle_acc = sum(p["oracle"] for p in per_problem) / n
    sel_acc = sum(p["selected_correct"] for p in per_problem) / n
    exists_but_missed = sum(1 for p in per_problem if p["oracle"] and not p["selected_correct"])
    print(f"\n=== E3b: judge vs gold (n={n}) ===")
    print(f"oracle accuracy (correct candidate exists): {100*oracle_acc:.1f}")
    print(f"judge-selected accuracy:                    {100*sel_acc:.1f}")
    print(f"oracle gap (missed despite existing):       {exists_but_missed}/{n} = {100*exists_but_missed/n:.1f}%")

    # E8 calibration over all candidates with scores
    pairs = [(cr["score"], cr["correct"]) for p in per_problem for cr in p["candidates"]
             if cr["score"] is not None]
    m = len(pairs)
    briers, bins = [], [[0, 0.0, 0.0] for _ in range(15)]
    for s, y in pairs:
        conf = sigmoid(s)
        briers.append((conf - y) ** 2)
        b = min(14, int(conf * 15))
        bins[b][0] += 1; bins[b][1] += conf; bins[b][2] += y
    ece = sum(cnt * abs(cs / cnt - ys / cnt) for cnt, cs, ys in bins if cnt) / m
    # selection AUC: P(score_correct > score_wrong) over random cross pairs
    pos = sorted(s for s, y in pairs if y == 1)
    neg = sorted(s for s, y in pairs if y == 0)
    if pos and neg:
        import bisect
        wins = sum(bisect.bisect_left(neg, s) for s in pos)
        ties = sum(bisect.bisect_right(neg, s) - bisect.bisect_left(neg, s) for s in pos)
        auc = (wins + 0.5 * ties) / (len(pos) * len(neg))
    else:
        auc = float("nan")
    print(f"\n=== E8: calibration over {m} candidates ===")
    print(f"candidate-pool correctness rate: {100*sum(y for _,y in pairs)/m:.1f}")
    print(f"Brier: {sum(briers)/m:.4f}  ECE(15): {ece:.4f}  selection AUC: {auc:.4f}")
    print("reliability bins (conf_range, n, mean_conf, frac_correct):")
    for i, (cnt, cs, ys) in enumerate(bins):
        if cnt:
            print(f"  [{i/15:.2f},{(i+1)/15:.2f}) n={cnt:5d} conf={cs/cnt:.3f} acc={ys/cnt:.3f}")

    out = args.out or f"{args.run}/judge_vs_gold_analysis.json"
    json.dump({
        "n_problems": n, "oracle_acc": oracle_acc, "selected_acc": sel_acc,
        "oracle_gap": exists_but_missed / n, "n_candidates": m,
        "brier": sum(briers)/m, "ece15": ece, "auc": auc,
        "per_problem": per_problem,
    }, open(out, "w"))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
