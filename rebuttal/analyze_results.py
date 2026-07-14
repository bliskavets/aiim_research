#!/usr/bin/env python3
"""Aggregate AAAI-2027 runs: accuracy, bootstrap CIs, multi-seed spread, and paired
baseline-vs-SAGE significance (McNemar + paired bootstrap).

Reads the per-problem jsonl logs written by the experiment scripts. Missing runs are
skipped with a note, so this can be run at any point while the queues are in flight.

Usage:
    python analyze_results.py [--logs logs] [--boot 10000]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_correct(path_glob: str, key: str = "is_correct") -> Optional[Dict[int, bool]]:
    """Return {index: correct_bool} from the newest jsonl matching path_glob, or None."""
    files = sorted(glob.glob(path_glob))
    if not files:
        return None
    out: Dict[int, bool] = {}
    with open(files[-1]) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = rec.get("index")
            if key in rec:
                val = bool(rec[key])
            elif "prompt_followed" in rec:
                val = bool(rec["prompt_followed"])
            else:
                continue
            if idx is not None:
                out[int(idx)] = val
    return out or None


def _acc(d: Dict[int, bool]) -> float:
    return sum(d.values()) / len(d) if d else 0.0


def _bootstrap_ci(vals: List[int], n_boot: int, rng: random.Random, alpha: float = 0.05) -> Tuple[float, float]:
    if not vals:
        return (0.0, 0.0)
    n = len(vals)
    means = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return (lo, hi)


def _mcnemar_p(paired: List[Tuple[bool, bool]]) -> Optional[float]:
    """Two-sided McNemar (baseline, sage) discordant pairs; normal approx with continuity."""
    b = sum(1 for base, sage in paired if base and not sage)  # baseline right, sage wrong
    c = sum(1 for base, sage in paired if not base and sage)  # sage right, baseline wrong
    if b + c == 0:
        return None
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # survival of chi-square with 1 dof = erfc(sqrt(chi2/2))
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return p


def summarize_run(name: str, d: Optional[Dict[int, bool]], n_boot: int, rng: random.Random) -> Optional[dict]:
    if d is None:
        print(f"  [missing] {name}")
        return None
    vals = [int(v) for v in d.values()]
    lo, hi = _bootstrap_ci(vals, n_boot, rng)
    return {"name": name, "n": len(d), "acc": _acc(d), "ci": (lo, hi), "data": d}


def paired_report(base: dict, sage: dict, n_boot: int, rng: random.Random) -> str:
    common = sorted(set(base["data"]) & set(sage["data"]))
    if not common:
        return "no common indices"
    paired = [(base["data"][i], sage["data"][i]) for i in common]
    diffs = [int(s) - int(b) for b, s in paired]
    # paired bootstrap CI on the accuracy difference
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    dlo, dhi = means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]
    p = _mcnemar_p(paired)
    delta = sum(diffs) / n
    psig = f"p={p:.4g}" if p is not None else "p=n/a"
    return (f"delta={delta*100:+.1f}pt (95% CI [{dlo*100:+.1f}, {dhi*100:+.1f}]), "
            f"McNemar {psig}, n_paired={n}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    L = args.logs
    rng = random.Random(12345)
    B = args.boot

    print("=" * 78)
    print("MATH-500 (exact-match)")
    print("=" * 78)
    base_math = summarize_run("baseline (seed42, greedy)", _load_correct(f"{L}/a1_baseline_full/math500_eval_*.jsonl"), B, rng)
    if base_math:
        print(f"  baseline: {base_math['acc']*100:.1f}  95% CI [{base_math['ci'][0]*100:.1f}, {base_math['ci'][1]*100:.1f}]  (n={base_math['n']})")
    sage_math_runs = []
    for s in (42, 7, 123):
        r = summarize_run(f"SAGE seed{s}", _load_correct(f"{L}/sage_math_full_s{s}/mmin_1/math500_eval_*.jsonl"), B, rng)
        if r:
            print(f"  SAGE seed{s}: {r['acc']*100:.1f}  95% CI [{r['ci'][0]*100:.1f}, {r['ci'][1]*100:.1f}]")
            sage_math_runs.append(r)
    if sage_math_runs:
        accs = [r["acc"] for r in sage_math_runs]
        mean = sum(accs) / len(accs)
        sd = (sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)) ** 0.5 if len(accs) > 1 else 0.0
        print(f"  SAGE across {len(accs)} seeds: {mean*100:.1f} +/- {sd*100:.1f}")
        if base_math:
            print(f"  paired (seed42): {paired_report(base_math, sage_math_runs[0], B, rng)}")

    print("=" * 78)
    print("MMLU-Pro STEM (letter-match)")
    print("=" * 78)
    for s in (42, 7, 123):
        b = summarize_run(f"baseline seed{s}", _load_correct(f"{L}/c3_mmlu_baseline_s{s}/mmlu_pro_baseline.jsonl"), B, rng)
        g = summarize_run(f"SAGE seed{s}", _load_correct(f"{L}/c3_mmlu_sage_s{s}/mmlu_pro_sage.jsonl"), B, rng)
        if b:
            print(f"  baseline seed{s}: {b['acc']*100:.1f}  95% CI [{b['ci'][0]*100:.1f}, {b['ci'][1]*100:.1f}]")
        if g:
            print(f"  SAGE     seed{s}: {g['acc']*100:.1f}  95% CI [{g['ci'][0]*100:.1f}, {g['ci'][1]*100:.1f}]")
        if b and g:
            print(f"    paired: {paired_report(b, g, B, rng)}")

    print("=" * 78)
    print("IFEval (prompt-level, instruction-following-eval)")
    print("=" * 78)
    for name, gl in (("baseline", "a3_ifeval_baseline"), ("BoN", "a3_ifeval_bon"), ("SAGE", "a3_ifeval_sage")):
        r = summarize_run(name, _load_correct(f"{L}/{gl}/ifeval_*.jsonl", key="prompt_followed"), B, rng)
        if r:
            print(f"  {name}: prompt-acc {r['acc']*100:.1f}  95% CI [{r['ci'][0]*100:.1f}, {r['ci'][1]*100:.1f}]  (n={r['n']})")

    print("=" * 78)
    print("E2 same-model baselines (MATH-500)")
    print("=" * 78)
    for mode in ("self_refine", "reflexion"):
        r = summarize_run(mode, _load_correct(f"{L}/e2_{mode}_math_s42/math500_eval_*.jsonl"), B, rng)
        if r:
            print(f"  {mode}: {r['acc']*100:.1f}  95% CI [{r['ci'][0]*100:.1f}, {r['ci'][1]*100:.1f}]")


if __name__ == "__main__":
    main()
