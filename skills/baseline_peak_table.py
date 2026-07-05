#!/usr/bin/env python
"""
baseline_peak_table.py — paper-table generator: Ours (baseline + our per-token shaping)
vs the baseline, per dataset:

  | method | <DS> peak reward | <DS> min steps to reach baseline's peak reward | ...

Definitions (computed on the rolling mean, window --window, of --metric):
  - peak reward: max_t rolling(metric)[t]
  - min steps to reach baseline's peak: first t with rolling(metric)[t] >= baseline's peak
    ("—" if never reached within the run; for the baseline row this is its own
    first-touch-of-peak step).

Model-agnostic: point --dirs at the folder(s) holding that model's logs in our standard
naming train_{dataset}_{suffix}.log (e.g. rerun the same setups for Llama and pass those
dirs). First dir containing the file wins.

Usage:
  python skills/baseline_peak_table.py \
    --dirs experiments/exp_077_dapo_shaped \
    --baseline-suffix dapo --baseline-label DAPO \
    --ours-suffix dapo_shaped --ours-label "Ours (DAPO + shaping)" \
    --datasets gsm8k math500 bigmath
"""
import argparse
import os
import re

DS_LABELS = {"gsm8k": "GSM8k", "math500": "MATH-500", "bigmath": "BigMath Int",
             "omnimath": "Omni-MATH"}


def read_series(dirs, dataset, suffix, metric):
    for d in dirs:
        p = os.path.join(d, f"train_{dataset}_{suffix}.log")
        if os.path.exists(p):
            vals = [float(m.group(1)) for m in
                    re.finditer(re.escape(metric) + r"':\s*([-\d.eE+]+)", open(p).read())]
            if vals:
                return vals
    return None


def rolling(xs, w):
    """Full-window rolling mean only (min_periods=w): [(step_1idx, value), ...].
    Early partial windows are excluded — a 2-sample 'peak' is noise, not a result."""
    return [(i + 1, sum(xs[i - w + 1:i + 1]) / w) for i in range(w - 1, len(xs))]


def peak(series):
    return max(v for _, v in series)


def steps_to_reach(series, target):
    for step, v in series:
        if v >= target:
            return step                      # 1-indexed training step
    return None


def fmt_steps(s):
    return str(s) if s is not None else "—"


def make_table(dirs, datasets, ours_suffix, base_suffix, ours_label, base_label,
               metric, window):
    cols, ours_cells, base_cells, missing = [], [], [], []
    for ds in datasets:
        lab = DS_LABELS.get(ds, ds)
        cols += [f"{lab} peak reward", f"{lab} min steps to reach baseline's peak reward"]
        so = read_series(dirs, ds, ours_suffix, metric)
        sb = read_series(dirs, ds, base_suffix, metric)
        if so is None or sb is None:
            missing.append(ds)
            ours_cells += ["n/a", "n/a"]; base_cells += ["n/a", "n/a"]
            continue
        ro, rb = rolling(so, window), rolling(sb, window)
        bp = peak(rb)
        ours_cells += [f"{peak(ro):+.2f}", fmt_steps(steps_to_reach(ro, bp))]
        base_cells += [f"{bp:+.2f}", fmt_steps(steps_to_reach(rb, bp))]
    lines = ["| method | " + " | ".join(cols) + " |",
             "|-" * (len(cols) + 1) + "|",
             f"| {ours_label} | " + " | ".join(ours_cells) + " |",
             f"| {base_label} | " + " | ".join(base_cells) + " |"]
    if missing:
        lines.append(f"\n_missing logs for: {', '.join(missing)}_")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--baseline-suffix", required=True)
    ap.add_argument("--ours-suffix", required=True)
    ap.add_argument("--baseline-label", default=None)
    ap.add_argument("--ours-label", default="Ours")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "bigmath"])
    ap.add_argument("--metric", default="reward_answer_boxed/mean")
    ap.add_argument("--window", type=int, default=30)
    a = ap.parse_args()
    base_label = a.baseline_label or a.baseline_suffix.upper()
    print(make_table(a.dirs, a.datasets, a.ours_suffix, a.baseline_suffix,
                     a.ours_label, base_label, a.metric, a.window))


if __name__ == "__main__":
    main()
