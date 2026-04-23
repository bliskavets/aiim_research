"""
compare_vs_027.py — live comparison of exp_028 (flipped GTPO-EMA) against
exp_027 (GRPO baseline) on the same Big-Math int-2000 setup.

Safe to re-run at any time during exp_028 training. Reads both train.log
files, parses them, and plots reward / KL / fmt_exact / ans_exact on a
shared time axis. Output: figures/compare_027_vs_028.png.
"""
import re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(THIS_DIR, "figures")

LOGS = [
    ("exp_027 GRPO baseline",        "tab:gray",
     os.path.join(THIS_DIR, "..", "exp_027_bigmath_int2k_grpo_baseline", "train.log")),
    ("exp_028 flipped GTPO-EMA",     "tab:green",
     os.path.join(THIS_DIR, "train.log")),
]

PATTERN = re.compile(r"\{'loss':.*?'epoch': \d+\.\d+\}")


def parse_log(path):
    recs = []
    try:
        with open(path) as f:
            text = f.read()
    except FileNotFoundError:
        return []
    for i, m in enumerate(PATTERN.finditer(text)):
        try:
            d = eval(m.group())
            d["step"] = i + 1
            recs.append(d)
        except Exception:
            pass
    return recs


def smooth(v, w=15):
    a = np.array([x if x is not None else np.nan for x in v], dtype=float)
    if len(a) < w:
        return a
    k = np.ones(w) / w
    p = np.pad(a, (w // 2, w - w // 2 - 1), mode="edge")
    return np.convolve(p, k, mode="valid")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    series = []
    for name, color, path in LOGS:
        recs = parse_log(path)
        print(f"  {name}: {len(recs)} steps")
        series.append((name, color, recs))

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        "Big-Math integer-2000 · Llama-3.2-3B · 500 steps · bs=4×8 gens — "
        "exp_027 GRPO baseline vs exp_028 flipped GTPO-EMA",
        fontsize=12, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    panels = [
        (gs[0, 0], "reward",                           "Total Reward (max 9.5)",   False),
        (gs[0, 1], "kl",                               "KL Divergence (symlog)",   True),
        (gs[1, 0], "rewards/reward_format_exact/mean", "Format Exact (max 3.0)",   False),
        (gs[1, 1], "rewards/reward_answer_exact/mean", "Answer Exact (max 3.0)",   False),
    ]
    for gs_loc, key, label, is_kl in panels:
        ax = fig.add_subplot(gs_loc)
        for name, color, recs in series:
            if not recs:
                continue
            steps = [r["step"] for r in recs]
            vals  = np.array([r.get(key, np.nan) for r in recs], dtype=float)
            ax.plot(steps, vals, color=color, alpha=0.15, linewidth=0.6)
            ax.plot(steps, smooth(vals), color=color, linewidth=2.2, label=name)
        if key == "reward":
            ax.axhline(9.5, color="red", linestyle=":", alpha=0.5, label="ceiling 9.5")
            ax.set_ylim(-3, 10.5)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("step"); ax.grid(True, alpha=0.3)
        if is_kl:
            ax.set_yscale("symlog", linthresh=0.01)
        ax.legend(fontsize=8, loc="best")

    out = os.path.join(OUT_DIR, "compare_027_vs_028.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # One-line summaries
    for name, _, recs in series:
        if not recs:
            print(f"  {name}: no data"); continue
        pk = max(recs, key=lambda d: d["reward"])
        pk_i = recs.index(pk) + 1
        d = recs[-1]
        print(f"  {name:30s} steps={len(recs):3d}  peak={pk['reward']:+5.2f} @ {pk_i:3d}  "
              f"reward@last={d['reward']:+5.2f}  fmt_ex={d['rewards/reward_format_exact/mean']:4.2f}  "
              f"ans_ex={d['rewards/reward_answer_exact/mean']:+5.2f}  kl={d['kl']:.3f}")


if __name__ == "__main__":
    main()
