"""
plot_passrate.py — per-step pass-rate proxies and "wasted gradient" share, exp_049.

exp_049 does not log per-row rollouts (unlike exp_040), so a true per-example
pass-rate over training is not recoverable. The closest aggregate we can derive
from train.log is:

  pass_numeric_any : frac of batches where reward_answer_numeric/mean > 0  →
                     at least 1 of 4 generations matched the numeric ground truth
  pass_numeric_maj : frac of batches where reward_answer_numeric/mean >= 0.5 →
                     at least 2 of 4 generations matched (1.5+1.5-0.5-0.5)/4 = 0.5
  pass_exact_any   : frac of batches where reward_answer_exact/mean > 0     →
                     at least 1 generation had any tagged + correct/close answer
  pass_exact_top   : frac of batches where reward_answer_exact/mean >= 1.5  →
                     at least 2 of 4 generations matched the exact ground truth
                     in tagged format (or equivalent)
  frac_zero_std    : per-step trainer field — fraction of prompts in the batch
                     where all 4 generations got the same reward (no advantage
                     gradient for that prompt; "wasted" sample)

A rolling-mean over the last 50 steps is overlaid for each curve.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)

METHODS = {
    "grpo":             ("GRPO baseline",                "#64748b"),
    "grpo_s_entropy":   ("GRPO-S seq-level entropy",     "#d97706"),
    "gtpo_conf":        ("GTPO per-token confidence",    "#059669"),
    "gtpo_ema_flipped": ("GTPO-EMA flipped",             "#4f46e5"),
}

PATTERNS = {
    "ans_e":   r"'rewards/reward_answer_exact/mean':\s*([-\d.]+)",
    "ans_n":   r"'rewards/reward_answer_numeric/mean':\s*([-\d.]+)",
    "fzs":     r"'frac_reward_zero_std':\s*([-\d.]+)",
}


def extract(p):
    if not os.path.exists(p):
        return None
    with open(p) as f:
        txt = f.read()
    return {k: [float(m.group(1)) for m in re.finditer(rx, txt)]
            for k, rx in PATTERNS.items()}


def rolling(xs, w=50):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "exp_049 — Big-Math int-2000, Llama-3.2-3B · pass-rate proxies (rolling-50)\n"
        "no per-row rollout logs in this run; proxies derived from batch-mean tiers",
        fontsize=12, weight="bold")

    panels = [
        ("pass_num_any",  "frac batches with ≥1/4 numeric-correct (ans_num_mean > 0)"),
        ("pass_num_maj",  "frac batches with ≥2/4 numeric-correct (ans_num_mean ≥ 0.5)"),
        ("pass_exact_top","frac batches with ≥2/4 exact-correct (ans_exact_mean ≥ 1.5)"),
        ("fzs",           "frac_reward_zero_std (wasted prompts: all 4 same reward)"),
    ]

    series = {}
    for method in METHODS:
        d = extract(os.path.join(HERE, f"train_{method}.log"))
        if not d:
            continue
        ans_e = d["ans_e"]
        ans_n = d["ans_n"]
        series[method] = {
            "pass_num_any":   [1.0 if x > 0      else 0.0 for x in ans_n],
            "pass_num_maj":   [1.0 if x >= 0.5   else 0.0 for x in ans_n],
            "pass_exact_top": [1.0 if x >= 1.5   else 0.0 for x in ans_e],
            "fzs":            d["fzs"],
        }

    for ax, (key, title) in zip(axes.flat, panels):
        for m, (label, color) in METHODS.items():
            if m not in series or not series[m].get(key):
                continue
            ys = rolling(series[m][key], w=50)
            ax.plot(range(len(ys)), ys, color=color, label=label, lw=1.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8, loc="best")

    out = os.path.join(HERE, "figures", "exp049_passrate_proxies.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}")

    print("\nlast-50 rolling means:")
    for m, (label, _) in METHODS.items():
        if m not in series:
            continue
        s = series[m]
        n = len(s["pass_num_any"])
        if n == 0:
            continue
        sl = slice(max(0, n - 50), n)
        print(f"  {label:30s} steps={n:3d}  "
              f"num_any={sum(s['pass_num_any'][sl])/min(50,n):.2f}  "
              f"num_maj={sum(s['pass_num_maj'][sl])/min(50,n):.2f}  "
              f"exact_top={sum(s['pass_exact_top'][sl])/min(50,n):.2f}  "
              f"fzs={sum(s['fzs'][sl])/min(50,n):.2f}")


if __name__ == "__main__":
    main()
