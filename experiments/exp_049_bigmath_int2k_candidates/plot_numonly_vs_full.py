"""
plot_numonly_vs_full.py — head-to-head: same 4 methods, full reward vs numeric-only.

Produces a 2x4 grid:
  rows = panels (reward, num_any rolling, num_maj rolling, KL)
  cols = methods (grpo, grpo_s_entropy, gtpo_conf, gtpo_ema_flipped)
each cell overlays the full-reward curve and the numonly curve for that method.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)

METHODS = ["grpo", "grpo_s_entropy", "gtpo_conf", "gtpo_ema_flipped"]
TITLES = {
    "grpo": "GRPO baseline",
    "grpo_s_entropy": "GRPO-S seq entropy",
    "gtpo_conf": "GTPO per-token conf",
    "gtpo_ema_flipped": "GTPO-EMA flipped",
}
COLORS = {"full": "#64748b", "numonly": "#4f46e5"}

PATTERNS = {
    "reward": r"'reward':\s*([-\d.]+)",
    "ans_n":  r"'rewards/reward_answer_numeric/mean':\s*([-\d.]+)",
    "kl":     r"'kl':\s*([-\d.]+)",
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
    data = {}
    for m in METHODS:
        data[(m, "full")] = extract(os.path.join(HERE, f"train_{m}.log"))
        data[(m, "numonly")] = extract(os.path.join(HERE, f"train_{m}_numonly.log"))

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle(
        "exp_049 — full reward (format+exact+numeric) vs numonly (numeric only) · Big-Math int-2000",
        fontsize=12, weight="bold")

    rows = [
        ("reward", "total reward (raw)", None),
        ("num_any", "rolling-50 ≥1/4 numeric-correct", (-0.02, 1.02)),
        ("num_maj", "rolling-50 ≥2/4 numeric-correct", (-0.02, 1.02)),
        ("kl", "rolling-50 KL divergence", None),
    ]

    for ri, (key, ylabel, ylim) in enumerate(rows):
        for ci, m in enumerate(METHODS):
            ax = axes[ri, ci]
            for cfg in ["full", "numonly"]:
                d = data[(m, cfg)]
                if not d:
                    continue
                if key == "reward":
                    ys = rolling(d["reward"], w=20)
                elif key == "num_any":
                    pn = [1.0 if x > 0 else 0.0 for x in d["ans_n"]]
                    ys = rolling(pn, w=50)
                elif key == "num_maj":
                    pn = [1.0 if x >= 0.5 else 0.0 for x in d["ans_n"]]
                    ys = rolling(pn, w=50)
                elif key == "kl":
                    ys = rolling(d["kl"], w=20)
                ax.plot(range(len(ys)), ys, color=COLORS[cfg], label=cfg, lw=1.4)
            if ri == 0:
                ax.set_title(TITLES[m], fontsize=11)
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if ri == 3:
                ax.set_xlabel("step", fontsize=9)
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="best")

    out = os.path.join(HERE, "figures", "exp049_numonly_vs_full.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")

    # Print summary table
    print(f"\n{'method':22s}  {'cfg':9s}  {'r_L50':>8s}  {'num_any_L50':>11s}  {'num_maj_L50':>11s}  {'KL_L50':>7s}")
    for m in METHODS:
        for cfg in ["full", "numonly"]:
            d = data[(m, cfg)]
            if not d:
                continue
            tail = lambda xs, k=50: sum(xs[-k:])/min(k, len(xs)) if xs else 0.0
            pn_any = [1.0 if x > 0 else 0.0 for x in d["ans_n"]]
            pn_maj = [1.0 if x >= 0.5 else 0.0 for x in d["ans_n"]]
            print(f"{m:22s}  {cfg:9s}  {tail(d['reward']):>+8.3f}  "
                  f"{tail(pn_any):>11.2f}  {tail(pn_maj):>11.2f}  {tail(d['kl']):>7.4f}")


if __name__ == "__main__":
    main()
