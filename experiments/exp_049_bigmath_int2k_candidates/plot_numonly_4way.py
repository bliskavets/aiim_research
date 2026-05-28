"""
plot_numonly_4way.py — clean 4-way comparison of the numonly runs.

No format / exact-tag rewards anywhere; the only signal is reward_answer_numeric.
Lets us see how each shaping method behaves on pure "did the model emit the
correct number" supervision, with no tag template pressure.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)

METHODS = {
    "grpo":             ("GRPO baseline",              "#64748b"),
    "grpo_s_entropy":   ("GRPO-S seq-level entropy",   "#d97706"),
    "gtpo_conf":        ("GTPO per-token confidence",  "#059669"),
    "gtpo_ema_flipped": ("GTPO-EMA flipped",           "#4f46e5"),
}

PATTERNS = {
    "reward": r"'reward':\s*([-\d.]+)",
    "ans_n":  r"'rewards/reward_answer_numeric/mean':\s*([-\d.]+)",
    "kl":     r"'kl':\s*([-\d.]+)",
    "fzs":    r"'frac_reward_zero_std':\s*([-\d.]+)",
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
    data = {m: extract(os.path.join(HERE, f"train_{m}_numonly.log"))
            for m in METHODS}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "exp_049 numonly — Big-Math int-2000, Llama-3.2-3B · 4-way head-to-head\n"
        "only reward_answer_numeric used (no format_* / answer_exact rewards)",
        fontsize=12, weight="bold")

    panels = [
        ("reward",  "total reward (= answer_numeric mean, rolling-20)",  20, None),
        ("num_any", "frac batches with ≥1/4 numeric-correct (rolling-50)", 50, (-0.02, 1.02)),
        ("num_maj", "frac batches with ≥2/4 numeric-correct (rolling-50)", 50, (-0.02, 1.02)),
        ("kl",      "KL divergence (rolling-20)",                          20, None),
    ]

    series = {}
    for m in METHODS:
        d = data[m]
        if not d:
            continue
        pn_any = [1.0 if x > 0 else 0.0 for x in d["ans_n"]]
        pn_maj = [1.0 if x >= 0.5 else 0.0 for x in d["ans_n"]]
        series[m] = {"reward": d["reward"], "num_any": pn_any,
                     "num_maj": pn_maj, "kl": d["kl"]}

    for ax, (key, title, w, ylim) in zip(axes.flat, panels):
        for m, (label, color) in METHODS.items():
            if m not in series:
                continue
            ys = rolling(series[m][key], w=w)
            ax.plot(range(len(ys)), ys, color=color, label=label, lw=1.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8, loc="best")

    out = os.path.join(HERE, "figures", "exp049_numonly_4way.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}")

    print("\nlast-50 summary (numonly only):")
    print(f"  {'method':32s}  {'reward_L50':>10s}  {'num_any_L50':>11s}  {'num_maj_L50':>11s}  {'KL_L50':>7s}")
    for m, (label, _) in METHODS.items():
        if m not in series:
            continue
        s = series[m]
        tail = lambda xs, k=50: sum(xs[-k:])/min(k, len(xs))
        print(f"  {label:32s}  {tail(s['reward']):>+10.3f}  "
              f"{tail(s['num_any']):>11.2f}  {tail(s['num_maj']):>11.2f}  {tail(s['kl']):>7.4f}")


if __name__ == "__main__":
    main()
