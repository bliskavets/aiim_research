"""
plot_metrics.py — 4-way comparison for exp_051 (tag-masked shaping, full reward).

Parses train_<method>.log for each method and overlays total reward,
answer_exact, format_exact and KL. Same shape as exp_049/plot_metrics.py.
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
    "reward":        r"'reward':\s*([-\d.]+)",
    "answer_exact":  r"'rewards/reward_answer_exact/mean':\s*([-\d.]+)",
    "format_exact":  r"'rewards/reward_format_exact/mean':\s*([-\d.]+)",
    "kl":            r"'kl':\s*([-\d.]+)",
}


def extract(log_path):
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        txt = f.read()
    return {k: [float(m.group(1)) for m in re.finditer(p, txt)]
            for k, p in PATTERNS.items()}


def smooth(xs, w=10):
    if len(xs) < w:
        return xs
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out


def main():
    data = {m: extract(os.path.join(HERE, f"train_{m}.log")) for m in METHODS}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("exp_051 — Big-Math int-2000, Qwen3-4B · tag-masked shaping (full reward) · "
                 "GRPO vs 3 candidates, identical hyperparameters seed 3407",
                 fontsize=12, weight="bold")
    panels = [("reward", "Total reward"), ("answer_exact", "Answer-exact reward"),
              ("format_exact", "Format-exact reward"), ("kl", "KL divergence")]

    for ax, (key, title) in zip(axes.flat, panels):
        for m, (label, color) in METHODS.items():
            d = data[m]
            if not d or not d[key]:
                continue
            ys = smooth(d[key])
            ax.plot(range(len(ys)), ys, color=color, label=label, lw=1.6)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8, loc="best")

    out = os.path.join(HERE, "figures", "exp051_4way_comparison.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}")

    print("\nlast-50 mean reward:")
    for m, (label, _) in METHODS.items():
        d = data[m]
        if d and d["reward"]:
            r = d["reward"]
            tail = r[-50:]
            print(f"  {label:32s} r@L50={sum(tail)/len(tail):+.3f}  "
                  f"peak={max(r):+.3f}  steps={len(r)}")
        else:
            print(f"  {label:32s} (no log yet)")


if __name__ == "__main__":
    main()
