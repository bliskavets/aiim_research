"""
plot_reward_dynamics.py — reward dynamics: GRPO baseline + per-token methods
with important-token masking (tag-masked shaping).

Single panel, rolling-mean smoothed total reward, exp_050 runs only:
  GRPO baseline           (no shaping, mask n/a)
  GTPO per-token conf     (tag-masked shaping)
  GTPO-EMA flipped        (tag-masked shaping)
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)

CURVES = [
    ("grpo",             "GRPO baseline",                     "#64748b"),
    ("gtpo_conf",        "GTPO per-token confidence (tag-masked)",  "#059669"),
    ("gtpo_ema_flipped", "GTPO-EMA flipped (tag-masked)",     "#4f46e5"),
]


def rolling(xs, w=20):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out


def extract_reward(p):
    if not os.path.exists(p):
        return None
    txt = open(p).read()
    return [float(m.group(1)) for m in re.finditer(r"'reward':\s*([-\d.]+)", txt)]


def main():
    fig, ax = plt.subplots(figsize=(12, 6.5))

    summary = []
    for method, label, color in CURVES:
        rewards = extract_reward(os.path.join(HERE, f"train_{method}.log"))
        if not rewards:
            continue
        ys = rolling(rewards, w=20)
        ax.plot(range(len(ys)), ys, color=color, lw=2.0, label=label)
        last50 = sum(rewards[-50:]) / min(50, len(rewards))
        summary.append((label, last50, max(rewards), len(rewards)))

    ax.set_title(
        "exp_050 — reward dynamics: GRPO baseline vs per-token methods with important-token masking\n"
        "Big-Math int-2000, Llama-3.2-3B, 500 steps, full reward set, rolling-20 smoothed",
        fontsize=11, weight="bold")
    ax.set_xlabel("training step", fontsize=11)
    ax.set_ylabel("total reward (rolling-20 mean)", fontsize=11)
    ax.axhline(0, color="#64748b", lw=0.6, ls="--", alpha=0.6)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")

    # annotate last-50 means as small text on the right edge
    xmax = 500
    for method, label, color in CURVES:
        rewards = extract_reward(os.path.join(HERE, f"train_{method}.log"))
        if not rewards:
            continue
        last50 = sum(rewards[-50:]) / min(50, len(rewards))
        ax.text(xmax + 5, last50, f"  L50 = {last50:+.2f}",
                color=color, fontsize=9, va="center", weight="bold")

    out = os.path.join(HERE, "figures", "exp050_reward_dynamics_baseline_vs_masked.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved {out}")

    print("\nlast-50 mean reward:")
    for label, last50, peak, n in summary:
        print(f"  {label:50s}  L50={last50:+.3f}  peak={peak:+.2f}  steps={n}")


if __name__ == "__main__":
    main()
