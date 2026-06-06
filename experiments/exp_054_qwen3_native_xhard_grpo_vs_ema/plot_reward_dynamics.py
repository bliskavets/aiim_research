"""
plot_reward_dynamics.py — reward dynamics for exp_054 (Qwen3-4B):
GRPO baseline (no mask) + GTPO-EMA-flipped (6-tag mask) on
Big-Math integer ∩ llama8b_solve_rate <= 0.125.

Only the two methods that were actually launched in run_054.sh.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)

CURVES = [
    ("grpo",             "GRPO baseline (no mask)",                            "#64748b"),
    ("gtpo_ema_flipped", "GTPO-EMA flipped (tag-masked, 6 special tokens)",    "#4f46e5"),
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

    for method, label, color in CURVES:
        rewards = extract_reward(os.path.join(HERE, f"train_{method}.log"))
        if not rewards:
            continue
        ys = rolling(rewards, w=20)
        ax.plot(range(len(ys)), ys, color=color, lw=2.0, label=label)
        last50 = sum(rewards[-50:]) / min(50, len(rewards))
        ax.text(len(ys) + 5, last50, f"  L50 = {last50:+.2f}",
                color=color, fontsize=9, va="center", weight="bold")

    ax.set_title(
        "exp_054 — Qwen3-4B reward dynamics: GRPO baseline vs GTPO-EMA-flipped (tag-masked)\n"
        "Big-Math integer ∩ Llama-8B<=0.125, 1000 steps, full reward set, rolling-20 smoothed",
        fontsize=11, weight="bold")
    ax.set_xlabel("training step", fontsize=11)
    ax.set_ylabel("total reward (rolling-20 mean)", fontsize=11)
    ax.axhline(0, color="#64748b", lw=0.6, ls="--", alpha=0.6)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    out = os.path.join(HERE, "figures", "exp054_reward_dynamics.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
