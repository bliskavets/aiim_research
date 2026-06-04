"""
plot_grpo_only.py — exp_053 GRPO baseline alone, rolling-20 smoothed.

Isolates the baseline trajectory on the extra-hard subset to show
the effect of ng=16 + 1000 steps (vs exp_052's ng=4 + 500 steps).
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)


def rolling(xs, w=20):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out


def main():
    p = os.path.join(HERE, "train_grpo.log")
    txt = open(p).read()
    r = [float(m.group(1)) for m in re.finditer(r"'reward':\s*([-\d.]+)", txt)]
    ys = rolling(r, w=20)
    last50 = sum(r[-50:]) / min(50, len(r))
    peak = max(r)
    peak_step = r.index(peak) + 1

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(range(len(ys)), ys, color="#64748b", lw=2.2, label="GRPO baseline (no mask)")
    ax.axhline(0, color="#64748b", lw=0.6, ls="--", alpha=0.6)

    # annotate end and peak
    ax.text(len(ys) + 5, last50, f"  L50 = {last50:+.2f}",
            color="#64748b", fontsize=11, va="center", weight="bold")
    ax.scatter([peak_step], [peak], color="#d97706", s=60, zorder=5)
    ax.text(peak_step, peak + 0.25, f"peak {peak:+.2f} @ step {peak_step}",
            color="#d97706", fontsize=9, ha="center", weight="bold")

    ax.set_title(
        "exp_053 — Qwen3-4B GRPO baseline reward dynamics\n"
        "Big-Math integer ∩ Llama-8B<=0.125, ng=16, 1000 steps, rolling-20",
        fontsize=11, weight="bold")
    ax.set_xlabel("training step", fontsize=11)
    ax.set_ylabel("total reward (rolling-20 mean)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")

    out = os.path.join(HERE, "figures", "exp053_grpo_only_reward_dynamics.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved {out}")
    print(f"  steps logged: {len(r)}")
    print(f"  last-50 mean: {last50:+.3f}")
    print(f"  peak: {peak:+.3f} @ step {peak_step}")


if __name__ == "__main__":
    main()
