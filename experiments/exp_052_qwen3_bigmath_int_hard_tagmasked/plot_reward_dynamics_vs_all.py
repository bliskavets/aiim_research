"""
plot_reward_dynamics_vs_all.py — 3-panel reward dynamics across the
tag-mask sweep so far:

  left   = exp_050  Llama-3.2-3B   Big-Math int-2000
  middle = exp_051  Qwen3-4B       Big-Math int-2000
  right  = exp_052  Qwen3-4B       Big-Math integer ∩ llama8b_solve_rate<0.3

Same 4 methods on each panel, same colors, shared y-axis so the
difficulty / model-strength regime is visible at a glance.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/mnt/data/aiim_research"
PANELS = [
    ("exp_050  Llama-3.2-3B  ·  Big-Math int-2000",
     os.path.join(REPO, "experiments/exp_050_bigmath_int2k_tagmasked")),
    ("exp_051  Qwen3-4B  ·  Big-Math int-2000",
     os.path.join(REPO, "experiments/exp_051_qwen3_bigmath_int2k_tagmasked")),
    ("exp_052  Qwen3-4B  ·  Big-Math integer ∩ Llama-8B<0.3",
     os.path.join(REPO, "experiments/exp_052_qwen3_bigmath_int_hard_tagmasked")),
]

METHODS = [
    ("grpo",             "GRPO baseline (no mask)",                  "#64748b"),
    ("grpo_s_entropy",   "GRPO-S seq entropy (mask n/a)",            "#d97706"),
    ("gtpo_conf",        "GTPO per-token conf (tag-masked)",         "#059669"),
    ("gtpo_ema_flipped", "GTPO-EMA flipped (tag-masked)",            "#4f46e5"),
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
    return [float(m.group(1)) for m in re.finditer(r"'reward':\s*([-\d.]+)", open(p).read())]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharey=True)
    fig.suptitle(
        "tag-masked per-token shaping — reward dynamics across exp_050 / exp_051 / exp_052",
        fontsize=12, weight="bold")

    for ax, (title, root) in zip(axes, PANELS):
        for method, label, color in METHODS:
            rewards = extract_reward(os.path.join(root, f"train_{method}.log"))
            if not rewards:
                continue
            ys = rolling(rewards, w=20)
            ax.plot(range(len(ys)), ys, color=color, lw=1.7, label=label)
            last50 = sum(rewards[-50:]) / min(50, len(rewards))
            ax.text(len(ys) + 5, last50, f"  {last50:+.2f}",
                    color=color, fontsize=8.5, va="center", weight="bold")
        ax.set_title(title, fontsize=10.5, weight="bold")
        ax.set_xlabel("training step")
        ax.axhline(0, color="#64748b", lw=0.6, ls="--", alpha=0.6)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 555)

    axes[0].set_ylabel("total reward (rolling-20 mean)", fontsize=11)
    axes[0].legend(fontsize=8.5, loc="lower right")

    out = os.path.join(REPO, "experiments/exp_052_qwen3_bigmath_int_hard_tagmasked",
                       "figures", "exp052_vs_exp051_vs_exp050_reward_dynamics.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
