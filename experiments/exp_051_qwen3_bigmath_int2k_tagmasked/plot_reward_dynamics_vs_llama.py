"""
plot_reward_dynamics_vs_llama.py — side-by-side reward dynamics, Llama vs Qwen3.

Left panel: exp_050 (Llama-3.2-3B, max_seq=2560)
Right panel: exp_051 (Qwen3-4B,   max_seq=4096)

Same 4 methods on each panel, same colors, shared y-axis. Shows at a
glance which methods transfer the tag-masked-shaping win from Llama
to Qwen3.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO  = "/mnt/data/aiim_research"
LLAMA = os.path.join(REPO, "experiments/exp_050_bigmath_int2k_tagmasked")
QWEN3 = os.path.join(REPO, "experiments/exp_051_qwen3_bigmath_int2k_tagmasked")

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
    txt = open(p).read()
    return [float(m.group(1)) for m in re.finditer(r"'reward':\s*([-\d.]+)", txt)]


def plot_panel(ax, root, title):
    for method, label, color in METHODS:
        rewards = extract_reward(os.path.join(root, f"train_{method}.log"))
        if not rewards:
            continue
        ys = rolling(rewards, w=20)
        ax.plot(range(len(ys)), ys, color=color, lw=1.8, label=label)
        last50 = sum(rewards[-50:]) / min(50, len(rewards))
        ax.text(len(ys) + 5, last50, f"  {last50:+.2f}",
                color=color, fontsize=8.5, va="center", weight="bold")
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel("training step")
    ax.axhline(0, color="#64748b", lw=0.6, ls="--", alpha=0.6)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 540)


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
    fig.suptitle(
        "tag-masked per-token shaping — transfer Llama → Qwen3 · Big-Math int-2000, 500 steps · rolling-20",
        fontsize=12, weight="bold")
    plot_panel(axL, LLAMA, "exp_050 — Llama-3.2-3B-Instruct (max_seq=2560)")
    plot_panel(axR, QWEN3, "exp_051 — Qwen/Qwen3-4B (max_seq=4096)")
    axL.set_ylabel("total reward (rolling-20 mean)", fontsize=11)
    axL.legend(fontsize=8.5, loc="lower right")

    out = os.path.join(QWEN3, "figures", "exp051_vs_exp050_reward_dynamics.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
