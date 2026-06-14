"""
plot_answer_boxed_dynamics.py — exp_057 reward_answer_boxed only (rolling-20).

Isolates the strict-correctness signal: did the model put the right integer
inside \\boxed{N} after closing </think>. Possible values per batch mean:
  +3.0  every rollout had correct boxed integer
   0.0  no \\boxed{} found
  -1.5  boxed found but integer wrong (no partial credit)
A mean of, say, +1.5 means roughly half the rollouts got it right.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)

CURVES = [
    ("grpo",             "GRPO baseline (no mask)",                  "#64748b"),
    ("grpo_s_entropy",   "GRPO-S seq-level entropy",                 "#d97706"),
    ("gtpo_conf",        "GTPO per-token confidence (tag-masked)",   "#059669"),
    ("gtpo_ema_flipped", "GTPO-EMA flipped (tag-masked)",            "#4f46e5"),
]


def rolling(xs, w=20):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out


def extract_boxed(p):
    if not os.path.exists(p):
        return None
    txt = open(p).read()
    return [float(m.group(1)) for m in re.finditer(
        r"'rewards/reward_answer_boxed/mean':\s*([-\d.]+)", txt)]


def main():
    fig, ax = plt.subplots(figsize=(12, 6.5))

    for method, label, color in CURVES:
        ys_raw = extract_boxed(os.path.join(HERE, f"train_{method}.log"))
        if not ys_raw:
            continue
        ys = rolling(ys_raw, w=20)
        ax.plot(range(len(ys)), ys, color=color, lw=2.0, label=label)
        last50 = sum(ys_raw[-50:]) / min(50, len(ys_raw))
        ax.text(len(ys) + 5, last50, f"  L50 = {last50:+.2f}",
                color=color, fontsize=9, va="center", weight="bold")

    ax.set_title(
        "exp_057 — reward_answer_boxed dynamics (strict integer match in \\boxed{})\n"
        "Qwen3-4B, Omni-MATH integer subset (1971), max +3.0 (per batch mean), rolling-20 smoothed",
        fontsize=11, weight="bold")
    ax.set_xlabel("training step", fontsize=11)
    ax.set_ylabel("reward_answer_boxed (rolling-20 mean)", fontsize=11)
    ax.axhline(0, color="#64748b", lw=0.6, ls="--", alpha=0.6)
    ax.axhline(3.0, color="#059669", lw=0.5, ls=":", alpha=0.4)
    ax.text(5, 3.0, "  +3.0 = all-correct ceiling", color="#059669", fontsize=8, va="bottom")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    out = os.path.join(HERE, "figures", "exp057_answer_boxed_dynamics.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
