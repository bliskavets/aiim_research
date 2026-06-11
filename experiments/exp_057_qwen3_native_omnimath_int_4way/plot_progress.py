"""
plot_progress.py — intermediate progress figure for exp_057 (run while training).

Parses the per-step metric dicts from train_<method>.log and draws a 2x2 panel
for the GRPO baseline (and any other methods that already have a log):
  (1) total reward (rolling-20)
  (2) reward components: answer_boxed / answer_numeric / format_thinking
  (3) completion clip-ratio + mean length
  (4) frac_reward_zero_std (within-group signal) + KL

Saves figures/exp057_progress.png. Honest about being mid-run: the title shows
each method's current step count.
"""
import ast
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)

METHODS = [
    ("grpo",             "GRPO baseline",                 "#64748b"),
    ("grpo_s_entropy",   "GRPO-S entropy",                "#d97706"),
    ("gtpo_conf",        "GTPO-Conf (tag-masked)",        "#059669"),
    ("gtpo_ema_flipped", "GTPO-EMA-flipped (tag-masked)", "#4f46e5"),
]


def rolling(xs, w=20):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out


def parse(path):
    if not os.path.exists(path):
        return []
    ds = []
    for m in re.finditer(r"\{'loss':.*?\}", open(path).read()):
        try:
            ds.append(ast.literal_eval(m.group(0)))
        except Exception:
            pass
    return ds


def col(ds, key):
    return [d.get(key, 0.0) for d in ds]


def main():
    data = {name: parse(os.path.join(HERE, f"train_{name}.log")) for name, _, _ in METHODS}
    present = [(n, lbl, c) for n, lbl, c in METHODS if data[n]]
    if not present:
        print("no train logs yet")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    (ax_r, ax_comp), (ax_clip, ax_sig) = axes

    for name, label, color in present:
        ds = data[name]
        steps = range(1, len(ds) + 1)
        ax_r.plot(steps, rolling(col(ds, "reward")), color=color, lw=2.0, label=label)

    ax_r.set_title("total reward (rolling-20)")
    ax_r.set_xlabel("step"); ax_r.set_ylabel("reward")
    ax_r.axhline(0, color="#94a3b8", lw=0.6, ls="--")
    ax_r.grid(alpha=0.3); ax_r.legend(fontsize=9, loc="lower right")

    # components: show for the first present method (baseline) to keep it readable
    bname, blabel, _ = present[0]
    bds = data[bname]
    steps = range(1, len(bds) + 1)
    for key, lbl, c in [
        ("rewards/reward_answer_boxed/mean",   "answer_boxed (max +3)",   "#059669"),
        ("rewards/reward_answer_numeric/mean", "answer_numeric (max +1.5)", "#d97706"),
        ("rewards/reward_format_thinking/mean", "format_thinking (max +2.5)", "#4f46e5"),
    ]:
        ax_comp.plot(steps, rolling(col(bds, key)), color=c, lw=1.8, label=lbl)
    ax_comp.set_title(f"reward components — {blabel} (rolling-20)")
    ax_comp.set_xlabel("step"); ax_comp.set_ylabel("component reward")
    ax_comp.axhline(0, color="#94a3b8", lw=0.6, ls="--")
    ax_comp.grid(alpha=0.3); ax_comp.legend(fontsize=9, loc="lower right")

    ax_clip.plot(steps, rolling(col(bds, "completions/clipped_ratio")), color="#dc2626", lw=1.8, label="clip ratio")
    ax_clip.set_ylim(0, 1.02)
    ax_clip.set_title(f"completion clipping & length — {blabel}")
    ax_clip.set_xlabel("step"); ax_clip.set_ylabel("clip ratio", color="#dc2626")
    ax_clip.grid(alpha=0.3)
    ax_len = ax_clip.twinx()
    ax_len.plot(steps, rolling(col(bds, "completions/mean_length")), color="#0891b2", lw=1.5, ls="--", label="mean length")
    ax_len.set_ylabel("mean completion length (tok)", color="#0891b2")

    ax_sig.plot(steps, rolling(col(bds, "frac_reward_zero_std")), color="#7c3aed", lw=1.8, label="frac_reward_zero_std")
    ax_sig.set_ylim(0, 1.02)
    ax_sig.set_title(f"within-group signal & KL — {blabel}")
    ax_sig.set_xlabel("step"); ax_sig.set_ylabel("frac_reward_zero_std", color="#7c3aed")
    ax_sig.grid(alpha=0.3)
    ax_kl = ax_sig.twinx()
    ax_kl.plot(steps, rolling(col(bds, "kl")), color="#b45309", lw=1.5, ls="--", label="KL")
    ax_kl.set_ylabel("KL", color="#b45309")

    status = ", ".join(f"{n}={len(data[n])}/1000" for n, _, _ in present)
    fig.suptitle(
        f"exp_057 — Qwen3-4B on Omni-MATH integer subset (1971), Qwen3 native format · IN PROGRESS [{status}]",
        fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(HERE, "figures", "exp057_progress.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"saved {out}  ({status})")


if __name__ == "__main__":
    main()
