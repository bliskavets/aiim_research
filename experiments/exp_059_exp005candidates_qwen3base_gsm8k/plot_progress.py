"""plot_progress.py — comparison figure for exp_059 (grpo vs gtpo_conf vs
grpo_s_conf on Qwen3-4B-Base / GSM8K). Parses train_<method>.log per-step dicts."""
import ast
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
METHODS = [
    ("grpo",        "GRPO baseline",            "#64748b"),
    ("gtpo_conf",   "GTPO-Conf (tag-masked)",   "#059669"),
    ("grpo_s_conf", "GRPO-S-Conf",              "#d97706"),
]


def rolling(xs, w=20):
    return [sum(xs[max(0, i - w + 1):i + 1]) / (i - max(0, i - w + 1) + 1) for i in range(len(xs))]


def parse(path):
    if not os.path.exists(path):
        return []
    return [ast.literal_eval(m.group(0)) for m in re.finditer(r"\{'loss':.*?\}", open(path).read())]


def col(ds, k):
    return [d.get(k, 0.0) for d in ds]


def main():
    data = {n: parse(os.path.join(HERE, f"train_{n}.log")) for n, _, _ in METHODS}
    present = [(n, l, c) for n, l, c in METHODS if data[n]]
    if not present:
        print("no logs yet"); return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    (ax_r, ax_fmt), (ax_ans, ax_kl) = axes

    for n, lbl, c in present:
        ds = data[n]
        ax_r.plot(range(1, len(ds) + 1), rolling(col(ds, "reward")), color=c, lw=2.0, label=lbl)
    ax_r.set_title("total reward (rolling-20)"); ax_r.set_xlabel("step"); ax_r.set_ylabel("reward")
    ax_r.axhline(0, color="#94a3b8", lw=0.6, ls="--"); ax_r.grid(alpha=0.3); ax_r.legend(fontsize=9, loc="lower right")

    for n, lbl, c in present:
        ds = data[n]
        ax_fmt.plot(range(1, len(ds) + 1), rolling(col(ds, "rewards/reward_format_exact/mean")), color=c, lw=1.8, label=lbl)
    ax_fmt.set_title("reward_format_exact (rolling-20, max +3)"); ax_fmt.set_xlabel("step")
    ax_fmt.axhline(0, color="#94a3b8", lw=0.6, ls="--"); ax_fmt.grid(alpha=0.3); ax_fmt.legend(fontsize=8)

    for n, lbl, c in present:
        ds = data[n]
        ax_ans.plot(range(1, len(ds) + 1), rolling(col(ds, "rewards/reward_answer_exact/mean")), color=c, lw=1.8, label=lbl)
    ax_ans.set_title("reward_answer_exact (rolling-20, max +3)"); ax_ans.set_xlabel("step")
    ax_ans.axhline(0, color="#94a3b8", lw=0.6, ls="--"); ax_ans.grid(alpha=0.3); ax_ans.legend(fontsize=8)

    for n, lbl, c in present:
        ds = data[n]
        ax_kl.plot(range(1, len(ds) + 1), rolling(col(ds, "kl")), color=c, lw=1.6, label=lbl)
    ax_kl.set_title("KL (rolling-20)"); ax_kl.set_xlabel("step"); ax_kl.grid(alpha=0.3); ax_kl.legend(fontsize=8)

    status = ", ".join(f"{n}={len(data[n])}" for n, _, _ in present)
    fig.suptitle(f"exp_059 — Qwen3-4B-Base / GSM8K (exp_005 candidates) [{status}]",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(HERE, "figures", "exp059_progress.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"saved {out}  ({status})")


if __name__ == "__main__":
    main()
