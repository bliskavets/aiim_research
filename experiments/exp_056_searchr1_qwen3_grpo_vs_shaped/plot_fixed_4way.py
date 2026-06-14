"""
plot_fixed_4way.py — exp_056 FINAL (shaping-bypass FIXED): grpo baseline vs the
3 now-actually-shaping methods on Search-R1.

Two findings, both visible here:
  1. EM: all 3 shaped methods land at the SAME ~0.196 as grpo — no shaping helps
     (and, suspiciously, the 3 shaped EM curves are ~identical to each other while
     grpo differs — flagged for investigation: rollouts look insensitive to the
     diverging updates).
  2. KL (log scale): ema stays sane (~0.07) but gtpo_conf EXPLODES (KL→1e7,
     grad_norm→5e8) and grpo_s also blows up (KL~1e3) — the shaped advantage
     magnitudes are uncontrolled (znorm scaling). Shaping runs, but conf/grpo_s
     are numerically unstable on this stack.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CURVES = [
    ("train_grpo.log",                "grpo baseline (1000)",        "#64748b"),
    ("train_gtpo_ema_flipped.log",    "gtpo_ema_flipped (500)",      "#4f46e5"),
    ("train_gtpo_conf.log",           "gtpo_conf (500)",             "#dc2626"),
    ("train_grpo_s_entropy.log",      "grpo_s_entropy (500)",        "#d97706"),
]


def rolling(xs, w=20):
    return [sum(xs[max(0, i - w + 1):i + 1]) / (i - max(0, i - w + 1) + 1)
            for i in range(len(xs))]


def extract(p, key):
    if not os.path.exists(p):
        return None
    return [float(m.group(1)) for m in
            re.finditer(re.escape(key) + r"':\s*([-\d.eE+]+)", open(p).read())]


def main():
    fig, (ax_em, ax_kl) = plt.subplots(1, 2, figsize=(15, 6))
    for fn, label, color in CURVES:
        em = extract(os.path.join(HERE, fn), "rewards/em/mean")
        kl = extract(os.path.join(HERE, fn), "kl")
        if not em:
            continue
        ys = rolling(em)
        ax_em.plot(range(len(ys)), ys, color=color, lw=2.0, label=f"{label}")
        ax_em.text(len(ys) + 3, ys[-1], f" {sum(em[-100:])/min(100,len(em)):.3f}",
                   color=color, fontsize=8, va="center", weight="bold")
        if kl:
            kpos = [max(k, 1e-4) for k in rolling(kl)]  # clip for log scale
            ax_kl.plot(range(len(kpos)), kpos, color=color, lw=2.0, label=label)

    ax_em.set_title("SQuAD-EM reward (rolling-20)", fontsize=12, weight="bold")
    ax_em.set_ylabel("EM reward"); ax_em.set_xlabel("training step")
    ax_em.axhline(0.20, color="#334155", lw=0.7, ls="--", alpha=0.6)
    ax_em.legend(fontsize=9, loc="upper right"); ax_em.grid(alpha=0.3)

    ax_kl.set_yscale("log")
    ax_kl.set_title("KL divergence (rolling-20, LOG scale)", fontsize=12, weight="bold")
    ax_kl.set_ylabel("KL (log)"); ax_kl.set_xlabel("training step")
    ax_kl.legend(fontsize=9, loc="upper left"); ax_kl.grid(alpha=0.3, which="both")

    fig.suptitle(
        "exp_056 FINAL (shaping-bypass FIXED) — Search-R1 (NQ+HotpotQA, wiki-18, Qwen3-4B)\n"
        "EM: no shaping method beats grpo (~0.196 all) — and the 3 shaped EM curves are ~identical (flagged).\n"
        "KL (log): ema stable (~0.07), but gtpo_conf EXPLODES (KL→1e7, grad→5e8) and grpo_s blows up — "
        "shaped-advantage magnitudes uncontrolled (znorm).",
        fontsize=10, weight="bold")
    out = os.path.join(HERE, "figures", "exp056_fixed_4way_final.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
