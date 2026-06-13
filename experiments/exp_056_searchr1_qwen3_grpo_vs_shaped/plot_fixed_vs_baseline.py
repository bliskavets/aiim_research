"""
plot_fixed_vs_baseline.py — INTERMEDIATE: does the now-actually-running shaping
diverge from plain GRPO on Search-R1?

After the shaping-bypass fix, gtpo_ema_flipped genuinely applies its per-token
EMA-confidence advantage (verified: shaped metrics logged every step). This
overlays, on EM (rolling-20) and KL:
  - grpo baseline (1000 steps, reference)
  - gtpo_ema_flipped FIXED (shaping ON, in progress)
  - gtpo_ema_flipped BYPASSED (old run, was silently plain GRPO)
The bypassed curve sat exactly on grpo; divergence of the FIXED curve from both
is the thing to watch.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)

CURVES = [
    ("train_grpo.log",                       "GRPO baseline (1000)",              "#64748b"),
    ("train_gtpo_ema_flipped.bypassed.log",  "GTPO-EMA BYPASSED (was plain GRPO)", "#a78bfa"),
    ("train_gtpo_ema_flipped.log",           "GTPO-EMA FIXED (shaping ON)",        "#dc2626"),
]


def rolling(xs, w=20):
    return [sum(xs[max(0, i - w + 1):i + 1]) / (i - max(0, i - w + 1) + 1)
            for i in range(len(xs))]


def extract(p, key):
    if not os.path.exists(p):
        return None
    txt = open(p).read()
    return [float(m.group(1)) for m in
            re.finditer(re.escape(key) + r"':\s*([-\d.eE+]+)", txt)]


def main():
    fig, (ax_em, ax_kl) = plt.subplots(1, 2, figsize=(15, 6))
    for fn, label, color in CURVES:
        em = extract(os.path.join(HERE, fn), "rewards/em/mean")
        kl = extract(os.path.join(HERE, fn), "kl")
        if not em:
            continue
        lw = 2.4 if "FIXED" in label else 1.8
        ys = rolling(em)
        ax_em.plot(range(len(ys)), ys, color=color, lw=lw, label=f"{label}  (n={len(em)})")
        ax_em.text(len(ys) + 3, ys[-1], f" {sum(em[-20:])/min(20,len(em)):.3f}",
                   color=color, fontsize=8, va="center", weight="bold")
        if kl:
            kys = rolling(kl)
            ax_kl.plot(range(len(kys)), kys, color=color, lw=lw, label=label)

    nfix = len(extract(os.path.join(HERE, "train_gtpo_ema_flipped.log"), "rewards/em/mean") or [])
    for ax, t, yl in [(ax_em, "SQuAD-EM reward (rolling-20)", "EM reward"),
                      (ax_kl, "KL divergence (rolling-20)", "KL")]:
        ax.axvline(nfix, color="#dc2626", lw=0.7, ls=":", alpha=0.5)
        ax.set_title(t, fontsize=12, weight="bold")
        ax.set_ylabel(yl); ax.set_xlabel("training step"); ax.grid(alpha=0.3)
    ax_em.axhline(0.20, color="#334155", lw=0.7, ls="--", alpha=0.6)
    ax_em.legend(fontsize=8.5, loc="upper right")
    ax_kl.legend(fontsize=8.5, loc="upper left")

    fig.suptitle(
        "exp_056 INTERMEDIATE — Search-R1 (Qwen3-4B): does ACTUAL shaping diverge from GRPO?\n"
        f"After the shaping-bypass fix, GTPO-EMA-flipped genuinely shapes (n={nfix} steps so far). "
        "Dotted line = current extent of the FIXED run.\n"
        "BYPASSED curve = the old run that silently ran plain GRPO (sat on baseline).",
        fontsize=10.5, weight="bold")
    out = os.path.join(HERE, "figures", "exp056_fixed_vs_baseline_intermediate.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=140)
    print(f"saved {out}  (fixed ema n={nfix})")


if __name__ == "__main__":
    main()
