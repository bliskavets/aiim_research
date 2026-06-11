"""
plot_grpo_vs_grpo_s.py — INTERMEDIATE comparison for exp_056 (Search-R1, Qwen3-4B):
GRPO baseline (1000 steps, done) vs GRPO-S seq-level entropy (in progress).

Headline metric is the SQuAD-EM reward (rewards/em/mean == reward here).
Per-step EM is bimodal (0 / ~0.8) on ng=4, so judge by the rolling-50 window.
KL is the secondary panel (shaping is expected to act as a mild KL regularizer).
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)

CURVES = [
    ("grpo",           "GRPO baseline (1000 steps)",          "#64748b"),
    ("grpo_s_entropy", "GRPO-S seq-level entropy (OOM @785)", "#d97706"),
]


def rolling(xs, w=50):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out


def extract(p, key):
    if not os.path.exists(p):
        return None
    txt = open(p).read()
    pat = re.escape(key) + r"':\s*([-\d.eE+]+)"
    return [float(m.group(1)) for m in re.finditer(pat, txt)]


def main():
    fig, (ax_em, ax_kl) = plt.subplots(1, 2, figsize=(15, 6))

    n_steps = {}
    for method, label, color in CURVES:
        log = os.path.join(HERE, f"train_{method}.log")
        em = extract(log, "rewards/em/mean")
        kl = extract(log, "kl")
        if not em:
            continue
        n_steps[method] = len(em)

        ys = rolling(em, w=50)
        ax_em.plot(range(len(ys)), ys, color=color, lw=2.0,
                   label=f"{label}  (n={len(em)})")
        last50 = sum(em[-50:]) / min(50, len(em))
        ax_em.text(len(ys) + 5, ys[-1], f"  L50={last50:.3f}",
                   color=color, fontsize=9, va="center", weight="bold")

        if kl:
            kys = rolling(kl, w=50)
            ax_kl.plot(range(len(kys)), kys, color=color, lw=2.0, label=label)
            kl50 = sum(kl[-50:]) / min(50, len(kl))
            ax_kl.text(len(kys) + 5, kys[-1], f"  {kl50:.3f}",
                       color=color, fontsize=9, va="center", weight="bold")

    smaller = min(n_steps.values()) if n_steps else 0
    for ax in (ax_em, ax_kl):
        ax.axvline(smaller, color="#94a3b8", lw=0.8, ls=":", alpha=0.8)
        ax.grid(alpha=0.3)
        ax.set_xlabel("training step", fontsize=11)

    ax_em.set_title("SQuAD-EM reward (rolling-50)", fontsize=11, weight="bold")
    ax_em.set_ylabel("EM reward", fontsize=11)
    ax_em.axhline(0.20, color="#64748b", lw=0.6, ls="--", alpha=0.6)
    ax_em.text(2, 0.205, "grpo plateau ~0.20", fontsize=8, color="#64748b")
    ax_em.legend(fontsize=9, loc="upper right")

    ax_kl.set_title("KL divergence (rolling-50)", fontsize=11, weight="bold")
    ax_kl.set_ylabel("KL", fontsize=11)
    ax_kl.legend(fontsize=9, loc="upper left")

    fig.suptitle(
        "exp_056 — Search-R1 (NQ+HotpotQA, wiki-18, Qwen3-4B)\n"
        "GRPO baseline (1000 steps) vs GRPO-S seq-level entropy (OOM-crashed @ step 785)\n"
        "ng=4, max_completion=4096, multi-turn retrieval; dotted line = where GRPO-S stopped\n"
        "EM trajectories near-identical (data-order-bound); KL differs (shaping = milder drift) — no measurable EM gain",
        fontsize=10, weight="bold")

    out = os.path.join(HERE, "figures", "exp056_grpo_vs_grpo_s_intermediate.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=140)
    print(f"saved {out}")
    for m, n in n_steps.items():
        print(f"  {m}: {n} steps logged")


if __name__ == "__main__":
    main()
