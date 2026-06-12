"""
plot_all_methods.py — exp_056 final comparison on Search-R1 (NQ+HotpotQA,
wiki-18, Qwen3-4B native). All 4 methods, EM reward (rolling-50) + KL.

Run lengths differ (each stopped at plateau or OOM):
  grpo             1000 (full)
  grpo_s_entropy    785 (OOM in backward @ step 785)
  gtpo_ema_flipped  785 (OOM in backward @ step 785, same seeded long batch)
  gtpo_conf         402 (stopped at plateau, before the 785 OOM batch)

Headline: the 4 EM curves are indistinguishable (shaping does not move EM on
Search-R1 at this budget); only KL differs (shaped methods drift less).
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)

CURVES = [
    ("grpo",             "GRPO baseline (1000)",            "#64748b"),
    ("grpo_s_entropy",   "GRPO-S seq entropy (785, OOM)",   "#d97706"),
    ("gtpo_ema_flipped", "GTPO-EMA flipped (785, OOM)",     "#4f46e5"),
    ("gtpo_conf",        "GTPO-conf (402, stopped)",        "#059669"),
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

    for method, label, color in CURVES:
        log = os.path.join(HERE, f"train_{method}.log")
        em = extract(log, "rewards/em/mean")
        kl = extract(log, "kl")
        if not em:
            continue
        ys = rolling(em, 50)
        ax_em.plot(range(len(ys)), ys, color=color, lw=2.0, label=label)
        l100 = sum(em[-100:]) / min(100, len(em))
        ax_em.text(len(ys) + 4, ys[-1], f" {l100:.3f}", color=color,
                   fontsize=8, va="center", weight="bold")
        if kl:
            kys = rolling(kl, 50)
            ax_kl.plot(range(len(kys)), kys, color=color, lw=2.0, label=label)

    ax_em.set_title("SQuAD-EM reward (rolling-50)", fontsize=12, weight="bold")
    ax_em.set_ylabel("EM reward")
    ax_em.set_xlabel("training step")
    ax_em.axhline(0.20, color="#334155", lw=0.7, ls="--", alpha=0.7)
    ax_em.text(4, 0.205, "all methods ≈ 0.20", fontsize=8, color="#334155")
    ax_em.legend(fontsize=8.5, loc="upper right")
    ax_em.grid(alpha=0.3)

    ax_kl.set_title("KL divergence (rolling-50)", fontsize=12, weight="bold")
    ax_kl.set_ylabel("KL")
    ax_kl.set_xlabel("training step")
    ax_kl.legend(fontsize=8.5, loc="upper left")
    ax_kl.grid(alpha=0.3)

    fig.suptitle(
        "exp_056 — Search-R1 (NQ+HotpotQA, wiki-18, Qwen3-4B native), 4 methods\n"
        "EM curves are indistinguishable — neither shaping NOR GRPO itself moves EM "
        "(policy barely drifts, EM is data-order-bound). Only KL differs.\n"
        "ng=4, lr 5e-6 cosine, max_completion=4096, tag-mask on shaped methods; "
        "GTPO-conf shows no length-explosion collapse here",
        fontsize=10.5, weight="bold")

    out = os.path.join(HERE, "figures", "exp056_all_methods_comparison.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
