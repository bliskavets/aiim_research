"""
plot_grpo_only.py — standalone GRPO baseline dynamics on Search-R1
(NQ+HotpotQA, wiki-18 retrieval, Qwen3-4B native format).

4 panels: EM reward (rolling-50 + raw), KL, completion length, and
retrieval/finish behavior (n_searches + frac_finish_answer). 1000 steps.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
COLOR = "#64748b"
ACCENT = "#059669"


def rolling(xs, w=50):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out


def extract(p, key):
    txt = open(p).read()
    pat = re.escape(key) + r"':\s*([-\d.eE+]+)"
    return [float(m.group(1)) for m in re.finditer(pat, txt)]


def main():
    log = os.path.join(HERE, "train_grpo.log")
    em = extract(log, "rewards/em/mean")
    kl = extract(log, "kl")
    clen = extract(log, "completion_length")
    nse = extract(log, "searchr1/n_searches_mean")
    fa = extract(log, "searchr1/frac_finish_answer")
    n = len(em)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    (ax_em, ax_kl), (ax_len, ax_beh) = axes

    # EM
    ax_em.plot(range(n), em, color=ACCENT, lw=0.6, alpha=0.25, label="per-step EM")
    ax_em.plot(range(n), rolling(em, 50), color=ACCENT, lw=2.2, label="rolling-50")
    overall = sum(em) / n
    ax_em.axhline(overall, color="#334155", lw=0.8, ls="--",
                  label=f"overall {overall:.3f}")
    ax_em.set_title("SQuAD-EM reward", fontsize=12, weight="bold")
    ax_em.set_ylabel("EM reward")
    ax_em.legend(fontsize=9, loc="upper right")

    # KL
    ax_kl.plot(range(len(kl)), kl, color="#b91c1c", lw=0.6, alpha=0.3)
    ax_kl.plot(range(len(kl)), rolling(kl, 50), color="#b91c1c", lw=2.0)
    ax_kl.set_title(f"KL divergence (max {max(kl):.1f}, last-50 {sum(kl[-50:])/50:.3f})",
                    fontsize=12, weight="bold")
    ax_kl.set_ylabel("KL")

    # completion length
    ax_len.plot(range(len(clen)), clen, color="#7c3aed", lw=0.6, alpha=0.3)
    ax_len.plot(range(len(clen)), rolling(clen, 50), color="#7c3aed", lw=2.0)
    ax_len.axhline(4096, color="#94a3b8", lw=0.8, ls=":")
    ax_len.text(2, 4096 * 0.96, "max_completion=4096", fontsize=8, color="#64748b")
    ax_len.set_title(f"completion length (mean {sum(clen)/len(clen):.0f}, max {max(clen):.0f})",
                     fontsize=12, weight="bold")
    ax_len.set_ylabel("tokens")
    ax_len.set_xlabel("training step")

    # behavior
    ax_beh.plot(range(len(nse)), rolling(nse, 50), color="#0891b2", lw=2.0,
                label="n_searches/rollout")
    ax_beh2 = ax_beh.twinx()
    ax_beh2.plot(range(len(fa)), rolling(fa, 50), color="#ca8a04", lw=2.0,
                 label="frac_finish_answer")
    ax_beh.set_title("retrieval & finish behavior", fontsize=12, weight="bold")
    ax_beh.set_ylabel("mean searches/rollout", color="#0891b2")
    ax_beh2.set_ylabel("frac finishing with <answer>", color="#ca8a04")
    ax_beh2.set_ylim(0, 1.05)
    ax_beh.set_xlabel("training step")
    l1, lab1 = ax_beh.get_legend_handles_labels()
    l2, lab2 = ax_beh2.get_legend_handles_labels()
    ax_beh.legend(l1 + l2, lab1 + lab2, fontsize=9, loc="lower right")

    for ax in (ax_em, ax_kl, ax_len, ax_beh):
        ax.grid(alpha=0.3)

    fig.suptitle(
        "exp_056 — GRPO baseline on Search-R1 (NQ+HotpotQA, wiki-18 retrieval, Qwen3-4B native)\n"
        f"ng=4, lr 5e-6 cosine, max_completion=4096, max_turns=4, topk=3 — {n} steps",
        fontsize=12, weight="bold")
    out = os.path.join(HERE, "figures", "exp056_grpo_only.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=140)
    print(f"saved {out}  ({n} steps, EM overall {overall:.3f}, last50 {sum(em[-50:])/50:.3f})")


if __name__ == "__main__":
    main()
