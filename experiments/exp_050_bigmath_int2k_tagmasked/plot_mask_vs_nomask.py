"""
plot_mask_vs_nomask.py — direct head-to-head: same method, mask vs no-mask.

For each of the 4 methods, overlay the exp_049 (no tag-mask) and exp_050
(with tag-mask) trajectories. grpo and grpo_s_entropy are controls (mask
is a no-op for them) — large cross-exp gaps here are run-to-run variance.
gtpo_conf and gtpo_ema_flipped are the real test — the mask should help
them learn the tag format.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/mnt/data/aiim_research"
NOMASK = os.path.join(REPO, "experiments/exp_049_bigmath_int2k_candidates")
MASKED = os.path.join(REPO, "experiments/exp_050_bigmath_int2k_tagmasked")

METHODS = [
    ("grpo",             "GRPO baseline                       (mask: n/a)"),
    ("grpo_s_entropy",   "GRPO-S seq-level entropy            (mask: n/a)"),
    ("gtpo_conf",        "GTPO per-token confidence           (mask: ACTIVE)"),
    ("gtpo_ema_flipped", "GTPO-EMA flipped                    (mask: ACTIVE)"),
]

PATTERNS = {
    "reward":       r"'reward':\s*([-\d.]+)",
    "answer_exact": r"'rewards/reward_answer_exact/mean':\s*([-\d.]+)",
    "format_exact": r"'rewards/reward_format_exact/mean':\s*([-\d.]+)",
    "kl":           r"'kl':\s*([-\d.]+)",
}


def extract(p):
    if not os.path.exists(p):
        return None
    with open(p) as f:
        txt = f.read()
    return {k: [float(m.group(1)) for m in re.finditer(rx, txt)]
            for k, rx in PATTERNS.items()}


def smooth(xs, w=20):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out


def main():
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle(
        "exp_050 (tag-mask) vs exp_049 (no mask) — Big-Math int-2000, Llama-3.2-3B\n"
        "rows = metrics; cols = methods. mask active on cols 3,4; cols 1,2 are control.",
        fontsize=12, weight="bold")

    rows = [("reward", "total reward (rolling-20)"),
            ("answer_exact", "answer_exact mean"),
            ("format_exact", "format_exact mean"),
            ("kl", "KL divergence")]

    for ri, (key, ylabel) in enumerate(rows):
        for ci, (method, title) in enumerate(METHODS):
            ax = axes[ri, ci]
            d_no = extract(os.path.join(NOMASK, f"train_{method}.log"))
            d_mk = extract(os.path.join(MASKED, f"train_{method}.log"))
            if d_no and d_no[key]:
                ax.plot(smooth(d_no[key]), color="#d97706", label="no mask (exp_049)", lw=1.4)
            if d_mk and d_mk[key]:
                ax.plot(smooth(d_mk[key]), color="#4f46e5", label="mask (exp_050)", lw=1.4)
            if ri == 0:
                ax.set_title(title, fontsize=9)
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if ri == 3:
                ax.set_xlabel("step", fontsize=9)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="best")

    out = os.path.join(MASKED, "figures", "exp050_vs_exp049_mask_effect.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")

    print(f"\n{'method':22s}  {'cfg':18s}  {'r_L50':>7s}  {'ans_e_L50':>9s}  {'fmt_e_L50':>9s}  {'exact_top':>9s}  {'KL_L50':>7s}")
    for method, _ in METHODS:
        for cfg, root in [("exp_049 (no-mask)", NOMASK), ("exp_050 (mask)", MASKED)]:
            d = extract(os.path.join(root, f"train_{method}.log"))
            if not d:
                continue
            tail = lambda xs, k=50: sum(xs[-k:])/min(k, len(xs)) if xs else 0.0
            # exact_top: frac batches with answer_exact_mean >= 1.5
            pe = [1.0 if x >= 1.5 else 0.0 for x in d["answer_exact"]]
            print(f"{method:22s}  {cfg:18s}  {tail(d['reward']):>+7.3f}  "
                  f"{tail(d['answer_exact']):>+9.3f}  {tail(d['format_exact']):>+9.3f}  "
                  f"{tail(pe):>9.2f}  {tail(d['kl']):>7.4f}")


if __name__ == "__main__":
    main()
