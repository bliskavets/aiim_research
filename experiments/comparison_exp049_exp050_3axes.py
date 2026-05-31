"""
comparison_exp049_exp050_3axes.py
----------------------------------
Cross-experiment summary: 4 methods × 3 reward configs × 4 metrics, all
on Big-Math int-2000 / Llama-3.2-3B / bs=1×ng=4 / seed 3407 / 500 steps.

Configs:
  full     — exp_049, full reward (format_exact + format_approx + answer_exact + answer_numeric)
  numonly  — exp_049, only reward_answer_numeric
  tag-mask — exp_050, full reward + tag-mask off per-token shaping on format-tag tokens

Each panel is a grouped bar chart: x = method, hue = config.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/mnt/data/aiim_research"
EXP049 = os.path.join(REPO, "experiments/exp_049_bigmath_int2k_candidates")
EXP050 = os.path.join(REPO, "experiments/exp_050_bigmath_int2k_tagmasked")
OUT = os.path.join(REPO, "experiments/comparison_exp049_exp050_3axes.png")

METHODS = [
    ("grpo",             "GRPO\nbaseline"),
    ("grpo_s_entropy",   "GRPO-S\nseq entropy"),
    ("gtpo_conf",        "GTPO\nper-token conf"),
    ("gtpo_ema_flipped", "GTPO-EMA\nflipped"),
]

CONFIGS = [
    ("full",     "full reward (exp_049)",            EXP049, "{m}",            "#64748b"),
    ("numonly",  "numeric-only reward (exp_049)",    EXP049, "{m}_numonly",   "#d97706"),
    ("tagmask",  "full reward + tag-mask (exp_050)", EXP050, "{m}",           "#4f46e5"),
]

PATTERNS = {
    "reward":       r"'reward':\s*([-\d.]+)",
    "answer_exact": r"'rewards/reward_answer_exact/mean':\s*([-\d.]+)",
    "format_exact": r"'rewards/reward_format_exact/mean':\s*([-\d.]+)",
    "answer_num":   r"'rewards/reward_answer_numeric/mean':\s*([-\d.]+)",
}


def extract_last50(log_path):
    if not os.path.exists(log_path):
        return None
    txt = open(log_path).read()
    out = {}
    for k, rx in PATTERNS.items():
        xs = [float(m.group(1)) for m in re.finditer(rx, txt)]
        if not xs:
            out[k] = None
            continue
        out[k] = sum(xs[-50:]) / min(50, len(xs))
    # derive exact_top (frac batches with answer_exact mean ≥ 1.5)
    xs_ae = [float(m.group(1)) for m in re.finditer(PATTERNS["answer_exact"], txt)]
    if xs_ae:
        pe = [1.0 if x >= 1.5 else 0.0 for x in xs_ae]
        out["exact_top"] = sum(pe[-50:]) / min(50, len(pe))
    else:
        out["exact_top"] = None
    return out


def main():
    # collect: data[method][cfg] = {metrics}
    data = {}
    for method, _ in METHODS:
        data[method] = {}
        for cfg_key, _, root, tmpl, _ in CONFIGS:
            fname = tmpl.format(m=method) + ".log"
            data[method][cfg_key] = extract_last50(os.path.join(root, "train_" + fname))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "exp_049 + exp_050 summary — Big-Math int-2000, Llama-3.2-3B (500 steps each)\n"
        "4 methods × 3 reward/mask configs · last-50 mean per metric",
        fontsize=12, weight="bold")

    panels = [
        ("reward",       "total reward (last-50 mean)"),
        ("answer_exact", "answer_exact reward (last-50 mean)"),
        ("format_exact", "format_exact reward (last-50 mean)"),
        ("exact_top",    "exact_top — frac batches with ≥2/4 correct in tags"),
    ]

    n_methods = len(METHODS)
    n_cfgs = len(CONFIGS)
    width = 0.25
    x = np.arange(n_methods)

    for ax, (key, title) in zip(axes.flat, panels):
        for ci, (cfg_key, cfg_label, _, _, color) in enumerate(CONFIGS):
            ys = []
            for method, _ in METHODS:
                v = data[method][cfg_key]
                ys.append(v[key] if v and v.get(key) is not None else np.nan)
            offsets = x + (ci - (n_cfgs - 1) / 2) * width
            bars = ax.bar(offsets, ys, width=width, color=color,
                          label=cfg_label, edgecolor="#1e293b", linewidth=0.4)
            # annotate above the bar
            for bx, by in zip(offsets, ys):
                if np.isnan(by):
                    continue
                va = "bottom" if by >= 0 else "top"
                dy = 0.04 if by >= 0 else -0.04
                ax.text(bx, by + dy, f"{by:.2f}", ha="center", va=va, fontsize=7.5,
                        color="#1e293b")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in METHODS], fontsize=8.5)
        ax.axhline(0, color="#64748b", lw=0.6, ls="--", alpha=0.6)
        ax.grid(alpha=0.3, axis="y")
    axes[0, 0].legend(fontsize=7.5, loc="upper left")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"saved {OUT}")

    # ── Console summary table ─────────────────────────────────────────────
    print(f"\n{'method':22s}  {'config':28s}  {'r_L50':>7s}  {'ans_e_L50':>9s}  {'fmt_e_L50':>9s}  {'exact_top':>9s}")
    for method, _ in METHODS:
        for cfg_key, cfg_label, _, _, _ in CONFIGS:
            d = data[method][cfg_key]
            if not d:
                print(f"{method:22s}  {cfg_label:28s}  (no log)")
                continue
            def fmt(v, w, prec, sign=""):
                if v is None:
                    return ("-" * (w - 2)).rjust(w)
                return f"{v:>{sign}{w}.{prec}f}"
            print(f"{method:22s}  {cfg_label:28s}  "
                  f"{fmt(d['reward'], 7, 3, '+')}  "
                  f"{fmt(d['answer_exact'], 9, 3, '+')}  "
                  f"{fmt(d['format_exact'], 9, 3, '+')}  "
                  f"{fmt(d['exact_top'], 9, 2)}")


if __name__ == "__main__":
    main()
