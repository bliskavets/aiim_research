"""Plot training metrics for exp_020 (GTPO per-token entropy)."""
import re
import os
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

EXP_DIR  = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(EXP_DIR, "train.log")
OUT_DIR  = os.path.join(EXP_DIR, "figures")
LOG_017  = os.path.join(EXP_DIR, "..", "exp_017_bigmath_llama_int16gen", "train.log")

PATTERN = re.compile(r"\{'loss':.*?'epoch': \d+\.\d+\}")
COLOR = "tab:red"
EXP_LABEL = "exp_020 GTPO"
CUSTOM_KEYS = [
    ("gtpo/mean_token_advantage", "Mean Token Advantage"),
    ("gtpo/mean_entropy",         "Mean Entropy"),
    ("gtpo/frac_pos",             "Fraction O+ Sequences"),
]
CUSTOM_TITLE = "exp_020 — GTPO (entropy) Internal Metrics"


def parse_log(path):
    records = []
    try:
        with open(path) as f:
            text = f.read()
    except FileNotFoundError:
        return []
    for i, m in enumerate(PATTERN.finditer(text)):
        try:
            d = eval(m.group())
            d["step"] = i + 1
            records.append(d)
        except Exception:
            pass
    return records


def smooth(values, w=20):
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    padded = np.pad(arr, (w // 2, w - w // 2 - 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def quick_summary(records):
    if not records:
        return {}
    rewards = [r.get("reward") or 0.0 for r in records]
    peak = max(rewards)
    return {
        "steps":    len(records),
        "step1":    records[0].get("reward"),
        "latest":   records[-1].get("reward"),
        "peak_reward": peak,
        "peak_step":   rewards.index(peak) + 1,
    }


def plot_dashboard(records, out_path, title, color=COLOR):
    steps = [r["step"] for r in records]
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)
    panels = [
        (gs[0, :2], "reward",                                 "Total Reward"),
        (gs[0, 2:], "kl",                                     "KL Divergence"),
        (gs[1, 0],  "rewards/reward_format_exact/mean",       "Format Exact"),
        (gs[1, 1],  "rewards/reward_format_approximate/mean", "Format Approx"),
        (gs[1, 2],  "rewards/reward_answer_exact/mean",       "Answer Exact"),
        (gs[1, 3],  "rewards/reward_answer_numeric/mean",     "Answer Numeric"),
        (gs[2, 0],  "loss",                                   "Loss"),
        (gs[2, 1],  "grad_norm",                              "Grad Norm"),
        (gs[2, 2],  "completion_length",                      "Completion Length"),
        (gs[2, 3],  "learning_rate",                          "Learning Rate"),
    ]
    for gs_loc, key, label in panels:
        ax = fig.add_subplot(gs_loc)
        vals = np.array([r.get(key, np.nan) for r in records], dtype=float)
        ax.plot(steps, vals, color=color, alpha=0.15, linewidth=0.8)
        ax.plot(steps, smooth(vals), color=color, linewidth=2)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Step", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_custom(records, out_path):
    steps = [r["step"] for r in records]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(CUSTOM_TITLE, fontsize=13, fontweight="bold")
    for ax, (key, label) in zip(axes, CUSTOM_KEYS):
        vals = np.array([r.get(key, np.nan) for r in records], dtype=float)
        if (~np.isnan(vals)).any():
            ax.plot(steps, vals, alpha=0.2, linewidth=0.8, color=COLOR)
            ax.plot(steps, smooth(vals), linewidth=2, color=COLOR)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_comparison(rec_self, rec_017, out_path):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"exp_017 (GRPO baseline) vs {EXP_LABEL} — Big-Math · Llama-3.2-3B · 16 gens",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)
    panels = [
        (gs[0, :2], "reward",                                 "Total Reward"),
        (gs[0, 2:], "kl",                                     "KL Divergence"),
        (gs[1, 0],  "rewards/reward_format_exact/mean",       "Format Exact"),
        (gs[1, 1],  "rewards/reward_format_approximate/mean", "Format Approx"),
        (gs[1, 2],  "rewards/reward_answer_exact/mean",       "Answer Exact"),
        (gs[1, 3],  "rewards/reward_answer_numeric/mean",     "Answer Numeric"),
    ]
    for gs_loc, key, label in panels:
        ax = fig.add_subplot(gs_loc)
        for records, clr, name in [
            (rec_017,  "tab:blue", "exp_017 GRPO"),
            (rec_self, COLOR,      EXP_LABEL),
        ]:
            if not records:
                continue
            steps = [r["step"] for r in records]
            vals = np.array([r.get(key, np.nan) for r in records], dtype=float)
            ax.plot(steps, vals, color=clr, alpha=0.12, linewidth=0.8)
            ax.plot(steps, smooth(vals), color=clr, linewidth=2, label=name)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Step", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rec = parse_log(LOG_PATH)
    if not rec:
        print("No records.")
        return
    print(f"Parsed {len(rec)} steps")
    summary = quick_summary(rec)
    print(f"  Peak: {summary['peak_reward']:.3f} @ step {summary['peak_step']}")
    print(f"  Latest: {summary['latest']}")
    with open(os.path.join(OUT_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    title = "exp_020 — GTPO (entropy) · Llama-3.2-3B · Big-Math (integer) · 16 gens · bf16"
    plot_dashboard(rec, os.path.join(OUT_DIR, "dashboard.png"), title)
    plot_custom(rec, os.path.join(OUT_DIR, "gtpo_metrics.png"))

    rec_017 = parse_log(LOG_017)
    if rec_017:
        plot_comparison(rec, rec_017, os.path.join(OUT_DIR, "compare_017_vs_020.png"))


if __name__ == "__main__":
    main()
