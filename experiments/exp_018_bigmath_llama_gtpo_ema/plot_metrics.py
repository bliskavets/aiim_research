"""
Plot training metrics for exp_018 (GTPO-EMA).
Usage: python plot_metrics.py [--log PATH] [--out DIR] [--compare LOG017]

Generates:
  figures/dashboard.png      — 10-panel training dashboard
  figures/reward_detail.png  — 4-panel reward breakdown
  figures/compare_017.png    — overlay comparison with exp_017
  figures/gtpo_ema_metrics.png — GTPO-EMA specific metrics
  figures/metrics_summary.json
"""

import re
import os
import argparse
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

EXP_DIR     = os.path.dirname(os.path.abspath(__file__))
LOG_PATH    = os.path.join(EXP_DIR, "train.log")
OUT_DIR     = os.path.join(EXP_DIR, "figures")
LOG_017     = os.path.join(EXP_DIR, "..", "exp_017_bigmath_llama_int16gen", "train.log")

PATTERN = re.compile(r"\{'loss':.*?'epoch': \d+\.\d+\}")


def parse_log(path: str) -> list[dict]:
    records = []
    try:
        with open(path) as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Log not found: {path}")
        return []
    for i, m in enumerate(PATTERN.finditer(text)):
        try:
            d = eval(m.group())
            d["step"] = i + 1
            records.append(d)
        except Exception:
            pass
    return records


def smooth(values, w: int = 20) -> np.ndarray:
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    padded = np.pad(arr, (w // 2, w - w // 2 - 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def quick_summary(records: list[dict]) -> dict:
    if not records:
        return {}
    rewards    = [r.get("reward") or 0.0 for r in records]
    peak_reward = max(rewards)
    peak_step   = rewards.index(peak_reward) + 1
    first, last = records[0], records[-1]
    return {
        "steps":    len(records),
        "step1":    {k: first.get(k) for k in ("reward", "rewards/reward_format_exact/mean", "kl")},
        "step500":  {k: records[499].get(k) if len(records) >= 500 else None
                     for k in ("reward", "rewards/reward_format_exact/mean", "kl")},
        "step1000": {k: last.get(k) for k in ("reward", "rewards/reward_format_exact/mean", "kl")},
        "peak_reward": peak_reward,
        "peak_step":   peak_step,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_dashboard(records: list[dict], out_path: str, title: str):
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

    color = "tab:orange"
    for gs_loc, key, label in panels:
        ax = fig.add_subplot(gs_loc)
        vals = np.array([r.get(key, np.nan) for r in records], dtype=float)
        ax.plot(steps, vals, color=color, alpha=0.15, linewidth=0.8)
        ax.plot(steps, smooth(vals), color=color, linewidth=2, label="exp_018")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Step", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Reward detail
# ─────────────────────────────────────────────────────────────────────────────

def plot_reward_detail(records: list[dict], out_path: str):
    steps = [r["step"] for r in records]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("exp_018 — GTPO-EMA · Reward Functions Detail", fontsize=13, fontweight="bold")

    keys = [
        ("rewards/reward_format_exact/mean",       "Format Exact (+3.0 max)"),
        ("rewards/reward_format_approximate/mean", "Format Approx (+2.0 max)"),
        ("rewards/reward_answer_exact/mean",       "Answer Exact (+3.0 max)"),
        ("rewards/reward_answer_numeric/mean",     "Answer Numeric (+1.5 max)"),
    ]
    for ax, (key, label) in zip(axes.flat, keys):
        vals = np.array([r.get(key, np.nan) for r in records], dtype=float)
        ax.plot(steps, vals, alpha=0.15, linewidth=0.8, color="tab:orange")
        ax.plot(steps, smooth(vals), linewidth=2, color="tab:orange")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# GTPO-EMA specific metrics
# ─────────────────────────────────────────────────────────────────────────────

def plot_gtpo_ema_metrics(records: list[dict], out_path: str):
    steps = [r["step"] for r in records]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("exp_018 — GTPO-EMA Internal Metrics", fontsize=13, fontweight="bold")

    panels = [
        ("gtpo_ema/mean_confidence",      "Mean Confidence (C_i,t)"),
        ("gtpo_ema/mean_token_advantage", "Mean Token Advantage"),
        ("gtpo_ema/frac_pos",             "Fraction O+ Sequences"),
    ]
    for ax, (key, label) in zip(axes, panels):
        vals = np.array([r.get(key, np.nan) for r in records], dtype=float)
        has_data = ~np.isnan(vals)
        if has_data.any():
            ax.plot(steps, vals, alpha=0.2, linewidth=0.8, color="tab:purple")
            ax.plot(steps, smooth(vals), linewidth=2, color="tab:purple")
        else:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison: exp_017 vs exp_018
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(rec018: list[dict], rec017: list[dict], out_path: str):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "exp_017 (GRPO baseline) vs exp_018 (GTPO-EMA) — Big-Math · Llama-3.2-3B · 16 gens",
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

        for records, color, name in [
            (rec017, "tab:blue",   "exp_017 GRPO"),
            (rec018, "tab:orange", "exp_018 GTPO-EMA"),
        ]:
            if not records:
                continue
            steps = [r["step"] for r in records]
            vals  = np.array([r.get(key, np.nan) for r in records], dtype=float)
            ax.plot(steps, vals, color=color, alpha=0.12, linewidth=0.8)
            ax.plot(steps, smooth(vals), color=color, linewidth=2, label=name)

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Step", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main(log_path: str = LOG_PATH, out_dir: str = OUT_DIR, log_017: str = LOG_017):
    os.makedirs(out_dir, exist_ok=True)

    rec018 = parse_log(log_path)
    if not rec018:
        print("No records found in exp_018 log — training may not have started yet.")
        return

    print(f"Parsed {len(rec018)} steps from {log_path}")

    summary = quick_summary(rec018)
    print(f"\nSummary (exp_018):")
    r1 = summary["step1"]
    print(f"  Step 1:  reward={r1.get('reward', '?')},  kl={r1.get('kl', '?')}")
    if summary["step500"].get("reward") is not None:
        r5 = summary["step500"]
        print(f"  Step 500: reward={r5['reward']:.3f},  kl={r5['kl']:.4f}")
    print(f"  Peak:    reward={summary['peak_reward']:.3f} @ step {summary['peak_step']}")
    rl = summary["step1000"]
    print(f"  Latest:  reward={rl.get('reward', '?')},  kl={rl.get('kl', '?')}")

    summary_path = os.path.join(out_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    title = (
        "exp_018 — GTPO-EMA · Llama-3.2-3B · Big-Math (integer) · "
        "16 gens · batch=4 · bf16 · 1000 steps"
    )
    plot_dashboard(rec018, os.path.join(out_dir, "dashboard.png"), title)
    plot_reward_detail(rec018, os.path.join(out_dir, "reward_detail.png"))
    plot_gtpo_ema_metrics(rec018, os.path.join(out_dir, "gtpo_ema_metrics.png"))

    rec017 = parse_log(log_017)
    if rec017:
        print(f"Parsed {len(rec017)} steps from exp_017 log")
        plot_comparison(rec018, rec017, os.path.join(out_dir, "compare_017_vs_018.png"))
    else:
        print("exp_017 log not found — skipping comparison plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",     default=LOG_PATH)
    parser.add_argument("--out",     default=OUT_DIR)
    parser.add_argument("--log017",  default=LOG_017)
    args = parser.parse_args()
    main(args.log, args.out, args.log017)
