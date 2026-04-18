"""
Plot training metrics for exp_017.
Usage: python plot_metrics.py [--log PATH] [--out DIR]

Reads train.log (or path from --log), generates figures/ in experiment folder.
Can also be called from compare_experiments.py to overlay on multi-exp charts.
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

EXP_DIR  = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(EXP_DIR, "train.log")
OUT_DIR  = os.path.join(EXP_DIR, "figures")

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


def smooth(values: list, w: int = 20) -> np.ndarray:
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    padded = np.pad(arr, (w // 2, w - w // 2 - 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def quick_summary(records: list[dict]) -> dict:
    if not records:
        return {}
    rewards = [r.get("reward") or 0.0 for r in records]
    peak_reward = max(rewards)
    peak_step   = rewards.index(peak_reward) + 1
    first, last = records[0], records[-1]
    return {
        "steps": len(records),
        "step1": {
            "reward":       first.get("reward"),
            "format_exact": first.get("rewards/reward_format_exact/mean"),
            "kl":           first.get("kl"),
        },
        "step500": {
            "reward":       records[499].get("reward") if len(records) >= 500 else None,
            "format_exact": records[499].get("rewards/reward_format_exact/mean") if len(records) >= 500 else None,
            "kl":           records[499].get("kl") if len(records) >= 500 else None,
        },
        "step1000": {
            "reward":       last.get("reward"),
            "format_exact": last.get("rewards/reward_format_exact/mean"),
            "kl":           last.get("kl"),
        },
        "peak_reward": peak_reward,
        "peak_step":   peak_step,
    }


def plot_dashboard(records: list[dict], out_path: str, title: str):
    steps = [r["step"] for r in records]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    panels = [
        (gs[0, :2], "reward",                                     "Total Reward"),
        (gs[0, 2:], "kl",                                         "KL Divergence"),
        (gs[1, 0],  "rewards/reward_format_exact/mean",           "Format Exact"),
        (gs[1, 1],  "rewards/reward_format_approximate/mean",     "Format Approx"),
        (gs[1, 2],  "rewards/reward_answer_exact/mean",           "Answer Exact"),
        (gs[1, 3],  "rewards/reward_answer_numeric/mean",         "Answer Numeric"),
        (gs[2, 0],  "loss",                                       "Loss"),
        (gs[2, 1],  "grad_norm",                                  "Grad Norm"),
        (gs[2, 2],  "completion_length",                          "Completion Length"),
        (gs[2, 3],  "learning_rate",                              "Learning Rate"),
    ]

    color = "tab:blue"
    for gs_loc, key, label in panels:
        ax = fig.add_subplot(gs_loc)
        vals = np.array([r.get(key, np.nan) for r in records], dtype=float)
        ax.plot(steps, vals, color=color, alpha=0.15, linewidth=0.8)
        ax.plot(steps, smooth(vals), color=color, linewidth=2, label="exp_017")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Step", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_reward_detail(records: list[dict], out_path: str):
    steps = [r["step"] for r in records]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("exp_017 — Reward Functions Detail", fontsize=13, fontweight="bold")

    keys = [
        ("rewards/reward_format_exact/mean",       "Format Exact (+3.0 max)"),
        ("rewards/reward_format_approximate/mean", "Format Approx (+2.0 max)"),
        ("rewards/reward_answer_exact/mean",       "Answer Exact (+3.0 max)"),
        ("rewards/reward_answer_numeric/mean",     "Answer Numeric (+1.5 max)"),
    ]
    for ax, (key, label) in zip(axes.flat, keys):
        vals = np.array([r.get(key, np.nan) for r in records], dtype=float)
        ax.plot(steps, vals, alpha=0.15, linewidth=0.8, color="tab:blue")
        ax.plot(steps, smooth(vals), linewidth=2, color="tab:blue")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main(log_path: str = LOG_PATH, out_dir: str = OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    records = parse_log(log_path)
    if not records:
        print("No records found — log may be empty or not started yet.")
        return

    print(f"Parsed {len(records)} steps from {log_path}")

    summary = quick_summary(records)
    print(f"\nSummary:")
    print(f"  Step 1:    reward={summary['step1'].get('reward', '?')},  kl={summary['step1'].get('kl', '?')}")
    if summary.get("step500", {}).get("reward") is not None:
        print(f"  Step 500:  reward={summary['step500']['reward']:.3f},  kl={summary['step500']['kl']:.4f}")
    print(f"  Peak:      reward={summary['peak_reward']:.3f} @ step {summary['peak_step']}")
    print(f"  Latest:    reward={summary['step1000'].get('reward', '?')},  kl={summary['step1000'].get('kl', '?')}")

    # Save summary JSON
    summary_path = os.path.join(out_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    title = (
        "exp_017 — GRPO Llama-3.2-3B · Big-Math (integer) · "
        "16 gens · batch=4 · bf16 · 1000 steps"
    )
    plot_dashboard(records, os.path.join(out_dir, "dashboard.png"), title)
    plot_reward_detail(records, os.path.join(out_dir, "reward_detail.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=LOG_PATH)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()
    main(args.log, args.out)
