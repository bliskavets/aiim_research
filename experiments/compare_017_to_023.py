"""
compare_017_to_023.py
---------------------
Overlay plot comparing exp_017 (GRPO baseline) with exp_018-023.

Usage: python compare_017_to_023.py [--out DIR]
Saves to: experiments/figures_comparison/
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

EXP_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(EXP_ROOT, "figures_comparison")

PATTERN = re.compile(r"\{'loss':.*?'epoch': \d+\.\d+\}")

EXPERIMENTS = [
    ("exp_017", "GRPO baseline",    "tab:blue",   "exp_017_bigmath_llama_int16gen"),
    ("exp_018", "GTPO-EMA",         "tab:orange", "exp_018_bigmath_llama_gtpo_ema"),
    ("exp_019", "GRPO-S entropy",   "tab:green",  "exp_019_bigmath_llama_grpos_entropy"),
    ("exp_020", "GTPO entropy",     "tab:red",    "exp_020_bigmath_llama_gtpo_entropy"),
    ("exp_021", "GTPO-Conf",        "tab:purple", "exp_021_bigmath_llama_gtpo_conf"),
    ("exp_022", "GTPO binary",      "tab:brown",  "exp_022_bigmath_llama_gtpo_binary"),
    ("exp_023", "GTPO-EMA binary",  "tab:pink",   "exp_023_bigmath_llama_gtpo_ema_binary"),
]


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


def main(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    experiments = []
    for short, label, color, subdir in EXPERIMENTS:
        log = os.path.join(EXP_ROOT, subdir, "train.log")
        records = parse_log(log)
        status = f"({len(records)} steps)" if records else "(no log)"
        print(f"  {short}: {label} — {status}")
        experiments.append((short, label, color, records))

    # ── Main overlay dashboard ────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "exp_017 → exp_023 · GRPO variants on Big-Math (integer) · Llama-3.2-3B · 16 gens · bs=4 · bf16",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)
    panels = [
        (gs[0, :2], "reward",                                 "Total Reward"),
        (gs[0, 2:], "kl",                                     "KL Divergence"),
        (gs[1, 0],  "rewards/reward_format_exact/mean",       "Format Exact (+3.0 max)"),
        (gs[1, 1],  "rewards/reward_format_approximate/mean", "Format Approx (+2.0 max)"),
        (gs[1, 2],  "rewards/reward_answer_exact/mean",       "Answer Exact (+3.0 max)"),
        (gs[1, 3],  "rewards/reward_answer_numeric/mean",     "Answer Numeric (+1.5 max)"),
    ]
    for gs_loc, key, label in panels:
        ax = fig.add_subplot(gs_loc)
        for short, exp_label, color, records in experiments:
            if not records:
                continue
            steps = [r["step"] for r in records]
            vals = np.array([r.get(key, np.nan) for r in records], dtype=float)
            ax.plot(steps, vals, color=color, alpha=0.08, linewidth=0.6)
            ax.plot(steps, smooth(vals), color=color, linewidth=2, label=f"{short} {exp_label}")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Step", fontsize=8)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    out_path = os.path.join(out_dir, "all_experiments_overlay.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # ── Reward-only comparison with log-KL twin axis ──────────────────────────
    fig, (ax_r, ax_kl) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("exp_017 → exp_023 · Reward & KL", fontsize=12, fontweight="bold")
    for short, exp_label, color, records in experiments:
        if not records:
            continue
        steps = [r["step"] for r in records]
        rewards = np.array([r.get("reward", np.nan) for r in records], dtype=float)
        kls     = np.array([r.get("kl", np.nan) for r in records], dtype=float)
        ax_r.plot(steps,  rewards, color=color, alpha=0.08, linewidth=0.6)
        ax_r.plot(steps,  smooth(rewards), color=color, linewidth=2,
                  label=f"{short} {exp_label}")
        ax_kl.plot(steps, kls, color=color, alpha=0.08, linewidth=0.6)
        ax_kl.plot(steps, smooth(kls), color=color, linewidth=2,
                   label=f"{short} {exp_label}")
    ax_r.set_title("Total Reward", fontweight="bold")
    ax_r.set_xlabel("Step"); ax_r.grid(True, alpha=0.3); ax_r.legend(fontsize=8)
    ax_kl.set_title("KL Divergence (log)", fontweight="bold")
    ax_kl.set_xlabel("Step"); ax_kl.grid(True, alpha=0.3); ax_kl.legend(fontsize=8)
    ax_kl.set_yscale("symlog", linthresh=0.01)
    plt.tight_layout()
    out_path2 = os.path.join(out_dir, "reward_kl_overlay.png")
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path2}")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {}
    for short, exp_label, color, records in experiments:
        if not records:
            summary[short] = {"status": "no log"}
            continue
        rewards = [r.get("reward") or 0.0 for r in records]
        peak = max(rewards)
        summary[short] = {
            "label":         exp_label,
            "steps":         len(records),
            "step1_reward":  records[0].get("reward"),
            "latest_reward": records[-1].get("reward"),
            "peak_reward":   peak,
            "peak_step":     rewards.index(peak) + 1,
            "latest_kl":     records[-1].get("kl"),
        }
    out_json = os.path.join(out_dir, "comparison_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_json}")

    print("\n=== SUMMARY ===")
    for short, s in summary.items():
        if "status" in s:
            print(f"  {short}: {s['status']}")
            continue
        print(f"  {short} ({s['label']}): peak={s['peak_reward']:.2f} @ step {s['peak_step']}, "
              f"latest={s['latest_reward']:.2f}, kl={s['latest_kl']:.3f}, steps={s['steps']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.out)
