"""
compare_conf_methods_gsm8k.py
-----------------------------
For each confidence/proof variant on GSM8K (Llama-3.2-3B, 500 steps), plot
a 2-panel "method vs GRPO baseline (exp_001)" comparison.

Variants covered:
  - exp_005 GTPO-Conf   (original)
  - exp_024 GTPO-Conf   (repro of exp_005 with byte-identical code)
  - exp_005 GRPO-S-Conf (original)
  - exp_024 GRPO-S-Conf (repro)
  - exp_025 pure-proof GTPO-EMA (new)

Plus a 6-method overlay is saved as well.

Usage: python compare_conf_methods_gsm8k.py [--out DIR]
Saves to: experiments/figures_comparison/
"""
import re, os, argparse, json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

EXP_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(EXP_ROOT, "figures_comparison")

PATTERN = re.compile(r"\{'loss':.*?'epoch': \d+\.\d+\}")

BASELINE = ("exp_001", "GRPO baseline (exp_001)", "tab:gray",
            "exp_001_grpo_llama32_gsm8k/train.log")

# Each entry: (short, label, color, output-png-filename, log-path)
VARIANTS = [
    ("exp_005-GTPO",  "GTPO-Conf original (exp_005)",
     "tab:blue",   "gsm8k_exp005_gtpo_conf_vs_grpo.png",
     "exp_005_confidence_gtpo_grpos/train_gtpo_conf.log"),
    ("exp_024-GTPO",  "GTPO-Conf repro (exp_024)",
     "tab:blue",   "gsm8k_exp024_gtpo_conf_vs_grpo.png",
     "exp_024_repro_exp005_confidence/train_gtpo_conf.log"),
    ("exp_005-GRPOS", "GRPO-S-Conf original (exp_005)",
     "tab:orange", "gsm8k_exp005_grpos_conf_vs_grpo.png",
     "exp_005_confidence_gtpo_grpos/train_grpo_s_conf.log"),
    ("exp_024-GRPOS", "GRPO-S-Conf repro (exp_024)",
     "tab:orange", "gsm8k_exp024_grpos_conf_vs_grpo.png",
     "exp_024_repro_exp005_confidence/train_grpo_s_conf.log"),
    ("exp_025-PROOF", "Pure-proof GTPO-EMA (exp_025)",
     "tab:red",    "gsm8k_exp025_pure_proof_vs_grpo.png",
     "exp_025_pure_proof_gtpo_ema/train_gtpo_ema_proof.log"),
    ("exp_026-FLIP",  "Flipped GTPO-EMA (exp_026)",
     "tab:green",  "gsm8k_exp026_flipped_vs_grpo.png",
     "exp_026_flipped_conf_gtpo_ema/train_gtpo_ema_flipped.log"),
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


def smooth(values, w=15):
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    padded = np.pad(arr, (w // 2, w - w // 2 - 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_pair(out_path, baseline_records, baseline_label, baseline_color,
              variant_records, variant_label, variant_color, suptitle):
    """2-panel (reward, KL) plot: baseline vs one variant."""
    fig, (ax_r, ax_kl) = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    for records, label, color in [
        (baseline_records, baseline_label, baseline_color),
        (variant_records,  variant_label,  variant_color),
    ]:
        if not records:
            continue
        steps = [r["step"] for r in records]
        rewards = np.array([r.get("reward", np.nan) for r in records], dtype=float)
        kls     = np.array([r.get("kl", np.nan) for r in records], dtype=float)
        ax_r.plot(steps,  rewards, color=color, alpha=0.12, linewidth=0.7)
        ax_r.plot(steps,  smooth(rewards), color=color, linewidth=2.2, label=label)
        ax_kl.plot(steps, kls, color=color, alpha=0.12, linewidth=0.7)
        ax_kl.plot(steps, smooth(kls), color=color, linewidth=2.2, label=label)

    ax_r.axhline(9.5, color="red", linestyle=":", alpha=0.45, linewidth=1.0,
                 label="max reward 9.5")
    ax_r.set_title("Total Reward", fontweight="bold")
    ax_r.set_xlabel("Step"); ax_r.grid(True, alpha=0.3); ax_r.legend(fontsize=8)
    ax_r.set_ylim(-2.5, 10.5)

    ax_kl.set_title("KL Divergence (symlog)", fontweight="bold")
    ax_kl.set_xlabel("Step"); ax_kl.grid(True, alpha=0.3); ax_kl.legend(fontsize=8)
    ax_kl.set_yscale("symlog", linthresh=0.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Parse baseline once
    base_short, base_label, base_color, base_subpath = BASELINE
    base_records = parse_log(os.path.join(EXP_ROOT, base_subpath))
    print(f"  {base_short}: {base_label} — {len(base_records)} steps")

    # ── Per-variant 2-panel plots ─────────────────────────────────────────
    summary = {}
    for short, label, color, fname, subpath in VARIANTS:
        records = parse_log(os.path.join(EXP_ROOT, subpath))
        print(f"  {short}: {label} — {len(records)} steps")
        out_path = os.path.join(out_dir, fname)
        plot_pair(
            out_path=out_path,
            baseline_records=base_records, baseline_label=base_label, baseline_color=base_color,
            variant_records=records, variant_label=label, variant_color=color,
            suptitle=f"GSM8K · {label}  vs  GRPO baseline",
        )
        if records:
            rewards = [r.get("reward") or 0.0 for r in records]
            peak = max(rewards)
            summary[short] = {
                "label": label,
                "steps": len(records),
                "peak_reward": peak,
                "peak_step": rewards.index(peak) + 1,
                "latest_reward": records[-1].get("reward"),
                "latest_fmt_exact": records[-1].get("rewards/reward_format_exact/mean"),
                "latest_ans_exact": records[-1].get("rewards/reward_answer_exact/mean"),
                "latest_kl": records[-1].get("kl"),
            }

    # Baseline summary entry
    if base_records:
        rewards = [r.get("reward") or 0.0 for r in base_records]
        peak = max(rewards)
        summary[base_short] = {
            "label": base_label,
            "steps": len(base_records),
            "peak_reward": peak,
            "peak_step": rewards.index(peak) + 1,
            "latest_reward": base_records[-1].get("reward"),
            "latest_fmt_exact": base_records[-1].get("rewards/reward_format_exact/mean"),
            "latest_ans_exact": base_records[-1].get("rewards/reward_answer_exact/mean"),
            "latest_kl": base_records[-1].get("kl"),
        }

    out_json = os.path.join(out_dir, "gsm8k_conf_methods_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_json}")

    # ── Combined 6-method overlay for context ────────────────────────────
    fig, (ax_r, ax_kl) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "GSM8K: GRPO baseline vs confidence-based variants (all on one plot)",
        fontsize=12, fontweight="bold",
    )
    all_series = [(base_short, base_label, base_color, "-", base_records)] + [
        (s, l, c, ls, parse_log(os.path.join(EXP_ROOT, p)))
        for (s, l, c, _, p), ls in zip(VARIANTS, ["-", "--", "-", "--", "-", "-"])
    ]
    for short, label, color, ls, records in all_series:
        if not records:
            continue
        steps = [r["step"] for r in records]
        rewards = np.array([r.get("reward", np.nan) for r in records], dtype=float)
        kls     = np.array([r.get("kl", np.nan) for r in records], dtype=float)
        ax_r.plot(steps,  rewards, color=color, linestyle=ls, alpha=0.08, linewidth=0.5)
        ax_r.plot(steps,  smooth(rewards), color=color, linestyle=ls,
                  linewidth=2, label=label)
        ax_kl.plot(steps, kls, color=color, linestyle=ls, alpha=0.08, linewidth=0.5)
        ax_kl.plot(steps, smooth(kls), color=color, linestyle=ls,
                   linewidth=2, label=label)
    ax_r.axhline(9.5, color="red", linestyle=":", alpha=0.45, linewidth=1.0)
    ax_r.set_title("Total Reward", fontweight="bold")
    ax_r.set_xlabel("Step"); ax_r.grid(True, alpha=0.3); ax_r.legend(fontsize=7)
    ax_r.set_ylim(-2.5, 10.5)
    ax_kl.set_title("KL Divergence (symlog)", fontweight="bold")
    ax_kl.set_xlabel("Step"); ax_kl.grid(True, alpha=0.3); ax_kl.legend(fontsize=7)
    ax_kl.set_yscale("symlog", linthresh=0.01)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "gsm8k_conf_methods_all_overlay.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    print("\n=== SUMMARY ===")
    order = ["exp_001", "exp_005-GTPO", "exp_024-GTPO",
             "exp_005-GRPOS", "exp_024-GRPOS", "exp_025-PROOF", "exp_026-FLIP"]
    for short in order:
        if short not in summary: continue
        s = summary[short]
        print(f"  {short:14s} ({s['label']:38s}): "
              f"peak={s['peak_reward']:+5.2f} @ {s['peak_step']:3d}, "
              f"r@final={s['latest_reward']:+5.2f}, kl={s['latest_kl']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.out)
