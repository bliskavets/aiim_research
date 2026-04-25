"""
compare_034_035.py
------------------
Overlay: Qwen3-4B pure-proof GTPO-EMA (exp_034) vs flipped GTPO-EMA (exp_035)
on GSM8K, 500 steps, bs=1 × 4 gens.

  exp_034  pure-proof (not flipped) — stopped at step 270
  exp_035  flipped pure-proof       — 500 steps complete

Saves:
  experiments/figures_comparison/gsm8k_qwen3_034_vs_035.png
"""
import re, os, json, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(EXP_ROOT, "figures_comparison")

PATTERN = re.compile(r"\{'loss':.*?'epoch': \d+\.\d+\}")

RUNS = [
    ("exp_034  pure-proof (not flipped)", "tab:orange",
     "exp_034_qwen3_pure_proof_gtpo_ema/train_gtpo_ema_proof.log"),
    ("exp_035  flipped pure-proof",       "tab:blue",
     "exp_035_qwen3_flipped_conf_gtpo_ema/train_gtpo_ema_flipped.log"),
]


def parse_log(path):
    records = []
    try:
        with open(path) as f:
            text = f.read()
    except FileNotFoundError:
        print(f"  WARNING: {path} not found")
        return []
    for i, m in enumerate(PATTERN.finditer(text)):
        try:
            d = eval(m.group())
            d["step"] = i + 1
            records.append(d)
        except Exception:
            pass
    return records


def smooth(v, w=20):
    a = np.array([x if x is not None else np.nan for x in v], dtype=float)
    if len(a) < w:
        return a
    k = np.ones(w) / w
    p = np.pad(a, (w // 2, w - w // 2 - 1), mode="edge")
    return np.convolve(p, k, mode="valid")


def last_n_avg(vals, n=50):
    v = [x for x in vals[-n:] if x is not None]
    return sum(v) / len(v) if v else float("nan")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    series = []
    for label, color, subpath in RUNS:
        recs = parse_log(os.path.join(EXP_ROOT, subpath))
        print(f"  {label}: {len(recs)} steps")
        series.append((label, color, recs))

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(
        "GSM8K · Qwen3-4B · LoRA r=64 · α₁=0.9 α₂=0.1 λ=0.9\n"
        "exp_034: pure-proof GTPO-EMA (stopped @270)  vs  exp_035: flipped pure-proof GTPO-EMA (500 steps)",
        fontsize=12, fontweight="bold",
    )

    metrics = [
        ("reward",                         "Reward",           axes[0, 0]),
        ("kl",                             "KL divergence",    axes[0, 1]),
        ("rewards/reward_answer_exact/mean","Answer reward",    axes[1, 0]),
        ("completion_length",              "Completion length", axes[1, 1]),
    ]

    summary = {}
    for label, color, recs in series:
        steps = [r["step"] for r in recs]
        peak_reward = max((r["reward"] for r in recs), default=float("nan"))
        peak_step   = next(r["step"] for r in recs if r["reward"] == peak_reward)
        rewards     = [r["reward"] for r in recs]
        kls         = [r.get("kl", None) for r in recs]
        l50_r       = last_n_avg(rewards, 50)
        l50_kl      = last_n_avg([k for k in kls if k is not None], 50)
        summary[label] = dict(
            steps=len(recs), peak_reward=peak_reward, peak_step=peak_step,
            last50_reward=round(l50_r, 3), last50_kl=round(l50_kl, 4),
        )

        for key, title, ax in metrics:
            vals = [r.get(key) for r in recs]
            raw  = np.array([v if v is not None else np.nan for v in vals], dtype=float)
            sm   = smooth(vals)
            ax.plot(steps, raw,  color=color, alpha=0.18, linewidth=0.8)
            ax.plot(steps, sm,   color=color, alpha=0.90, linewidth=2.0, label=label)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.3)

    # annotations
    ax_r = axes[0, 0]
    ax_r.axhline(9.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="ceiling 9.5")
    for label, color, recs in series:
        s = summary[label]
        ax_r.axvline(s["peak_step"], color=color, linestyle=":", linewidth=1.2, alpha=0.7)
        ax_r.text(s["peak_step"] + 2, 0.3, f"peak@{s['peak_step']}", color=color, fontsize=8)

    for _, _, ax in metrics:
        ax.legend(fontsize=8)

    # summary table
    print("\n--- Summary ---")
    print(f"{'Exp':<45} {'steps':>6} {'peak':>6} {'@step':>6} {'L50 r':>7} {'L50 KL':>8}")
    for label, s in summary.items():
        print(f"{label:<45} {s['steps']:>6} {s['peak_reward']:>6.2f} {s['peak_step']:>6} "
              f"{s['last50_reward']:>7.3f} {s['last50_kl']:>8.4f}")

    out_json = os.path.join(OUT_DIR, "gsm8k_qwen3_034_035_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    out_png = os.path.join(OUT_DIR, "gsm8k_qwen3_034_vs_035.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
