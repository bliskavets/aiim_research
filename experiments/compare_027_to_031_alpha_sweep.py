"""
compare_027_to_031_alpha_sweep.py
---------------------------------
Overlay exp_027 GRPO baseline with the four alpha-sweep runs of the
flipped pure-proof GTPO-EMA on Big-Math int-2000.

exp_027 α=1.0 / 0.0  GRPO baseline (no shaping)
exp_028 α=0.9 / 0.1  weak bonus
exp_029 α=0.7 / 0.3  stronger bonus
exp_030 α=0.5 / 0.5  equal weight
exp_031 α=0.3 / 0.7  bonus dominates

Saves:
  experiments/figures_comparison/bigmath_alpha_sweep_overlay.png
  experiments/figures_comparison/bigmath_alpha_sweep_reward_kl.png
  experiments/figures_comparison/bigmath_alpha_sweep_summary.json
"""
import re, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

EXP_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(EXP_ROOT, "figures_comparison")

PATTERN = re.compile(r"\{'loss':.*?'epoch': \d+\.\d+\}")

# (label, color, path)
RUNS = [
    ("exp_027 α=1.0/0.0 baseline", "tab:gray",
     "exp_027_bigmath_int2k_grpo_baseline/train.log"),
    ("exp_028 α=0.9/0.1",          "tab:green",
     "exp_028_bigmath_int2k_flipped_gtpo_ema/train.log"),
    ("exp_029 α=0.7/0.3",          "tab:blue",
     "exp_029_bigmath_flipped_a07_a03/train.log"),
    ("exp_030 α=0.5/0.5",          "tab:orange",
     "exp_030_bigmath_flipped_a05_a05/train.log"),
    ("exp_031 α=0.3/0.7",          "tab:red",
     "exp_031_bigmath_flipped_a03_a07/train.log"),
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


def smooth(v, w=15):
    a = np.array([x if x is not None else np.nan for x in v], dtype=float)
    if len(a) < w:
        return a
    k = np.ones(w) / w
    p = np.pad(a, (w // 2, w - w // 2 - 1), mode="edge")
    return np.convolve(p, k, mode="valid")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    series = []
    for label, color, subpath in RUNS:
        path = os.path.join(EXP_ROOT, subpath)
        recs = parse_log(path)
        print(f"  {label}: {len(recs)} steps")
        series.append((label, color, recs))

    # ── 4-panel dashboard ────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "Big-Math integer-2000 · Llama-3.2-3B · 500 steps · bs=4 × 8 gens — "
        "alpha sweep of flipped pure-proof GTPO-EMA",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.22)
    panels = [
        (gs[0, 0], "reward",                           "Total Reward (max 9.5)", False),
        (gs[0, 1], "kl",                               "KL Divergence (symlog)", True),
        (gs[1, 0], "rewards/reward_format_exact/mean", "Format Exact (max 3.0)", False),
        (gs[1, 1], "rewards/reward_answer_exact/mean", "Answer Exact (max 3.0)", False),
    ]
    for gs_loc, key, label, is_kl in panels:
        ax = fig.add_subplot(gs_loc)
        for name, color, recs in series:
            if not recs:
                continue
            steps = [r["step"] for r in recs]
            vals  = np.array([r.get(key, np.nan) for r in recs], dtype=float)
            ax.plot(steps, vals, color=color, alpha=0.10, linewidth=0.5)
            ax.plot(steps, smooth(vals), color=color, linewidth=2.2, label=name)
        if key == "reward":
            ax.axhline(9.5, color="red", linestyle=":", alpha=0.4, label="ceiling 9.5")
            ax.set_ylim(-3, 10.5)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("step"); ax.grid(True, alpha=0.3)
        if is_kl:
            ax.set_yscale("symlog", linthresh=0.01)
        ax.legend(fontsize=8, loc="best")

    out1 = os.path.join(OUT_DIR, "bigmath_alpha_sweep_overlay.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out1}")

    # ── Reward + KL focus ────────────────────────────────────────────
    fig, (ax_r, ax_kl) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("alpha sweep on Big-Math int-2000 — reward and KL",
                 fontsize=12, fontweight="bold")
    for name, color, recs in series:
        if not recs:
            continue
        steps = [r["step"] for r in recs]
        rw = np.array([r.get("reward", np.nan) for r in recs], dtype=float)
        kl = np.array([r.get("kl", np.nan) for r in recs], dtype=float)
        ax_r.plot(steps,  rw, color=color, alpha=0.10, linewidth=0.5)
        ax_r.plot(steps,  smooth(rw), color=color, linewidth=2.2, label=name)
        ax_kl.plot(steps, kl, color=color, alpha=0.10, linewidth=0.5)
        ax_kl.plot(steps, smooth(kl), color=color, linewidth=2.2, label=name)
    ax_r.axhline(9.5, color="red", linestyle=":", alpha=0.4)
    ax_r.set_title("Total Reward", fontweight="bold")
    ax_r.set_xlabel("step"); ax_r.grid(True, alpha=0.3); ax_r.legend(fontsize=8)
    ax_r.set_ylim(-3, 10.5)
    ax_kl.set_title("KL Divergence (symlog)", fontweight="bold")
    ax_kl.set_xlabel("step"); ax_kl.grid(True, alpha=0.3); ax_kl.legend(fontsize=8)
    ax_kl.set_yscale("symlog", linthresh=0.01)
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "bigmath_alpha_sweep_reward_kl.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")

    # ── Summary JSON ─────────────────────────────────────────────────
    import statistics as st
    summary = {}
    for name, _, recs in series:
        if not recs:
            summary[name] = {"status": "no data"}; continue
        pk = max(recs, key=lambda r: r["reward"])
        pk_i = recs.index(pk) + 1
        l50 = recs[-50:]
        summary[name] = {
            "steps":          len(recs),
            "peak":           pk["reward"],
            "peak_step":      pk_i,
            "reward_last":    recs[-1]["reward"],
            "reward_l50_avg": st.mean([x["reward"] for x in l50]),
            "fmt_l50_avg":    st.mean([x["rewards/reward_format_exact/mean"] for x in l50]),
            "ans_l50_avg":    st.mean([x["rewards/reward_answer_exact/mean"] for x in l50]),
            "kl_l50_avg":     st.mean([x["kl"] for x in l50]),
        }
    out3 = os.path.join(OUT_DIR, "bigmath_alpha_sweep_summary.json")
    with open(out3, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out3}")


if __name__ == "__main__":
    main()
