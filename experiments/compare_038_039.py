"""
compare_038_039.py — comparison plots: exp_038 (GRPO baseline) vs exp_039 (GTPO-EMA-flipped).
Both: Qwen3-4B, Big-Math int-2000, bs=4, gens=8, 1000 steps, max_seq=4096.

Usage:  python experiments/compare_038_039.py
Saves:  experiments/figures_comparison/compare_038_039_reward.png
        experiments/figures_comparison/compare_038_039_kl.png
        experiments/figures_comparison/compare_038_039_dashboard.png
"""
import re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "figures_comparison")
PATTERN = re.compile(r"\{'loss':.*?'epoch': \d+\.\d+\}")

EXPS = {
    "exp_038": {
        "log":   os.path.join(BASE, "exp_038_qwen3_bigmath_grpo", "train.log"),
        "label": "exp_038 GRPO baseline",
        "color": "tab:blue",
    },
    "exp_039": {
        "log":   os.path.join(BASE, "exp_039_qwen3_bigmath_pure_proof_gtpo_ema", "train.log"),
        "label": "exp_039 GTPO-EMA-flipped",
        "color": "tab:orange",
    },
}

TITLE = "Qwen3-4B · Big-Math int-2000 · bs=4 · gens=8 · 1000 steps"


def parse_log(path):
    records = []
    with open(path) as f:
        text = f.read()
    for i, m in enumerate(PATTERN.finditer(text)):
        try:
            d = eval(m.group())
            d["step"] = i + 1
            records.append(d)
        except Exception:
            pass
    return records


def smooth(v, w=20):
    a = np.array(v, dtype=float)
    if len(a) < w:
        return a
    k = np.ones(w) / w
    p = np.pad(a, (w // 2, w - w // 2 - 1), mode="edge")
    return np.convolve(p, k, mode="valid")


def get_series(records, key):
    return np.array([d.get(key, np.nan) for d in records], dtype=float)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data = {}
    for name, cfg in EXPS.items():
        r = parse_log(cfg["log"])
        print(f"{cfg['label']}: {len(r)} steps")
        data[name] = {"records": r, "steps": [d["step"] for d in r], **cfg}

    # ── 1. Reward comparison ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, d in data.items():
        r = get_series(d["records"], "reward")
        ax.plot(d["steps"], r, color=d["color"], alpha=0.18, linewidth=0.7)
        ax.plot(d["steps"], smooth(r), color=d["color"], linewidth=2.5, label=d["label"])
    ax.axhline(9.5, color="red", linestyle=":", alpha=0.4, label="ceiling 9.5")

    # stats annotation
    for i, (name, d) in enumerate(data.items()):
        r = get_series(d["records"], "reward")
        l50 = float(np.nanmean(r[-50:])) if len(r) >= 50 else float(np.nanmean(r))
        l10 = float(np.nanmean(r[-10:])) if len(r) >= 10 else float(np.nanmean(r))
        peak = float(np.nanmax(r)); peak_s = int(np.nanargmax(r)) + 1
        ypos = 0.20 - i * 0.13
        ax.text(0.01, ypos,
                f"{d['label']}:  peak={peak:.2f}@s{peak_s}  L50={l50:.3f}  L10={l10:.3f}",
                transform=ax.transAxes, fontsize=9, color=d["color"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    ax.set_title(f"Reward — {TITLE}", fontweight="bold")
    ax.set_xlabel("step"); ax.set_ylabel("reward")
    ax.set_ylim(-3, 10.5); ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "compare_038_039_reward.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")

    # ── 2. KL comparison ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, d in data.items():
        kl = get_series(d["records"], "kl")
        ax.plot(d["steps"], kl, color=d["color"], alpha=0.18, linewidth=0.7)
        ax.plot(d["steps"], smooth(kl), color=d["color"], linewidth=2.5, label=d["label"])
    ax.set_yscale("symlog", linthresh=0.001)

    for i, (name, d) in enumerate(data.items()):
        kl = get_series(d["records"], "kl")
        kl50 = float(np.nanmean(kl[-50:])) if len(kl) >= 50 else float(np.nanmean(kl))
        ypos = 0.95 - i * 0.10
        ax.text(0.01, ypos, f"{d['label']}:  KL_L50={kl50:.5f}",
                transform=ax.transAxes, fontsize=9, color=d["color"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    ax.set_title(f"KL Divergence (symlog) — {TITLE}", fontweight="bold")
    ax.set_xlabel("step"); ax.set_ylabel("KL")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "compare_038_039_kl.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")

    # ── 3. 4-panel dashboard ─────────────────────────────────────────
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f"exp_038 vs exp_039 — {TITLE}", fontsize=11, fontweight="bold")

    panels = [
        (axs[0, 0], "reward",                            "Total Reward",       True,  False),
        (axs[0, 1], "kl",                                "KL Divergence",      False, True),
        (axs[1, 0], "rewards/reward_format_exact/mean",  "Format Exact",       True,  False),
        (axs[1, 1], "rewards/reward_answer_exact/mean",  "Answer Exact",       True,  False),
    ]
    for ax, key, label, show_ceil, is_kl in panels:
        for name, d in data.items():
            vals = get_series(d["records"], key)
            ax.plot(d["steps"], vals, color=d["color"], alpha=0.15, linewidth=0.6)
            ax.plot(d["steps"], smooth(vals), color=d["color"], linewidth=2.0, label=d["label"])
        if show_ceil and key == "reward":
            ax.axhline(9.5, color="red", linestyle=":", alpha=0.4)
            ax.set_ylim(-3, 10.5)
        if is_kl:
            ax.set_yscale("symlog", linthresh=0.001)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("step"); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "compare_038_039_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n=== Summary ===")
    for name, d in data.items():
        r   = get_series(d["records"], "reward")
        kl  = get_series(d["records"], "kl")
        fmt = get_series(d["records"], "rewards/reward_format_exact/mean")
        ans = get_series(d["records"], "rewards/reward_answer_exact/mean")
        n = len(r)
        print(f"\n{d['label']} ({n} steps):")
        print(f"  reward:  peak={np.nanmax(r):.3f}@s{int(np.nanargmax(r))+1}  "
              f"L50={np.nanmean(r[-50:]):.3f}  L10={np.nanmean(r[-10:]):.3f}")
        print(f"  KL:      L50={np.nanmean(kl[-50:]):.5f}  last={kl[-1]:.5f}")
        print(f"  fmt_ex:  last={fmt[-1]:.2f}")
        print(f"  ans_ex:  last={ans[-1]:.3f}")


if __name__ == "__main__":
    main()
