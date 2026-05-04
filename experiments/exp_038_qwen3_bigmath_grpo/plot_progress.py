"""
plot_progress.py — exp_038 training dashboard.
Reads train.log, plots reward / KL / format / answer-exact over steps.

Usage:  python plot_progress.py
Saves:  figures/progress_dashboard.png, figures/progress_reward.png
"""
import re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR  = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(EXP_DIR, "train.log")
OUT_DIR  = os.path.join(EXP_DIR, "figures")
EXP_TAG  = "exp_038 — GRPO baseline · Qwen3-4B · Big-Math int-2000"

PATTERN = re.compile(r"\{'loss':.*?'epoch': \d+\.\d+\}")


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
    a = np.array([x if x is not None else np.nan for x in v], dtype=float)
    if len(a) < w:
        return a
    k = np.ones(w) / w
    p = np.pad(a, (w // 2, w - w // 2 - 1), mode="edge")
    return np.convolve(p, k, mode="valid")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    r = parse_log(LOG_PATH)
    if not r:
        print(f"No records in {LOG_PATH}"); return
    steps = [d["step"] for d in r]
    print(f"{len(r)} steps parsed")

    # ── 4-panel dashboard ─────────────────────────────────────────────
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"{EXP_TAG} — {len(r)}/1000 steps", fontsize=11, fontweight="bold")

    panels = [
        (axs[0, 0], "reward",                            "Total Reward (max 9.5)"),
        (axs[0, 1], "kl",                                "KL Divergence"),
        (axs[1, 0], "rewards/reward_format_exact/mean",  "Format Exact (max 3.0)"),
        (axs[1, 1], "rewards/reward_answer_exact/mean",  "Answer Exact (max 3.0)"),
    ]
    for ax, key, label in panels:
        vals = np.array([d.get(key, np.nan) for d in r], dtype=float)
        ax.plot(steps, vals, color="tab:blue", alpha=0.2, linewidth=0.6)
        ax.plot(steps, smooth(vals), color="tab:blue", linewidth=2.0)
        if key == "reward":
            ax.axhline(9.5, color="red", linestyle=":", alpha=0.5, label="ceiling 9.5")
            ax.set_ylim(-3, 10.5)
        if key == "kl":
            ax.set_yscale("symlog", linthresh=0.001)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("step"); ax.grid(True, alpha=0.3)
        if key == "reward":
            ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "progress_dashboard.png")
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")

    # ── Reward-only (large) ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4.5))
    rewards = np.array([d.get("reward", np.nan) for d in r], dtype=float)
    ax.plot(steps, rewards, color="tab:blue", alpha=0.25, linewidth=0.8)
    ax.plot(steps, smooth(rewards), color="tab:blue", linewidth=2.4, label="reward (smoothed w=20)")
    ax.axhline(9.5, color="red", linestyle=":", alpha=0.5, label="ceiling 9.5")
    ax.set_title(EXP_TAG, fontweight="bold")
    ax.set_xlabel("step"); ax.set_ylabel("reward"); ax.grid(True, alpha=0.3)
    ax.set_ylim(-3, 10.5)
    pk_i = int(np.nanargmax(rewards)); pk = rewards[pk_i]
    ax.plot([pk_i + 1], [pk], "ro", markersize=8)
    ax.annotate(f"peak {pk:+.2f}@step{pk_i+1}",
                xy=(pk_i+1, pk), xytext=(10, -15), textcoords="offset points",
                fontsize=9, color="red")
    ax.legend(fontsize=9); plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "progress_reward.png")
    plt.savefig(out2, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Saved: {out2}")

    last = r[-1]
    print(f"\nstep {len(r)}: reward={last['reward']:+.3f}  "
          f"fmt_ex={last.get('rewards/reward_format_exact/mean', 0):.2f}  "
          f"ans_ex={last.get('rewards/reward_answer_exact/mean', 0):+.3f}  "
          f"kl={last.get('kl', 0):.4f}")


if __name__ == "__main__":
    main()
