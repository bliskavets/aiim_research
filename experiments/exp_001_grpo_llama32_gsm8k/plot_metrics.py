"""
Plot all training metrics from train.log for exp_001.
Saves individual + summary figures to figures/ subdirectory.
"""

import re
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

LOG_PATH = os.path.join(os.path.dirname(__file__), "train.log")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Parse log ────────────────────────────────────────────────────────────────

def parse_log(path):
    records = []
    pattern = re.compile(
        r"\{'loss':.*?'epoch': \d+\.\d+\}(?=\n)"
    )
    with open(path) as f:
        text = f.read()
    for i, m in enumerate(pattern.finditer(text)):
        try:
            d = eval(m.group())
            d["step"] = i + 1
            records.append(d)
        except Exception:
            pass
    return records

records = parse_log(LOG_PATH)
print(f"Parsed {len(records)} steps from log")

steps = [r["step"] for r in records]

# ── Helper ───────────────────────────────────────────────────────────────────

def get(records, key):
    return [r.get(key) for r in records]

def smooth(values, w=10):
    """Simple moving average."""
    values = np.array([v if v is not None else np.nan for v in values], dtype=float)
    if w <= 1:
        return values
    kernel = np.ones(w) / w
    padded = np.pad(values, (w // 2, w - w // 2 - 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid")

STYLE = dict(alpha=0.25, linewidth=1)
SMOOTH_STYLE = dict(linewidth=2)

def plot_metric(ax, steps, values, label, color="steelblue", smooth_w=10, ylabel=None):
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    ax.plot(steps, arr, color=color, label=f"{label} (raw)", **STYLE)
    ax.plot(steps, smooth(arr, smooth_w), color=color, label=f"{label} (smooth)", **SMOOTH_STYLE)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel or label)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# ── Figure 1: Rewards ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle("Exp 001 — Rewards", fontsize=14, fontweight="bold")

metrics_rewards = [
    ("reward",                                "Total Reward",          "tab:blue"),
    ("reward_std",                            "Reward Std",            "tab:orange"),
    ("rewards/reward_format_exact/mean",      "Format Exact (mean)",   "tab:green"),
    ("rewards/reward_format_approximate/mean","Format Approx (mean)",  "tab:red"),
    ("rewards/reward_answer_exact/mean",      "Answer Exact (mean)",   "tab:purple"),
    ("rewards/reward_answer_numeric/mean",    "Answer Numeric (mean)", "tab:brown"),
]
for ax, (key, label, color) in zip(axes.flat, metrics_rewards):
    plot_metric(ax, steps, get(records, key), label, color=color)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "rewards.png"), dpi=150)
plt.close()
print("Saved rewards.png")

# ── Figure 2: Loss, Grad Norm, LR ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Exp 001 — Loss / Grad Norm / LR", fontsize=14, fontweight="bold")

plot_metric(axes[0], steps, get(records, "loss"),      "Loss",      color="tab:blue")
plot_metric(axes[1], steps, get(records, "grad_norm"), "Grad Norm", color="tab:orange")
plot_metric(axes[2], steps, get(records, "learning_rate"), "LR",    color="tab:green", smooth_w=1)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "loss_gradnorm_lr.png"), dpi=150)
plt.close()
print("Saved loss_gradnorm_lr.png")

# ── Figure 3: KL divergence ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4))
fig.suptitle("Exp 001 — KL Divergence from Base Model", fontsize=13, fontweight="bold")
plot_metric(ax, steps, get(records, "kl"), "KL", color="tab:red")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "kl_divergence.png"), dpi=150)
plt.close()
print("Saved kl_divergence.png")

# ── Figure 4: Completion lengths ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("Exp 001 — Completion Lengths", fontsize=13, fontweight="bold")

plot_metric(axes[0], steps, get(records, "completion_length"), "Mean Length", color="tab:blue")
plot_metric(axes[1], steps, get(records, "completions/clipped_ratio"), "Clipped Ratio", color="tab:orange")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "completion_lengths.png"), dpi=150)
plt.close()
print("Saved completion_lengths.png")

# ── Figure 5: Clip ratios ─────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Exp 001 — GRPO Clip Ratios", fontsize=13, fontweight="bold")

plot_metric(axes[0], steps, get(records, "clip_ratio/low_mean"),    "Low Mean",    color="tab:blue")
plot_metric(axes[1], steps, get(records, "clip_ratio/high_mean"),   "High Mean",   color="tab:orange")
plot_metric(axes[2], steps, get(records, "clip_ratio/region_mean"), "Region Mean", color="tab:green")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "clip_ratios.png"), dpi=150)
plt.close()
print("Saved clip_ratios.png")

# ── Figure 6: Reward std per function ────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Exp 001 — Reward Std per Function", fontsize=13, fontweight="bold")

std_metrics = [
    ("rewards/reward_format_exact/std",      "Format Exact Std",   "tab:green"),
    ("rewards/reward_format_approximate/std","Format Approx Std",  "tab:red"),
    ("rewards/reward_answer_exact/std",      "Answer Exact Std",   "tab:purple"),
    ("rewards/reward_answer_numeric/std",    "Answer Numeric Std", "tab:brown"),
]
for ax, (key, label, color) in zip(axes.flat, std_metrics):
    plot_metric(ax, steps, get(records, key), label, color=color)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reward_stds.png"), dpi=150)
plt.close()
print("Saved reward_stds.png")

# ── Figure 7: Big summary dashboard ──────────────────────────────────────────

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Exp 001 — Full Training Dashboard\nGRPO + LoRA · Llama-3.2-3B · GSM8K · 500 steps · A100 80GB",
             fontsize=14, fontweight="bold")

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

panels = [
    (gs[0, :2], "reward",                                "Total Reward",         "tab:blue"),
    (gs[0, 2:], "kl",                                    "KL Divergence",        "tab:red"),
    (gs[1, 0],  "rewards/reward_format_exact/mean",      "Format Exact",         "tab:green"),
    (gs[1, 1],  "rewards/reward_format_approximate/mean","Format Approx",        "tab:olive"),
    (gs[1, 2],  "rewards/reward_answer_exact/mean",      "Answer Exact",         "tab:purple"),
    (gs[1, 3],  "rewards/reward_answer_numeric/mean",    "Answer Numeric",       "tab:brown"),
    (gs[2, 0],  "loss",                                  "Loss",                 "tab:blue"),
    (gs[2, 1],  "grad_norm",                             "Grad Norm",            "tab:orange"),
    (gs[2, 2],  "completion_length",                     "Completion Length",    "tab:cyan"),
    (gs[2, 3],  "learning_rate",                         "Learning Rate",        "tab:gray"),
]

for gs_loc, key, label, color in panels:
    ax = fig.add_subplot(gs_loc)
    arr = np.array([r.get(key) if r.get(key) is not None else np.nan for r in records], dtype=float)
    ax.plot(steps, arr, color=color, alpha=0.2, linewidth=1)
    ax.plot(steps, smooth(arr, 15), color=color, linewidth=2)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlabel("Step", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

plt.savefig(os.path.join(OUT_DIR, "dashboard.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved dashboard.png")

# ── Save parsed metrics as full JSON ─────────────────────────────────────────
with open(os.path.join(os.path.dirname(__file__), "metrics_full.json"), "w") as f:
    json.dump(records, f, indent=2)
print("Saved metrics_full.json")

print(f"\nAll figures saved to: {OUT_DIR}")
print(f"Files: {os.listdir(OUT_DIR)}")
