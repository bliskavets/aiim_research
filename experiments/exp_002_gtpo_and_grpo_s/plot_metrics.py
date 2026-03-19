"""
Plot and compare training metrics for exp_002: GTPO vs GRPO-S vs GRPO (baseline from exp_001).
"""
import re, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

EXP_DIR = os.path.dirname(__file__)
OUT_DIR = "/workspace/figures_exp002"
os.makedirs(OUT_DIR, exist_ok=True)

EXP001_LOG = "/workspace/train_grpo_baseline.log"
GTPO_LOG   = os.path.join(EXP_DIR, "train_gtpo.log")
GRPOS_LOG  = os.path.join(EXP_DIR, "train_grpo_s.log")

PATTERN = re.compile(r"\{'loss':.*?'epoch': \d+\.\d+\}(?=\n)")

def parse_log(path):
    records = []
    with open(path) as f:
        text = f.read()
    for i, m in enumerate(PATTERN.finditer(text)):
        try:
            d = eval(m.group())
            d["step"] = i + 1
            records.append(d)
        except: pass
    return records

def get(records, key):
    return [r.get(key) for r in records]

def smooth(values, w=15):
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    if w <= 1: return arr
    kernel = np.ones(w) / w
    padded = np.pad(arr, (w//2, w - w//2 - 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid")

grpo  = parse_log(EXP001_LOG)
gtpo  = parse_log(GTPO_LOG)
grpos = parse_log(GRPOS_LOG)

print(f"Parsed: GRPO={len(grpo)}, GTPO={len(gtpo)}, GRPO-S={len(grpos)} steps")

COLORS = {"GRPO": "tab:gray", "GTPO": "tab:blue", "GRPO-S": "tab:orange"}

def plot_triple(ax, key, label, smooth_w=15):
    for name, records, color in [("GRPO", grpo, COLORS["GRPO"]),
                                   ("GTPO", gtpo, COLORS["GTPO"]),
                                   ("GRPO-S", grpos, COLORS["GRPO-S"])]:
        steps = [r["step"] for r in records]
        vals = np.array([r.get(key) if r.get(key) is not None else np.nan for r in records], dtype=float)
        ax.plot(steps, vals, color=color, alpha=0.15, linewidth=0.8)
        ax.plot(steps, smooth(vals, smooth_w), color=color, linewidth=2, label=name)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlabel("Step", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

# ── Figure 1: Main comparison dashboard ──────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Exp 002 — GTPO vs GRPO-S vs GRPO (baseline)\nLlama-3.2-3B · GSM8K · 500 steps · A100 80GB",
             fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

panels = [
    (gs[0, :2], "reward",                             "Total Reward"),
    (gs[0, 2:], "kl",                                 "KL Divergence"),
    (gs[1, 0],  "rewards/reward_format_exact/mean",   "Format Exact Reward"),
    (gs[1, 1],  "rewards/reward_format_approximate/mean", "Format Approx Reward"),
    (gs[1, 2],  "rewards/reward_answer_exact/mean",   "Answer Exact Reward"),
    (gs[1, 3],  "rewards/reward_answer_numeric/mean", "Answer Numeric Reward"),
    (gs[2, 0],  "loss",                               "Loss"),
    (gs[2, 1],  "grad_norm",                          "Grad Norm"),
    (gs[2, 2],  "completion_length",                  "Completion Length"),
    (gs[2, 3],  "learning_rate",                      "Learning Rate"),
]
for gs_loc, key, label in panels:
    ax = fig.add_subplot(gs_loc)
    plot_triple(ax, key, label)

plt.savefig(os.path.join(OUT_DIR, "dashboard_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved dashboard_comparison.png")

# ── Figure 2: Reward comparison (main result) ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Exp 002 — Reward Comparison: GTPO vs GRPO-S vs GRPO", fontsize=13, fontweight="bold")
plot_triple(axes[0], "reward", "Total Reward (smoothed)")
plot_triple(axes[1], "rewards/reward_format_exact/mean", "Format Exact Reward")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reward_comparison.png"), dpi=150)
plt.close()
print("Saved reward_comparison.png")

# ── Figure 3: KL divergence ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
plot_triple(ax, "kl", "KL Divergence from Base Model")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "kl_comparison.png"), dpi=150)
plt.close()
print("Saved kl_comparison.png")

# ── Figure 4: All reward functions ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Exp 002 — Reward Functions Comparison", fontsize=13, fontweight="bold")
plot_triple(axes[0,0], "rewards/reward_format_exact/mean",       "Format Exact")
plot_triple(axes[0,1], "rewards/reward_format_approximate/mean", "Format Approximate")
plot_triple(axes[1,0], "rewards/reward_answer_exact/mean",       "Answer Exact")
plot_triple(axes[1,1], "rewards/reward_answer_numeric/mean",     "Answer Numeric")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reward_functions_comparison.png"), dpi=150)
plt.close()
print("Saved reward_functions_comparison.png")

# ── Figure 5: Loss & grad norm ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Exp 002 — Loss & Grad Norm", fontsize=13, fontweight="bold")
plot_triple(axes[0], "loss",      "Loss")
plot_triple(axes[1], "grad_norm", "Grad Norm")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "loss_gradnorm.png"), dpi=150)
plt.close()
print("Saved loss_gradnorm.png")

# ── Save summary metrics JSON ────────────────────────────────────────────────
def summarize(records):
    if not records: return {}
    last = records[-1]
    mid  = records[len(records)//2]
    first = records[0]
    return {
        "steps": len(records),
        "step1":  {"reward": first.get("reward"), "format_exact": first.get("rewards/reward_format_exact/mean"), "kl": first.get("kl")},
        "step250": {"reward": mid.get("reward"),  "format_exact": mid.get("rewards/reward_format_exact/mean"),  "kl": mid.get("kl")},
        "step500": {"reward": last.get("reward"), "format_exact": last.get("rewards/reward_format_exact/mean"), "kl": last.get("kl")},
        "train_loss": last.get("loss"),
    }

summary = {"grpo": summarize(grpo), "gtpo": summarize(gtpo), "grpo_s": summarize(grpos)}
with open('/workspace/metrics_summary_exp002.json', "w") as f:
    json.dump(summary, f, indent=2)
print("Saved metrics_summary.json")
print("\nSummary:")
for name, s in summary.items():
    print(f"  {name.upper()}: step1 reward={s['step1']['reward']:.3f} → step500 reward={s['step500']['reward']:.3f}, format={s['step500']['format_exact']}, kl={s['step500']['kl']:.4f}")
