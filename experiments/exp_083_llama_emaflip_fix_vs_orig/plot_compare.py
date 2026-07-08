"""
exp_083 — Llama-3.2-3B-Instruct (exp_050 harness), Big-Math: GRPO vs gtpo_ema_flipped
(ORIGINAL, pre-FIX) vs gtpo_ema_flipped_fixed. 4-panel like exp_050:
total reward / answer_exact / format_exact / length.
"""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CURVES = [
    ("grpo",                    "GRPO",                       "#94a3b8"),
    ("gtpo_ema_flipped",        "gtpo_ema_flipped (ORIGINAL)", "#f59e0b"),
    ("gtpo_ema_flipped_fixed",  "gtpo_ema_flipped (FIXED)",    "#dc2626"),
]
PANELS = [("reward", "'reward'", "total reward"),
          ("reward_answer_exact/mean", "reward_answer_exact/mean", "answer_exact reward"),
          ("reward_format_exact/mean", "reward_format_exact/mean", "format_exact reward"),
          ("completions/mean_length", "completions/mean_length", "completion length")]


def rolling(xs, w=20):
    return [sum(xs[max(0, i-w+1):i+1]) / (i-max(0, i-w+1)+1) for i in range(len(xs))]


def col(p, key):
    if not os.path.exists(p):
        return None
    pat = re.escape(key) + r"':\s*([-\d.eE+]+)" if not key.startswith("'") else re.escape(key[1:-1]) + r"':\s*([-\d.eE+]+)"
    return [float(m.group(1)) for m in re.finditer(pat, open(p).read())]


def lm(x, n=50):
    return sum(x[-n:]) / min(n, len(x)) if x else float("nan")

fig, axes = plt.subplots(2, 2, figsize=(15, 9)); axes = axes.ravel()
rows = []
for ax, (metric, key, lab) in zip(axes, PANELS):
    for suf, name, c in CURVES:
        p = os.path.join(HERE, f"train_bigmath_{suf}.log")
        ys = col(p, "reward" if metric == "reward" else metric)
        if metric == "reward":  # 'reward' also matches reward_* keys; grab exact 'reward':
            raw = open(p).read() if os.path.exists(p) else ""
            ys = [float(m.group(1)) for m in re.finditer(r"[^_]'reward':\s*([-\d.eE+]+)", raw)]
        if ys:
            r = rolling(ys); ax.plot(range(len(r)), r, color=c, lw=2.0, label=f"{name} ({lm(ys):+.2f})")
            if metric == "reward":
                rows.append((suf, lm(ys)))
    ax.set_title(f"Big-Math — {lab}", fontsize=12, weight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best"); ax.set_xlabel("step")
    if metric != "completions/mean_length":
        ax.axhline(0, color="#cbd5e1", lw=0.6, ls="--")

fig.suptitle("exp_083 — Llama-3.2-3B-Instruct (exp_050 harness), Big-Math int-2k: "
             "GRPO vs gtpo_ema_flipped ORIGINAL vs FIXED (500 steps)", fontsize=13, weight="bold")
out = os.path.join(HERE, "figures", "exp083_emaflip_fix_vs_orig.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out, dpi=140)
print(f"saved {out}")
for suf, v in rows:
    print(f"  {suf:26s} total_reward L50 {v:+.2f}")
