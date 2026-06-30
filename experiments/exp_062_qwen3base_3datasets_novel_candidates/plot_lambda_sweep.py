"""
exp_062 — EMA-lambda sweep for gtpo_ema_flipped(FIXED) across all 4 setups.
4 columns (datasets) x 2 rows (answer_boxed, length). Each panel overlays GRPO
(grey) + FIXED at lambda in {0.1,0.3,0.5,0.7,0.8,0.9} (0.9 = the original FIXED).
"""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
DATASETS = [("gsm8k", "GSM8K (easy)"), ("math500", "MATH-500 (medium)"),
            ("bigmath", "Big-Math int-2k"), ("omnimath", "Omni-MATH (hard)")]
# lambda -> colour (low=blue ... high=red); 0.9 is the original FIXED
LAMS = [("0.1", "#1d4ed8"), ("0.3", "#0891b2"), ("0.5", "#16a34a"),
        ("0.7", "#d97706"), ("0.8", "#dc2626"), ("0.9*", "#7c3aed")]


def rolling(xs, w=30):
    return [sum(xs[max(0, i-w+1):i+1]) / (i-max(0, i-w+1)+1) for i in range(len(xs))]


def col(p, k):
    if not os.path.exists(p):
        return None
    return [float(m.group(1)) for m in re.finditer(re.escape(k)+r"':\s*([-\d.eE+]+)", open(p).read())]


def lm(x, n=50):
    return sum(x[-n:]) / min(n, len(x)) if x else float("nan")


def logpath(ds, lam):
    # lambda=0.9 = the original FIXED run (different filename)
    if lam == "0.9*":
        return os.path.join(HERE, f"train_{ds}_gtpo_ema_flipped_fixed.log")
    return os.path.join(HERE, f"train_{ds}_lam{lam}.log")


fig, axes = plt.subplots(2, 4, figsize=(20, 9))
rows = []
for j, (ds, ds_lab) in enumerate(DATASETS):
    axb, axl = axes[0][j], axes[1][j]
    g = os.path.join(HERE, f"train_{ds}_grpo.log")
    gb, gl = col(g, "reward_answer_boxed/mean"), col(g, "completions/mean_length")
    if gb:
        axb.plot(range(len(rolling(gb))), rolling(gb), color="#94a3b8", lw=2.4, ls="--", label=f"GRPO ({lm(gb):+.2f})")
    if gl:
        axl.plot(range(len(rolling(gl))), rolling(gl), color="#94a3b8", lw=2.4, ls="--", label=f"GRPO ({lm(gl):.0f})")
    for lam, c in LAMS:
        p = logpath(ds, lam)
        bx, cl = col(p, "reward_answer_boxed/mean"), col(p, "completions/mean_length")
        if bx:
            axb.plot(range(len(rolling(bx))), rolling(bx), color=c, lw=1.8, label=f"λ={lam} ({lm(bx):+.2f})")
        if cl:
            axl.plot(range(len(rolling(cl))), rolling(cl), color=c, lw=1.8, label=f"λ={lam} ({lm(cl):.0f})")
        if bx or cl:
            rows.append((ds, lam, lm(cl) if cl else float("nan"), lm(bx) if bx else float("nan"), len(bx or cl)))
    axb.set_title(f"{ds_lab} — answer_boxed", fontsize=11, weight="bold")
    axb.axhline(0, color="#cbd5e1", lw=0.6, ls="--"); axb.grid(alpha=0.3); axb.legend(fontsize=7, loc="lower right")
    axb.set_xlabel("step"); axb.set_ylabel("boxed reward")
    axl.set_title(f"{ds_lab} — length", fontsize=11, weight="bold")
    axl.grid(alpha=0.3); axl.legend(fontsize=7, loc="upper right"); axl.set_xlabel("step"); axl.set_ylabel("tokens")

fig.suptitle("exp_062 — gtpo_ema_flipped(FIXED) EMA-λ sweep across all setups (Qwen3-4B-Base, 300 steps)\n"
             "λ∈{0.1,0.3,0.5,0.7,0.8,0.9*} (0.9*=original FIXED) vs GRPO (grey dashed). top=boxed, bottom=length.",
             fontsize=11, weight="bold")
out = os.path.join(HERE, "figures", "exp062_lambda_sweep.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(out, dpi=140)
print(f"saved {out}")
print(f"\n{'dataset':10s} {'lambda':>7s} {'L50_len':>9s} {'L50_boxed':>10s} {'steps':>6s}")
for ds, lam, ml, mb, n in rows:
    print(f"{ds:10s} {lam:>7s} {ml:>9.0f} {mb:>+10.2f} {n:>6d}")
