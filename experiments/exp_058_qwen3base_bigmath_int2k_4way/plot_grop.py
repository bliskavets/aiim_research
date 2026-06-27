"""
exp_058 — GROP (Group Relative Overlong Punishment, arXiv:2508.04349 App.D) vs
GRPO baseline, bare gtpo_ema_flipped, and the best fixed-L gated (L=1536).
2x2: completion length, answer_boxed reward, and GROP's difficulty-regime
fractions + solve-rate over training (which explains its behaviour).
"""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CURVES = [
    ("train_grpo.log",                        "GRPO baseline",           "#64748b", "-"),
    ("train_gtpo_ema_flipped.log",            "gtpo_ema_flipped (bare)", "#dc2626", "-"),
    ("train_gtpo_ema_lenpen_gated_L1536.log", "fixed-L gated (L=1536)",  "#2563eb", "-"),
    ("train_gtpo_ema_flipped_grop.log",       "GROP (paper App.D)",      "#16a34a", "-"),
]


def rolling(xs, w=30):
    return [sum(xs[max(0, i-w+1):i+1]) / (i-max(0, i-w+1)+1) for i in range(len(xs))]


def col(p, k):
    if not os.path.exists(p):
        return None
    return [float(m.group(1)) for m in re.finditer(re.escape(k)+r"':\s*([-\d.eE+]+)", open(p).read())]


def lm(x, n=50):
    return sum(x[-n:]) / min(n, len(x)) if x else float("nan")


fig, ((axl, axb), (axr, axc)) = plt.subplots(2, 2, figsize=(15, 11))
rows = []
for fn, label, c, ls in CURVES:
    p = os.path.join(HERE, fn)
    cl = col(p, "completions/mean_length"); bx = col(p, "reward_answer_boxed/mean")
    focal = "grop" in fn
    lw = 2.6 if focal else 1.8
    if cl:
        ys = rolling(cl); axl.plot(range(len(ys)), ys, color=c, ls=ls, lw=lw, label=f"{label} ({lm(cl):.0f})")
    if bx:
        ys = rolling(bx); axb.plot(range(len(ys)), ys, color=c, ls=ls, lw=lw, label=f"{label} ({lm(bx):+.2f})")
    if cl or bx:
        rows.append((label, lm(cl) if cl else float("nan"), lm(bx) if bx else float("nan"), len(bx or cl)))

axl.set_title("completion length (rolling-30)", fontsize=11, weight="bold")
axl.set_xlabel("step"); axl.set_ylabel("tokens"); axl.grid(alpha=0.3); axl.legend(fontsize=9, loc="upper left")
axb.set_title("answer_boxed reward (rolling-30)", fontsize=11, weight="bold")
axb.set_xlabel("step"); axb.set_ylabel("boxed reward"); axb.axhline(0, color="#cbd5e1", lw=0.6, ls="--")
axb.grid(alpha=0.3); axb.legend(fontsize=9, loc="lower right")

# GROP regime fractions + solve rate
gp = os.path.join(HERE, "train_gtpo_ema_flipped_grop.log")
for key, lab, c in [("gtpo_ema_flipped_grop/frac_hard", "hard (no penalty)", "#dc2626"),
                    ("gtpo_ema_flipped_grop/frac_easy", "easy (penalize correct)", "#16a34a"),
                    ("gtpo_ema_flipped_grop/frac_medium", "medium", "#d97706")]:
    v = col(gp, key)
    if v:
        ys = rolling(v); axr.plot(range(len(ys)), ys, color=c, lw=2.0, label=f"{lab} ({lm(v):.2f})")
axr.set_title("GROP difficulty-regime fractions over training\n(hard → NO penalty by design)", fontsize=11, weight="bold")
axr.set_xlabel("step"); axr.set_ylabel("fraction of groups"); axr.grid(alpha=0.3); axr.legend(fontsize=9, loc="best")

fc = col(gp, "gtpo_ema_flipped_grop/frac_correct"); pen = col(gp, "gtpo_ema_flipped_grop/pen_absmean")
if fc:
    ys = rolling(fc); axc.plot(range(len(ys)), ys, color="#7c3aed", lw=2.0, label=f"frac_correct ({lm(fc):.2f})")
if pen:
    ys = rolling(pen); axc.plot(range(len(ys)), ys, color="#0ea5e9", lw=2.0, label=f"|penalty| applied ({lm(pen):.3f})")
axc.set_title("GROP solve-rate & applied penalty over training", fontsize=11, weight="bold")
axc.set_xlabel("step"); axc.set_ylabel("value"); axc.grid(alpha=0.3); axc.legend(fontsize=9, loc="best")

fig.suptitle("exp_058 — GROP (Group Relative Overlong Punishment, arXiv:2508.04349 App.D) on gtpo_ema_flipped\n"
             "Qwen3-4B-Base, Big-Math int-2000, 300 steps. Difficulty gating disables the penalty under collapse "
             "(most groups become 'hard' → no penalty).",
             fontsize=10, weight="bold")
out = os.path.join(HERE, "figures", "exp058_grop.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(out, dpi=140)
print(f"saved {out}")
print(f"\n{'method':28s} {'L50_len':>9s} {'L50_boxed':>10s} {'steps':>6s}")
for label, ml, mb, n in rows:
    print(f"{label:28s} {ml:>9.0f} {mb:>+10.2f} {n:>6d}")
