"""
exp_058 — the two follow-ups: (1) GROP on plain GRPO at reward-level (paper-
faithful, working base) and (2) the FIXED gtpo_ema_flipped (shaped advantage on
the full group, not the degenerate B=1 compute_loss). vs GRPO baseline and the
bare (broken) gtpo_ema_flipped. Two panels: completion length, answer_boxed.
"""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CURVES = [
    ("train_grpo.log",                       "GRPO baseline",                 "#64748b", "-",  False),
    ("train_gtpo_ema_flipped.log",           "gtpo_ema_flipped (bare/broken)","#dc2626", "-",  False),
    ("train_gtpo_ema_lenpen_gated_L1536.log","fixed-L gated (L=1536)",        "#0ea5e9", "--", False),
    ("train_grpo_grop.log",                  "GROP @ GRPO (paper, reward)",   "#d97706", "-",  True),
    ("train_gtpo_ema_flipped_fixed.log",     "gtpo_ema_flipped FIXED (group)","#16a34a", "-",  True),
]


def rolling(xs, w=30):
    return [sum(xs[max(0, i-w+1):i+1]) / (i-max(0, i-w+1)+1) for i in range(len(xs))]


def col(p, k):
    if not os.path.exists(p):
        return None
    return [float(m.group(1)) for m in re.finditer(re.escape(k)+r"':\s*([-\d.eE+]+)", open(p).read())]


def lm(x, n=50):
    return sum(x[-n:]) / min(n, len(x)) if x else float("nan")


fig, (axl, axb) = plt.subplots(1, 2, figsize=(15, 6))
rows = []
for fn, label, c, ls, focal in CURVES:
    p = os.path.join(HERE, fn)
    cl = col(p, "completions/mean_length"); bx = col(p, "reward_answer_boxed/mean")
    lw = 2.7 if focal else 1.8
    if cl:
        ys = rolling(cl); axl.plot(range(len(ys)), ys, color=c, ls=ls, lw=lw, label=f"{label} ({lm(cl):.0f})")
        if focal: axl.text(len(ys)+3, ys[-1], f" {lm(cl):.0f}", color=c, fontsize=8.5, va="center", weight="bold")
    if bx:
        ys = rolling(bx); axb.plot(range(len(ys)), ys, color=c, ls=ls, lw=lw, label=f"{label} ({lm(bx):+.2f})")
        if focal: axb.text(len(ys)+3, ys[-1], f" {lm(bx):+.2f}", color=c, fontsize=8.5, va="center", weight="bold")
    if cl or bx:
        rows.append((label, lm(cl) if cl else float("nan"), lm(bx) if bx else float("nan"), len(bx or cl)))

axl.set_title("completion length (rolling-30)", fontsize=11, weight="bold")
axl.set_xlabel("step"); axl.set_ylabel("tokens"); axl.grid(alpha=0.3); axl.legend(fontsize=8.5, loc="upper left")
axb.set_title("answer_boxed reward (rolling-30)", fontsize=11, weight="bold")
axb.set_xlabel("step"); axb.set_ylabel("boxed reward"); axb.axhline(0, color="#cbd5e1", lw=0.6, ls="--")
axb.grid(alpha=0.3); axb.legend(fontsize=8.5, loc="lower right")

fig.suptitle("exp_058 — fixing gtpo_ema_flipped + GROP on a working base (Qwen3-4B-Base, Big-Math int-2000)\n"
             "Computing the shaped advantage on the FULL group (not degenerate B=1) removes the length explosion AND the "
             "reward inversion: FIXED matches GRPO quality at shorter length. GROP@GRPO also controls length faithfully.",
             fontsize=10, weight="bold")
out = os.path.join(HERE, "figures", "exp058_fix_grop.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(out, dpi=140)
print(f"saved {out}")
print(f"\n{'method':34s} {'L50_len':>9s} {'L50_boxed':>10s} {'steps':>6s}")
for label, ml, mb, n in rows:
    print(f"{label:34s} {ml:>9.0f} {mb:>+10.2f} {n:>6d}")
