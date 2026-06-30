"""
exp_062 — the 4 non-entropy candidates on Big-Math int-2000 (exp_058 setup),
overlaid with GRPO / GROP@GRPO / gtpo_ema_flipped(FIXED) reused from exp_058.
Two panels: answer_boxed reward and completion length (rolling-30).
"""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CURVES = [
    ("train_bigmath_grpo.log",                   "GRPO",            "#64748b", False),
    ("train_bigmath_grpo_grop.log",              "GROP @ GRPO",     "#94a3b8", False),
    ("train_bigmath_gtpo_ema_flipped_fixed.log", "flipped FIXED",   "#0ea5e9", False),
    ("train_bigmath_sign_gate.log",              "sign_gate (6A)",  "#16a34a", True),
    ("train_bigmath_pos_discount.log",           "pos_discount",    "#d97706", True),
    ("train_bigmath_raw_c.log",                  "raw_C (no EMA)",  "#7c3aed", True),
    ("train_bigmath_ref_delta.log",              "ref_delta (3A)",  "#dc2626", True),
]


def rolling(xs, w=30):
    return [sum(xs[max(0, i-w+1):i+1]) / (i-max(0, i-w+1)+1) for i in range(len(xs))]


def col(p, k):
    if not os.path.exists(p):
        return None
    return [float(m.group(1)) for m in re.finditer(re.escape(k)+r"':\s*([-\d.eE+]+)", open(p).read())]


def lm(x, n=50):
    return sum(x[-n:]) / min(n, len(x)) if x else float("nan")


fig, (axb, axl) = plt.subplots(1, 2, figsize=(15, 6))
rows = []
for fn, lab, c, focal in CURVES:
    p = os.path.join(HERE, fn)
    bx = col(p, "reward_answer_boxed/mean"); cl = col(p, "completions/mean_length")
    lw = 2.5 if focal else 1.5
    a = 1.0 if focal else 0.55
    if bx:
        ys = rolling(bx); axb.plot(range(len(ys)), ys, color=c, lw=lw, alpha=a, label=f"{lab} ({lm(bx):+.2f})")
        if focal: axb.text(len(ys)+3, ys[-1], f" {lm(bx):+.2f}", color=c, fontsize=8, va="center", weight="bold")
    if cl:
        ys = rolling(cl); axl.plot(range(len(ys)), ys, color=c, lw=lw, alpha=a, label=f"{lab} ({lm(cl):.0f})")
    if bx or cl:
        rows.append((lab, lm(cl) if cl else float("nan"), lm(bx) if bx else float("nan"), len(bx or cl)))

axb.set_title("answer_boxed reward (rolling-30)", fontsize=11, weight="bold")
axb.axhline(0, color="#cbd5e1", lw=0.6, ls="--"); axb.grid(alpha=0.3); axb.legend(fontsize=8.5, loc="lower right")
axb.set_xlabel("step"); axb.set_ylabel("boxed reward")
axl.set_title("completion length (rolling-30)", fontsize=11, weight="bold")
axl.grid(alpha=0.3); axl.legend(fontsize=8.5, loc="upper right"); axl.set_xlabel("step"); axl.set_ylabel("tokens")

fig.suptitle("exp_062 — non-entropy candidates on Big-Math int-2000 (exp_058 setup, Qwen3-4B-Base, 300 steps)\n"
             "bold = candidates; thin = GRPO / GROP@GRPO / gtpo_ema_flipped(FIXED) (reused from exp_058)",
             fontsize=10, weight="bold")
out = os.path.join(HERE, "figures", "exp062_bigmath_compare.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(out, dpi=140)
print(f"saved {out}")
print(f"\n{'method':18s} {'L50_len':>9s} {'L50_boxed':>10s} {'steps':>6s}")
for lab, ml, mb, n in rows:
    print(f"{lab:18s} {ml:>9.0f} {mb:>+10.2f} {n:>6d}")
