"""
exp_068 — nucleus-C (dynamic top-p k for C) vs GRPO and the prior best
(pos_discount FIXED λ0.7 top_k=5). Per dataset: boxed (top) + length (bottom),
overlaying nucleus top_p∈{0.7,0.8,0.9,0.95}. Intermediate-safe.
"""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
DATASETS = [("gsm8k", "GSM8K (easy)"), ("math500", "MATH-500 (medium)"),
            ("bigmath", "Big-Math int-2k"), ("omnimath", "Omni-MATH (hard)")]
CURVES = [
    ("grpo",                "GRPO",           "#94a3b8", False),
    ("posdisc_lam0.7_k5",   "pos_disc k=5",   "#16a34a", False),
    ("nucleus_p0.7",        "nucleus p=0.7",  "#1d4ed8", True),
    ("nucleus_p0.8",        "nucleus p=0.8",  "#0891b2", True),
    ("nucleus_p0.9",        "nucleus p=0.9",  "#d97706", True),
    ("nucleus_p0.95",       "nucleus p=0.95", "#dc2626", True),
]


def rolling(xs, w=30):
    return [sum(xs[max(0, i-w+1):i+1]) / (i-max(0, i-w+1)+1) for i in range(len(xs))]


def col(p, k):
    if not os.path.exists(p):
        return None
    return [float(m.group(1)) for m in re.finditer(re.escape(k)+r"':\s*([-\d.eE+]+)", open(p).read())]


def lm(x, n=50):
    return sum(x[-n:]) / min(n, len(x)) if x else float("nan")


fig, axes = plt.subplots(2, 4, figsize=(20, 9))
rows = []
for j, (ds, ds_lab) in enumerate(DATASETS):
    axb, axl = axes[0][j], axes[1][j]
    for suf, lab, c, focal in CURVES:
        p = os.path.join(HERE, f"train_{ds}_{suf}.log")
        bx = col(p, "reward_answer_boxed/mean"); cl = col(p, "completions/mean_length")
        lw = 2.2 if focal else 1.6
        a = 1.0 if focal else 0.6
        if bx:
            ys = rolling(bx); axb.plot(range(len(ys)), ys, color=c, lw=lw, alpha=a, label=f"{lab} ({lm(bx):+.2f})")
        if cl:
            ys = rolling(cl); axl.plot(range(len(ys)), ys, color=c, lw=lw, alpha=a, label=f"{lab} ({lm(cl):.0f})")
        if bx or cl:
            rows.append((ds, lab, lm(cl) if cl else float("nan"), lm(bx) if bx else float("nan"), len(bx or cl)))
    axb.set_title(f"{ds_lab} — answer_boxed", fontsize=11, weight="bold")
    axb.axhline(0, color="#cbd5e1", lw=0.6, ls="--"); axb.grid(alpha=0.3); axb.legend(fontsize=7.5, loc="lower right")
    axb.set_xlabel("step"); axb.set_ylabel("boxed reward")
    axl.set_title(f"{ds_lab} — length", fontsize=11, weight="bold")
    axl.grid(alpha=0.3); axl.legend(fontsize=7.5, loc="upper right"); axl.set_xlabel("step"); axl.set_ylabel("tokens")

fig.suptitle("exp_068 — nucleus-C (dynamic top-p k) vs GRPO & pos_discount k=5 (FIXED λ0.7) "
             "(Qwen3-4B-Base, 300 steps) [intermediate]\nnucleus: n=#{prefix probs ≤ p}, min_k=1; sampling stays 1.0",
             fontsize=11, weight="bold")
out = os.path.join(HERE, "figures", "exp068_nucleus.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(out, dpi=140)
print(f"saved {out}")
print(f"\n{'dataset':10s} {'method':16s} {'L50_len':>9s} {'L50_boxed':>10s} {'steps':>6s}")
for ds, m, ml, mb, n in rows:
    print(f"{ds:10s} {m:16s} {ml:>9.0f} {mb:>+10.2f} {n:>6d}")
