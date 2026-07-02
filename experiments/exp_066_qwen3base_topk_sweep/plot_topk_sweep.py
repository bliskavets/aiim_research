"""
exp_066 — top_k sweep for confidence C on pos_discount + FIXED λ=0.7.
Per dataset (col): boxed reward (top) & length (bottom), overlaying top_k∈{5,10,20,40}
+ GRPO. k=20 is the default (reused from exp_063). Intermediate-safe.
"""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
DATASETS = [("gsm8k", "GSM8K (easy)"), ("math500", "MATH-500 (medium)"),
            ("bigmath", "Big-Math int-2k"), ("omnimath", "Omni-MATH (hard)")]
# (suffix, label, color)
CURVES = [
    ("grpo",                "GRPO",        "#94a3b8"),
    ("posdisc_lam0.7_k5",   "k=5",         "#1d4ed8"),
    ("posdisc_lam0.7_k10",  "k=10",        "#0891b2"),
    ("posdisc_lam0.7_k20",  "k=20 (def)",  "#16a34a"),
    ("posdisc_lam0.7_k40",  "k=40",        "#dc2626"),
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
    for suf, lab, c in CURVES:
        p = os.path.join(HERE, f"train_{ds}_{suf}.log")
        bx = col(p, "reward_answer_boxed/mean"); cl = col(p, "completions/mean_length")
        lw = 1.5 if suf == "grpo" else 2.1
        ls = "--" if suf == "grpo" else "-"
        if bx:
            ys = rolling(bx); axb.plot(range(len(ys)), ys, color=c, lw=lw, ls=ls, label=f"{lab} ({lm(bx):+.2f})")
        if cl:
            ys = rolling(cl); axl.plot(range(len(ys)), ys, color=c, lw=lw, ls=ls, label=f"{lab} ({lm(cl):.0f})")
        if bx or cl:
            rows.append((ds, lab, lm(cl) if cl else float("nan"), lm(bx) if bx else float("nan"), len(bx or cl)))
    axb.set_title(f"{ds_lab} — answer_boxed", fontsize=11, weight="bold")
    axb.axhline(0, color="#cbd5e1", lw=0.6, ls="--"); axb.grid(alpha=0.3); axb.legend(fontsize=8, loc="lower right")
    axb.set_xlabel("step"); axb.set_ylabel("boxed reward")
    axl.set_title(f"{ds_lab} — length", fontsize=11, weight="bold")
    axl.grid(alpha=0.3); axl.legend(fontsize=8, loc="upper right"); axl.set_xlabel("step"); axl.set_ylabel("tokens")

fig.suptitle("exp_066 — top_k sweep for C on pos_discount + FIXED λ0.7 (Qwen3-4B-Base, 300 steps) [intermediate]\n"
             "C = -mean over top-k logprobs; k∈{5,10,20(def),40}. top=boxed, bottom=length.",
             fontsize=11, weight="bold")
out = os.path.join(HERE, "figures", "exp066_topk_sweep.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(out, dpi=140)
print(f"saved {out}")
print(f"\n{'dataset':10s} {'k':>10s} {'L50_len':>9s} {'L50_boxed':>10s} {'steps':>6s}")
for ds, m, ml, mb, n in rows:
    print(f"{ds:10s} {m:>10s} {ml:>9.0f} {mb:>+10.2f} {n:>6d}")
