"""
exp_061 — GROP @ GRPO vs FIXED gtpo_ema_flipped across 3 datasets.
One column per dataset (gsm8k / math500 / omnimath), two rows:
  top    = answer_boxed reward (rolling-30)
  bottom = completion length (rolling-30)
Both setups overlaid per dataset. Reads train_<ds>_<method>.log.
"""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
DATASETS = [("gsm8k", "GSM8K (easy)"), ("math500", "MATH-500 (medium)"), ("omnimath", "Omni-MATH (hard)")]
METHODS = [("grpo_grop", "GROP @ GRPO", "#d97706"),
           ("gtpo_ema_flipped_fixed", "gtpo_ema_flipped FIXED", "#16a34a")]


def rolling(xs, w=30):
    return [sum(xs[max(0, i-w+1):i+1]) / (i-max(0, i-w+1)+1) for i in range(len(xs))]


def col(p, k):
    if not os.path.exists(p):
        return None
    return [float(m.group(1)) for m in re.finditer(re.escape(k)+r"':\s*([-\d.eE+]+)", open(p).read())]


def lm(x, n=50):
    return sum(x[-n:]) / min(n, len(x)) if x else float("nan")


fig, axes = plt.subplots(2, 3, figsize=(18, 9))
summary = []
for j, (ds, ds_lab) in enumerate(DATASETS):
    axb, axl = axes[0][j], axes[1][j]
    for m, m_lab, c in METHODS:
        p = os.path.join(HERE, f"train_{ds}_{m}.log")
        bx = col(p, "reward_answer_boxed/mean"); cl = col(p, "completions/mean_length")
        if bx:
            ys = rolling(bx); axb.plot(range(len(ys)), ys, color=c, lw=2.0, label=f"{m_lab} ({lm(bx):+.2f})")
        if cl:
            ys = rolling(cl); axl.plot(range(len(ys)), ys, color=c, lw=2.0, label=f"{m_lab} ({lm(cl):.0f})")
        if bx or cl:
            summary.append((ds, m_lab, lm(cl) if cl else float("nan"), lm(bx) if bx else float("nan"), len(bx or cl)))
    axb.set_title(f"{ds_lab} — answer_boxed", fontsize=11, weight="bold")
    axb.axhline(0, color="#cbd5e1", lw=0.6, ls="--"); axb.grid(alpha=0.3); axb.legend(fontsize=8.5, loc="lower right")
    axb.set_xlabel("step"); axb.set_ylabel("boxed reward")
    axl.set_title(f"{ds_lab} — length", fontsize=11, weight="bold")
    axl.grid(alpha=0.3); axl.legend(fontsize=8.5, loc="upper left"); axl.set_xlabel("step"); axl.set_ylabel("tokens")

fig.suptitle("exp_061 — GROP@GRPO vs FIXED gtpo_ema_flipped across 3 datasets (Qwen3-4B-Base, integer-answer, 300 steps)\n"
             "top=answer_boxed reward, bottom=completion length; per dataset both setups overlaid.",
             fontsize=11, weight="bold")
out = os.path.join(HERE, "figures", "exp061_compare.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(out, dpi=140)
print(f"saved {out}")
print(f"\n{'dataset':10s} {'method':26s} {'L50_len':>9s} {'L50_boxed':>10s} {'steps':>6s}")
for ds, m, ml, mb, n in summary:
    print(f"{ds:10s} {m:26s} {ml:>9.0f} {mb:>+10.2f} {n:>6d}")
