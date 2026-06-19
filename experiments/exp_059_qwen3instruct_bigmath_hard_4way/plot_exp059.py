"""
exp_059 — 4-method comparison on Qwen3-4B-INSTRUCT, HARD Big-Math (llama8b<0.3).
FIXED (non-bypassed) trainers; shaped metrics logged every step.

Headline: with shaping ACTUALLY applied, on the instruct model WITH headroom
(boxed ~0.65/3.0 at start, far from saturated), GRPO baseline LEARNS (boxed
0.66->0.78) while ALL three shaped methods UNDERPERFORM and DECLINE
(0.65 -> 0.39-0.51). KL stays tiny for all; completion length ~uniform near the
cap (instruct thinking-mode verbose) — so the shaped degradation here is direct
(bad advantage), not the length-explosion (exp_058 base) or KL-blowup (exp_056).
"""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CURVES = [
    ("train_grpo.log",              "grpo baseline",     "#64748b"),
    ("train_grpo_s_entropy.log",    "grpo_s_entropy",    "#d97706"),
    ("train_gtpo_conf.log",         "gtpo_conf",         "#059669"),
    ("train_gtpo_ema_flipped.log",  "gtpo_ema_flipped",  "#dc2626"),
]


def rolling(xs, w=30):
    return [sum(xs[max(0, i-w+1):i+1]) / (i-max(0, i-w+1)+1) for i in range(len(xs))]


def col(p, k):
    if not os.path.exists(p): return None
    return [float(m.group(1)) for m in re.finditer(re.escape(k)+r"':\s*([-\d.eE+]+)", open(p).read())]


fig, (axb, axr) = plt.subplots(1, 2, figsize=(15, 6))
for fn, label, c in CURVES:
    bx = col(os.path.join(HERE, fn), "reward_answer_boxed/mean")
    rw = col(os.path.join(HERE, fn), "'reward")
    if bx:
        ys = rolling(bx); axb.plot(range(len(ys)), ys, color=c, lw=2.0, label=f"{label}")
        axb.text(len(ys)+3, ys[-1], f" {sum(bx[-100:])/min(100,len(bx)):.2f}", color=c, fontsize=8, va="center", weight="bold")
    if rw:
        ys = rolling(rw); axr.plot(range(len(ys)), ys, color=c, lw=2.0, label=label)

axb.set_title("answer_boxed reward (rolling-30)\n+3 correct / -1.5 wrong / 0 none", fontsize=11, weight="bold")
axb.set_xlabel("step"); axb.set_ylabel("boxed reward"); axb.axhline(0, color="#94a3b8", lw=0.6, ls="--"); axb.grid(alpha=0.3); axb.legend(fontsize=8.5, loc="upper left")
axr.set_title("total reward (rolling-30)", fontsize=11, weight="bold")
axr.set_xlabel("step"); axr.set_ylabel("reward"); axr.axhline(0, color="#94a3b8", lw=0.6, ls="--"); axr.grid(alpha=0.3); axr.legend(fontsize=8.5, loc="upper left")

fig.suptitle("exp_059 — Qwen3-4B-INSTRUCT, HARD Big-Math (llama8b<0.3), 4 methods (shaping ACTUALLY applied)\n"
             "Not saturated (boxed starts ~0.65/3.0). grpo LEARNS (boxed ->0.78); all 3 shaped methods DECLINE "
             "(->0.39-0.51). No shaping beats grpo — completes the base+instruct negative result.",
             fontsize=10, weight="bold")
out = os.path.join(HERE, "figures", "exp059_4way_instruct_hard.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.91]); fig.savefig(out, dpi=140)
print(f"saved {out}")
