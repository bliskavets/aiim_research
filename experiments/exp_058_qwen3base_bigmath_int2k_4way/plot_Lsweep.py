"""
exp_058 — L-sweep for the two length-penalty methods.
Both panels (completion length + answer_boxed reward) over the L knee values
{3096, 2048, 1536} for gtpo_ema_lenpen (solid) and gtpo_ema_lenpen_gated (dashed),
with the GRPO baseline as a grey reference. Reads train_<method>_L<L>.log; any
missing run is silently skipped, so the plot is buildable mid-sweep.
"""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)

# L value -> colour (smaller L = tighter penalty = warmer)
L_COLOR = {3096: "#2563eb", 2048: "#7c3aed", 1536: "#dc2626"}
L_LIST = [3096, 2048, 1536]
# method -> linestyle / label
METHODS = [("gtpo_ema_lenpen", "-", "lenpen"),
           ("gtpo_ema_lenpen_gated", "--", "gated")]


def rolling(xs, w=30):
    return [sum(xs[max(0, i-w+1):i+1]) / (i-max(0, i-w+1)+1) for i in range(len(xs))]


def col(p, k):
    if not os.path.exists(p):
        return None
    return [float(m.group(1))
            for m in re.finditer(re.escape(k)+r"':\s*([-\d.eE+]+)", open(p).read())]


def last_mean(xs, n=50):
    return sum(xs[-n:]) / min(n, len(xs)) if xs else float("nan")


fig, (axl, axb) = plt.subplots(1, 2, figsize=(15, 6))

# GRPO baseline reference (grey)
for ax, key in ((axl, "completions/mean_length"), (axb, "reward_answer_boxed/mean")):
    g = col(os.path.join(HERE, "train_grpo.log"), key)
    if g:
        ys = rolling(g)
        ax.plot(range(len(ys)), ys, color="#94a3b8", lw=1.6, ls=":",
                label=f"GRPO baseline ({last_mean(g):+.2f})" if ax is axb
                else f"GRPO baseline ({last_mean(g):.0f})", zorder=1)

rows = []
for L in L_LIST:
    for method, ls, short in METHODS:
        fn = os.path.join(HERE, f"train_{method}_L{L}.log")
        ln = col(fn, "completions/mean_length")
        bx = col(fn, "reward_answer_boxed/mean")
        if not ln and not bx:
            continue
        c = L_COLOR[L]
        lab = f"{short} L={L} (n={len(bx or ln)})"
        if ln:
            ys = rolling(ln)
            axl.plot(range(len(ys)), ys, color=c, ls=ls, lw=2.0, label=lab)
        if bx:
            ys = rolling(bx)
            axb.plot(range(len(ys)), ys, color=c, ls=ls, lw=2.0, label=lab)
        rows.append((short, L, last_mean(ln) if ln else float("nan"),
                     last_mean(bx) if bx else float("nan"), len(bx or ln)))

axl.set_title("completion length (rolling-30, mean tokens/gen)", fontsize=11, weight="bold")
axl.set_xlabel("step"); axl.set_ylabel("tokens"); axl.grid(alpha=0.3)
axl.legend(fontsize=8.5, loc="upper left")
axb.set_title("answer_boxed reward (rolling-30)\n+3 correct / -1.5 wrong / 0 none",
              fontsize=11, weight="bold")
axb.set_xlabel("step"); axb.set_ylabel("boxed reward")
axb.axhline(0, color="#cbd5e1", lw=0.6, ls="--"); axb.grid(alpha=0.3)
axb.legend(fontsize=8.5, loc="lower right")

fig.suptitle("exp_058 — length-penalty L-sweep (Qwen3-4B-Base, Big-Math int-2000, alpha_len=0.005)\n"
             "L in {3096, 2048, 1536}; solid=gtpo_ema_lenpen, dashed=gated. Smaller L = tighter length cap.",
             fontsize=10, weight="bold")
out = os.path.join(HERE, "figures", "exp058_lenpen_Lsweep.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(out, dpi=140)
print(f"saved {out}")
print(f"\n{'method':22s} {'L':>5s} {'L50_len':>9s} {'L50_boxed':>10s} {'steps':>6s}")
for short, L, ml, mb, n in rows:
    print(f"{short:22s} {L:>5d} {ml:>9.0f} {mb:>+10.2f} {n:>6d}")
