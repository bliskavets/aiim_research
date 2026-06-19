"""
exp_058 — L-sweep for the two length-penalty methods.
2x2 layout:
  top row    — time series (rolling-30): completion length, answer_boxed reward
  bottom row — SUMMARY vs L (last-50-step mean): length-vs-L and boxed-vs-L, one
               marker per L for each method. The bottom row makes every L value
               (3096 / 2048 / 1536) an explicit x-position so none get lost in
               the overlapping time-series tangle.
solid / circle = gtpo_ema_lenpen ; dashed / square = gtpo_ema_lenpen_gated.
GRPO baseline shown as a grey reference. Reads train_<method>_L<L>.log; any
missing run is silently skipped, so the plot is buildable mid-sweep.
"""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)

L_COLOR = {3096: "#2563eb", 2048: "#7c3aed", 1536: "#dc2626"}
L_LIST = [3096, 2048, 1536]
METHODS = [("gtpo_ema_lenpen", "-", "o", "lenpen"),
           ("gtpo_ema_lenpen_gated", "--", "s", "gated")]


def rolling(xs, w=30):
    return [sum(xs[max(0, i-w+1):i+1]) / (i-max(0, i-w+1)+1) for i in range(len(xs))]


def col(p, k):
    if not os.path.exists(p):
        return None
    return [float(m.group(1))
            for m in re.finditer(re.escape(k)+r"':\s*([-\d.eE+]+)", open(p).read())]


def last_mean(xs, n=50):
    return sum(xs[-n:]) / min(n, len(xs)) if xs else float("nan")


fig, ((axl, axb), (axsl, axsb)) = plt.subplots(2, 2, figsize=(15, 11))

# ---- GRPO baseline reference (grey) on the time-series + as h-lines on summary
g_len = col(os.path.join(HERE, "train_grpo.log"), "completions/mean_length")
g_box = col(os.path.join(HERE, "train_grpo.log"), "reward_answer_boxed/mean")
if g_len:
    ys = rolling(g_len)
    axl.plot(range(len(ys)), ys, color="#94a3b8", lw=1.6, ls=":",
             label=f"GRPO baseline ({last_mean(g_len):.0f})", zorder=1)
    axsl.axhline(last_mean(g_len), color="#94a3b8", lw=1.4, ls=":", label="GRPO baseline")
if g_box:
    ys = rolling(g_box)
    axb.plot(range(len(ys)), ys, color="#94a3b8", lw=1.6, ls=":",
             label=f"GRPO baseline ({last_mean(g_box):+.2f})", zorder=1)
    axsb.axhline(last_mean(g_box), color="#94a3b8", lw=1.4, ls=":", label="GRPO baseline")

# ---- per (L, method): time-series + collect summary points
summary = {short: {"L": [], "len": [], "box": []} for _, _, _, short in METHODS}
rows = []
for L in L_LIST:
    for method, ls, mk, short in METHODS:
        fn = os.path.join(HERE, f"train_{method}_L{L}.log")
        ln = col(fn, "completions/mean_length")
        bx = col(fn, "reward_answer_boxed/mean")
        if not ln and not bx:
            continue
        c = L_COLOR[L]
        n = len(bx or ln)
        if ln:
            ys = rolling(ln)
            axl.plot(range(len(ys)), ys, color=c, ls=ls, lw=1.8, label=f"{short} L={L} (n={n})")
        if bx:
            ys = rolling(bx)
            axb.plot(range(len(ys)), ys, color=c, ls=ls, lw=1.8, label=f"{short} L={L} (n={n})")
        summary[short]["L"].append(L)
        summary[short]["len"].append(last_mean(ln) if ln else float("nan"))
        summary[short]["box"].append(last_mean(bx) if bx else float("nan"))
        rows.append((short, L, last_mean(ln) if ln else float("nan"),
                     last_mean(bx) if bx else float("nan"), n))

# ---- summary rows: metric vs L (x reversed so tighter knee is on the right)
for method, ls, mk, short in METHODS:
    s = summary[short]
    if not s["L"]:
        continue
    pts = sorted(zip(s["L"], s["len"], s["box"]), reverse=True)
    Ls = [p[0] for p in pts]
    cols = [L_COLOR[L] for L in Ls]
    axsl.plot(Ls, [p[1] for p in pts], color="#475569", ls=ls, lw=1.4, zorder=2)
    axsb.plot(Ls, [p[2] for p in pts], color="#475569", ls=ls, lw=1.4, zorder=2)
    axsl.scatter(Ls, [p[1] for p in pts], c=cols, s=90, marker=mk, zorder=3,
                 edgecolors="k", linewidths=0.6, label=short)
    axsb.scatter(Ls, [p[2] for p in pts], c=cols, s=90, marker=mk, zorder=3,
                 edgecolors="k", linewidths=0.6, label=short)
    for L, vl, vb in pts:
        axsl.annotate(f"{vl:.0f}", (L, vl), textcoords="offset points", xytext=(0, 7),
                      ha="center", fontsize=8, color=L_COLOR[L])
        axsb.annotate(f"{vb:+.2f}", (L, vb), textcoords="offset points", xytext=(0, 7),
                      ha="center", fontsize=8, color=L_COLOR[L])

axl.set_title("completion length (rolling-30, mean tokens/gen)", fontsize=11, weight="bold")
axl.set_xlabel("step"); axl.set_ylabel("tokens"); axl.grid(alpha=0.3)
axl.legend(fontsize=8.5, loc="upper left")
axb.set_title("answer_boxed reward (rolling-30)\n+3 correct / -1.5 wrong / 0 none", fontsize=11, weight="bold")
axb.set_xlabel("step"); axb.set_ylabel("boxed reward")
axb.axhline(0, color="#cbd5e1", lw=0.6, ls="--"); axb.grid(alpha=0.3)
axb.legend(fontsize=8.5, loc="lower right")

for ax in (axsl, axsb):
    ax.set_xticks(L_LIST); ax.set_xlabel("length-penalty knee L (tighter →)")
    ax.invert_xaxis(); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="best")
axsl.set_title("SUMMARY: final length vs L (last-50 mean)", fontsize=11, weight="bold")
axsl.set_ylabel("tokens")
axsb.set_title("SUMMARY: final boxed reward vs L (last-50 mean)", fontsize=11, weight="bold")
axsb.set_ylabel("boxed reward")

fig.suptitle("exp_058 — length-penalty L-sweep (Qwen3-4B-Base, Big-Math int-2000, alpha_len=0.005)\n"
             "L in {3096, 2048, 1536}; solid/circle=gtpo_ema_lenpen, dashed/square=gated. "
             "Smaller L = tighter length cap. Colour = L (blue 3096 / purple 2048 / red 1536).",
             fontsize=10, weight="bold")
out = os.path.join(HERE, "figures", "exp058_lenpen_Lsweep.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out, dpi=140)
print(f"saved {out}")
print(f"\n{'method':22s} {'L':>5s} {'L50_len':>9s} {'L50_boxed':>10s} {'steps':>6s}")
for short, L, ml, mb, n in rows:
    print(f"{short:22s} {L:>5d} {ml:>9.0f} {mb:>+10.2f} {n:>6d}")
