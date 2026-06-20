"""
exp_058 — adaptive length penalty: gtpo_ema_adaptlen (always-on) and
gtpo_ema_adaptlen_gated (low-temp difficulty gate), vs GRPO baseline and the
best fixed-L config from the L-sweep (gated L=1536). The adaptive knee floats
with each group's own length distribution; the penalty is bounded in [-0.5, 0].

Top row    — time series (rolling-30): completion length, answer_boxed reward.
Bottom row — the adaptive knee L over training (mean_L), and the penalty
             activity (pen_rel_absmean) / gate fraction. Reads each method's log;
             missing runs are skipped so the plot is buildable mid-run.
"""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)

CURVES = [
    ("train_grpo.log",                       "GRPO baseline",        "#94a3b8", ":"),
    ("train_gtpo_ema_lenpen_gated_L1536.log","fixed-L gated (L=1536)","#0ea5e9", "--"),
    ("train_gtpo_ema_adaptlen.log",          "adaptlen (always)",    "#16a34a", "-"),
    ("train_gtpo_ema_adaptlen_gated.log",    "adaptlen gated",       "#dc2626", "-"),
]


def rolling(xs, w=30):
    return [sum(xs[max(0, i-w+1):i+1]) / (i-max(0, i-w+1)+1) for i in range(len(xs))]


def col(p, k):
    if not os.path.exists(p):
        return None
    return [float(m.group(1))
            for m in re.finditer(re.escape(k)+r"':\s*([-\d.eE+]+)", open(p).read())]


def last_mean(xs, n=50):
    return sum(xs[-n:]) / min(n, len(xs)) if xs else float("nan")


fig, ((axl, axb), (axL, axp)) = plt.subplots(2, 2, figsize=(15, 11))

rows = []
for fn, label, c, ls in CURVES:
    focal = "adaptlen" in fn                      # the two new methods = focal
    lw    = 2.7 if focal else 1.5
    alpha = 1.0 if focal else 0.5
    z     = 5 if focal else 1
    p = os.path.join(HERE, fn)
    cl = col(p, "completions/mean_length")
    bx = col(p, "reward_answer_boxed/mean")
    if cl:
        ys = rolling(cl); axl.plot(range(len(ys)), ys, color=c, ls=ls, lw=lw, alpha=alpha,
                                   zorder=z, label=f"{label} ({last_mean(cl):.0f})")
        if focal:
            axl.text(len(ys)+3, ys[-1], f" {last_mean(cl):.0f}", color=c, fontsize=8.5,
                     va="center", weight="bold", zorder=6)
    if bx:
        ys = rolling(bx); axb.plot(range(len(ys)), ys, color=c, ls=ls, lw=lw, alpha=alpha,
                                   zorder=z, label=f"{label} ({last_mean(bx):+.2f})")
        if focal:
            axb.text(len(ys)+3, ys[-1], f" {last_mean(bx):+.2f}", color=c, fontsize=8.5,
                     va="center", weight="bold", zorder=6)
    if cl or bx:
        rows.append((label, last_mean(cl) if cl else float("nan"),
                     last_mean(bx) if bx else float("nan"), len(bx or cl)))

# bottom-left: adaptive knee L (only the two adaptive methods log it)
for fn, label, c, ls in CURVES[2:]:
    mk = "gtpo_ema_adaptlen_gated/mean_L" if "gated" in fn else "gtpo_ema_adaptlen/mean_L"
    L = col(os.path.join(HERE, fn), mk)
    if L:
        ys = rolling(L); axL.plot(range(len(ys)), ys, color=c, ls=ls, lw=2.0,
                                  label=f"{label} (L→{last_mean(L):.0f})")

# bottom-right: penalty activity (abs mean) + gate fraction for the gated one
for fn, label, c, ls in CURVES[2:]:
    base = "gtpo_ema_adaptlen_gated" if "gated" in fn else "gtpo_ema_adaptlen"
    pa = col(os.path.join(HERE, fn), f"{base}/pen_rel_absmean")
    if pa:
        ys = rolling(pa); axp.plot(range(len(ys)), ys, color=c, ls=ls, lw=2.0,
                                   label=f"{label} pen|·| ({last_mean(pa):.3f})")
gf = col(os.path.join(HERE, "train_gtpo_ema_adaptlen_gated.log"),
         "gtpo_ema_adaptlen_gated/gate_frac")
if gf:
    ys = rolling(gf); axp.plot(range(len(ys)), ys, color="#a855f7", ls="-.", lw=1.6,
                               label=f"gate_frac ({last_mean(gf):.2f})")

axl.set_title("completion length (rolling-30, mean tokens/gen)", fontsize=11, weight="bold")
axl.set_xlabel("step"); axl.set_ylabel("tokens"); axl.grid(alpha=0.3); axl.legend(fontsize=8.5, loc="upper left")
axb.set_title("answer_boxed reward (rolling-30)\n+3 correct / -1.5 wrong / 0 none", fontsize=11, weight="bold")
axb.set_xlabel("step"); axb.set_ylabel("boxed reward"); axb.axhline(0, color="#cbd5e1", lw=0.6, ls="--")
axb.grid(alpha=0.3); axb.legend(fontsize=8.5, loc="lower right")
axL.set_title("adaptive knee L over training (rolling-30)", fontsize=11, weight="bold")
axL.set_xlabel("step"); axL.set_ylabel("L (tokens)"); axL.grid(alpha=0.3); axL.legend(fontsize=9, loc="best")
axp.set_title("penalty activity |pen_rel| + gate fraction (rolling-30)", fontsize=11, weight="bold")
axp.set_xlabel("step"); axp.set_ylabel("value"); axp.grid(alpha=0.3); axp.legend(fontsize=9, loc="best")

fig.suptitle("exp_058 — ADAPTIVE length penalty (Qwen3-4B-Base, Big-Math int-2000)\n"
             "knee L=max((Lmin+Lmax)/2, Lmean) per group; penalty in [-0.5,0]. "
             "always-on vs gated (t=0/0.5 difficulty gate), vs GRPO & best fixed-L.",
             fontsize=10, weight="bold")
out = os.path.join(HERE, "figures", "exp058_adaptlen.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(out, dpi=140)
print(f"saved {out}")
print(f"\n{'method':28s} {'L50_len':>9s} {'L50_boxed':>10s} {'steps':>6s}")
for label, ml, mb, n in rows:
    print(f"{label:28s} {ml:>9.0f} {mb:>+10.2f} {n:>6d}")
