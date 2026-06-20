"""
exp_058 — focused 3-way comparison (Qwen3-4B-Base, Big-Math int-2000):
  GRPO baseline           — reference (no shaping)
  gtpo_ema_flipped (bare) — the length-explosion collapse the penalty fixes
  fixed-L gated (L=1536)  — best length-penalty config from the L-sweep
Two panels: completion length and answer_boxed reward (rolling-30).
"""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CURVES = [
    ("train_grpo.log",                        "GRPO baseline",            "#64748b", "-"),
    ("train_gtpo_ema_flipped.log",            "gtpo_ema_flipped (bare)",  "#dc2626", "-"),
    ("train_gtpo_ema_lenpen_gated_L1536.log", "fixed-L gated (L=1536)",   "#2563eb", "-"),
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


fig, (axl, axb) = plt.subplots(1, 2, figsize=(15, 6))
rows = []
for fn, label, c, ls in CURVES:
    p = os.path.join(HERE, fn)
    cl = col(p, "completions/mean_length")
    bx = col(p, "reward_answer_boxed/mean")
    if cl:
        ys = rolling(cl); axl.plot(range(len(ys)), ys, color=c, ls=ls, lw=2.2, label=f"{label} ({last_mean(cl):.0f})")
        axl.text(len(ys)+3, ys[-1], f" {last_mean(cl):.0f}", color=c, fontsize=8.5, va="center", weight="bold")
    if bx:
        ys = rolling(bx); axb.plot(range(len(ys)), ys, color=c, ls=ls, lw=2.2, label=f"{label} ({last_mean(bx):+.2f})")
        axb.text(len(ys)+3, ys[-1], f" {last_mean(bx):+.2f}", color=c, fontsize=8.5, va="center", weight="bold")
    if cl or bx:
        rows.append((label, last_mean(cl) if cl else float("nan"), last_mean(bx) if bx else float("nan"), len(bx or cl)))

axl.set_title("completion length (rolling-30, mean tokens/gen)", fontsize=11, weight="bold")
axl.set_xlabel("step"); axl.set_ylabel("tokens"); axl.grid(alpha=0.3); axl.legend(fontsize=9, loc="upper left")
axb.set_title("answer_boxed reward (rolling-30)\n+3 correct / -1.5 wrong / 0 none", fontsize=11, weight="bold")
axb.set_xlabel("step"); axb.set_ylabel("boxed reward"); axb.axhline(0, color="#cbd5e1", lw=0.6, ls="--")
axb.grid(alpha=0.3); axb.legend(fontsize=9, loc="lower right")

fig.suptitle("exp_058 — GRPO baseline vs gtpo_ema_flipped (bare) vs fixed-L gated (L=1536)\n"
             "Qwen3-4B-Base, Big-Math int-2000. Bare gtpo_ema_flipped collapses via length explosion; "
             "the gated fixed-L penalty recovers it to near-GRPO quality.",
             fontsize=10, weight="bold")
out = os.path.join(HERE, "figures", "exp058_baseline_compare.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(out, dpi=140)
print(f"saved {out}")
print(f"\n{'method':28s} {'L50_len':>9s} {'L50_boxed':>10s} {'steps':>6s}")
for label, ml, mb, n in rows:
    print(f"{label:28s} {ml:>9.0f} {mb:>+10.2f} {n:>6d}")
