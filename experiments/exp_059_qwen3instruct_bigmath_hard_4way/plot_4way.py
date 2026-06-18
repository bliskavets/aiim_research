"""
exp_058 — 4-method comparison on Qwen3-4B-BASE (Big-Math int-2000).
First valid shaping test on a non-instruction-tuned model (FIXED non-bypassed
trainers; shaped metrics logged on every step).

Two panels tell the whole story:
  1. answer_boxed reward (correctness): grpo and grpo_s_entropy steadily LEARN
     (the base model has headroom, unlike saturated instruct); gtpo_conf lags;
     gtpo_ema_flipped COLLAPSES to ~0.
  2. completion length: gtpo_ema_flipped's length EXPLODES (~640 -> 3400+ tokens)
     — the per-token EMA bonus rewards flat/uncertain tokens -> the model rambles
     and stops emitting a parseable \\boxed{} answer (same length-explosion failure
     as exp_043/047). The others stay ~640.
"""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CURVES = [
    ("train_grpo.log",              "grpo baseline",       "#64748b"),
    ("train_grpo_s_entropy.log",    "grpo_s_entropy",      "#d97706"),
    ("train_gtpo_conf.log",         "gtpo_conf",           "#059669"),
    ("train_gtpo_ema_flipped.log",  "gtpo_ema_flipped",    "#dc2626"),
]


def rolling(xs, w=30):
    return [sum(xs[max(0, i-w+1):i+1]) / (i-max(0, i-w+1)+1) for i in range(len(xs))]


def col(p, k):
    if not os.path.exists(p): return None
    return [float(m.group(1)) for m in re.finditer(re.escape(k)+r"':\s*([-\d.eE+]+)", open(p).read())]


fig, (axb, axl) = plt.subplots(1, 2, figsize=(15, 6))
for fn, label, c in CURVES:
    bx = col(os.path.join(HERE, fn), "reward_answer_boxed/mean")
    cl = col(os.path.join(HERE, fn), "completions/mean_length")
    if bx:
        ys = rolling(bx); axb.plot(range(len(ys)), ys, color=c, lw=2.0, label=f"{label} (n={len(bx)})")
        axb.text(len(ys)+3, ys[-1], f" {sum(bx[-100:])/min(100,len(bx)):+.2f}", color=c, fontsize=8, va="center", weight="bold")
    if cl:
        ys = rolling(cl); axl.plot(range(len(ys)), ys, color=c, lw=2.0, label=label)

axb.set_title("answer_boxed reward (rolling-30)\n+3 correct / -1.5 wrong / 0 none", fontsize=11, weight="bold")
axb.set_xlabel("step"); axb.set_ylabel("boxed reward"); axb.axhline(0, color="#94a3b8", lw=0.6, ls="--"); axb.grid(alpha=0.3); axb.legend(fontsize=8.5, loc="center right")
axl.set_title("completion length (rolling-30, mean tokens/gen)", fontsize=11, weight="bold")
axl.set_xlabel("step"); axl.set_ylabel("tokens"); axl.grid(alpha=0.3); axl.legend(fontsize=8.5, loc="upper left")

fig.suptitle("exp_058 — Qwen3-4B-BASE (pretrained), Big-Math int-2000, 4 methods (shaping ACTUALLY applied)\n"
             "Base model HAS headroom: grpo & grpo_s_entropy LEARN (boxed 0.7->1.9). gtpo_conf lags. "
             "gtpo_ema_flipped COLLAPSES via length-explosion (640->3400 tok, boxed->0).",
             fontsize=10, weight="bold")
out = os.path.join(HERE, "figures", "exp058_4way_base_model.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.91]); fig.savefig(out, dpi=140)
print(f"saved {out}")
