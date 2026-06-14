"""Quick look at the in-progress exp_058 grpo run on Qwen3-4B-BASE."""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
log = os.path.join(HERE, "train_grpo.log")
t = open(log).read()


def col(k):
    return [float(m.group(1)) for m in re.finditer(re.escape(k) + r"':\s*([-\d.eE+]+)", t)]


reward = col("'reward")
boxed  = col("rewards/reward_answer_boxed/mean")
numeric = col("rewards/reward_answer_numeric/mean")
clen   = col("completions/mean_length")
n = len(reward)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
x = range(n)
ax1.plot(x, reward, "o-", color="#0f766e", lw=1.8, ms=3, label="total reward")
ax1.plot(range(len(boxed)), boxed, "s-", color="#dc2626", lw=1.4, ms=2.5, label="answer_boxed mean (+3 correct/-1.5 wrong)")
ax1.plot(range(len(numeric)), numeric, "^-", color="#d97706", lw=1.2, ms=2.5, label="answer_numeric mean")
ax1.axhline(0, color="#94a3b8", lw=0.6, ls="--")
ax1.set_title(f"exp_058 grpo on Qwen3-4B-BASE — reward ({n} steps)", fontsize=11, weight="bold")
ax1.set_xlabel("step"); ax1.set_ylabel("reward"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

ax2.plot(range(len(clen)), clen, "o-", color="#7c3aed", lw=1.6, ms=3)
ax2.set_title("completion length (mean tokens/gen)", fontsize=11, weight="bold")
ax2.set_xlabel("step"); ax2.set_ylabel("tokens"); ax2.grid(alpha=0.3)

fig.suptitle("exp_058 IN-PROGRESS (grpo only, early) — Qwen3-4B-Base, Big-Math int-2000. "
             "NOTE: launched at ng=8/max_seq=6656 (127 s/it, too slow); relaunching at exp_051 config (ng=4/4096).",
             fontsize=9.5)
out = os.path.join(HERE, "figures", "exp058_current_grpo.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out, dpi=140)
print(f"saved {out} | steps={n} reward[-1]={reward[-1] if reward else 'NA'} boxed[-1]={boxed[-1] if boxed else 'NA'}")
