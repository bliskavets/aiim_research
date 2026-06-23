"""
analyze_diag.py — why does bare gtpo_ema_flipped explode in length?

Diagnostic finding: under unsloth's B=1 microbatching, compute_loss shapes ONE
completion at a time, but the flipped-EMA shaping is designed for a full group of
num_generations completions. With a single completion the per-position group
normalization (Σ over O+/O- at each t) collapses to 1 and the per-polarity z-norm
divides by ~0 std → the shaped advantage degenerates to a per-sequence CONSTANT
(occasionally blown up to ±6 by numerical noise) that (a) does not reward correct
completions, (b) drifts to INVERT the reward signal, and (c) is mildly positively
correlated with length. Hence: ramble longer, abandon concise correct answers.

This script works at the SEQUENCE level (the per-token bins are constant within a
completion, so per-position structure is not the mechanism). It produces:
  figures/exp058_diag_mechanism.png  (4 panels)
"""
import os, json, math
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
SHAPE = os.path.join(HERE, "diag", "diag_gtpo_ema_flipped_shape.jsonl")


def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def roll(xs, w=21):
    return [sum(xs[max(0, i-w+1):i+1]) / (i-max(0, i-w+1)+1) for i in range(len(xs))]


def corr(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs)/n, sum(ys)/n
    cov = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
    return cov/math.sqrt(vx*vy) if vx > 0 and vy > 0 else float("nan")


rows = load(SHAPE)
# B=1: one completion per record
D = [dict(step=r["step"], L=r["len"][0], sa=r["seq_adv"][0],
          pos=r["is_pos"][0], madv=r["mean_tok_adv"][0]) for r in rows]
steps_sorted = sorted({d["step"] for d in D})

# per-step aggregates
byk = defaultdict(list)
for d in D:
    byk[d["step"]].append(d)
mlen, advP, advN = [], [], []
for s in steps_sorted:
    g = byk[s]
    mlen.append(sum(d["L"] for d in g)/len(g))
    p = [d["madv"] for d in g if d["sa"] > 0]
    n = [d["madv"] for d in g if d["sa"] < 0]
    advP.append(sum(p)/len(p) if p else float("nan"))
    advN.append(sum(n)/len(n) if n else float("nan"))


def fill(xs):                                  # forward-fill NaN for a smooth line
    out, last = [], float("nan")
    for x in xs:
        if x == x:
            last = x
        out.append(last)
    return out


fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=(15, 11))

# A: length + boxed over steps
axA.plot(steps_sorted, roll(mlen), color="#dc2626", lw=2.2, label="mean completion length")
axA.axhline(3584, color="#fca5a5", lw=0.8, ls="--", label="max_completion (3584)")
axA.set_title("A. completion length over training (the explosion)", fontsize=10, weight="bold")
axA.set_xlabel("step"); axA.set_ylabel("tokens"); axA.grid(alpha=0.3); axA.legend(fontsize=8.5, loc="best")

# B: mean shaped advantage O+ vs O- over training -> the inversion crossover
axB.plot(steps_sorted, roll(fill(advP)), color="#2563eb", lw=2.2, label="genuine O+ (correct, seq_adv>0)")
axB.plot(steps_sorted, roll(fill(advN)), color="#dc2626", lw=2.2, label="genuine O- (wrong, seq_adv<0)")
axB.axhline(0, color="#94a3b8", lw=0.6)
axB.set_title("B. mean shaped advantage: O+ vs O- over training\n"
              "O+ pinned ~-0.47; O- drifts UP -> crossover = reward INVERSION", fontsize=10, weight="bold")
axB.set_xlabel("step"); axB.set_ylabel("mean per-token advantage"); axB.grid(alpha=0.3); axB.legend(fontsize=8.5, loc="best")

# C: corr(length, mean_adv) in a sliding window over steps (signal-only)
win = 60
cs, cv = [], []
for i, s in enumerate(steps_sorted):
    lo = s - win
    w = [d for d in D if lo <= d["step"] <= s and d["sa"] != 0]
    if len(w) >= 15:
        c = corr([d["L"] for d in w], [d["madv"] for d in w])
        if c == c:
            cs.append(s); cv.append(c)
axC.plot(cs, cv, color="#7c3aed", lw=2.0, label=f"corr(length, shaped-adv), {win}-step window")
axC.axhline(0, color="#94a3b8", lw=0.6, ls="--")
axC.set_title("C. corr(completion length, shaped advantage)\nconsistently POSITIVE -> a length incentive", fontsize=10, weight="bold")
axC.set_xlabel("step"); axC.set_ylabel("Pearson r"); axC.grid(alpha=0.3); axC.legend(fontsize=8.5, loc="best")

# D: distribution of per-completion mean advantage (degeneracy: pinned values + blow-up tail)
vals = [d["madv"] for d in D]
axD.hist(vals, bins=80, color="#0ea5e9", alpha=0.85)
axD.axvline(0, color="#94a3b8", lw=0.6)
blow = sum(1 for v in vals if abs(v) > 3) / len(vals)
axD.set_title(f"D. distribution of per-completion shaped advantage\n"
              f"z-norm over near-constant single completion -> spikes + blow-up tail "
              f"(|adv|>3 in {blow*100:.0f}% of completions, max {max(abs(v) for v in vals):.1f})",
              fontsize=10, weight="bold")
axD.set_xlabel("mean per-token shaped advantage (per completion)"); axD.set_ylabel("count"); axD.grid(alpha=0.3)

fig.suptitle("exp_058 — gtpo_ema_flipped length-explosion diagnostic (Qwen3-4B-Base, Big-Math int-2000, 420 steps)\n"
             "ROOT CAUSE: unsloth B=1 microbatching collapses the group-normalized shaping -> degenerate, "
             "reward-inverting, length-correlated advantages.",
             fontsize=11, weight="bold")
out = os.path.join(HERE, "figures", "exp058_diag_mechanism.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(out, dpi=140)
print(f"saved {out}")

# numeric summary
def winstat(lo, hi):
    w = [d for d in D if lo <= d["step"] <= hi and d["sa"] != 0]
    p = [d["madv"] for d in w if d["sa"] > 0]; n = [d["madv"] for d in w if d["sa"] < 0]
    return (sum(p)/len(p) if p else float("nan"), sum(n)/len(n) if n else float("nan"),
            corr([d["L"] for d in w], [d["madv"] for d in w]),
            sum(d["L"] for d in w)/len(w))
mx = max(steps_sorted)
for nm, lo, hi in [("early", 0, 100), ("mid", 150, 250), ("late", 300, mx)]:
    p, n, c, L = winstat(lo, hi)
    print(f"{nm:5s}: meanL={L:.0f}  O+adv={p:+.2f}  O-adv={n:+.2f}  (O+−O−={p-n:+.2f})  corr(L,adv)={c:+.2f}")
