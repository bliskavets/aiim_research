"""
exp_064 — O+ vs O- overlaid: C(t) and logprob(t) vs position on one figure
(2 panels), with ±std bands. O+ = correct rollouts, O- = incorrect.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
BIN = 64
DS = sys.argv[1] if len(sys.argv) > 1 else "gsm8k"
GROUPS = [("correct", "O+ (correct)", "#16a34a"), ("incorrect", "O- (incorrect)", "#dc2626")]


def binned(n, s, s2):
    P = len(n); nb = P // BIN
    n = n[:nb*BIN].reshape(nb, BIN).sum(1)
    s = s[:nb*BIN].reshape(nb, BIN).sum(1)
    s2 = s2[:nb*BIN].reshape(nb, BIN).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / n, np.nan)
        std = np.sqrt(np.clip(np.where(n > 0, s2 / n - mean**2, np.nan), 0, None))
    return (np.arange(nb) + 0.5) * BIN, mean, std, n


d = np.load(os.path.join(HERE, "diag", f"posstats_{DS}.npz"))
gen_calls = int(d["gen_calls"][0])
fig, (axC, axL) = plt.subplots(1, 2, figsize=(14, 5))
for g, lab, c in GROUPS:
    xC, mC, sC, nC = binned(d[f"{g}_n"], d[f"{g}_C"], d[f"{g}_C2"])
    xL, mL, sL, _ = binned(d[f"{g}_n"], d[f"{g}_lp"], d[f"{g}_lp2"])
    v = nC > 0
    axC.plot(xC[v], mC[v], color=c, lw=2.2, label=lab)
    axC.fill_between(xC[v], (mC-sC)[v], (mC+sC)[v], color=c, alpha=0.10)
    axL.plot(xL[v], mL[v], color=c, lw=2.2, label=lab)
    axL.fill_between(xL[v], (mL-sL)[v], (mL+sL)[v], color=c, alpha=0.10)
axC.set_title("C(t) = -mean_topk log p", fontsize=11, weight="bold")
axC.set_xlabel("token position"); axC.set_ylabel("C (±std)"); axC.grid(alpha=0.3); axC.legend(fontsize=9)
axL.set_title("logprob(t) of sampled token", fontsize=11, weight="bold")
axL.set_xlabel("token position"); axL.set_ylabel("log p (±std)"); axL.grid(alpha=0.3); axL.legend(fontsize=9)
fig.suptitle(f"exp_064 [{DS}] — O+ vs O- : C(t) and logprob(t) vs position "
             f"(gen_calls={gen_calls}, bin={BIN})", fontsize=12, weight="bold")
out = os.path.join(HERE, "figures", f"exp064_posstats_{DS}_OplusOminus.png")
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out, dpi=140)
print(f"saved {out}")
