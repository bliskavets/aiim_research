"""
exp_064 — 3 separate figures (overall / O+ / O-) of C(t) and logprob(t) vs position.
O+ = successful (correct) rollouts, O- = unsuccessful (incorrect) — the paper's
terminal-reward partition (arXiv:2508.04349), which is exactly the correct/incorrect
split we logged. Each figure has 2 panels: C(t)±std and logprob(t)±std (binned).
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
BIN = 64
DS = sys.argv[1] if len(sys.argv) > 1 else "gsm8k"
GROUPS = [("all", "overall", "#334155"), ("correct", "O+ (correct)", "#16a34a"),
          ("incorrect", "O- (incorrect)", "#dc2626")]


def binned(n, s, s2):
    P = len(n); nb = P // BIN
    n = n[:nb*BIN].reshape(nb, BIN).sum(1)
    s = s[:nb*BIN].reshape(nb, BIN).sum(1)
    s2 = s2[:nb*BIN].reshape(nb, BIN).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / n, np.nan)
        std = np.sqrt(np.clip(np.where(n > 0, s2 / n - mean**2, np.nan), 0, None))
    x = (np.arange(nb) + 0.5) * BIN
    return x, mean, std, n


d = np.load(os.path.join(HERE, "diag", f"posstats_{DS}.npz"))
gen_calls = int(d["gen_calls"][0])

for g, glab, c in GROUPS:
    xc, mC, sC, nc = binned(d[f"{g}_n"], d[f"{g}_C"], d[f"{g}_C2"])
    xl, mL, sL, _ = binned(d[f"{g}_n"], d[f"{g}_lp"], d[f"{g}_lp2"])
    valid = nc > 0
    fig, (axC, axL) = plt.subplots(1, 2, figsize=(14, 5))
    axC.plot(xc[valid], mC[valid], color=c, lw=2)
    axC.fill_between(xc[valid], (mC-sC)[valid], (mC+sC)[valid], color=c, alpha=0.15)
    axC.set_title(f"C(t) = -mean_topk log p", fontsize=11, weight="bold")
    axC.set_xlabel("token position"); axC.set_ylabel("C (±std)"); axC.grid(alpha=0.3)
    axL.plot(xl[valid], mL[valid], color=c, lw=2)
    axL.fill_between(xl[valid], (mL-sL)[valid], (mL+sL)[valid], color=c, alpha=0.15)
    axL.set_title(f"logprob(t) of sampled token", fontsize=11, weight="bold")
    axL.set_xlabel("token position"); axL.set_ylabel("log p (±std)"); axL.grid(alpha=0.3)
    fig.suptitle(f"exp_064 [{DS}] — {glab} : C(t) and logprob(t) vs position "
                 f"(gen_calls={gen_calls}, bin={BIN})", fontsize=12, weight="bold")
    tag = {"all": "overall", "correct": "Oplus", "incorrect": "Ominus"}[g]
    out = os.path.join(HERE, "figures", f"exp064_posstats_{DS}_{tag}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")
    m = nc > 0
    print(f"  {glab}: C {mC[m][0]:.2f}->{mC[m][-1]:.2f}  logp {mL[m][0]:.3f}->{mL[m][-1]:.3f}  "
          f"tokens={int(nc.sum())}")
