"""
analyze_posstats.py — turn diag/posstats_<ds>.npz into per-position profiles of
C_{i,t} (confidence) and logprob_{i,t}, to inform an adaptive pos_discount.
For each dataset: 3 panels (C vs pos, logprob vs pos, coverage=count vs pos),
overlaying all / correct / incorrect rollouts, with ±std bands. Positions are
binned (default 64-token bins) for a readable curve.
"""
import os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
BIN = 64


def load(npz):
    d = np.load(npz)
    groups = {}
    for g in ("all", "correct", "incorrect"):
        n = d[f"{g}_n"]; C = d[f"{g}_C"]; C2 = d[f"{g}_C2"]; lp = d[f"{g}_lp"]; lp2 = d[f"{g}_lp2"]
        groups[g] = dict(n=n, C=C, C2=C2, lp=lp, lp2=lp2)
    return groups, int(d["gen_calls"][0]) if "gen_calls" in d else -1


def binned(n, s, s2, bins):
    """mean/std per bin from per-position count/sum/sumsq."""
    P = len(n); nb = P // bins
    n = n[:nb*bins].reshape(nb, bins).sum(1)
    s = s[:nb*bins].reshape(nb, bins).sum(1)
    s2 = s2[:nb*bins].reshape(nb, bins).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / n, np.nan)
        var = np.where(n > 0, s2 / n - mean**2, np.nan)
    std = np.sqrt(np.clip(var, 0, None))
    x = (np.arange(nb) + 0.5) * bins
    return x, mean, std, n


COLORS = {"all": "#334155", "correct": "#16a34a", "incorrect": "#dc2626"}

for npz in sorted(glob.glob(os.path.join(HERE, "diag", "posstats_*.npz"))):
    ds = os.path.basename(npz)[len("posstats_"):-len(".npz")]
    groups, gen_calls = load(npz)
    fig, (axC, axL, axN) = plt.subplots(1, 3, figsize=(18, 5))
    for g, gr in groups.items():
        x, mC, sC, nb = binned(gr["n"], gr["C"], gr["C2"], BIN)
        _, mL, sL, _ = binned(gr["n"], gr["lp"], gr["lp2"], BIN)
        c = COLORS[g]
        axC.plot(x, mC, color=c, lw=2, label=g); axC.fill_between(x, mC-sC, mC+sC, color=c, alpha=0.12)
        axL.plot(x, mL, color=c, lw=2, label=g); axL.fill_between(x, mL-sL, mL+sL, color=c, alpha=0.12)
        axN.plot(x, nb, color=c, lw=2, label=g)
    axC.set_title(f"{ds}: confidence C = -mean_topk log p  vs position", fontsize=10, weight="bold")
    axC.set_xlabel("token position"); axC.set_ylabel("C (±std)"); axC.grid(alpha=0.3); axC.legend(fontsize=9)
    axL.set_title(f"{ds}: logprob of sampled token vs position", fontsize=10, weight="bold")
    axL.set_xlabel("token position"); axL.set_ylabel("log p (±std)"); axL.grid(alpha=0.3); axL.legend(fontsize=9)
    axN.set_title(f"{ds}: coverage (tokens observed) vs position", fontsize=10, weight="bold")
    axN.set_xlabel("token position"); axN.set_ylabel("token count"); axN.grid(alpha=0.3); axN.legend(fontsize=9)
    fig.suptitle(f"exp_064 — GRPO per-position C / logprob profile [{ds}] (gen_calls={gen_calls}, bin={BIN})",
                 fontsize=11, weight="bold")
    out = os.path.join(HERE, "figures", f"exp064_posstats_{ds}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out, dpi=140)
    print(f"saved {out}")
    # quick numeric summary for adaptive-discount design (all rollouts)
    x, mC, _, nb = binned(groups["all"]["n"], groups["all"]["C"], groups["all"]["C2"], BIN)
    valid = nb > 0
    print(f"  [{ds}] C(pos): first-bin={mC[valid][0]:.3f}  last-valid-bin={mC[valid][-1]:.3f}  "
          f"min={np.nanmin(mC):.3f} max={np.nanmax(mC):.3f}")
