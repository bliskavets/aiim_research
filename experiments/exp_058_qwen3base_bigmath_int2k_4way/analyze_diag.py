"""
analyze_diag.py — why does bare gtpo_ema_flipped explode in length?
Reads diag/diag_gtpo_ema_flipped_{shape,gens}.jsonl and produces:
  figures/exp058_diag_mechanism.png  (4 panels)
  DIAG_LENGTH_EXPLOSION.md           (findings + sample generations)

Panels:
  A. shaped token-advantage vs RELATIVE position in the completion (O+ / O-),
     averaged over early / mid / late training windows.
  B. EMA-confidence & raw confidence vs relative position (same windows).
  C. the key relationship: per-position shaped token-advantage vs EMA-confidence
     (pooled over positions & steps) — tests "are low-confidence tokens rewarded?"
  D. completion length over steps + per-completion (length vs sum-of-token-adv)
     correlation over steps.
"""
import os, json, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
SHAPE = os.path.join(HERE, "diag", "diag_gtpo_ema_flipped_shape.jsonl")
GENS  = os.path.join(HERE, "diag", "diag_gtpo_ema_flipped_gens.jsonl")
NB = 10


def load(path):
    rows = []
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                try: rows.append(json.loads(line))
                except Exception: pass
    return rows


def nanmean_cols(list_of_lists):
    """column-wise mean ignoring NaN/None; returns NB-list."""
    out = []
    for b in range(NB):
        vals = [r[b] for r in list_of_lists if r and r[b] is not None and not (isinstance(r[b], float) and math.isnan(r[b]))]
        out.append(sum(vals)/len(vals) if vals else float("nan"))
    return out


shape = load(SHAPE)
gens  = load(GENS)
steps = [r["step"] for r in shape] if shape else []
maxstep = max(steps) if steps else 0
# windows
def window(rows, lo, hi):
    return [r for r in rows if lo <= r["step"] <= hi]
W = [("early (0-25%)", 0, maxstep*0.25),
     ("mid (35-65%)",  maxstep*0.35, maxstep*0.65),
     ("late (75-100%)",maxstep*0.75, maxstep)]
WC = ["#16a34a", "#d97706", "#dc2626"]

fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=(15, 11))
xb = [(b+0.5)/NB for b in range(NB)]

# Panel A: token-advantage vs position, O+/O- per window
for (name, lo, hi), c in zip(W, WC):
    ws = window(shape, lo, hi)
    if not ws: continue
    pos = nanmean_cols([r["tok_adv_bins_pos"] for r in ws])
    neg = nanmean_cols([r["tok_adv_bins_neg"] for r in ws])
    axA.plot(xb, pos, color=c, ls="-",  lw=2.0, marker="o", ms=3, label=f"O+ {name}")
    axA.plot(xb, neg, color=c, ls="--", lw=2.0, marker="s", ms=3, label=f"O- {name}")
axA.axhline(0, color="#94a3b8", lw=0.6)
axA.set_title("A. shaped token-advantage vs relative position\n(solid=O+ correct, dashed=O- incorrect)", fontsize=10, weight="bold")
axA.set_xlabel("relative position in completion (0=start,1=end)"); axA.set_ylabel("mean token advantage")
axA.grid(alpha=0.3); axA.legend(fontsize=7.5, ncol=3, loc="best")

# Panel B: EMA & confidence vs position (late window)
for (name, lo, hi), c in zip(W, WC):
    ws = window(shape, lo, hi)
    if not ws: continue
    emap = nanmean_cols([r["ema_bins_pos"] for r in ws])
    eman = nanmean_cols([r["ema_bins_neg"] for r in ws])
    axB.plot(xb, emap, color=c, ls="-",  lw=2.0, marker="o", ms=3, label=f"EMA O+ {name}")
    axB.plot(xb, eman, color=c, ls="--", lw=2.0, marker="s", ms=3, label=f"EMA O- {name}")
axB.set_title("B. EMA-confidence vs relative position\n(C = -mean top-k logp; HIGHER = more peaked/decisive)", fontsize=10, weight="bold")
axB.set_xlabel("relative position"); axB.set_ylabel("EMA(confidence)")
axB.grid(alpha=0.3); axB.legend(fontsize=7.5, ncol=3, loc="best")

# Panel C: token-adv vs EMA, pooled across positions+steps (late window) — the mechanism
def scatter_adv_vs_ema(rows, advkey, emakey, ax, color, label):
    xs, ys = [], []
    for r in rows:
        a, e = r[advkey], r[emakey]
        for b in range(NB):
            if a[b] is not None and e[b] is not None and not (isinstance(a[b],float) and math.isnan(a[b])) and not (isinstance(e[b],float) and math.isnan(e[b])):
                xs.append(e[b]); ys.append(a[b])
    if xs:
        ax.scatter(xs, ys, s=10, alpha=0.35, color=color, label=label)
        # corr
        n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
        cov=sum((xs[i]-mx)*(ys[i]-my) for i in range(n)); vx=sum((x-mx)**2 for x in xs); vy=sum((y-my)**2 for y in ys)
        r=cov/math.sqrt(vx*vy) if vx>0 and vy>0 else float("nan")
        return r
    return float("nan")
late = window(shape, maxstep*0.5, maxstep)
rp = scatter_adv_vs_ema(late, "tok_adv_bins_pos", "ema_bins_pos", axC, "#2563eb", "O+ (correct)")
rn = scatter_adv_vs_ema(late, "tok_adv_bins_neg", "ema_bins_neg", axC, "#dc2626", "O- (incorrect)")
axC.axhline(0, color="#94a3b8", lw=0.6)
axC.set_title(f"C. token-advantage vs EMA-confidence (steps>50%, per position-bin)\n"
              f"corr O+={rp:+.2f}  O-={rn:+.2f}  (negative => LOW-confidence tokens get MORE advantage)",
              fontsize=10, weight="bold")
axC.set_xlabel("EMA(confidence)  [higher = more decisive]"); axC.set_ylabel("shaped token advantage")
axC.grid(alpha=0.3); axC.legend(fontsize=8, loc="best")

# Panel D: length over steps + corr(length, sum_tok_adv) per step
def roll(xs, w=20): return [sum(xs[max(0,i-w+1):i+1])/(i-max(0,i-w+1)+1) for i in range(len(xs))]
mlen = [sum(r["len"])/len(r["len"]) for r in shape]
axD.plot(steps, roll(mlen), color="#dc2626", lw=2.0, label="mean completion length (roll-20)")
axD.set_xlabel("step"); axD.set_ylabel("tokens", color="#dc2626"); axD.tick_params(axis="y", labelcolor="#dc2626")
axD.grid(alpha=0.3)
# correlation length vs sum_tok_adv per step (needs >=2 completions/step; with B=1 microbatch this is per-step over the few logged)
corr_steps, corr_vals = [], []
# aggregate per step across the microbatch records
from collections import defaultdict
bystep = defaultdict(lambda: {"len": [], "sa": []})
for r in shape:
    bystep[r["step"]]["len"] += r["len"]; bystep[r["step"]]["sa"] += r["sum_tok_adv"]
for s in sorted(bystep):
    L=bystep[s]["len"]; SA=bystep[s]["sa"]
    if len(L)>=3:
        n=len(L); mL=sum(L)/n; mS=sum(SA)/n
        cov=sum((L[i]-mL)*(SA[i]-mS) for i in range(n)); vL=sum((x-mL)**2 for x in L); vS=sum((x-mS)**2 for x in SA)
        if vL>0 and vS>0: corr_steps.append(s); corr_vals.append(cov/math.sqrt(vL*vS))
axD2 = axD.twinx()
if corr_vals:
    axD2.plot(corr_steps, roll(corr_vals), color="#2563eb", lw=1.6, label="corr(len, Σtok_adv) (roll-20)")
    axD2.axhline(0, color="#93c5fd", lw=0.6, ls="--")
axD2.set_ylabel("corr(length, Σ token-adv)", color="#2563eb"); axD2.tick_params(axis="y", labelcolor="#2563eb")
axD.set_title("D. length over training + within-step corr(length, Σ token-advantage)", fontsize=10, weight="bold")

fig.suptitle("exp_058 — gtpo_ema_flipped length-explosion diagnostic (Qwen3-4B-Base, Big-Math int-2000)\n"
             "Does the shaping reward low-confidence / later / longer tokens? (panels A-D)",
             fontsize=11, weight="bold")
out = os.path.join(HERE, "figures", "exp058_diag_mechanism.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig(out, dpi=140)
print(f"saved {out}")
print(f"steps={len(shape)} (max {maxstep}); corr O+={rp:+.3f} O-={rn:+.3f}")

# ---- findings doc with sample long-incorrect generations ----
if gens:
    # pick the longest incorrect completions from late training
    late_g = [g for g in gens if g["step"] >= maxstep*0.6]
    samples = []
    for g in late_g:
        for c in g["gens"]:
            if not c["boxed_correct"]:
                samples.append((g["step"], c["len"], c))
    samples.sort(key=lambda x: -x[1])
    print(f"longest late incorrect lengths: {[s[1] for s in samples[:5]]}")
