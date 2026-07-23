import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sig=lambda x: 1.0/(1.0+math.exp(-x)) if -30<x<30 else (0.0 if x<0 else 1.0)
pairs=[]
for s in (42,7,123):
    d=json.load(open(f"results_aaai2027/judge_vs_gold_s{s}.json"))
    for p in d["per_problem"]:
        for c in p["candidates"]:
            if c["score"] is not None:
                pairs.append((sig(c["score"]), c["correct"]))
B=15
conf=np.zeros(B); acc=np.zeros(B); cnt=np.zeros(B)
for cf,y in pairs:
    b=min(B-1,int(cf*B)); conf[b]+=cf; acc[b]+=y; cnt[b]+=1
mc=np.where(cnt>0,conf/np.maximum(cnt,1),np.nan)
ma=np.where(cnt>0,acc/np.maximum(cnt,1),np.nan)
brier=np.mean([(cf-y)**2 for cf,y in pairs])
ece=np.nansum(cnt*np.abs(mc-ma))/cnt.sum()
print(f"pooled n={len(pairs)} brier={brier:.4f} ece={ece:.4f}")

fig,ax=plt.subplots(1,2,figsize=(7.4,3.1),gridspec_kw={"width_ratios":[1,1]})
# reliability
ax[0].plot([0,1],[0,1],"--",color="gray",lw=1,label="perfect calibration")
m=cnt>0
ax[0].plot(mc[m],ma[m],"o-",color="#1f6fb2",lw=1.8,ms=4,label="self-judge margin")
ax[0].set_xlabel("predicted confidence  $\\sigma(s)$"); ax[0].set_ylabel("empirical accuracy")
ax[0].set_xlim(0,1); ax[0].set_ylim(0,1); ax[0].set_title(f"Reliability (Brier {brier:.3f}, ECE {ece:.3f})",fontsize=9)
ax[0].legend(fontsize=7,loc="upper left"); ax[0].grid(alpha=.25)
# mass histogram
centers=(np.arange(B)+0.5)/B
ax[1].bar(centers,cnt/cnt.sum(),width=1/B*0.9,color="#8bbfe0",edgecolor="#1f6fb2",lw=.5)
ax[1].set_xlabel("predicted confidence  $\\sigma(s)$"); ax[1].set_ylabel("fraction of candidates")
ax[1].set_xlim(0,1); ax[1].set_title("Confidence mass",fontsize=9); ax[1].grid(alpha=.25)
plt.tight_layout()
out="/root/aiim_2/aiim_research/papers/aaai27/aaai27_submission/figures/judge_reliability.pdf"
plt.savefig(out,bbox_inches="tight"); print("saved",out)
