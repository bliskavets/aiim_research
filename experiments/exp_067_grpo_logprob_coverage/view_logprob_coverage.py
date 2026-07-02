"""
view_logprob_coverage.py — inspect probability-mass coverage of top-k from dumped
GRPO rollout logprobs (exp_067). For a dump dir diag/lpdump_<ds>:
  - coverage(k) = Σ_{j≤k} p_(j)  (p sorted desc) per token; report mean/min/max/
    percentiles across all valid tokens, for k = 1..K.
  - top-p (nucleus): for p thresholds, distribution of k needed to reach mass p.
  - fraction of SAMPLED tokens whose prob is within the top-k.
Outputs a table (stdout) + figures/exp067_coverage_<ds>.png. Basis for adaptive
top-k / nucleus / adaptive-nucleus design.

Usage: python view_logprob_coverage.py <ds>   (ds = dump tag, e.g. gsm8k)
"""
import os, sys, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
DS = sys.argv[1] if len(sys.argv) > 1 else "gsm8k"
DUMP = os.path.join(HERE, "diag", f"lpdump_{DS}")


def load_tokens(dump):
    """Return (cov (N,K) cumulative coverage per valid token, sampled_p (N,) prob of
    sampled token, K)."""
    lps, samps = [], []
    files = sorted(glob.glob(os.path.join(dump, "step_*.npz")))
    for f in files:
        d = np.load(f)
        tk = d["topk_lp"].astype(np.float32)        # (G,T,K) sorted-desc logprob
        cm = d["completion_mask"].astype(bool)      # (G,T)
        sp = d["sampled_lp"].astype(np.float32)     # (G,T)
        G, T, K = tk.shape
        m = cm[:, :T]
        lps.append(tk[m])                           # (n_valid, K)
        samps.append(sp[:, :T][m])
    if not lps:
        raise SystemExit(f"no dumps in {dump}")
    LP = np.concatenate(lps, 0)                     # (N, K) logprob
    SP = np.concatenate(samps, 0)                   # (N,)
    P = np.exp(LP)                                  # (N, K) prob (sorted desc)
    cov = np.cumsum(P, axis=1)                      # (N, K) coverage(k)
    return cov, np.exp(SP), P.shape[1], len(files)


cov, sampled_p, K, nfiles = load_tokens(DUMP)
N = cov.shape[0]
print(f"[{DS}] files={nfiles}  valid tokens={N}  K={K}")

ks = [k for k in [1, 2, 3, 5, 10, 20, 40, 64, 128] if k <= K]
pcts = [5, 25, 50, 75, 95]
print(f"\ncoverage(k) = prob mass in top-k  (across {N} tokens)")
print(f"{'k':>4} {'mean':>7} {'min':>7} {'p5':>7} {'p50':>7} {'p95':>7} {'max':>7}")
for k in ks:
    c = cov[:, k-1]
    print(f"{k:>4} {c.mean():>7.3f} {c.min():>7.3f} "
          f"{np.percentile(c,5):>7.3f} {np.percentile(c,50):>7.3f} "
          f"{np.percentile(c,95):>7.3f} {c.max():>7.3f}")

print(f"\nk needed to reach top-p mass  (nucleus)")
print(f"{'p':>5} {'mean_k':>7} {'p50':>5} {'p95':>5} {'%>K':>6}")
for p in [0.5, 0.8, 0.9, 0.95, 0.99]:
    reached = cov >= p                              # (N,K)
    has = reached.any(1)
    kneed = np.where(has, reached.argmax(1) + 1, K + 1)
    over = 100.0 * (~has).mean()
    kk = kneed[has]
    print(f"{p:>5} {kk.mean():>7.1f} {int(np.percentile(kk,50)):>5} {int(np.percentile(kk,95)):>5} {over:>6.1f}")

# fraction of sampled tokens within top-k (sampled prob >= k-th largest prob)
print(f"\nfraction of SAMPLED tokens within top-k")
kth = np.exp(np.sort(cov, axis=1))  # not used; compute from P directly below
# recompute P from cov (P_1=cov_1, P_j=cov_j-cov_{j-1})
P = np.diff(np.concatenate([np.zeros((N, 1)), cov], axis=1), axis=1)
for k in [1, 3, 5, 10, 20]:
    if k <= K:
        thresh = P[:, k-1]                          # k-th largest prob
        frac = (sampled_p >= thresh - 1e-9).mean()
        print(f"  top-{k:<3}: {frac*100:5.1f}%")

# ---- figure ----
fig, (axc, axk) = plt.subplots(1, 2, figsize=(15, 5.5))
kx = np.arange(1, K + 1)
mean_c = cov.mean(0)
axc.plot(kx, mean_c, color="#2563eb", lw=2, label="mean")
axc.fill_between(kx, np.percentile(cov, 5, 0), np.percentile(cov, 95, 0), color="#2563eb", alpha=0.15, label="p5–p95")
axc.plot(kx, np.percentile(cov, 50, 0), color="#16a34a", lw=1.5, ls="--", label="median")
axc.plot(kx, cov.min(0), color="#dc2626", lw=1, ls=":", label="min")
for kv in [5, 20]:
    axc.axvline(kv, color="#94a3b8", lw=0.7, ls="--")
axc.set_title(f"[{DS}] probability mass covered by top-k", fontsize=12, weight="bold")
axc.set_xlabel("k"); axc.set_ylabel("coverage Σ p_(1..k)"); axc.set_ylim(0, 1.02); axc.grid(alpha=0.3); axc.legend(fontsize=9)

# k-needed histogram for p=0.9
reached = cov >= 0.9
kneed = np.where(reached.any(1), reached.argmax(1) + 1, K + 1)
axk.hist(kneed, bins=np.arange(1, K + 2), color="#7c3aed", alpha=0.85)
axk.set_title(f"[{DS}] k needed to cover top-p=0.9 (nucleus)", fontsize=12, weight="bold")
axk.set_xlabel("k for mass ≥ 0.9"); axk.set_ylabel("token count"); axk.grid(alpha=0.3)
axk.set_xlim(0, min(K + 1, 60))

out = os.path.join(HERE, "figures", f"exp067_coverage_{DS}.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.suptitle(f"exp_067 — GRPO rollout logprob coverage [{DS}] "
             f"(N={N} tokens, K={K})", fontsize=12, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out, dpi=140)
print(f"\nsaved {out}")
