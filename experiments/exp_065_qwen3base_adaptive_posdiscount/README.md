# exp_065 — adaptive pos_discount variants (base FIXED λ=0.7)

Shortlist of adaptive exploration-bonus discounts g_{i,t} (multiplies α₂ only) on
top of gtpo_ema_flipped(FIXED) λ=0.7, grounded in exp_064's per-position C/logprob
profiles. Batch-relative stats (m,sd,C_ref,s_ref over valid tokens) => self-norming.

- **adisc_p1**  — position-only, hyperbolic w/ floor: g = f + (1−f)·τ/(τ+t) (f=0.3, τ=1024).
- **adisc_c1**  — surprisal weight: g = clip(s/s_ref, g_min, 1), s = −logp(o_t).
- **adisc_pc1** — position × prefix-decisiveness: [f+(1−f)τ/(τ+t)]·σ((EMA(C)−m)/sd).
- **adisc_pc2** — early-decisiveness (boost allowed): clip((C/C_ref)·τ/(τ+t), g_min, 1.5).

All group-visible FIXED pattern (no B=1 bug); pure g-functions unit-tested
(`tests/test_adaptive_discount.py`). 4 variants × 4 datasets (gsm8k/math500/bigmath/
omnimath) = 16 runs, same hyperparams as exp_058/062. Overlay vs GRPO + pos_discount λ0.7.

## Run
```
./run_adaptive.sh        # or chain_adaptive.sh (waits for exp_064 posstats2, smokes, then runs)
```
