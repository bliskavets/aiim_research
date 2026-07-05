# exp_075 — final combo: gate + branch + budget (roadmap setup 5/5)

See `analysis/exp055-070_deep_analysis.md` §5.5. The paper-candidate assembly.

**Setup:** zero-variance gate (exp_071) + bounded branching signal h (exp_072) +
length-invariant bonus budget (exp_073, replaces posdisc), λ=0.7, k=5, α₁=0.9, α₂=0.1.
All three principled ingredients, no heuristic decay.

Queued last so per-ingredient results (071–073) arrive first; if a component underperforms
there, re-assembly is cheap (config-level).

**Run:** `bash run_setup.sh`, `python plot_compare.py` → `figures/exp075_final_combo.png`
(vs best posdisc λ0.7 k5 + GRPO). Metrics: `final_combo/gated`, `final_combo/mean_h`.

## Results (300 steps, L50 boxed / len)

| dataset | GRPO | best (posdisc λ0.7 k5) | **final_combo (gate+branch+budget)** |
|---|---|---|---|
| gsm8k    | +2.02 / 414 | +2.49 / 317 | +2.35 / 301 |
| math500  | +0.94 / 942 | **+1.63** / 635 | +1.49 / 585 |
| bigmath  | +1.51 / 622 | **+1.81** / 529 | +1.78 / 578 |
| omnimath | **−0.23** / 957 | −0.33 / 733 | −0.34 / 822 |

**Most BALANCED shaped method** — no dataset collapse, competitive everywhere, and on
omnimath (−0.34) it matches posdisc and is the best-behaved shaped run on the hard set (the
gate removes the zero-variance noise, the bounded branching signal removes the reciprocal
pathology). But it does NOT beat the tuned posdisc overall (below on gsm8k/math500/bigmath).
The three principled ingredients (gate + bounded h + budget) give ROBUSTNESS and a clean
theoretical story rather than a raw win over the well-tuned heuristic — exactly the
safety/principle narrative for the paper, with α₂ still un-tuned for the h-scale.
