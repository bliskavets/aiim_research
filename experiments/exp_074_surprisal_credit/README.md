# exp_074 — surprisal credit (roadmap setup 4/5)

See `analysis/exp055-070_deep_analysis.md` §2 (exp_064), §5.4. Minimal-machinery variant.

**Idea:** exp_064 showed the clean per-token informativeness signal is the sampled token's
surprisal s = −log p(o_t). Setup: `Ã = A_grpo + α₂·g(t)·z_polarity(s)` — reward surprising
tokens in correct rollouts, punish confident tokens in wrong ones, forgive exploration.
Additive on the GRPO scalar (no cold-start dead signal), needs NO top-k forward (one
logprob gather) → cheapest method in the family. posdisc λ0.7 kept.

**Run:** `bash run_setup.sh`, `python plot_compare.py` → `figures/exp074_surprisal_credit.png`.
Metric `surprisal_credit/mean_s`.

## Results (300 steps, L50 boxed / len) — NEGATIVE

| dataset | GRPO | best (posdisc λ0.7 k5) | **surprisal_credit** |
|---|---|---|---|
| gsm8k    | +2.02 / 414 | **+2.49** / 317 | +2.06 / 375 |
| math500  | +0.94 / 942 | **+1.63** / 635 | +0.51 / 600 |
| bigmath  | +1.51 / 622 | **+1.81** / 529 | +0.81 / 630 |
| omnimath | **−0.23** / 957 | −0.33 / 733 | −0.63 / 656 |

**The cheap variant fails** (≈GRPO on gsm8k, WORSE than GRPO on math500/bigmath/omnimath).
Additive per-polarity z(−log p(o_t)) is too weak/noisy a signal: the realized-token
surprisal is dominated by routine high-prob tokens, and z-normalizing it injects
low-information direction that the head-truncated C avoids. Takeaway: the value is in the
top-k HEAD statistic (C / branching h), not the single sampled-token surprisal — quantifies
what the extra forward buys. Confirms exp_072's head-signal as the right estimator.
