# exp_074 — surprisal credit (roadmap setup 4/5)

See `analysis/exp055-070_deep_analysis.md` §2 (exp_064), §5.4. Minimal-machinery variant.

**Idea:** exp_064 showed the clean per-token informativeness signal is the sampled token's
surprisal s = −log p(o_t). Setup: `Ã = A_grpo + α₂·g(t)·z_polarity(s)` — reward surprising
tokens in correct rollouts, punish confident tokens in wrong ones, forgive exploration.
Additive on the GRPO scalar (no cold-start dead signal), needs NO top-k forward (one
logprob gather) → cheapest method in the family. posdisc λ0.7 kept.

**Run:** `bash run_setup.sh`, `python plot_compare.py` → `figures/exp074_surprisal_credit.png`.
Metric `surprisal_credit/mean_s`.

## Results

_(in progress)_
