# exp_072 — bounded branching signal (roadmap setup 2/5)

See `analysis/exp055-070_deep_analysis.md` §1.1, §5.2. Core estimator contribution.

**Idea:** replace C = −mean top-k logp with the **normalized entropy of the renormalized
top-k head**: `h = H(p̃)/log k ∈ [0,1]` (p̃ = softmax over the top-5 logprobs). Peaked
token → h≈0, contested/branching token → h≈1. Then O+ bonus ∝ EMA(h) (reward branch
points on correct rollouts), O− penalty ∝ 1−EMA(h) (blame peaked wrong tokens).
Bounded by construction — no reciprocal, no ε-blowup, robust to k (unlike raw C which
dilutes at k=20 and reverses meaning at k=1). posdisc λ0.7 kept; k=5.

**Run:** `bash run_setup.sh`, `python plot_compare.py` → `figures/exp072_branch_entropy.png`
(vs best posdisc λ0.7 k5 + GRPO). Metric `branch_entropy/mean_h`.

## Results (300 steps, L50 boxed / len)

| dataset | GRPO | best (posdisc λ0.7 k5) | **branch_entropy h (k5)** |
|---|---|---|---|
| gsm8k    | +2.02 / 414 | **+2.49** / 317 | +2.29 / 275 |
| math500  | +0.94 / 942 | **+1.63** / 635 | +1.62 / 610 |
| bigmath  | +1.51 / 622 | **+1.81** / 529 | +1.78 / 624 |
| omnimath | **−0.23** / 957 | −0.33 / 733 | −0.38 / 898 |

**Bounded branching signal is stable and competitive but does not beat the raw-C best.**
mean_h ≈ 0.11–0.14 (matches exp_067 bimodality: ~85% peaked tokens → h≈0, signal on
branch points). Ties posdisc on math500 (+1.62 vs +1.63) and bigmath (+1.78 vs +1.81),
below it on gsm8k (+2.29 vs +2.49). Still below GRPO on omnimath (−0.38 vs −0.23) — the
bounded signal removes the collapse risk and the k-dilution/sign-reversal pathology, but on
its own it is ~neutral-to-slightly-worse than the raw-C reciprocal at these settings.
Value for the paper: it is the *clean, robust* estimator (bounded, k-robust, no reciprocal)
that MATCHES the tuned heuristic — the safety/principled story. Worth an α₂ sweep (the raw-C
best was tuned; h has a different scale) and the k∈{3,20} robustness ablation.
