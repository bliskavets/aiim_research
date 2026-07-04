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

## Results

_(in progress)_
