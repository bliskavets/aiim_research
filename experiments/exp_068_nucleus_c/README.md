# exp_068 — dynamic nucleus (top-p) k for confidence C

C_{i,t} with a per-token adaptive k: n = #{leading tokens with cumulative prob ≤ top_p}
(min_k=1), C = −mean of those n logprobs. Probabilities used ONLY to pick n; C from
logprobs. Base = gtpo_ema_flipped (FIXED) + pos_discount + λ=0.5.
Sweep top_p ∈ {0.7,0.8,0.9,0.95}; rollout sampling stays top_p=1.0 (unchanged).
Compare vs GRPO and pos_discount(FIXED,λ0.5,top_k=5). 4 datasets. src/nucleus_c*.py
(nucleus math CPU-unit-tested). run_nucleus.sh (20 runs). plot: exp068_nucleus.png.
