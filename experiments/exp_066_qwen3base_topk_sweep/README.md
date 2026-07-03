# exp_066 — top_k sweep for confidence C (base pos_discount + FIXED λ=0.7)

C_{i,t} = -mean over top-k logprobs. Swept k ∈ {1,3,5,10,20,40} × {gsm8k, math500,
bigmath, omnimath} on the prior-best setup (pos_discount + gtpo_ema_flipped FIXED
λ=0.7). k=20 = default. Plot: figures/exp066_topk_sweep.png.

## Results (300 steps, L50 boxed / length)

| dataset | GRPO | k=1 | k=3 | k=5 | k=10 | k=20 | k=40 |
|---|---|---|---|---|---|---|---|
| gsm8k    | +2.02 | −0.01 | **+2.62** | +2.49 | +2.54 | +2.60 | +2.44 |
| math500  | +0.94 | +0.00 | **+1.67** | +1.63 | +1.46 | +1.39 | +1.44 |
| bigmath  | +1.51 | +0.02 | **+1.93** | +1.81 | +1.80 | +1.67 | +1.73 |
| omnimath | **−0.23** | +0.00 | −0.38 | −0.33 | −0.41 | −0.50 | −0.53 |

## Findings
- **k=1 COLLAPSES everywhere** (length → 3584 max, boxed → 0). C from the single
  top logprob (= −log max-prob) is degenerate: EMA of it + flipped shaping farms
  length. Too sharp.
- **k=3 is the sweet spot** — best or near-best on gsm8k (+2.62), math500 (+1.67),
  bigmath (+1.93, clear); small k (3–5) beats the default k=20 on all three
  learnable datasets (bigmath +0.26, math500 +0.28 over k=20).
- **Inverted-U in k**: k=1 degenerate → k=3–5 optimal → k≥20 dilutes. A few top
  logprobs give a SHARP, meaningful confidence; many average in the near-zero tail.
- **omnimath (hard)**: all shaped configs negative; plain GRPO (−0.23) still best;
  among k, k=5 least bad.
- Matches exp_067 coverage: distributions are bimodal (peaked majority + flat
  decision-point minority) → small-k C is sharp on the peaked mass; motivates
  top-p / adaptive-nucleus C.

**Practical takeaway: use k≈3 (not the default 20) for C — beats GRPO by a wide
margin on easy/medium/big-math; avoid k=1 (collapse).**
