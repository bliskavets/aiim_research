# exp_063 — pos_discount + EMA-λ combo (Qwen3-4B-Base, 4 datasets)

Combines the two exp_062 winners: **pos_discount** (gentle position discount
`g(t)=τ/(τ+t)` on the α₂ exploration bonus) with a **lower EMA λ** (the λ-sweep
showed 0.9 is suboptimal; 0.5–0.7 is better). Two setups × 4 datasets = 8 runs.

- **pos_discount + λ=0.5**
- **pos_discount + λ=0.7**

Same group-visible FIXED pattern (no B=1 bug), same hyperparameters as exp_058's
last setups (ng=4, bs=1, ga=4, lr 5e-6, 300 steps, seed 3407, integer reward).
Datasets: gsm8k / math500 / omnimath / bigmath.

Code reused from exp_062 (`src/novel_trainers.py` PosDiscountTrainer; `--lam`
override generalized to any EMA method). Smoke-tested: lam override applies to
pos_discount, used_group_shaped=1.0.

## Run
```
HF_TOKEN=... ./run_combo.sh          # 8 runs
python plot_combo.py                 # figures/exp063_combo.png (vs GRPO / pos_discount λ0.9 / FIXED λ0.7)
```

## Results
(filled after the run)

## Results (300 steps, L50; figures/exp063_combo.png, exp063_grpo_vs_combo.png)

L50 boxed (length):

| dataset | GRPO | pos_disc λ0.9 | FIXED λ0.7 | COMBO λ0.5 | COMBO λ0.7 |
|---|---|---|---|---|---|
| GSM8K    | +2.02 (414) | +2.50 (294) | +2.06 (331) | +2.59 (287) | **+2.60** (310) |
| MATH-500 | +0.94 (942) | +1.34 (686) | +1.17 (599) | +1.23 (674) | **+1.39** (628) |
| Big-Math | +1.51 (622) | +1.54 (548) | **+1.86** (499) | +1.62 (613) | +1.67 (537) |
| Omni-MATH| **−0.23** (957) | −0.38 (1068) | −0.55 (742) | −0.53 (834) | −0.50 (851) |

**Findings**
- **The two levers stack on the learnable datasets:** COMBO (pos_discount + lower
  λ) beats pos_discount-alone (λ0.9) on GSM8K (+2.60 vs +2.50), MATH-500 (+1.39 vs
  +1.34) and Big-Math (+1.67 vs +1.54) — and beats GRPO on all three, usually
  shorter. **λ=0.7 ≥ λ=0.5 in the combo** consistently.
- **Exception — Big-Math:** the pure λ-lever (FIXED λ0.7, +1.86) still beats the
  combo (+1.67). Here adding pos_discount slightly dilutes the strong low-λ effect;
  the levers don't always stack.
- **Hard (Omni-MATH):** combo stays below GRPO (−0.50 vs −0.23), like every shaped
  variant — the base is too weak for the shaping to help.
- **Best all-round config: pos_discount + λ=0.7** — beats GRPO on 3/4 datasets and
  improves on pos_discount-alone everywhere it matters. Big-Math is better served
  by FIXED λ0.7 alone, and hard tasks by plain GRPO.
