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
