# exp_070 — rank-FLOOR adaptive-k for the confidence signal C

## Idea

Corrected version of exp_069. Instead of *capping* k at 5 (which let k=1 dominate and
collapsed), we **floor** k at 5 and let it grow only when the model sampled from the tail:

```
r_t = 1-indexed rank of the sampled token   (= #{tokens with strictly greater logprob} + 1)
k_t = max(r_t, 5) = clamp(r_t, min_k=5, cap=256)
C_it = -(1/k_t) Σ_{j<k_t} logπ_(j)          # mean of the top-k_t log-probs (sorted desc)
```

- argmax-sampled tokens (~83% of tokens, per exp_069 `mean_k`≈1.18 → most rank 1) → **k=5**,
  which is exactly the *stable* fixed value from exp_066.
- Only tokens the model actually sampled from the tail get k>5 (up to a compute cap of 256;
  exp_067 says nucleus/rank p95≈20–28, so 256 is generous).

So the dominant behaviour is stable fixed k=5, with mild widening on "surprising" tokens —
the opposite failure mode of exp_069 (k≤5, where k=1 dominated → collapse).

## Config

Base = `gtpo_ema_flipped` (FIXED, group-visible) + `pos_discount`, matching the prior best:
- λ (EMA) = 0.7, `pos_tau` = 1024, α₁=0.9, α₂=0.1
- `min_k` = 5 (floor), `rank_cap` = 256 (compute cap)
- Qwen3-4B-Base, ng=4, bs=1, ga=4, lr 5e-6 cosine, 300 steps, seed 3407, max_seq 4096
- Sampling unchanged (temperature=1.0, top_p=1.0)
- Datasets: gsm8k / math500 / bigmath / omnimath

## Run

```bash
bash run_rankfloor.sh       # rank_floor_c × 4 datasets (uses /root/aiim/venv python)
python plot_rankfloor.py    # figures/exp070_rankfloor.png
```

Baselines `train_{ds}_grpo.log`, `train_{ds}_posdisc_lam0.7_k5.log` (exp_063/066) and the
exp_069 `train_{ds}_rank_c.log` (k≤5) are reused as comparison curves.

## Files

- `src/rank_c.py` — `rank_C` = `clamp(rank, min_k, cap)`; here min_k=5, cap=256.
- `src/rank_c_trainer.py` — `RankCTrainer`; logs `rank_c/mean_k` (expect ≳5).
- `tests/test_rank_c.py` — CPU unit tests incl. floor@5 and grow-on-tail. 7/7 pass.

## Results

_(in progress)_
