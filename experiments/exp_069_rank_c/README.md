# exp_069 — rank-based adaptive-k for the confidence signal C

## Idea

Instead of a fixed top-k (exp_066: best k≈3–5) or a nucleus/top-p k (exp_068: collapses
at every p), pick the per-token k from the **rank of the actually-sampled token** in the
model's descending-logprob distribution:

```
r_t = 1-indexed rank of the sampled token   (= #{tokens with strictly greater logprob} + 1)
k_t = clamp(r_t, min_k=1, cap=5)
C_it = -(1/k_t) Σ_{j<k_t} logπ_(j)          # mean of the top-k_t log-probs (sorted desc)
```

So the model sampling its argmax (`r=1`) → `k=1` (sharp `−log p_max` on confident tokens);
sampling from the tail → larger k (up to `cap`). Probabilities are **not** used — the rank
is computed over the full vocab from logits; C is computed from log-probs, exactly like the
fixed-top-k C.

**Why this might beat nucleus (exp_068):** nucleus grew k on *flat-but-argmax-picked*
positions, diluting C and destabilising training (all top_p collapsed into length-farming).
Here k is small *exactly* on the confident tokens where a sharp signal is wanted, and only
grows when the model genuinely sampled off-peak. Contrast with fixed k=1 (exp_066), which
collapsed because *every* token got k=1; here k=1 is reserved for argmax tokens.

## Config

Base = `gtpo_ema_flipped` (FIXED, group-visible) + `pos_discount`, matching the prior best:
- λ (EMA) = 0.7, `pos_tau` = 1024, α₁=0.9, α₂=0.1
- `rank_cap` = 5, `min_k` = 1
- Qwen3-4B-Base, ng=4, bs=1, ga=4, lr 5e-6 cosine, 300 steps, seed 3407, max_seq 4096
- Sampling unchanged (temperature=1.0, top_p=1.0)
- Datasets: gsm8k / math500 / bigmath / omnimath
- Reward: format_thinking + answer_boxed (±3) + answer_numeric (integer exact-match)

## Run

```bash
bash run_rank.sh          # rank_c × 4 datasets (uses /root/aiim/venv python)
python plot_rank.py       # figures/exp069_rank.png — rank_c vs GRPO vs pos_disc k=5
```

Baselines `train_{ds}_grpo.log` and `train_{ds}_posdisc_lam0.7_k5.log` are reused from
exp_063 / exp_066.

## Files

- `src/rank_c.py` — `rank_C` (pure, unit-tested) + `rank_C_from_model_chunked` (one forward/microbatch).
- `src/rank_c_trainer.py` — `RankCTrainer(GroupShapedBase)`; logs `rank_c/mean_k`.
- `tests/test_rank_c.py` — CPU unit tests (rank→k mapping, cap, min_k floor, batch). 5/5 pass.

## Results

_(in progress)_
