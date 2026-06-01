# exp_052 — exp_051 on a harder Big-Math subset (Qwen3-4B · Llama-8B-hard · tag-masked shaping)

Same setup as exp_051 (Qwen3-4B, tag-masked per-token shaping, full reward set, 4 methods head-to-head), but the dataset is filtered to keep only problems that are hard for Llama-8B.

## Motivation

exp_051 (Qwen3 on Big-Math int-2000) hit a near-ceiling regime — exact_top 0.66 for the GRPO baseline, KL ~0.01-0.03 across all methods, very little room for shaping to differ. gtpo_conf with tag-mask still nudged ahead (Δ+0.31 vs baseline) but gtpo_ema_flipped lost (Δ-0.67).

Hypothesis: harder problems leave more headroom above the baseline floor, so shaped methods that converged near the same place on exp_051 may separate more clearly on exp_052.

## Dataset

Big-Math-RL-Verified filtered through two conjunctive predicates:

1. `is_integer_answer(answer)` — same as exp_051 (parse to int)
2. `is_llama8b_hard(llama8b_solve_rate)` — keep only examples where the dataset's recorded Llama-3.1-8B-Instruct pass rate is below 0.3

Filter pipeline observed (full Big-Math train split, 251,122 rows):
- total: 251,122
- after integer-answer filter: 131,812
- after Llama-8B solve_rate < 0.3 filter: 45,273
- after `shuffle(seed=3407) + select(2000)`: **2000 examples** (mean Llama-8B solve_rate ≈ 0.115)

Filter logic lives inline in `train.py::prepare_dataset` so the dataset is constructed deterministically at training time from the SOLVE_RATE_THRESHOLD constant. No separate dataset artifact stored.

## Setup vs exp_051

Identical to exp_051 except for the dataset filter. Verbatim copies of `train.py`, `src/*`, `tests/*`, `plot_metrics.py`, `plot_reward_dynamics.py`, `requirements.txt`. SOLVE_RATE_THRESHOLD=0.3 and a second `ds.filter(is_llama8b_hard)` pass are the only deltas.

| field | exp_051 | exp_052 |
|---|---|---|
| dataset slice | integer-answer, first 2000 | integer-answer ∩ Llama-8B<0.3, first 2000 |
| mean Llama-8B solve_rate of slice | ~0.50 (proxy, full dist) | ~0.11 |
| model | Qwen3-4B | Qwen3-4B |
| max_seq | 4096 | 4096 |
| reward | full set | full set |
| tag-mask | active for gtpo_conf/gtpo_ema_flipped | active for gtpo_conf/gtpo_ema_flipped |
| training args | bs=1 × ga=4 × ng=4, 500 steps, lr 5e-6 cosine, seed 3407 | same |

## Methods

| method | shaping | tag-mask effect |
|---|---|---|
| `grpo` | none (baseline) | no mask (per protocol — run unmasked) |
| `grpo_s_entropy` | seq-level entropy weighting | mask active but no-op (seq-level shaping) |
| `gtpo_conf` | per-token confidence bonus | mask active |
| `gtpo_ema_flipped` | per-token EMA-flipped advantages | mask active |

## Files

```
README.md               this file
requirements.txt        numpy<2.3 overlay
run_052.sh              docker launcher, 4 methods sequential
plot_metrics.py         4-way reward / ans_e / fmt_e / KL grid
plot_reward_dynamics.py single-panel rolling-20 reward
train.py                method-switch trainer, full reward, tag-mask wiring,
                        + is_llama8b_hard filter in prepare_dataset
src/                    same trainers/utils as exp_051
tests/                  6 shaping + 4 tag-mask unit tests
```

## Results

(to be filled in once training finishes)

| method | reward L50 | peak | answer_exact L50 | format_exact L50 | exact_top | KL L50 |
|---|---|---|---|---|---|---|
| grpo               | tbd | tbd | tbd | tbd | tbd | tbd |
| grpo_s_entropy     | tbd | tbd | tbd | tbd | tbd | tbd |
| gtpo_conf          | tbd | tbd | tbd | tbd | tbd | tbd |
| gtpo_ema_flipped   | tbd | tbd | tbd | tbd | tbd | tbd |
