# Experiment 029: Flipped pure-proof GTPO-EMA on Big-Math integer-2000

## Purpose

Test whether the flipped pure-proof GTPO-EMA (from exp_026) — which
rewards *low-peakedness* tokens in O+ and penalizes *high-peakedness*
tokens in O- — behaves differently from the GRPO baseline (exp_027) on
a harder dataset than GSM8K.

On GSM8K the reward ceiling (9.5) was hit by both baseline and flipped
variants, so the methods were indistinguishable at convergence. Big-Math
integer answers are harder — baseline exp_017 hit peak 9.5 but did not
stay there, and many runs collapsed mid-training. The hope is that the
shaping difference will be resolvable here.

## What changes vs exp_026

1. **Dataset** — SynthLabsAI/Big-Math-RL-Verified, filtered to integer
   answers, shuffled with seed 3407, **first 2000 examples** kept.
2. **Training budget** — 500 steps, bs=4, `num_generations=8`,
   `max_completion_length=2048`. Matches exp_027 exactly.
3. **O+/O- split driven by `reward_answer_exact`**, not by the z-scored
   group advantage. Mask is stashed into `src/reward_cache.py` during
   scoring and read by the trainer in `_compute_loss`. Threshold chosen
   at **`1.0`** so:
     - O+ = exact (3.0), strip (1.5), within-10% (1.0)
     - O- = within-20% (0.5), no-format (0.0), wrong (-1.5)
   This is stricter than exp_022 (`>=0.0`, which counted no-format as O+).

## Method (unchanged from exp_026, restated for Big-Math)

For o_i ∈ O+ active at step t:

    r̃⁺_{i,t} = α₁ · 1 + α₂ · (1/EMA(C)_{i,t} / Σ_{k∈O⁺_t} 1/EMA(C)_{k,t}) · d_t

For o_j ∈ O- active at step t:

    r̃⁻_{j,t} = −α₁ + α₂ · (EMA(C)_{j,t} / Σ_{k∈O⁻_t} EMA(C)_{k,t}) · h_t · (−1)

Final Ã⁺ / Ã⁻ are z-normed separately over the active tokens in each
group (Def 1.5). Conservation (Prop 2.3) holds for α₁+α₂=1.

## Config

| | Value |
|---|---|
| Model | meta-llama/Llama-3.2-3B-Instruct (LoRA r=64) |
| Dataset | Big-Math int-2000, shuffle seed=3407 |
| Steps | 500 |
| bs × grad_accum × gens | 4 × 1 × 8 (32 seqs/step) |
| LR | 5e-6, cosine, warmup 10% |
| max_prompt_length | min(p99+1, 512) |
| max_completion_length | 2048 |
| α₁, α₂, λ, top_k | 0.9, 0.1, 0.9, 20 |
| answer_exact threshold | 1.0 |
| random_state | 3407 |
| bf16 | true |

## Files

- `src/ema_flipped_utils.py` — flipped shaping, takes external `is_pos` mask
- `src/gtpo_ema_flipped_trainer.py` — trainer reading `_CACHE` at compute_loss
- `src/reward_cache.py` — module-level O+ mask cache
- `tests/test_ema_flipped_bigmath.py` — 7 unit tests
- `train.py` — full Big-Math int-2000 training script
- `run_028.sh` — Docker launcher
