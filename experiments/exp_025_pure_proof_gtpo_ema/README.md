# Experiment 025: Pure-Proof GTPO-EMA on GSM8K

## Purpose

Test the token-shaping formula **exactly as stated in
`experiments/proof/GTPO-EMA-full.txt` (Def 1.4)** against the existing
confidence-based variants on the same dataset and config as exp_005
(`GTPO-Conf`) and exp_024 (`exp_005 repro`).

Goal: decide whether the theoretical formulation wins over the pragmatic
z-norm variants (exp_005, exp_010) at fixed compute.

## Method (what changes vs exp_005 / exp_010)

Shaped per-token rewards for o_i ∈ O⁺ active at step t (|o_i| ≥ t):

    r̃⁺_{i,t} = α₁ · r_i + α₂ · (EMA_{i,t} / Σ_{k∈O⁺_t} EMA_{k,t}) · d_t
    r̃⁺ = 0  if o_i ∉ O⁺_t

For o_j ∈ O⁻ active:

    r̃⁻_{j,t} = −α₁ + α₂ · (1/EMA_{j,t} / Σ_{k∈O⁻_t} 1/EMA_{k,t}) · h_t · (−1)

Final advantages (Def 1.5):

    Ã⁺_{i,t} = (r̃⁺_{i,t} − mean(R̃⁺)) / std(R̃⁺)
    Ã⁻_{j,t} = (r̃⁻_{j,t} − mean(R̃⁻)) / std(R̃⁻)

z-norm is taken over active tokens in each group separately.

Differences vs `exp_005` (`GTPO-Conf`):
  - uses **EMA-smoothed** confidence (λ=0.9), not raw C (Prop 3.1 says Var ≈ 19× lower)
  - no `log(1+·)` compression before weighting
  - O⁻ penalty uses raw `1/EMA`, not `log(1+1/C)`
  - α₁+α₂=1 (Prop 2.3 conservation of reward mass). exp_005 used α₁=1.0, α₂=0.1.

Differences vs `exp_010` (`GTPO-EMA v2`, current best):
  - keeps the `·d_t` multiplier and Σ-normalization instead of z-norm on the bonus
  - does not z-norm `base` and `bonus` separately

Everything else (seeds, LoRA config, LR schedule, GSM8K tokens/format,
reward functions, max_seq=2048, 500 steps, 4 generations, bs=1, grad_accum=4)
is byte-identical to exp_005.

## Config

| | Value |
|---|---|
| Model | meta-llama/Llama-3.2-3B-Instruct (LoRA r=64) |
| Dataset | GSM8K (train), 7473 examples |
| Steps | 500, save every 250 |
| bs / grad_accum / gens | 1 × 4 × 4 |
| LR | 5e-6, cosine, warmup 10% |
| max_seq | 2048 |
| α₁, α₂, λ, top_k | 0.9, 0.1, 0.9, 20 |
| reward_threshold | 0.0 |
| random_state | 3407 (LoRA) |

## Reference numbers to beat

| Method | Step 250 | Step 500 | Peak | @Step | KL@500 |
|--------|----------|----------|------|-------|--------|
| GRPO baseline (exp_001) | 2.000 | 3.000 | 8.375 | 169 | 0.086 |
| GTPO-Conf (exp_005) | 2.875 | 2.375 | 9.500 | 268 | 0.069 |
| GTPO-Conf (exp_024 repro) | (TBD) | (TBD) | (TBD) | | |
| **GTPO-EMA-Proof (exp_025)** | ? | ? | ? | | |

## Note on the confidence formula

`C = -mean(top-k log π)` does not measure "uncertainty" the way the Def 1.1
prose suggests. A numerical check (see `tests/test_ema_proof_utils.py
::test_confidence_peaked_gt_flat`) shows that a **peaked** one-hot-ish
distribution yields a **LARGER** C than a **flat** uniform distribution,
because the low-probability tail inside the top-k dominates the mean.
The shaping therefore rewards tokens where the model has a single dominant
candidate with many unlikely alternatives in the top-k — roughly, "sharp
decisions", not "high entropy / exploration". The formula in the paper
and in this file is unchanged; only the intuitive labeling in the prose
is misleading. Worth revisiting in a later experiment.

## Files

- `src/ema_proof_utils.py` — EMA + pure-proof reward shaping
- `src/gtpo_ema_proof_trainer.py` — GTPOEMAProofTrainer
- `tests/test_ema_proof_utils.py` — 13 unit tests (conservation, active-mask,
  z-norm invariants, edge cases)
- `train_gtpo_ema_proof.py` — GSM8K training script (mirrors exp_005)
- `requirements.txt` — numpy<2.3 overlay
- `run_025.sh` — Docker launcher (matches exp_024 pattern)
