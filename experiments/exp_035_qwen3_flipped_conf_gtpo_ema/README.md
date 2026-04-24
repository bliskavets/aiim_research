# Experiment 035: Flipped pure-proof GTPO-EMA on GSM8K

## Why

`C = -mean(top-k log π)` numerically **grows** with peakedness, not with
Shannon entropy, contrary to the intuition in Def 1.1 of
`experiments/proof/GTPO-EMA-full.txt`. A one-hot-ish distribution yields
C ≈ 9.5; a uniform distribution yields C ≈ 4.6 (see
`tests/test_ema_flipped_utils.py::test_confidence_peaked_gt_flat`).

Consequence for exp_005/006/010/025: the bonus in O+ is proportional to
EMA(C), so it concentrates on **peaked/decisive** tokens; the penalty in
O- is proportional to 1/EMA(C), so it lands on **flat/hesitant** tokens.
That's the opposite of the "reward exploration / punish confident
mistakes" story the prose claims.

## What changes (minimal patch vs exp_025)

Only the signal roles between O+ and O- are swapped:

    O+: bonus_{i,t}   = (1/EMA(C)_{i,t} / Σ_{k∈O⁺_t} 1/EMA(C)_{k,t}) · d_t
    O-: penalty_{j,t} = (EMA(C)_{j,t}   / Σ_{k∈O⁻_t} EMA(C)_{k,t})   · h_t

Conservation (Prop 2.3) and the separate-z-norm step (Def 1.5) hold
unchanged — they depend only on the weights being positive, not on which
group they came from. Concretely, Σ_{i∈O⁺_t} shaped_pos = d_t and
Σ_{j∈O⁻_t} shaped_neg = −h_t for α₁+α₂=1, regardless of the swap.

Identical to exp_025 in every other respect:
  - Model, dataset, LoRA config, LR schedule, batch size, grad accum,
    num_generations, max_seq, reward functions, random_state=3407.
  - `α₁=0.9, α₂=0.1, λ=0.9, top_k=20, reward_threshold=0`.

## Prediction

If the swap direction is the semantically correct one (reward
exploration in O+, punish confident mistakes in O-), exp_035 should
either match the successful runs at the 9.5 ceiling, or — more
interestingly — show a **different trajectory** in the first 150-250
steps (where the bimodal collapse vs. success decision happens).
Lower early-run KL spikes would be consistent with a less aggressive
shaping that doesn't yank decisive tokens around while the model is
still learning the format.

Given that exp_024/025 already showed run-to-run variance of the
"collapse-or-succeed" sort dominates the method gap on GSM8K at this
scale, a single seed here is still not conclusive. It will at least
tell us whether the swap doesn't make things worse.

## Files

- `src/ema_flipped_utils.py` — swap-aware reward shaping
- `src/gtpo_ema_flipped_trainer.py` — `GTPOEMAFlippedTrainer`
- `tests/test_ema_flipped_utils.py` — 12 tests covering conservation
  (both groups), swapped ranking in O+, z-norm invariants, padding,
  and a small-EMA numerical bound
- `train_gtpo_ema_flipped.py` — GSM8K training script (same as exp_025)
- `requirements.txt` — `numpy<2.3`
- `run_026.sh` — Docker launcher
