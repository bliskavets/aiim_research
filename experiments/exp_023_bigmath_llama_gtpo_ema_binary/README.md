# Exp 023: GTPO-EMA with Binary O+/O- Split (answer_exact >= 0)

## Overview

GTPO-EMA (EMA-smoothed confidence shaping) where **O+/O- is determined by
raw binary correctness** from `reward_answer_exact` instead of z-scored
advantages.

**Hypothesis:** Combining the variance-reduction of EMA smoothing
(Proposition 3.1: ~19x variance reduction for λ=0.9) with a true binary
correctness signal should give the cleanest GTPO-EMA training yet.

## Method

Same cache-based approach as exp_022:

1. `reward_answer_exact` stashes per-sequence bool mask into shared cache
2. Trainer reads mask, converts to ±1 signed tensor, passes as `rewards`
   to `compute_gtpo_ema_advantages` with `reward_threshold=0.0`
3. The base_adv inside that utility becomes a z-scored binary signal; the
   EMA confidence bonus is applied per-token on top (normalized within
   O+/O- groups)

Threshold **0.0**: O+ = `{3.0, 1.5, 1.0, 0.5, 0.0}`, O- = `{-1.5}`.

## Config (same as exp_017/018)

| Parameter | Value |
|-----------|-------|
| Model | meta-llama/Llama-3.2-3B-Instruct |
| Dataset | SynthLabsAI/Big-Math-RL-Verified (integer filter) |
| num_generations | 16 |
| per_device_train_batch_size | 4 |
| max_steps | 1000 |
| learning_rate | 5e-6 (cosine, warmup 10%) |
| lora_rank | 64 |
| dtype | bf16 |
| **answer_exact threshold** | **0.0** |
| **EMA λ** | **0.9** |
| **top-k (confidence)** | **20** |

## Files

| File | Purpose |
|------|---------|
| `src/reward_cache.py` | Module-level cache for binary mask |
| `src/ema_confidence_utils.py` | EMA confidence utilities (same as exp_018) |
| `src/gtpo_ema_binary_trainer.py` | Trainer using cache-based O+/O- |
| `train.py` | Main training script with stashing reward function |
| `tests/test_exp023.py` | 8 unit tests (cache, mapping, EMA shaping) |

## Expected Differences from exp_018

- **Explicit binary semantics** vs exp_018's mix of raw_rewards/advantages
- **Clean reward threshold**: answer-based, not total-reward-based
- **Stable KL**: binary signal should reduce the large KL drift seen in exp_018
  (kl=4.13 at step 1000)

## Status

🔄 Queued after exp_018-022 chain finishes.
