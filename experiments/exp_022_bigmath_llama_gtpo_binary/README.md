# Exp 022: GTPO with Binary O+/O- Split (answer_exact >= 0)

## Overview

GTPO per-token entropy shaping where **O+/O- is determined by raw binary
correctness** (from `reward_answer_exact`), not by z-scored group advantages.

**Hypothesis:** Most GRPO variants (exp_018-021) use z-scored advantages for
O+/O- split, which always yields ~50/50 inside each group (by construction
of z-score). A true binary split by absolute correctness should give GTPO a
cleaner signal — especially when the whole group is correct or all-wrong.

## Method

1. `reward_answer_exact` computes its usual scalar reward AND stashes a
   per-sequence `torch.bool` mask into a shared module-level cache.
2. In `_compute_loss`, the trainer reads the mask and converts to a signed
   ±1 tensor, which is passed as `rewards` to `compute_gtpo_rewards` with
   `reward_threshold=0.0`. The existing utility splits O+/O- on the sign.
3. `answer_exact` reward levels:
   - **O+** (score ≥ 0): `{3.0 exact, 1.5 strip, 1.0 within-10%, 0.5 within-20%, 0.0 no-format}`
   - **O-** (score < 0): `{-1.5 wrong-answer-in-format}`

Threshold **0.0** treats "no format" as O+ (don't penalize sequences that
haven't learned format). Only *confidently wrong* answers are O-.

## Config (same as exp_017/020)

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

## Files

| File | Purpose |
|------|---------|
| `src/reward_cache.py` | Module-level cache for binary mask |
| `src/entropy_utils.py` | GTPO entropy utilities (same as exp_020) |
| `src/gtpo_binary_trainer.py` | Trainer using cache-based O+/O- |
| `train.py` | Main training script with stashing reward function |
| `tests/test_exp022.py` | 17 unit tests (cache, mapping, shaping) |

## Expected Differences from exp_020

- **Frac_pos** should reflect actual correctness (growing with training),
  not always ~0.5 as in z-scored versions
- **Early training**: Most sequences get 0.0 (no format) → most are O+ →
  reduced contrast for GTPO shaping (fallback is baseline GRPO-like)
- **Late training**: Clean binary signal, strong contrast for shaping

## Status

🔄 Queued after exp_018-021 chain finishes.
