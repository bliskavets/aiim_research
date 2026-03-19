# Experiment 002: GTPO and GRPO-S Replication

## Overview
Replication of ["GTPO and GRPO-S: Token and Sequence-Level Reward Shaping with Policy Entropy"](experiments/exp_002_gtpo_and_grpo_s/18825_GTPO_and_GRPO_S_Token_an.txt) (ICML 2026 submission).

Two new algorithms are implemented as subclasses of TRL's `GRPOTrainer`:
- **GTPO** — token-level entropy-weighted reward shaping (Eq. 3, 5, 7)
- **GRPO-S** — sequence-level entropy-weighted reward shaping (Eq. 8, 9, 10)

Both are compared against GRPO baseline (exp_001).

## Implementation

### Files
```
src/
├── entropy_utils.py    — entropy computation + reward shaping (tested)
├── gtpo_trainer.py     — GTPOTrainer(GRPOTrainer)
└── grpo_s_trainer.py   — GRPOSTrainer(GRPOTrainer)
tests/
└── test_entropy_utils.py — 11 unit tests (all passing)
train_gtpo.py           — GTPO training script
train_grpo_s.py         — GRPO-S training script
plot_metrics.py         — comparison plots
```

### Key Design Decisions
- **O+/O- split threshold:** reward > 0 → O+, reward ≤ 0 → O- (our rewards are continuous, not binary)
- **Entropy computed in `_compute_loss()`** rather than `_generate_and_score_completions()` to avoid Unsloth buffer splitting issues
- **No new libraries** — all logic uses PyTorch + existing TRL utilities

## Config

| Parameter | Value |
|-----------|-------|
| Model | meta-llama/Llama-3.2-3B-Instruct |
| Dataset | openai/gsm8k (train, 7473 examples) |
| Max steps | 500 |
| α₁, β₁ | 1.0 |
| α₂, β₂ | 0.1 |
| ε_entropy_low | 0.2 |
| ε_entropy_high | 0.28 |
| Learning rate | 5e-6 (cosine) |
| Batch size | 1 (grad accum=4) |
| Num generations | 4 |
| Hardware | NVIDIA A100 80GB PCIe |
| Docker image | unsloth/unsloth (2026.3.7) |

## Results

### Training Summary

| Method | Runtime | train_loss | Step 1 Reward | Step 500 Reward | Format@500 | KL@500 |
|--------|---------|------------|--------------|-----------------|------------|--------|
| GRPO (baseline) | 35 min | 0.000211 | -1.375 | **3.000** | **3.0** | 0.0855 |
| GTPO | 35 min | 0.000064 | -1.375 | 0.000 | 0.0 | 0.0229 |
| GRPO-S | 34 min | 0.000565 | -1.000 | **3.000** | **3.0** | 0.1172 |

### Observations

**GRPO-S:**
- Matches GRPO baseline in final reward (3.0) and format convergence
- **Higher KL (0.117 vs 0.085)** — more divergence from base model, consistent with stronger exploration
- Slightly faster per step (3.36s vs 3.46s) due to sequence-level IS weights

**GTPO:**
- Final reward = 0.0, format_exact = 0.0 — **did not converge to format**
- Much lower KL (0.023) — stayed close to base model
- Possible cause: per-token advantages with high variance destabilize early training; the model never learns the format template before entropy shaping takes over

**vs Paper:**
- Paper uses binary rewards {0,1} with Qwen2.5-7B/32B on AIME/MATH500
- We use continuous multi-component rewards with Llama-3.2-3B on GSM8K
- GRPO-S behavior matches paper's description (stable, matches GRPO ceiling)
- GTPO divergence may be due to: (1) continuous reward O+/O- split vs binary, (2) smaller model, (3) GSM8K is simpler than AIME (early saturation)

### Hypothesis for GTPO failure
The continuous reward split (reward > 0) causes many sequences to flip between O+/O- early in training when rewards are volatile around 0. This creates noisy per-token advantages that prevent format learning. Paper uses binary rewards where O+/O- is stable from the start.

**Next steps:**
- Try GTPO with hard binary split: O+ only when `reward_answer_numeric == 1.5`
- Try GTPO with more steps (1000) to see if it eventually converges
- Try with Qwen model (closer to paper's setup)

## Figures
- `figures/dashboard_comparison.png` — full comparison dashboard
- `figures/reward_comparison.png` — reward curves GTPO vs GRPO-S vs GRPO
- `figures/reward_functions_comparison.png` — per-reward-function breakdown
- `figures/kl_comparison.png` — KL divergence
- `figures/loss_gradnorm.png` — loss and gradient norm

## Unit Tests
```bash
cd experiments/exp_002_gtpo_and_grpo_s
python3 tests/test_entropy_utils.py
# Results: 11/11 passed
# Verified: reward mass conservation (Prop. 2.2, B.5), shapes, padding, edge cases
```
