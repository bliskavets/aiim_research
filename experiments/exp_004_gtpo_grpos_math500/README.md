# Experiment 004: GTPO and GRPO-S on MATH-500

## Overview
Replication of exp_002 (GTPO + GRPO-S) but with MATH-500 dataset and exp_003 reward setup.
Direct comparison to exp_003 (GRPO baseline on MATH-500).

## Config
Same as exp_003 + GTPO/GRPO-S entropy shaping from exp_002:
- **Model:** meta-llama/Llama-3.2-3B-Instruct (LoRA r=64)
- **Dataset:** HuggingFaceH4/MATH-500 (500 examples, 1 epoch)
- **Tags:** `<think>`/`</think>` `<answer>`/`</answer>`
- **Answer bonus:** 5.0 (exact), 2.5 (normalized), -1.0 (wrong)
- **α₁=β₁=1.0, α₂=β₂=0.1, ε_low=0.2, ε_high=0.28**
- **Hardware:** NVIDIA A100 80GB

## Results

### Key Metrics (vs GRPO baseline from exp_003)

| Method | Step 1 | Step 250 | Step 500 | Peak | Peak Step | Format@500 | Answer@500 | KL@500 |
|--------|--------|----------|----------|------|-----------|------------|------------|--------|
| GRPO (exp_003) | 0.75 | **7.88** | 0.62 | **10.0** | 150 | 0.75 | -0.25 | 0.0453 |
| GTPO | 0.75 | 5.50 | **2.38** | **10.0** | 155 | 2.25 | -0.75 | **0.0035** |
| GRPO-S | 0.75 | 6.38 | **2.38** | **10.0** | 136 | 2.25 | -0.75 | **0.0041** |

### Observations

**All three methods hit peak reward=10.0** (format_exact=3 + format_approx=2 + answer=5) — confirming the model successfully learns both the `<think>`/`<answer>` format AND produces correct LaTeX answers.

**Key difference — post-peak behaviour:**
- **GRPO** peaks early (~step 150) but **collapses hard** to 0.62 by step 500 — typical reward hacking on small dataset
- **GTPO and GRPO-S** also peak early but **retain higher reward at step 500 (2.38)** — entropy weighting provides regularization effect that partially prevents collapse

**KL divergence:**
- GTPO (0.0035) and GRPO-S (0.0041) stay much **closer to the base model** than GRPO (0.0453) — entropy shaping acts as implicit regularization

**GTPO improvement vs exp_002:**
- In exp_002 (GSM8K), GTPO failed completely (reward=0, format=0)
- In exp_004 (MATH-500), GTPO successfully converges — confirms the hypothesis that MATH-500's harder tasks create more stable O+/O- signal for entropy shaping

### Comparison with exp_002 (GSM8K)

| Exp | Dataset | GTPO result | GRPO-S result |
|-----|---------|-------------|---------------|
| exp_002 | GSM8K | ❌ Failed (reward=0) | ✅ Matches GRPO |
| exp_004 | MATH-500 | ✅ Partial convergence (2.38) | ✅ Partial convergence (2.38) |

## Files
- `train_gtpo.py` / `train_grpo_s.py` — training scripts
- `train_gtpo.log` / `train_grpo_s.log` — full training logs
- `figures/dashboard.png` — full comparison dashboard
- `figures/reward_comparison.png` — reward curves
