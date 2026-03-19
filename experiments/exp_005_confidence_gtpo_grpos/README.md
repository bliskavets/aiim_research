# Experiment 005: Confidence-Based GTPO and GRPO-S on GSM8K

## Overview
Modified versions of GTPO and GRPO-S (exp_002) replacing entropy H_i,t with
**confidence score** from "Deep Think with Confidence" (Meta, arXiv:2508.15260):

```
C_i,t = -mean_{v ∈ top-k}( log π(v | context) )    k=20
```

Small C → model is confident (focused); Large C → model is uncertain (spread).

Normalization: log(1 + C) compression + group-relative division.
For O- sequences: inverse confidence log(1 + 1/C) to penalize confident mistakes.

## Config
Same as exp_002 (GSM8K, Llama-3.2-3B, 500 steps) + confidence params:
- **top_k:** 20
- **α₁=β₁=1.0, α₂=β₂=0.1**
- **reward_threshold:** 0.0 (O+/O- split)

## Results

| Method | Step 1 | Step 250 | Step 500 | Peak | @Step | Format@500 | KL@500 |
|--------|--------|----------|----------|------|-------|------------|--------|
| GRPO baseline (exp_001) | -1.375 | 2.000 | **3.000** | 8.375 | 169 | **3.00** | 0.0855 |
| GRPO + seq-level entropy (exp_002) | -1.000 | 1.500 | **3.000** | 9.500 | 294 | **3.00** | 0.1172 |
| **GTPO-Conf (exp_005)** | -1.000 | **2.875** | 2.375 | **9.500** | 268 | 2.25 | 0.0691 |
| **GRPO-S-Conf (exp_005)** | -1.000 | 2.000 | 0.000 | 2.000 | 147 | 0.00 | **0.0233** |

## Observations

**GTPO-Conf:**
- Reached peak 9.5 at step 268 — comparable to best entropy-based method
- Step 500 reward 2.375 — partial convergence, similar to entropy GTPO on MATH-500
- KL 0.069 — lower than GRPO baseline, confidence shaping regularizes effectively
- **Improvement over entropy GTPO on GSM8K (exp_002)**: entropy GTPO failed completely (reward=0), confidence GTPO partially converged (reward=2.375) ✅

**GRPO-S-Conf:**
- Collapsed by step 500 (reward=0, format=0)
- Peak was only 2.0 at step 147 — never learned the format properly
- Very low KL (0.023) — stayed close to base model, didn't update enough
- Likely cause: sequence-level confidence averaging loses too much signal on GSM8K

## Key Insight
Confidence metric works better than entropy for **token-level** (GTPO-Conf) than
**sequence-level** (GRPO-S-Conf) shaping on GSM8K. The per-token signal is richer
and more informative at the granular level.

## Files
- `src/confidence_utils.py` — core confidence computation + reward shaping
- `src/gtpo_conf_trainer.py` — GTPOConfTrainer
- `src/grpo_s_conf_trainer.py` — GRPOSConfTrainer
- `tests/test_confidence_utils.py` — 12/12 unit tests passing
- `train_gtpo_conf.log` / `train_grpo_s_conf.log` — training logs
