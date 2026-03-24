# Experiment 008: GRPO Qwen3-4B on GSM8K

## Overview
Replication of exp_001 (GRPO + LoRA on GSM8K) with **Qwen/Qwen3-4B** instead of Llama-3.2-3B.
Identical hyperparameters, reward functions, dataset, and format.

## Config
| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3-4B |
| Method | GRPO + LoRA (r=64, alpha=64) |
| Dataset | openai/gsm8k (train, 7473 examples) |
| Max steps | 500 |
| LR | 5e-6 (cosine, warmup 10%) |
| Batch size | 1 (grad accum=4) |
| Num generations | 4 |
| Hardware | NVIDIA A100 80GB |
| Runtime | **2h 0min** (vs 35min for Llama — 3.4× slower due to longer generations) |

## Results vs Llama-3.2-3B (exp_001)

| Metric | Llama-3.2-3B | **Qwen3-4B** |
|--------|-------------|-------------|
| Step 1 reward | -1.375 | **4.625** |
| Step 100 reward | ~2.0 | **9.500** |
| Step 500 reward | 3.000 | 3.000 |
| Peak reward | 8.375 @step 169 | **9.500 @step 28** |
| Format@500 | 3.00 | 3.00 |
| KL@500 | 0.0855 | **0.0059** |
| Mean completion length | ~250 tokens | ~900 tokens |
| train_loss | 0.000211 | 0.000127 |

## Key Observations

**Qwen3-4B is significantly stronger:**
- Peak reward 9.5 reached at **step 28** (vs step 169 for Llama) — **6× faster convergence**
- Starting reward 4.625 at step 1 — Qwen already partially knows the format before any training
- Much lower KL (0.006 vs 0.086) — stays much closer to base model while achieving same final reward

**Thinking mode:** Qwen3-4B generates very long completions (mean 900 tokens, max 1779)
due to its built-in extended thinking capability. This causes:
- Slower training (15s/step vs 4s/step)
- `clipped_ratio > 0` on some steps — completions hit max_completion_length=2048

**Implication for future experiments:** With Qwen3-4B, 500 steps may be overkill —
the model converges within ~50-100 steps. Consider using fewer steps or early stopping.
