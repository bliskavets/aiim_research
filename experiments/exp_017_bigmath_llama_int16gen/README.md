# Exp 017: GRPO Baseline — Big-Math (Integer) · 16 Generations

## Overview

Baseline GRPO on SynthLabsAI/Big-Math-RL-Verified filtered to integer-only answers.
Key change vs exp_001: **num_generations=16** (vs 4) and **batch_size=4** (vs 1).

**Hypothesis:** Larger group size (16 generations per prompt) gives GRPO wider reward variance
per group → cleaner O+/O- separation → more stable advantage estimates →
faster/more stable convergence on this dataset.

## Config

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | meta-llama/Llama-3.2-3B-Instruct | No quantization |
| Dataset | SynthLabsAI/Big-Math-RL-Verified | Integer-answer filter |
| Dataset size | ~TBD (run to find) | After integer filter |
| max_steps | 1000 | 2× exp_001 |
| num_generations | **16** | vs 4 in exp_001 |
| per_device_train_batch_size | **4** | vs 1 in exp_001 |
| gradient_accumulation_steps | 1 | — |
| sequences/gradient step | 64 | = 4 × 16 |
| learning_rate | 5e-6 | Cosine, warmup 10% |
| lora_rank | 64 | alpha=64 |
| dtype | bf16 | No fp16, no 4bit |
| max_prompt_length | 99th percentile (capped 512) | Dynamic |
| max_completion_length | 768 | Enough for integer solutions |
| Hardware | NVIDIA A100 80GB | — |

## Reward Functions

| Function | Max Score | Description |
|----------|-----------|-------------|
| `reward_format_exact` | +3.0 | Exact format match |
| `reward_format_approximate` | +2.0 | Partial credit per tag |
| `reward_answer_exact` | +3.0 | Text match (multi-level) |
| `reward_answer_numeric` | +1.5 | Numeric equality |
| **Total max** | **+9.5** | — |

## Files

| File | Purpose |
|------|---------|
| `train.py` | Main GRPO training script |
| `verify_format.py` | Pre-training sanity check (run first!) |
| `tests/test_exp017.py` | Pytest unit tests (no GPU) |
| `plot_metrics.py` | Generate figures from train.log |
| `figures/` | Saved plots |

## Pre-Training Checklist

```bash
# 1. Run unit tests (no GPU needed)
cd experiments/exp_017_bigmath_llama_int16gen
pytest tests/ -v

# 2. Verify model format compliance (needs GPU + HF_TOKEN)
HF_TOKEN=<token> python verify_format.py

# 3. Check disk space
df -h /mnt/data

# 4. Clear Docker caches
for cid in $(docker ps -q); do
  docker exec $cid bash -c "rm -rf ~/.cache/huggingface/hub ~/.cache/torch 2>/dev/null || true"
done

# 5. Run training
HF_TOKEN=<token> python train.py 2>&1 | tee train.log
```

## Status

🔄 Pending

## Results

*(to be filled after run)*

| Step | Reward | Format Exact | Answer Exact | KL |
|------|--------|--------------|--------------|-----|
| 1 | — | — | — | — |
| 250 | — | — | — | — |
| 500 | — | — | — | — |
| 1000 | — | — | — | — |
| Peak | — | — | — | — |

## Estimated Runtime

With batch_size=4, num_generations=16 on A100 80GB:
- 64 sequences/step for loss computation
- Estimated ~60-120 sec/step (vs ~4 sec/step in exp_001 with batch=1, gens=4)
- Total estimate: **17-33 hours** for 1000 steps

## Observations

*(to be filled)*

## Recommendations

*(to be filled)*
