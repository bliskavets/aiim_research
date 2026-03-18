# Experiment 001: GRPO + LoRA fine-tuning of Llama 3.2 3B on GSM8K

## Overview
Baseline run of GRPO (Group Relative Policy Optimization) with LoRA on Llama-3.2-3B-Instruct,
trained on the GSM8K math reasoning dataset.

## Config
| Parameter | Value |
|-----------|-------|
| Model | meta-llama/Llama-3.2-3B-Instruct |
| Method | GRPO + LoRA |
| LoRA rank | 64 |
| LoRA alpha | 64 |
| Dataset | openai/gsm8k (train, 7473 examples) |
| Max steps | 500 |
| Batch size | 1 (grad accum=4, effective=4) |
| Num generations | 4 |
| Learning rate | 5e-6 (cosine, warmup 10%) |
| Optimizer | adamw_8bit |
| Max seq length | 2048 |
| fast_inference (vLLM) | True |
| Docker image | unsloth/unsloth (unsloth==2026.3.7) |
| Hardware | NVIDIA A100 80GB PCIe |

## Reward Functions
1. `reward_format_exact` — +3.0 for correct `<start_working_out>...<SOLUTION>` format
2. `reward_format_approximate` — partial credit per token (max +2.0)
3. `reward_answer_exact` — +3.0 exact, +1.5 strip match, ±1.0/0.5 within 10/20%
4. `reward_answer_numeric` — +1.5 correct number, -0.5 wrong

## Status
✅ **Completed** — 500/500 steps in 34 min 57 sec

## Results

### Key Metrics

| Step | Reward | Format Exact | Answer Numeric | KL |
|------|--------|-------------|----------------|----|
| 1 | -1.375 | 0.0 | 0.0 | 0.00044 |
| 50 | 2.25 | 0.75 | 0.375 | 0.0028 |
| 100 | 3.75 | 1.5 | 0.375 | 0.0075 |
| 250 | 5.75 | 2.25 | 0.375 | 0.036 |
| 500 | 3.00 | **3.0** | -0.5 | 0.085 |

### Training Summary
- **train_runtime:** 2097.28 sec (~35 min)
- **train_samples_per_second:** 0.954
- **train_steps_per_second:** 0.238
- **train_loss:** 0.000211
- **Final reward:** 3.0 (started at -1.375)

### Observations
- **Format reward converged to maximum (3.0)** by step ~500 — model fully learned the `<start_working_out>...<SOLUTION>` format
- **Reward peaked around step ~250** at ~5.75 combining format + correct answers
- **Final step shows format_exact=3.0 but answer_numeric=-0.5** — model prioritizes format over correct answers at end; likely needs more steps or adjusted reward weights
- **KL divergence grew from 0.00044 → 0.085** — model diverged significantly from base; could be reduced with lower learning rate or KL penalty
- **Completion lengths** ranged 150–665 tokens, mean ~250 — reasonable chain-of-thought length

### Recommendations for Next Experiments
- Increase `max_steps` to 1000-2000 for better answer accuracy convergence
- Try reducing `reward_format_exact` weight and increasing `reward_answer_numeric`
- Add KL penalty to stabilize training
- Evaluate on GSM8K test set to measure actual accuracy

## Files
- `train.py` — training script
- `train.log` — full training log
- `metrics.json` — per-5-step metrics (reward, loss, KL, etc.)
- `requirements.txt` — dependencies

## Notes on Setup
Had to update unsloth to 2026.3.7 (from 2026.1.4) to fix `device_synchronize` bug in compiled GRPO trainer.
NumPy was automatically downgraded to 2.2.6 (from 2.4.1) to satisfy numba compatibility.
