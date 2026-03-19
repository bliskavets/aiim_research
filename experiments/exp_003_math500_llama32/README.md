# Experiment 003: GRPO + LoRA on MATH-500

## Overview
GRPO fine-tuning of Llama-3.2-3B-Instruct on HuggingFaceH4/MATH-500.

## Changes vs exp_001 (GSM8K)
| Parameter | exp_001 | exp_003 |
|-----------|---------|---------|
| Dataset | openai/gsm8k (7473 train) | HuggingFaceH4/MATH-500 (500 test) |
| Reasoning tags | `<start_working_out>` / `<SOLUTION>` | `<think>` / `<answer>` |
| Correct answer bonus | 3.0 | **5.0** |
| Answer matching | exact + numeric | exact string match (LaTeX) |

## Config
- **Model:** meta-llama/Llama-3.2-3B-Instruct
- **Method:** GRPO + LoRA (r=64, alpha=64)
- **Max steps:** 500
- **LR:** 5e-6 (cosine, warmup 10%)
- **Hardware:** NVIDIA A100 80GB

## Reward Functions
1. `reward_format_exact` — +3.0 for correct `<think>...</think><answer>...</answer>` format
2. `reward_format_approximate` — partial credit per tag (max +2.0)
3. `reward_answer_exact` — **+5.0** exact match, +2.5 normalized match, -1.0 wrong

## Status
🟡 Running...

## Results
_To be filled after training._
