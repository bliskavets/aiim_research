# Experiment 003: GRPO + LoRA on MATH-500

## Overview
GRPO fine-tuning of Llama-3.2-3B-Instruct on HuggingFaceH4/MATH-500.

## Changes vs exp_001 (GSM8K)
| Parameter | exp_001 | exp_003 |
|-----------|---------|---------|
| Dataset | openai/gsm8k (7473 train) | HuggingFaceH4/MATH-500 (500 examples) |
| Reasoning tags | `<start_working_out>` / `<SOLUTION>` | `<think>` / `<answer>` |
| Correct answer bonus | 3.0 | **5.0** |
| Answer matching | exact + numeric | exact string match (LaTeX) |

## Config
- **Model:** meta-llama/Llama-3.2-3B-Instruct
- **Method:** GRPO + LoRA (r=64, alpha=64)
- **Max steps:** 500 (1 full epoch — dataset has 500 examples)
- **LR:** 5e-6 (cosine, warmup 10%)
- **Hardware:** NVIDIA A100 80GB PCIe
- **Runtime:** 63 min (8.4 sec/step — longer than GSM8K due to harder problems)

## Reward Functions
1. `reward_format_exact` — +3.0 for correct `<think>...</think><answer>...</answer>` format
2. `reward_format_approximate` — partial credit per tag (max +2.0)
3. `reward_answer_exact` — **+5.0** exact match, +2.5 normalized match, -1.0 wrong

## Status
✅ **Completed** — 500/500 steps

## Results

### Key Metrics

| Step | Reward | Format Exact | Answer Exact | KL |
|------|--------|-------------|-------------|----|
| 1 | 0.750 | 1.50 | -0.50 | 0.0004 |
| 100 | 1.500 | 1.50 | -0.50 | 0.0070 |
| 250 | **7.875** | **3.00** | **2.88** | 0.0300 |
| 500 | 0.625 | 0.75 | -0.25 | 0.0453 |

- **train_runtime:** 3786 sec (~63 min)
- **train_loss:** 0.000161
- **epoch:** 1.0 (exactly 1 epoch — 500 steps on 500 examples)

### Observations

**Peak at step ~250:** Model reaches format_exact=3.0 and answer_exact≈2.88 — model successfully learns both the `<think>`/`<answer>` format AND produces correct LaTeX answers. This is a strong result.

**Collapse after step 250:** Reward drops significantly by step 500 (reward 0.625, format 0.75). This is consistent with **reward hacking / overfitting** on a tiny 500-sample dataset — with only 500 examples, the model sees each sample once per epoch and may destabilize after the initial learning phase.

**Root cause of collapse:** With 500 training examples and num_generations=4, each step uses just 1 prompt × 4 completions. After ~250 steps, the model has seen all 500 examples once. Without more diverse data, GRPO's advantage estimates become noisy as the model oscillates.

### Recommendations
- **More steps with repetition:** Run 1000-2000 steps (2-4 epochs) to see if format re-stabilizes
- **Larger dataset:** Use full MATH (~7500 examples) to avoid overfitting
- **Lower LR after step 250:** Reduce lr to 1e-6 when model peaks to preserve format

## Files
- `train.py` — training script
- `train.log` — full training log
- `metrics.json` — per-5-step metrics
- `figures/dashboard.png` — full training dashboard
- `figures/rewards.png` — reward functions detail
