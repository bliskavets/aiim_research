# Experiment 036: Qwen3-4B repro of exp_024 (GTPO-Conf + GRPO-S-Conf on GSM8K)

## Purpose

Repeat exp_024 (repro of exp_005) with **Qwen3-4B** instead of Llama-3.2-3B.
exp_024 showed that on Llama-3.2-3B, the GTPO-Conf / GRPO-S-Conf ranking from
exp_005 did not reproduce — the win was run-to-run noise. This experiment asks
the same question for Qwen3-4B, and adds a data point for method comparison
across models.

## What changes vs exp_024

| Parameter | exp_024 | exp_036 |
|-----------|---------|---------|
| Model | `meta-llama/Llama-3.2-3B-Instruct` | `Qwen/Qwen3-4B` |
| max_seq_length | 2048 | 4096 |
| output_dir (GTPO-Conf) | `/workspace/outputs_exp024_gtpo_conf` | `/workspace/outputs_exp036_gtpo_conf` |
| output_dir (GRPO-S-Conf) | `/workspace/outputs_exp024_grpos_conf` | `/workspace/outputs_exp036_grpos_conf` |

Everything else is identical to exp_024: same src/, same reward functions,
same hyperparams (α₁=1.0, α₂=0.1, top_k=20, reward_threshold=0.0,
lr=5e-6, 500 steps, 4 gens, random_state=3407).

## Config

- Model: Qwen3-4B with LoRA r=64
- Dataset: GSM8K train (7473 examples)
- Steps: 500, bs=1 × grad_accum=4, 4 generations
- max_seq_length: 4096 (Qwen3 generates long thinking-mode completions)
- Confidence: top_k=20, α₁=β₁=1.0, α₂=β₂=0.1, reward_threshold=0.0

## Reference (exp_024, Llama-3.2-3B)

| Method | L50 avg | Peak | @Step |
|--------|---------|------|-------|
| GTPO-Conf | ~8.5 | 9.5 | ~200 |
| GRPO-S-Conf | collapsed | ~2.0 | — |

## Files

- `src/confidence_utils.py` — copied verbatim from exp_024
- `src/gtpo_conf_trainer.py` — copied verbatim from exp_024
- `src/grpo_s_conf_trainer.py` — copied verbatim from exp_024
- `train_gtpo_conf.py` — Qwen3-4B version
- `train_grpo_s_conf.py` — Qwen3-4B version
- `run_036.sh` — Docker launcher
