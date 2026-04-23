# Experiment 027: GRPO baseline on Big-Math integer-2000

## Purpose

Matched GRPO baseline for exp_028 (flipped pure-proof GTPO-EMA) on
**Big-Math-RL-Verified, integer-answer subset, first 2000 shuffled
examples**. Same model, same steps, same optimizer, same reward
functions — only the trainer and (in exp_028) the per-token shaping
differ.

## Config

| | Value |
|---|---|
| Model | meta-llama/Llama-3.2-3B-Instruct (LoRA r=64) |
| Dataset | SynthLabsAI/Big-Math-RL-Verified, integer-answer filter, 2000 after `shuffle(seed=3407)` |
| Steps | 500 |
| bs / grad_accum / num_generations | 4 × 1 × 8 (32 seqs/step) |
| LR | 5e-6, cosine, warmup 10% |
| max_prompt_length | min(p99+1, 512) |
| max_completion_length | 2048 |
| random_state | 3407 |
| bf16 | true |

Expected sequence budget per step: 4 prompts × 8 completions × (up to
2048 completion tokens) ≈ 65k training tokens. Matches exp_017 on a
smaller budget (500 vs 1000 steps, 8 vs 16 gens, 2048 vs 3072 comp
tokens).

## Reward functions (same as exp_017)

| Reward | Max | Meaning |
|---|---|---|
| `reward_format_exact`       | +3.0 | full tag match |
| `reward_format_approximate` | +2.0 | partial credit per tag |
| `reward_answer_exact`       | +3.0 / +1.5 / +1.0 / +0.5 / 0.0 / -1.5 | exact / strip / within-10% / within-20% / no-format / wrong |
| `reward_answer_numeric`     | +1.5 / -0.5 / 0.0 | numeric-eq / wrong / no-format |

Reward ceiling: 3.0 + 2.0 + 3.0 + 1.5 = **9.5**, same as exp_005 / exp_026.

## Files

- `train.py` — full training entrypoint
- `requirements.txt` — `numpy<2.3`
- `run_027.sh` — Docker launcher (matches exp_025/026 pattern)
