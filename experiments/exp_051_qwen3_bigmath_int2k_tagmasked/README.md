# exp_051 — Qwen3-4B port of exp_050 · tag-masked per-token shaping on Big-Math int-2000

Direct port of exp_050 (which showed that masking per-token shaping off on format-tag tokens makes GTPO-Conf / GTPO-EMA-Flipped beat the GRPO baseline on Llama-3.2-3B). Same 4 candidates, same hyperparameters, same dataset, same tag-mask intervention — only the base model and seq-length budget change.

## Hypothesis

On Llama-3.2-3B (exp_050), tag-masked per-token shaping flipped two failures into wins:
- gtpo_conf: exact_top 0.06 → 0.30, reward L50 +4.44 → +5.10
- gtpo_ema_flipped: exact_top 0.04 → 0.30, reward L50 +3.39 → +5.16

Both per-token methods crossed the GRPO baseline trajectory around step ~280 and stayed above it. Question: does the same mechanism transfer to Qwen3-4B?

## Setup

Identical to exp_050 except:

| field | exp_050 (Llama) | exp_051 (Qwen3) |
|---|---|---|
| model | meta-llama/Llama-3.2-3B-Instruct | Qwen/Qwen3-4B |
| max_seq_length | 2560 (512 prompt + 2048 completion) | 4096 (512 prompt + 3584 completion) |
| max_completion_tokens | 2048 | 3584 |
| gpu_memory_utilization | 0.55 | 0.55 |

Everything else verbatim from exp_050: bs=1 × grad_accum=4 × num_generations=4, 500 steps, lr 5e-6 cosine, warmup 0.1, wd 0.1, adamw_8bit, LoRA r=64/α=64/7 modules, full reward set (format_exact + format_approximate + answer_exact + answer_numeric), seed 3407 throughout.

## Methods and tag-mask scope

| method | shaping | tag-mask effect |
|---|---|---|
| `grpo` | none (baseline) | no mask (per user instruction — run unmasked as comparator) |
| `grpo_s_entropy` | seq-level entropy weighting (GRPO-S) | mask active but no-op — shaping is seq-level, not per-token |
| `gtpo_conf` | per-token confidence bonus (α₂=0.1, top-k=20) | mask active and effective |
| `gtpo_ema_flipped` | per-token EMA-flipped advantages (α₁=0.9, α₂=0.1, λ=0.9) | mask active and effective |

## Tag-mask mechanism

Same as exp_050: pre-compute token-id subsequences for the 4 format tags (`<start_working_out>`, `<end_working_out>`, `<SOLUTION>`, `</SOLUTION>`) using the Qwen3-4B tokenizer (both bare and " "-prefixed BPE variants). On positions where any subsequence matches in `completion_ids`, replace the per-token shaped advantage with the seq-level GRPO advantage. Content tokens keep their shaped advantage.

## Files

```
README.md               this file
requirements.txt        numpy<2.3 overlay (rest via /opt/venv)
run_051.sh              docker launcher, 4 methods sequential
plot_metrics.py         4-way reward / ans_e / fmt_e / KL grid
train.py                method-switch trainer, full reward, tag-mask wiring
src/                    same trainers/utils as exp_050
tests/                  6 shaping + 4 tag-mask unit tests
```

## Results

(to be filled in once `python plot_metrics.py` runs against the train logs)

| method | reward L50 | peak | answer_exact L50 | format_exact L50 | exact_top | KL L50 |
|---|---|---|---|---|---|---|
| grpo               | tbd | tbd | tbd | tbd | tbd | tbd |
| grpo_s_entropy     | tbd | tbd | tbd | tbd | tbd | tbd |
| gtpo_conf          | tbd | tbd | tbd | tbd | tbd | tbd |
| gtpo_ema_flipped   | tbd | tbd | tbd | tbd | tbd | tbd |
