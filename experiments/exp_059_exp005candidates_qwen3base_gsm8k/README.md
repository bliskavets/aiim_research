# exp_059 — exp_005 candidates (GTPO-Conf, GRPO-S-Conf) + GRPO on Qwen3-4B-Base / GSM8K

Re-run of **exp_005** (confidence-based GTPO / GRPO-S on GSM8K with the custom
`<start_working_out>…<end_working_out>` + `<SOLUTION>…</SOLUTION>` format) with the
**same hyperparameters**, but:

- **model → `Qwen/Qwen3-4B-Base`** (a BASE model — no instruction tuning; it must
  learn the format and answering from scratch via RL, which is the regime where
  format-token masking / shaping should matter most);
- the shaping runs through the **fixed injection framework** (`src/shaped_loss.py`)
  so it is actually applied. exp_005's original trainers override `_compute_loss`,
  which unsloth's compiled GRPO loss silently bypasses on this stack — see
  `../exp_057_qwen3_native_omnimath_int_4way/SHAPING_BYPASS_BUGFIX.md`. Each
  shaped run logs its `<method>/*` metrics so we can confirm shaping ran;
- a plain **`grpo`** baseline is run for comparison.

## Candidates
- **grpo** — baseline (no shaping).
- **gtpo_conf** — per-token confidence shaping (exp_005 `compute_gtpo_conf_rewards`):
  `C=-mean_top-k(log p)`, compress `log(1+C)`; O+ bonus ∝ `log(1+C)`, O- penalty ∝
  `log(1+1/C)`; separate z-norm; format tags masked (reverted to seq advantage).
- **grpo_s_conf** — sequence-level confidence shaping (`compute_grpo_s_conf_rewards`):
  mean per-sequence confidence shapes the seq reward, then group-normalized.

## Setup (exp_005 hyperparameters, verbatim)
| field | value |
|---|---|
| model | Qwen/Qwen3-4B-Base |
| dataset | openai/gsm8k (main, train) |
| format | `<start_working_out>…<end_working_out>` then `<SOLUTION>…</SOLUTION>` |
| rewards | format_exact (+3) + format_approximate (±0.5/tag) + answer_exact (+3/+1.5/+1/+0.5/−1.5) + answer_numeric (+1.5/−0.5) |
| max_seq_length | 2048 (prompt + completion) |
| LoRA | r=64, α=64, 7 modules |
| bs / ga / num_generations | 1 / 4 / 4 |
| lr / sched | 5e-6 cosine, warmup 0.1, wd 0.1, adamw_8bit |
| max_steps | 500 |
| shaping | top_k=20, α1=β1=1.0, α2=β2=0.1, reward_threshold=0.0 |
| seed | 3407 |
| gpu_memory_utilization | 0.60 (infra; lowered from exp_005's 0.9 for the extra confidence forward) |

## How to run
```bash
HF_TOKEN=<token> bash experiments/exp_059_exp005candidates_qwen3base_gsm8k/run_059.sh \
  > experiments/exp_059_exp005candidates_qwen3base_gsm8k/run_059.console.log 2>&1
```
Single method: `python train.py --method {grpo|gtpo_conf|grpo_s_conf}`.

## Results

_Pending — run in progress. Comparison plot: `figures/exp059_progress.png`._

| method | steps | reward L50 | format_exact L50 | answer_exact L50 | KL | shaping ran? |
|---|---|---|---|---|---|---|
| grpo         | — | — | — | — | — | n/a |
| gtpo_conf    | — | — | — | — | — | (gtpo_conf/* logged) |
| grpo_s_conf  | — | — | — | — | — | (grpo_s_conf/* logged) |

Conclusion: _to be filled._

## Files
```
train.py                       Qwen3-4B-Base + GSM8K, exp_005 format/rewards/hyperparams, method switch
src/confidence_utils.py        exp_005 confidence shaping math (verbatim)
src/gtpo_conf_trainer.py       GTPO-Conf via injection framework
src/grpo_s_conf_trainer.py     GRPO-S-Conf via injection framework
src/shaped_loss.py             injection helpers (from exp_057 fix)
src/format_tag_mask.py         multi-token tag masking
tests/test_confidence_utils.py shaping-math unit tests
plot_progress.py               4-panel comparison
run_059.sh                     sequential launcher (grpo, gtpo_conf, grpo_s_conf)
```
