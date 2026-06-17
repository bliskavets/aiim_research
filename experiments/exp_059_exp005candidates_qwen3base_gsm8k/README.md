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

All three methods ran 500 steps with shaping **actually applied** (each shaped
run logs its `<method>/*` metrics; gradients flow). Plot: `figures/exp059_progress.png`.

| method | steps | reward L50 | format_exact L50 (max +3) | answer_exact L50 (max +3) | answer_numeric L50 | KL | Δreward vs grpo |
|---|---|---|---|---|---|---|---|
| **grpo** (baseline)        | 500 | **+8.66** | +2.96 | +2.53 | +1.26 | 0.036 | — |
| grpo_s_conf (seq-level)    | 500 | **+8.84** | +2.97 | +2.65 | +1.28 | 0.055 | **+0.18 (≈tie)** |
| gtpo_conf (per-token)      | 500 | **+0.86** | +0.84 | −0.11 | +0.68 | 0.0022 | **−7.80** |

Reward trajectory (100-step blocks):

| steps | grpo | grpo_s_conf | gtpo_conf |
|---|---|---|---|
| 1–100   | −0.23 | +0.67 | −0.16 |
| 201–300 | +7.73 | +7.76 | +0.79 |
| 401–500 | +8.71 | +8.93 | +1.02 |

### Findings

**On Qwen3-4B-Base / GSM8K the result splits by shaping granularity:**

- **grpo** cleanly learns the format and solves GSM8K: reward −0.23 → +8.71
  (near the ~9.5 reward ceiling), format_exact +2.96/3, KL 0.036.
- **grpo_s_conf (sequence-level confidence)** ≈ **ties grpo** — it tracks the
  baseline step-for-step (+8.84 vs +8.66, within noise; marginally higher
  answer_exact). The mild seq-level reweighting (β2=0.1) does not distort the
  per-token gradient, so learning proceeds normally. It does **not beat** grpo.
- **gtpo_conf (per-token confidence)** **badly underperforms** (+0.86): it never
  learns the format (format_exact +0.84, answer_exact −0.11) and **KL ≈ 0.002**
  (the policy barely moves). The per-token confidence advantage is z-normalized
  and ~uncorrelated with the reward (the exp_057 caveat), so it injects a
  reward-misaligned gradient that prevents the base model from learning the task.

**Takeaway:** with the shaping actually applied (vs silently bypassed), **neither
confidence variant beats the GRPO baseline.** The sequence-level variant is benign
(ties); the **per-token variant is actively harmful** — strongest on the base
model, which has the format to learn from scratch. This is consistent with
exp_057 (per-token shaping drags Qwen3 off the reward signal) and **inverts the
original exp_005 conclusion** ("GTPO-Conf > GRPO-S-Conf"), which was almost
certainly measured with the shaping bypassed (= plain GRPO, run-to-run noise).

Note: `grpo_s_conf` uses a full-group microbatch (bs=num_generations, ga=1) so its
within-group re-normalization is valid; `grpo`/`gtpo_conf` use bs=1, ga=4 (robust
to single-sequence microbatches). Same effective 4 sequences / optimizer step.

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
