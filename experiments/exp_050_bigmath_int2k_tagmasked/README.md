# exp_050 — Big-Math int-2000, Llama-3.2-3B · tag-masked shaping (full reward)

Same 4 candidates as exp_049, **full reward set** (format_exact + format_approximate + answer_exact + answer_numeric), but the **per-token shaping bonus is masked off on format-tag tokens** (`<start_working_out>`, `<end_working_out>`, `<SOLUTION>`, `</SOLUTION>`).

## Hypothesis

In exp_049 the GTPO-Conf and GTPO-EMA-Flipped shaped variants learned to emit correct numbers (num_maj 0.62 / 0.50) but lost the format-tag structure (exact_top 0.06 / 0.04). The per-token shaping may be distorting the gradient on tag-control tokens — those tokens are highly peaked (the model is very sure when emitting `<SOLUTION>`), so they collect either a large positive or large negative per-token bonus, and that drowns out the format-reward signal that should be teaching tag use.

Fix: on tokens that belong to a format-tag substring, replace the per-token shaped advantage with the seq-level GRPO advantage (no shaping). Content tokens still receive the full per-token shaping. The model should now learn the format from the standard GRPO objective on tag tokens, while keeping the per-token confidence/EMA signal on content tokens.

## Setup

Identical to exp_049 except for the tag-mask intervention:

- Model: Llama-3.2-3B-Instruct (4-bit, LoRA r=64/α=64/7 modules)
- Dataset: Big-Math-RL-Verified, integer-answer filter, first 2000 shuffled (seed 3407)
- Reward: full set (format_exact, format_approximate, answer_exact, answer_numeric)
- Training: bs=1 × grad_accum=4 × num_generations=4 (16 seqs/step), 500 steps
- max_seq_length=2560, gpu_memory_utilization=0.55, lr 5e-6 cosine, warmup 0.1
- seed=3407 throughout

## Tag-mask mechanism

The 4 tag strings tokenize as fixed token-id subsequences in Llama-3.2-3B. We pre-compute these (bare and with leading space) and slide-window match across each completion to build a (B, T) boolean mask. On True positions the trainer replaces the per-token advantage with `seq_advantages[b]` (broadcast). On False positions the per-token shaped advantage is unchanged.

- Applied in: `GTPOConfTrainer`, `GTPOEMAFlippedTrainer`
- No-op in: `GRPOTrainer` (no per-token shaping), `GRPOSTrainer` (shaping is at sequence level, not per-token)

So the experiment effectively tests two methods (gtpo_conf and gtpo_ema_flipped) with tag-mask, plus two controls (grpo and grpo_s_entropy) that should reproduce their exp_049 counterparts within run-to-run noise.

## Methods

| method | shaping | tag-mask effect |
|---|---|---|
| `grpo` | none (baseline) | n/a |
| `grpo_s_entropy` | seq-level entropy weighting (GRPO-S) | n/a (mask irrelevant — shaping is at seq level) |
| `gtpo_conf` | per-token confidence bonus (α₂=0.1, top-k=20) | active |
| `gtpo_ema_flipped` | per-token EMA-flipped advantages (α₁=0.9, α₂=0.1, λ=0.9) | active |

## Files

```
README.md               this file
requirements.txt        same as exp_049 (numpy<2.3 overlay)
run_050.sh              docker run script for all 4 methods sequential
train.py                method-switch trainer, full reward, tag-mask wiring
src/
  __init__.py
  entropy_utils.py            (copied verbatim from exp_049/exp_002)
  grpo_s_trainer.py           (copied verbatim from exp_049)
  confidence_utils.py         (copied verbatim from exp_049/exp_005)
  gtpo_conf_trainer.py        + format_tag_patterns kwarg
  ema_flipped_utils.py        (copied verbatim from exp_049/exp_026)
  gtpo_ema_flipped_trainer.py + format_tag_patterns kwarg
  format_tag_mask.py          new: pattern encoding, mask building, apply
tests/
  test_methods.py             6 shaping tests from exp_049
  test_format_tag_mask.py     4 new tests for the mask logic
```

## Results

(to be filled in by `plot_metrics.py` once the run completes)

| method | reward L50 | peak | answer_exact L50 | format_exact L50 | KL L50 |
|---|---|---|---|---|---|
| grpo               | tbd | tbd | tbd | tbd | tbd |
| grpo_s_entropy     | tbd | tbd | tbd | tbd | tbd |
| gtpo_conf          | tbd | tbd | tbd | tbd | tbd |
| gtpo_ema_flipped   | tbd | tbd | tbd | tbd | tbd |
