# exp_054 — Qwen3-4B native format, GRPO vs GTPO-EMA-flipped on extra-hard Big-Math

Re-run of exp_053 using the **Qwen3 native format** instead of our custom `<start_working_out>` / `<SOLUTION>` tags. The custom-tag setup in exp_051/052/053 was confounded — Qwen3-4B was already trained with `<think>...</think>` as a structural protocol (registered as single special tokens 151667/151668) and `apply_chat_template` defaults to `enable_thinking=True`. Our system prompt asked for a different format; the model fought two competing format expectations, spent most of its rollout budget on `<think>` thinking, and hit 52%–76% completion clipping on hard subsets.

## What changed vs exp_053

| field | exp_053 (custom format) | exp_054 (Qwen3 native) |
|---|---|---|
| SYSTEM_PROMPT | "Place reasoning between `<start_working_out>` and `<end_working_out>`, answer between `<SOLUTION>...</SOLUTION>`" | "Solve step by step. Put your final integer answer inside `\boxed{}`, like `\boxed{42}`." |
| reward_funcs | format_exact + format_approximate + answer_exact + answer_numeric (4 funcs scoring our tags) | reward_format_thinking + reward_answer_boxed + reward_answer_numeric (3 funcs scoring Qwen3 native shape) |
| max possible reward | ~7.5 (3+1.5+3) | 5.5 (1+3+1.5) |
| tag-mask | 6 tags (4 ours + `<think>`/`</think>`) | 4 Qwen3-native tags (`<think>`, `</think>`, `<|im_start|>`, `<|im_end|>`) |

Everything else identical to exp_053: Qwen3-4B, max_seq=4096, gpu_memory_utilization=0.50, dataset = integer ∩ `llama8b_solve_rate < 0.125` (8000 examples, seed 3407), bs=1×ga=4×ng=16, 1000 steps, methods = grpo (no mask) + gtpo_ema_flipped (mask active).

## Reward components (exp_054)

```
reward_format_thinking  (soft):
  +1.0   exactly one matched <think>...</think> pair
  +0.5   no <think> blocks (Qwen3 chose to skip thinking — also fine)
  -0.5   multiple or mismatched blocks

reward_answer_boxed     (strict, integer-only):
  +3.0   \boxed{N} matches GT exactly
  -1.5   \boxed{N} found but N != GT
   0.0   no \boxed{} block

reward_answer_numeric   (fallback):
  +1.5   last number after </think> matches GT
  -0.5   last number is something else
   0.0   no number found
```

A perfect rollout reaches +5.5. A wrong-but-tried rollout sits around -1.0 to -2.0. Junk gets 0.0.

## Tag-mask details

`encode_tag_patterns` is called on 4 Qwen3-native tags. With Qwen3-4B tokenizer this yields 8 unique patterns (each tag × bare + leading-space variant). All 4 base tags are SINGLE special-token-ids (`<think>=151667`, `</think>=151668`, `<|im_start|>=151644`, `<|im_end|>=151645`), so the mask is one-token-precise on these. On positions matching any pattern in `completion_ids`, the per-token shaped advantage is replaced with the seq-level GRPO advantage.

Effect: GTPO-EMA-flipped's per-token EMA bonus does not land on the highly-peaked `</think>` close token (which caused the 76% clipping feedback loop in exp_053). Content tokens still receive the full shaping.

## Hypothesis

If exp_051/052/053 underperformed because of the format conflict, then exp_054 with native format should produce cleaner method comparison: lower clipping, smaller KL distortion from format-fighting, more interpretable Δ between GRPO and GTPO-EMA-flipped.

## Files

```
README.md
requirements.txt
run_054.sh                  docker launcher, 2 methods sequential
plot_metrics.py             4-metric grid
plot_reward_dynamics.py     single-panel rolling-20 reward
train.py                    rewritten for native format + 3 native rewards + 4-tag mask
src/                        same trainer modules as exp_053 (no shaping-code changes needed)
tests/test_format_tag_mask.py
                            14 tests including:
                              - exp_054 mask covers exactly the 4 Qwen3-native tags
                              - train.py does NOT carry exp_053's custom-tag constants
                              - encode_tag_patterns wires the 4 tags (not 6)
```

## Results

(to be filled in)

| method | reward L50 | peak | answer_boxed L50 | answer_numeric L50 | format_thinking L50 | clip% | KL |
|---|---|---|---|---|---|---|---|
| grpo               | tbd | tbd | tbd | tbd | tbd | tbd | tbd |
| gtpo_ema_flipped   | tbd | tbd | tbd | tbd | tbd | tbd | tbd |
