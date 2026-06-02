# exp_053 — Qwen3-4B GRPO vs GTPO-EMA-flipped on extra-hard Big-Math

Two-method stress test of the exp_052 finding (gtpo_ema_flipped won on hard subset) at a tougher difficulty + a larger compute budget.

## Setup

| field | value |
|---|---|
| model | Qwen/Qwen3-4B |
| max_seq | 4096 (512 prompt + 3584 completion) |
| gpu_memory_utilization | 0.50 (down from 0.55 — ng=16 needs more PyTorch activation memory than ng=4) |
| reward set | full (format_exact + format_approximate + answer_exact + answer_numeric) |
| dataset | Big-Math-RL-Verified, integer answer ∩ `llama8b_solve_rate < 0.125` (=1/8), shuffled seed 3407, 8000 examples |
| per_device_train_batch_size | 1 |
| gradient_accumulation_steps | 4 |
| num_generations | 16 |
| max_steps | 1000 |
| seed | 3407 |
| methods | `grpo` (no mask, control), `gtpo_ema_flipped` (tag-mask active) |

## Compute notes

- **Total sequences per gradient update**: 1 × 4 × 16 = **64** (4× exp_052)
- **Total gradient updates**: 1000 (2× exp_052)
- **Total compute**: ~16× exp_052 per method × 2 methods → estimated 4-5 days per method
- **Per-step memory**: same per_device_bs as exp_052, but ng=16 means 4× more rollouts in PyTorch forward/loss. gpu_memory_utilization dropped 0.55 → 0.50 to leave activation headroom.
- Optimal `(per_device_bs, ga)` chosen as `(1, 4)` — same per-step VRAM as exp_052 + safety margin for ng=16 activations. Trying per_device_bs > 1 with ng=16 at max_seq=4096 risks OOM.

## Hypothesis

exp_052 (Qwen3 on llama<0.3 subset) showed gtpo_ema_flipped winning by Δ +0.08 vs grpo baseline (the only positive shaping result on hard Qwen3). exp_053 tests this finding at:
1. Harder dataset (llama<0.125 vs <0.3)
2. Larger compute (ng=16 vs ng=4; 1000 steps vs 500)
3. With a single seed, but bigger sample (more rollouts per prompt → smoother advantage)

If the EMA-shaping advantage scales, exp_053 should show a clearer Δ vs baseline.

## Caveats from exp_052

- 52% completion clipping at max_completion=3584 on the harder subset → truncated rollouts feed noisy reward signal. exp_053 inherits this issue — if clipping stays this high, results need a follow-up at max_completion=6144.
- Single seed — ranking may not be stable.

## Files

```
README.md               this file
requirements.txt
run_053.sh              docker launcher, 2 methods sequential
plot_metrics.py         2-method × 4 metrics grid (skips methods without logs)
plot_reward_dynamics.py single-panel rolling-20 reward
train.py                method-switch trainer; --method ∈ {grpo, gtpo_ema_flipped}
src/                    same trainers/utils as exp_052 (format_tag_mask etc.)
tests/                  6 shaping + 4 tag-mask unit tests
```

## Results

(to be filled in after the run)

| method | reward L50 | peak | answer_exact L50 | format_exact L50 | exact_top | KL L50 | clip% |
|---|---|---|---|---|---|---|---|
| grpo               | tbd | tbd | tbd | tbd | tbd | tbd | tbd |
| gtpo_ema_flipped   | tbd | tbd | tbd | tbd | tbd | tbd | tbd |
