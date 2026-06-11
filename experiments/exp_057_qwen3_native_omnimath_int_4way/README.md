# exp_057 — Qwen3-4B native format, 4 methods on Omni-MATH integer subset

Exact re-run of **exp_055** (same model, same 4 methods, same hyperparameters,
same Qwen3 native `<think>...</think>` + `\boxed{}` format) on a **different,
harder dataset**: [KbsdJames/Omni-MATH](https://huggingface.co/datasets/KbsdJames/Omni-MATH),
restricted to its integer-answer subset.

## Why

exp_055 was an honest **null**: the easy Big-Math integer-2000 slice saturates
Qwen3-4B (~82% of the strict-answer ceiling at step 0), so per-token shaping had
no headroom — all 4 methods landed within ±0.13 reward. The standing
cross-experiment takeaway is "shaping only helps on weaker, non-saturated
baselines (Llama); on saturated Qwen3 it's a no-op."

Omni-MATH is competition-grade (olympiad/contest problems, difficulty 1–9.5,
mean ≈4.16). The GRPO baseline should be **far from saturated** here, so this is
the missing test: *given headroom on a strong model, does tag-masked shaping
(esp. `gtpo_conf`) finally beat plain GRPO?*

## Hypothesis

- **H1:** the GRPO baseline does NOT saturate on Omni-MATH (reward well below the
  ~+7 practical ceiling), unlike exp_055.
- **H2:** if H1 holds, `gtpo_conf` (the most reliable shaped variant) opens a
  measurable gap over GRPO — the result exp_055 couldn't show. Honest null is
  still a possible (and reportable) outcome if Omni-MATH is so hard that reward
  stays near the floor (no within-group advantage signal to shape).

## Setup

Identical to exp_055 except the dataset row.

| field | value |
|---|---|
| model | Qwen/Qwen3-4B |
| max_seq_length | 6656 (512 prompt + 6144 completion) |
| gpu_memory_utilization | 0.40 |
| LoRA | r=64, α=64, 7 modules (q,k,v,o,gate,up,down) |
| **dataset** | **KbsdJames/Omni-MATH, integer-answer subset (1971 problems), shuffled seed 3407, all kept** |
| per_device_train_batch_size | 1 |
| gradient_accumulation_steps | 4 |
| num_generations | 8 |
| max_steps | 1000 (stop early once plateau is visible) |
| lr / sched | 5e-6 cosine, warmup 0.1, wd 0.1, adamw_8bit |
| seed | 3407 (LoRA init, dataset shuffle, GRPOConfig) |

### Dataset construction (integer-answer filter)

Omni-MATH ships a single `test` split of 4428 problems with columns
`domain, difficulty, problem, solution, answer, source`. Answers are
competition-style, often LaTeX (`\boxed{60}`, `$30`, `1,700`, or non-integer
like `1 + \lceil n/2 \rceil`, `\frac{1}{2}`, `2\sqrt{3}`).

`is_integer_answer` keeps only answers that reduce to a plain signed integer
after a **minimal, safe** normalization (`train.py:_clean_integer`):
1. unwrap a single surrounding `\boxed{...}`,
2. strip surrounding inline-math `$`,
3. collapse thousands-grouped commas only (`1,700 → 1700`; **not** a European
   decimal like `3,7`, which is rejected),
4. keep iff the result matches `-?\d+`.

Result: **1971 / 4428** integer-answer problems. The `subset_size` cap (2000) is
larger than 1971, so the whole integer subset is used (shuffle just fixes order).

Difficulty of the kept subset: min 1.0, max 9.5, mean ≈4.16
(hist by rounded level: 1:89, 2:398, 3:15, 4:612, 5:596, 6:135, 7:54, 8:42, 9:26, 10:4).
Top domains: Algebra 917, Number Theory 436, Discrete Math 410, Applied 365,
Geometry 351.

### Format (Qwen3 native) — unchanged from exp_055

System prompt: *"Solve the problem step by step. Put your final integer answer
inside \boxed{}, like \boxed{42}."* `apply_chat_template` default leaves
`enable_thinking=True`, so the model emits `<think>...</think>` then the answer.

Three reward components (`train.py`):
- `reward_format_thinking`: +2.5 one matched `<think>...</think>` pair / +1.5 no thinking / -2.0 mismatched
- `reward_answer_boxed`: +3.0 correct integer in `\boxed{}` / -1.5 wrong / 0.0 no boxed (answer region is post-`</think>`; unclosed `<think>` → no answer reward)
- `reward_answer_numeric`: +1.5 correct last number after `</think>` / -0.5 wrong / 0.0 none

Practical max reward ≈ +7.

### Tag mask (per-token shaping trainers) — unchanged

`gtpo_conf` / `gtpo_ema_flipped` mask per-token shaping off on the 4 Qwen3-native
special tokens `<think>`, `</think>`, `<|im_start|>`, `<|im_end|>` (replaced by
the seq-level GRPO advantage there). No-op for `grpo` and `grpo_s_entropy`.

## Infra (this machine)

Runs **natively in a uv venv** — this box is itself an unprivileged docker
container with the GPU passed through, so the docker-wrapped launch from exp_055
(`run_055.sh`) is not usable here. Validated stack:
`unsloth 2026.3.7 · trl 0.23.1 · torch 2.9.1+cu128 · vllm 0.16.0 · transformers 4.57.6 · numpy 2.2.6`
on **NVIDIA H200 NVL 143 GB**. (Driver supports CUDA 12.8 max → torch/vllm pinned
to cu128 builds; default cu130 wheels error with "driver too old".)

## How to run

All four methods sequentially:

```bash
HF_TOKEN=<token> bash experiments/exp_057_qwen3_native_omnimath_int_4way/run_057.sh \
  > experiments/exp_057_qwen3_native_omnimath_int_4way/run_057.console.log 2>&1
```

Single method (venv active, `HF_TOKEN` exported):

```bash
source /root/aiim/venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_TOKEN=<token>
cd experiments/exp_057_qwen3_native_omnimath_int_4way
python train.py --method {grpo|grpo_s_entropy|gtpo_conf|gtpo_ema_flipped}
```

## Results

_In progress — last-50-step averages (same columns as exp_055). Methods stopped
early once a reward plateau is visible (stop-early workflow). Snapshot plot:
`figures/exp057_progress.png`._

| method | steps | reward L50 | answer_boxed L50 | answer_numeric L50 | format_thinking L50 | KL | clip% |
|---|---|---|---|---|---|---|---|
| grpo (baseline)        | 492 (stopped) | +2.56 | +1.27 | +0.63 | +0.65 | 0.010 | 46% |
| grpo_s_entropy         | — | — | — | — | — | — | — |
| gtpo_conf (tag-masked) | — | — | — | — | — | — | — |
| gtpo_ema_flipped       | — | — | — | — | — | — | — |

**grpo baseline (492 steps):** reward climbs +0.64 (first 50) → +2.56 (last 50),
peak rolling-20 +3.15 @ step 368. Far from the ~+7 ceiling — **non-saturated**,
the opposite of exp_055. Over training: format_thinking −0.55 → +0.65 (model
learns to close `<think>`), clip 72% → 46%, mean completion 5380 → 4575 tok,
`frac_reward_zero_std` 0.80 → 0.60 (more steps carry within-group signal). This
is the headroom exp_055 lacked — the shaped methods now have something to act on.

## Files

```
README.md                      this file
requirements.txt               numpy<2.3 overlay
run_057.sh                     native (no-docker) launcher, 4 methods sequential
plot_metrics.py                4-metric grid
plot_reward_dynamics.py        total-reward overlay (rolling-20)
plot_answer_boxed_dynamics.py  strict-answer reward overlay
train.py                       method-switch trainer, Omni-MATH integer filter, Qwen3 native format, 3 rewards, tag-mask
src/                           shaping trainers + format_tag_mask (verbatim from exp_055)
tests/test_methods.py          6 shaping correctness tests
```
