# exp_055 — Qwen3-4B native format, 4 methods on Big-Math int-2000

GRPO baseline vs three reward-shaped candidates on the **easy** Big-Math
integer-answer subset, using the **Qwen3 native format** (`<think>...</think>`
+ `\boxed{}`). This is the corrected re-run of exp_051, which used a custom
`<start_working_out>`/`<SOLUTION>` system prompt that fought Qwen3's
trained-in thinking-mode protocol.

## TL;DR result

All methods cluster within ±0.13 reward — on this saturated subset
(Qwen3-4B already at ~82% of the strict-answer ceiling out-of-the-box)
shaping has no measurable effect. There is no signal to amplify.

| method | steps | reward L50 | answer_boxed L50 | answer_numeric L50 | format_thinking L50 | KL | clip% |
|---|---|---|---|---|---|---|---|
| grpo (baseline)         | 738 | +5.76 | +2.45 | +1.22 | +2.08 | 0.008 | 11% |
| grpo_s_entropy          | 738 | +5.83 | +2.49 | +1.24 | +2.11 | 0.009 | 10% |
| gtpo_conf (tag-masked)  | 450 | +5.72 | +2.48 | +1.24 | +2.00 | 0.004 | 12% |
| gtpo_ema_flipped        | —   | not run (skipped — exp_054 already showed it ties/loses on Qwen3 native) |

Max practical reward ≈ +6 (format_thinking +2.5, answer_boxed +3.0, answer_numeric +1.5). All three methods were stopped early once their plateau was visible (steps logged above), to save GPU days.

Plots:
- `figures/exp055_reward_dynamics.png` — total reward, all methods overlaid (curves sit on top of each other)
- `figures/exp055_answer_boxed_dynamics.png` — strict integer-in-`\boxed{}` reward, shows saturation at ~82% ceiling

## Setup

| field | value |
|---|---|
| model | Qwen/Qwen3-4B |
| max_seq_length | 6656 (512 prompt + 6144 completion) |
| gpu_memory_utilization | 0.40 |
| LoRA | r=64, α=64, 7 modules (q,k,v,o,gate,up,down) |
| dataset | SynthLabsAI/Big-Math-RL-Verified, integer-answer filter, first 2000 shuffled (seed 3407), **no llama8b filter** (this is the "easy" slice) |
| per_device_train_batch_size | 1 |
| gradient_accumulation_steps | 4 |
| num_generations | 8 |
| max_steps | 1000 (all 3 methods stopped early) |
| lr / sched | 5e-6 cosine, warmup 0.1, wd 0.1, adamw_8bit |
| seed | 3407 (LoRA init, dataset shuffle, GRPOConfig) |

### Format (Qwen3 native)

System prompt: *"Solve the problem step by step. Put your final integer answer inside \boxed{}, like \boxed{42}."* `apply_chat_template` default leaves `enable_thinking=True`, so the model emits `<think>...</think>` then the answer.

Three reward components (in `train.py`):
- `reward_format_thinking`: +2.5 one matched `<think>...</think>` pair / +1.5 no thinking / -2.0 mismatched
- `reward_answer_boxed`: +3.0 correct integer in `\boxed{}` / -1.5 wrong / 0.0 no boxed (strict; answer region is post-`</think>`, or whole text if no thinking, or NONE if `<think>` opened-but-not-closed — blocks the "boxed-inside-unclosed-think" exploit)
- `reward_answer_numeric`: +1.5 correct last number after `</think>` / -0.5 wrong / 0.0 none

### Tag mask (for the per-token shaping trainers)

`gtpo_conf` / `gtpo_ema_flipped` mask per-token shaping off on 4 Qwen3-native special tokens: `<think>`, `</think>`, `<|im_start|>`, `<|im_end|>` (8 patterns with leading-space variants). On those positions the per-token shaped advantage is replaced by the seq-level GRPO advantage. No-op for `grpo` (no shaping) and `grpo_s_entropy` (seq-level shaping).

## How to run

All four methods sequentially (full 1000 steps each):

```bash
cd /mnt/data/aiim_research
HF_TOKEN=<token> bash experiments/exp_055_qwen3_native_int2k_4way/run_055.sh \
  > experiments/exp_055_qwen3_native_int2k_4way/run_055.console.log 2>&1
```

Single method:

```bash
# inside the unsloth/unsloth container with /opt/venv active (see run_055.sh)
python train.py --method {grpo|grpo_s_entropy|gtpo_conf|gtpo_ema_flipped}
```

Resume scripts (stop-early workflow): `run_055_resume_shaped.sh` (3 shaped methods), `run_055_resume_conf_ema.sh` (gtpo_conf + gtpo_ema_flipped only). They keep already-trained `train_*.log` intact.

**For full reproduction context on another machine (infra, stack versions, the whole shaping-research arc, why this experiment exists, what to expect, and a known seed pitfall) see `HANDOFF.md` in this folder.**

## Files

```
README.md                      this file
HANDOFF.md                     self-contained context for replicating on another machine
requirements.txt               numpy<2.3 overlay (rest via /opt/venv)
run_055.sh                     docker launcher, 4 methods sequential
run_055_resume_shaped.sh       resume: 3 shaped methods only
run_055_resume_conf_ema.sh     resume: gtpo_conf + gtpo_ema_flipped
plot_metrics.py                4-metric grid
plot_reward_dynamics.py        total-reward overlay (rolling-20)
plot_answer_boxed_dynamics.py  strict-answer reward overlay
train.py                       method-switch trainer, Qwen3 native format, 3 rewards, tag-mask wiring
src/
  format_tag_mask.py           tag → token-id patterns, build mask, blend advantages
  entropy_utils.py / grpo_s_trainer.py        GRPO-S (seq-level entropy)
  confidence_utils.py / gtpo_conf_trainer.py  GTPO per-token confidence
  ema_flipped_utils.py / gtpo_ema_flipped_trainer.py  GTPO-EMA-flipped
tests/
  test_methods.py              6 shaping correctness tests
train_grpo.log / train_grpo_s_entropy.log / train_gtpo_conf.log   training logs
figures/                       the two comparison plots
```
