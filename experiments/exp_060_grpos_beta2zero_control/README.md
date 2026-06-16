# exp_060 — CONTROL: GRPO-S with β2=0 vs GRPO

**Question (raised by operator):** in exp_057, with the shaping bug fixed, even
`grpo_s_entropy` underperformed plain GRPO (L50 +1.63 vs +2.56). Is that the
entropy shaping, or a bug in the GRPO-S code / gradient path?

**Test:** set the GRPO-S entropy-bonus weight **β2 = 0** and re-run on the exact
exp_057 setup (Qwen3-4B instruct, Omni-MATH integer subset, Qwen3 native format,
same hyperparameters). With β2=0, `compute_grpo_s_rewards` collapses to
`shaped = +β1` for O+ (reward>0) and `−β1` for O− — a **sign-binarized,
group-normalized advantage** ("GRPO with binarized rewards"). This is *not*
byte-identical to GRPO's continuous advantage, but it must **learn ~like GRPO**
if the GRPO-S injection and gradient path are correct.

- grpo_s(β2=0) **tracks** grpo ⇒ the GRPO-S code/grad path is sound; the entropy
  shaping (β2>0) is what dragged exp_057 down.
- grpo_s(β2=0) **diverges** from grpo ⇒ bug in the grpo_s path.

Both methods use the fixed injection framework (`src/shaped_loss.py`): the shaped
advantage is computed in `compute_loss` and injected into unsloth's compiled loss
(which owns the chunked gradient). Shaped metrics (`grpo_s/*`) are logged, so we
can confirm the shaping actually ran and gradients flowed.

## Setup
Identical to exp_057 except: only `grpo` and `grpo_s_entropy` are run, and
`SHAPING_CONFIG["grpo_s_entropy"]["beta2"] = 0.0`. Qwen3-4B (instruct), Omni-MATH
integer subset (1971), bs=1, ga=4, ng=8, lr 5e-6 cosine, seed 3407, max_seq 6656.
Early-stopped once the grpo vs grpo_s(β2=0) relationship is clear.

## How to run
```bash
HF_TOKEN=<token> bash experiments/exp_060_grpos_beta2zero_control/run_060.sh \
  > experiments/exp_060_grpos_beta2zero_control/run_060.console.log 2>&1
```

## Results

_Pending — run in progress. Comparison plot: `figures/exp060_progress.png`._

| method | steps | reward L50 | boxed L50 | grad_norm (mean) | shaping ran? |
|---|---|---|---|---|---|
| grpo                  | — | — | — | — | n/a |
| grpo_s_entropy (β2=0) | — | — | — | — | (grpo_s/* logged) |

Conclusion: _to be filled — does grpo_s(β2=0) track grpo?_
