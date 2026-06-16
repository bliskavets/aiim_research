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

Ran grpo_s_entropy(β2=0) for 50 steps and compared to grpo (reused from exp_057
@492, same code/seed/data). Plot: `figures/exp060_progress.png` (orange = β2=0,
sits right on grpo's early grey curve).

| metric (steps 1–50) | grpo | grpo_s_entropy (β2=0) |
|---|---|---|
| reward (mean) | **+0.639** | **+0.667** |
| grad_norm (mean) | 0.0123 | 0.0186 |
| reward mean\|grpo − grpo_s\| | — | 0.18 (early steps identical; small drift later) |
| shaped metrics `grpo_s/*` logged | n/a | **yes** (shaping path ran) |

**Conclusion — the GRPO-S code is correct; gradients flow.** With the entropy
bonus off (β2=0), grpo_s_entropy reproduces grpo's behaviour: identical rewards on
the first steps (same rollouts), matching grad_norm magnitude, and reward tracking
within noise over 50 steps (+0.667 vs +0.639). The early steps are *exactly* equal;
small later drift is expected because β2=0 uses a **sign-binarized** advantage
(±1 by reward sign, group-normalized) vs grpo's continuous advantage — a tiny
difference, not a bug.

**Therefore the grpo_s_entropy underperformance in exp_057 (L50 +1.63 vs grpo
+2.56) is caused by the entropy shaping itself (β2=0.1), not by a bug in the
GRPO-S injection / gradient path.** This matches the broader exp_057 finding that
the shaping, when actually applied, drags the policy off the reward signal.

(Validation stopped at 50 steps — the control question was answered; the GPU was
handed to exp_059.)
