# exp_062 — non-entropy credit-assignment candidates vs references (3 datasets)

Follow-up to exp_058/061. Goal: try credit-assignment ideas that do NOT use
entropy (to differ from arXiv:2508.04349), each on the **group-visible FIXED
pattern** (shaped advantage computed on the full num_generations group in
`_generate_and_score`, propagated per-token — never the B=1 degeneracy that broke
the original gtpo_ema_flipped). Qwen3-4B-Base, same hyperparameters as exp_058's
last setups (ng=4, bs=1, ga=4, lr 5e-6 cosine, 300 steps, seed 3407,
max_seq 4096, integer exact-match reward).

## Methods (7 overlaid per dataset)
References:
- **grpo** — plain baseline.
- **grpo_grop** — GROP length penalty as a reward term (reused from exp_061).
- **gtpo_ema_flipped_fixed** — entropy/confidence-flipped shaping, fixed (reused).

Non-entropy candidates (this experiment):
- **sign_gate** (6A) — FIXED shaping, then keep it only where its sign agrees with
  the GRPO advantage; else revert to GRPO (shaping never inverts the reward).
- **pos_discount** — FIXED shaping with a gentle position discount `g(t)=τ/(τ+t)`
  (τ=1024) on the α₂ bonus only (correctness term untouched). Softer than 1/√t.
- **raw_c** — same flipped formula but raw `C_{i,t}` (peakedness) instead of EMA(C).
- **ref_delta** (3A) — credit ∝ deviation from the frozen base:
  `δ = logπ_θ(o_t) − logπ_base(o_t)` (LoRA disabled for the reference), added on
  top of the GRPO advantage per polarity (cold-start-safe: δ=0 ⇒ plain GRPO).

## Datasets (integer-answer, exact-match)
GSM8K (easy, 2000) · MATH-500 integer subset (medium, 312) · Omni-MATH integer
subset (hard, 1971).

## Correctness guardrails (per the B=1 lesson)
- All shaping math is in pure functions `src/novel_shaping.py`, **CPU unit-tested**
  (`tests/test_candidates.py`, 9 tests: non-degeneracy, polarity direction,
  sign-gate logic, position-discount, ref-delta cold-start = GRPO).
- All candidates use one tested group-visible code path (`GroupShapedBase`).
- GPU smoke (2 steps) per candidate: `used_group_shaped=1.0` (2-D propagation OK);
  ref_delta validated over 15 steps (mean_abs_delta rises 0 → ~0.01 ⇒
  `disable_adapter` truly toggles the base; cold-start additive form avoids a dead
  start).

## Run
```
HF_TOKEN=... ./run_overnight.sh     # 15 runs (grpo + 4 candidates) x 3 datasets
python plot_compare.py              # figures/exp062_compare.png  (7 methods x 3 datasets)
```
grop@grpo and gtpo_ema_flipped_fixed logs are reused from exp_061.

## Results (300 steps each, last-50 mean; figures/exp062_compare.png)

L50 boxed reward (length in parens):

| method | GSM8K (easy) | MATH-500 (med) | Omni-MATH (hard) |
|---|---|---|---|
| GRPO                 | +2.02 (414) | +0.94 (942) | **−0.23** (957) |
| GROP @ GRPO          | +2.06 (352) | +0.96 (831) | **−0.12** (720) |
| flipped FIXED        | +2.01 (274) | +1.03 (546) | −0.40 (859) |
| **sign_gate** (6A)   | +2.21 (359) | +1.14 (707) | −0.60 (892) |
| **pos_discount**     | **+2.50** (294) | **+1.34** (686) | −0.38 (1068) |
| **raw_C** (no EMA)   | +2.44 (240) | +1.09 (585) | −0.42 (785) |
| **ref_delta** (3A)   | +1.67 (483) | +1.00 (755) | −0.59 (954) |

**Findings**
- **pos_discount is the standout:** beats every reference on BOTH easy (+2.50 vs
  GRPO +2.02) and medium (+1.34 vs +0.94) with controlled length — a genuine
  improvement over GRPO (rare in this project, where shaping usually ties/loses).
  The gentle `g(t)=τ/(τ+t)` discount on the exploration bonus (correctness term
  untouched) concentrates the bonus on early decision tokens without starving the
  answer.
- **raw_C and sign_gate also beat the references on easy & medium** (raw_C is the
  most concise everywhere; sign_gate's "never invert the reward" gate helps when
  there's enough correct signal). raw_C ≈ EMA-free is competitive → the EMA
  smoothing isn't load-bearing.
- **ref_delta is the weakest candidate** — below refs on easy (+1.67), ~GRPO on
  medium. The deviation-from-base signal stays tiny (|δ|~0.01) at lr 5e-6, so it
  is mostly GRPO + noise; on easy that noise hurts. Honest negative.
- **Difficulty cliff (consistent with exp_061):** on Omni-MATH (hard) ALL methods
  go negative and the **plain references win** (GROP@GRPO −0.12, GRPO −0.23);
  every shaped candidate underperforms (sign_gate worst −0.60, pos_discount longest
  1068). When the base rarely solves the task, the per-token shaping has little
  correct signal to amplify and adds variance.
- **Takeaway:** the position-discounted exploration bonus (pos_discount) is the
  most promising non-entropy idea — it's the only candidate that consistently
  beats GRPO where the model is actually learning (easy+medium). Hard-task regime
  needs either more steps or difficulty gating before shaping helps.
