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

## Results
(filled after the run — figures/exp062_compare.png)
