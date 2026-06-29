# exp_061 — GROP @ GRPO vs FIXED gtpo_ema_flipped across 3 datasets

Follow-up to exp_058. Compares the two length-healthy setups from exp_058 across
three integer-answer (exact-match verifiable) math datasets on **Qwen3-4B-Base**.

## Setups (2 per dataset, identical hyperparameters to exp_058's last setups)
- **grpo_grop** — plain GRPO + Group Relative Overlong Punishment (arXiv:2508.04349
  Appendix D) added as a **reward** term (paper-faithful injection).
- **gtpo_ema_flipped_fixed** — gtpo_ema_flipped with the shaped advantage computed
  on the **full group** in `_generate_and_score` (the B=1 fix from exp_058,
  see `../exp_058.../DIAG_LENGTH_EXPLOSION.md`).

Qwen3-4B-Base, ng=4, bs=1, ga=4, lr 5e-6 cosine, 300 steps, seed 3407,
max_seq 4096 (512 prompt + 3584 completion), reward = format_thinking +
answer_boxed(±3 exact integer) + answer_numeric, Qwen3-native tag-mask.

## Datasets (all integer-answer subset, exact-match verifiable)
| key | source | split | integer N |
|---|---|---|---|
| gsm8k    | openai/gsm8k (main)        | train | 7473 (all; gold after "####") |
| math500  | HuggingFaceH4/MATH-500     | test  | 312 (integer subset) |
| omnimath | KbsdJames/Omni-MATH        | test  | 1971 (integer subset) |

Difficulty spread: GSM8K (easy) → MATH-500 (medium) → Omni-MATH (hard).

## Run
```
HF_TOKEN=... ./run_overnight.sh        # 6 runs sequential, ~8h
python plot_compare.py                 # figures/exp061_compare.png
```

## Files
```
train.py            parameterized (--dataset {gsm8k,math500,omnimath} --method {grpo_grop,gtpo_ema_flipped_fixed})
run_overnight.sh    all 6 runs (dataset-outer)
plot_compare.py     3-column (per dataset) x 2-row (boxed / length) comparison
src/                fixed trainer + GROP helper (copied from exp_058)
```

## Results (300 steps each, last-50 mean; figures/exp061_compare.png)

| dataset | method | L50 len | L50 boxed |
|---|---|---|---|
| GSM8K (easy)    | GROP @ GRPO            | 352 | **+2.06** |
| GSM8K (easy)    | gtpo_ema_flipped FIXED | **274** | +2.01 |
| MATH-500 (med)  | GROP @ GRPO            | 831 | +0.96 |
| MATH-500 (med)  | gtpo_ema_flipped FIXED | **546** | **+1.03** |
| Omni-MATH (hard)| GROP @ GRPO           | 720 | **-0.12** |
| Omni-MATH (hard)| gtpo_ema_flipped FIXED | 859 | -0.40 |

**Findings**
- **No length explosion anywhere** — both setups stay length-healthy across all
  three datasets (max ~860 tok, ≪ the 3584 cap). The exp_058 fixes generalize.
- **Easy (GSM8K):** both learn strongly and tie on quality (+2.06 vs +2.01); the
  FIXED shaped method is more concise (274 vs 352 tok).
- **Medium (MATH-500):** FIXED wins on both axes — higher boxed (+1.03 vs +0.96)
  AND shorter (546 vs 831).
- **Hard (Omni-MATH):** both go slightly negative (the base model rarely solves
  this set in 300 steps), and here **GROP @ GRPO edges out** (-0.12 vs -0.40);
  FIXED shows mild length creep (859) with little correct signal to anchor its
  shaping. No collapse, but the shaped method's exploration bonus is less helpful
  when correctness is scarce.
- **Difficulty trend:** quality falls monotonically with difficulty (+2.0 → +1.0
  → negative) as expected; FIXED's edge over GROP erodes as the task gets harder
  (better → tied/better → worse), consistent with the shaped bonus needing some
  correct rollouts to be useful.
