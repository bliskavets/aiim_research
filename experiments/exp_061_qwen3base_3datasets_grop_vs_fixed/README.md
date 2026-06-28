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

## Results
(filled after the overnight run — see figures/exp061_compare.png)
