# Experiment 024: Reproduction of exp_005 (Confidence-Based GTPO and GRPO-S on GSM8K)

## Purpose
Re-run exp_005 with **byte-identical code** to verify that its win over the
GRPO baseline (exp_001) is reproducible and not a lucky seed / run-to-run
variance artefact. In particular: exp_005 GTPO-Conf reached peak reward
**9.500 @ step 268** vs exp_001 baseline peak **8.375 @ step 169** on the
same data (GSM8K, Llama-3.2-3B, 500 steps).

## What was copied verbatim from exp_005
- `src/confidence_utils.py` — identical
- `src/gtpo_conf_trainer.py` — identical
- `src/grpo_s_conf_trainer.py` — identical
- `src/__init__.py` — identical
- `tests/test_confidence_utils.py` — identical
- `train_gtpo_conf.py` — identical except `output_dir` and banner string
- `train_grpo_s_conf.py` — identical except `output_dir` and banner string

Seeds (`random_state=3407`), LR, optimizer, batch size, group size,
max_steps, reward functions, confidence hyper-params, all data-prep logic —
unchanged.

Sources of non-determinism still present: vLLM sampling, async dataloader
ordering, CUDA kernel non-associativity. So trajectories will differ
step-by-step even with identical code and seeds.

## Config (same as exp_005)
- Model: Llama-3.2-3B-Instruct (LoRA r=64)
- Dataset: GSM8K (train), 7473 examples
- Steps: 500, save every 250, bs=1×grad_accum=4, 4 generations
- Confidence: top_k=20, α₁=β₁=1.0, α₂=β₂=0.1, reward_threshold=0.0

## Reference numbers from exp_005

| Method | Step 250 | Step 500 | Peak | @Step | KL@500 |
|--------|----------|----------|------|-------|--------|
| GRPO baseline (exp_001) | 2.000 | 3.000 | 8.375 | 169 | 0.0855 |
| **GTPO-Conf (exp_005)** | **2.875** | 2.375 | **9.500** | 268 | 0.0691 |
| **GRPO-S-Conf (exp_005)** | 2.000 | 0.000 | 2.000 | 147 | 0.0233 |

## Success criteria (repro)
- GTPO-Conf reaches peak reward ≥ 8.5 within 500 steps (vs exp_001 baseline 8.375)
- GTPO-Conf final-500 format-exact consistent with exp_005's 2.25
- GRPO-S-Conf collapses again (reward ≈ 0 at step 500, format=0)

If (1) holds — the GTPO-Conf advantage is real. If not — it was noise.
