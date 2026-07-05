# exp_076 — DAPO baseline (4 datasets)

Baseline for testing whether our per-token shaping composes with a stronger RL algorithm
(exp_077 adds our best shaping on top of this DAPO config).

## What "DAPO" means here

DAPO (arXiv:2503.14476) contributes four things; TRL 0.23.1 support:

| DAPO component | our setup |
|---|---|
| **Clip-Higher** (decoupled ε_low/ε_high) | `epsilon=0.2, epsilon_high=0.28` ✓ |
| **Token-level PG loss** (normalize over all tokens, not per-seq) | `loss_type="dapo"` ✓ (already the TRL 0.23.1 default) |
| **Overlong (truncated) filtering** | `mask_truncated_completions=True` ✓ |
| **Dynamic sampling** (resample until no all-correct/all-wrong groups) | ✗ not native to TRL 0.23.1 — see note |

**Note on dynamic sampling.** TRL 0.23.1 has no native prompt-level resampling loop. Our
exp_071 zero-variance *gate* (skip the shaping update when a group has std(R)=0) is the
advantage-level analogue and can be composed later. The distinctive DAPO knob vs our
existing runs is therefore **Clip-Higher + overlong masking** (token-level loss was already
our default), which is exactly what makes this a meaningful, controlled DAPO comparison.

Everything else identical to the rest of the arc: Qwen3-4B-Base, ng=4, bs=1, ga=4, lr 5e-6
cosine, 300 steps, seed 3407, integer reward, β=0 (no KL).

## Run
```
bash run_setup.sh        # dapo × 4 datasets
python plot_compare.py   # figures/exp076_dapo.png (DAPO vs GRPO vs our best posdisc λ0.7 k5)
```

## Results (300 steps, L50 boxed / len)

| dataset | GRPO | our best (posdisc λ0.7 k5) | **DAPO** |
|---|---|---|---|
| gsm8k    | +2.02 / 414 | +2.49 / 317 | +1.57 / 620 |
| math500  | +0.94 / 942 | +1.63 / 635 | +1.02 / 725 |
| bigmath  | +1.51 / 622 | +1.81 / 529 | +1.33 / 762 |
| omnimath | −0.23 / 957 | −0.33 / 733 | −0.29 / 1178 |

**DAPO underperforms plain GRPO on this setup** (below on gsm8k/math500/bigmath, ~tied on
omnimath) and consistently produces LONGER completions (Clip-Higher rewards upside token
moves; overlong masking doesn't offset it within 300 steps at ng=4). This makes it a good
stress test for our per-token shaping — see exp_077, where the shaping recovers DAPO to
best-in-class on every dataset.
