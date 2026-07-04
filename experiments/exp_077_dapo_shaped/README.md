# exp_077 — our best per-token shaping ON TOP of DAPO (4 datasets)

Tests whether our per-token reward-shaping add-on (the core contribution) is
**algorithm-agnostic**: does it still help when the base RL algorithm is DAPO
(Clip-Higher + token-level loss + overlong masking, exp_076) rather than plain GRPO?

**Method:** `PosDiscountTrainer` = gtpo_ema_flipped (FIXED, group-visible) + pos_discount,
λ=0.7, top_k=5, α₁=0.9, α₂=0.1 — our best per-token setup — with the exp_076 DAPO
GRPOConfig knobs applied (epsilon 0.2/0.28, loss_type=dapo, mask_truncated_completions).
The shaping composes cleanly: GroupShapedBase injects the 2-D token advantage, TRL then
applies DAPO's clipping + token-level normalization on top.

**Key comparison (2×2):** {GRPO, DAPO} × {no shaping, +our shaping}
- GRPO                → `train_{ds}_grpo.log`
- GRPO + shaping      → `train_{ds}_posdisc_lam0.7_k5.log`
- DAPO               → `train_{ds}_dapo.log` (copied from exp_076 by run_setup.sh)
- DAPO + shaping     → `train_{ds}_dapo_shaped.log`

If (DAPO+shaping − DAPO) > 0 with a similar sign to (GRPO+shaping − GRPO), the add-on is
algorithm-agnostic — a strong paper claim.

## Run
```
bash run_setup.sh        # dapo_shaped × 4 datasets (also copies DAPO baseline logs)
python plot_compare.py   # figures/exp077_dapo_shaped.png
```

## Results

_(in progress)_
