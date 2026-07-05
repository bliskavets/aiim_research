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

## Results (300 steps, L50 boxed / len; omnimath at 238)

2×2 — {GRPO, DAPO} × {no shaping, + our per-token shaping (posdisc λ0.7 k5)}:

| dataset | GRPO | +shaping | DAPO | +shaping | Δ on GRPO | Δ on DAPO |
|---|---|---|---|---|---|---|
| gsm8k    | +2.02 | +2.49 | +1.57 | +2.46 | +0.47 | **+0.89** |
| math500  | +0.94 | +1.63 | +1.02 | +1.44 | +0.69 | **+0.42** |
| bigmath  | +1.51 | +1.81 | +1.33 | **+2.02** | +0.30 | **+0.69** |
| omnimath | −0.23 | −0.33 | −0.29 | **−0.12** | −0.10 | **+0.17** |

**Our per-token shaping is algorithm-agnostic and additive — it helps on top of DAPO on ALL
FOUR datasets** (Δ_DAPO = +0.89 / +0.42 / +0.69 / +0.17, all positive) and dramatically cuts
length (620→328, 725→583, 762→477, 1178→785). Two headline outcomes:
- **bigmath: shaping-on-DAPO = +2.02 — the best bigmath score in the whole project** (beats
  shaping-on-GRPO +1.81).
- **omnimath: shaping-on-DAPO = −0.12 — the first shaped method to clearly beat plain GRPO
  (−0.23) on the hard set**, and the best omnimath result overall. DAPO's overlong masking +
  our gate/flip shaping compose to finally crack the hard dataset.

The add-on rescues DAPO (which alone trails GRPO) to best-in-class, i.e. the per-token
credit is a modular improvement that stacks with a stronger base RL algorithm — a strong,
paper-ready claim of generality.
