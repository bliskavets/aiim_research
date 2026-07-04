# exp_071 — zero-variance gate (roadmap setup 1/5)

See `analysis/exp055-070_deep_analysis.md` §3, §5.1.

**Problem:** when a group's rewards have zero variance (std(R)=0 — e.g. all rollouts wrong
on omnimath: 40–50% of groups), TRL sets all advantages to 0, and `is_pos = adv > 0` sends
EVERY rollout to O− → full-strength z-normed penalty with zero correctness information.
Half of the omnimath updates are pure noise; plain GRPO does nothing there and wins.

**Setup:** current best (gtpo_ema_flipped FIXED + pos_discount, λ=0.7, k=5) + hard gate:
`group_has_signal(adv)` false → return plain-GRPO zeros for the group. Metric `zvgate/gated`
logs the gated fraction.

**Run:** `bash run_setup.sh` (omnimath first — the target), `python plot_compare.py`
→ `figures/exp071_zvgate.png` (vs best posdisc λ0.7 k5 + GRPO).

**Expected:** omnimath gap to GRPO closes/flips; no regression on gsm8k/math500/bigmath.

## Results (300 steps, L50 boxed / len)

| dataset | GRPO | best (posdisc λ0.7 k5) | **zvgate** |
|---|---|---|---|
| gsm8k    | +2.02 / 414 | +2.49 / 317 | **+2.49 / 314** (=best) |
| math500  | +0.94 / 942 | **+1.63** / 635 | +1.55 / 632 |
| bigmath  | +1.51 / 622 | +1.81 / 529 | **+1.84 / 586** |
| omnimath | **−0.23** / 957 | −0.33 / 733 | −0.48 / 831 |

Gate fires exactly as predicted (~40% of omnimath groups; 0.63 in the last window as
saturation grows). **No regression** on the easy/medium datasets (gsm8k identical to best,
bigmath slightly up). **Omnimath not fixed though:** transient improvement mid-run
(window 180–240: −0.07, briefly the best-ever shaped score there) but the last window fell
back to −0.42 — removing the zero-variance noise updates alone is insufficient; the shaping
signal on the ~60% non-degenerate groups still hurts on the hard tail. Takeaway: keep the
gate (free, principled, no downside) but omnimath needs signal-level changes (see exp_072/075).
