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

## Results

_(in progress)_
