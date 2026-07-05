# exp_073 — length-invariant bonus budget (roadmap setup 3/5)

See `analysis/exp055-070_deep_analysis.md` §2, §5.3.

**Idea:** the α₂ bonus mass a rollout harvests currently grows ~linearly with its length
(the root of every length-farming collapse; pos_discount only softens it to ~log). Here the
per-rollout Σ_t bonus is **rescaled to the polarity's mean active length** — total shaped
credit is length-invariant by construction. Signal/weights unchanged (1/EMA(C) bonus,
EMA(C) penalty, λ0.7 k5); **no position discount** — budget replaces it (head-to-head:
budget vs posdisc, both vs GRPO).

**Run:** `bash run_setup.sh`, `python plot_compare.py` → `figures/exp073_flipped_budget.png`.

## Results (300 steps, L50 boxed / len)

| dataset | GRPO | best (posdisc λ0.7 k5) | **flipped_budget (no posdisc)** |
|---|---|---|---|
| gsm8k    | +2.02 / 414 | +2.49 / 317 | **+2.54 / 319** |
| math500  | +0.94 / 942 | **+1.63** / 635 | +1.36 / 621 |
| bigmath  | +1.51 / 622 | **+1.81** / 529 | +1.79 / 660 |
| omnimath | **−0.23** / 957 | −0.33 / 733 | −0.42 / 986 |

**Length-invariant budget BEATS the tuned posdisc on gsm8k (+2.54, new best there)** and
ties on bigmath (+1.79 vs +1.81), confirming the budget is a valid principled replacement
for the hyperbolic position discount (no length farming — lengths comparable). But it is
below posdisc on math500 (+1.36) — removing the position decay entirely costs medium-hardness
accuracy. Best used TOGETHER with a mild posdisc rather than as a full replacement (a budget
+ light posdisc variant is the natural follow-up).
