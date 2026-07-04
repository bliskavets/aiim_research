# exp_073 — length-invariant bonus budget (roadmap setup 3/5)

See `analysis/exp055-070_deep_analysis.md` §2, §5.3.

**Idea:** the α₂ bonus mass a rollout harvests currently grows ~linearly with its length
(the root of every length-farming collapse; pos_discount only softens it to ~log). Here the
per-rollout Σ_t bonus is **rescaled to the polarity's mean active length** — total shaped
credit is length-invariant by construction. Signal/weights unchanged (1/EMA(C) bonus,
EMA(C) penalty, λ0.7 k5); **no position discount** — budget replaces it (head-to-head:
budget vs posdisc, both vs GRPO).

**Run:** `bash run_setup.sh`, `python plot_compare.py` → `figures/exp073_flipped_budget.png`.

## Results

_(in progress)_
