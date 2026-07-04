# exp_075 — final combo: gate + branch + budget (roadmap setup 5/5)

See `analysis/exp055-070_deep_analysis.md` §5.5. The paper-candidate assembly.

**Setup:** zero-variance gate (exp_071) + bounded branching signal h (exp_072) +
length-invariant bonus budget (exp_073, replaces posdisc), λ=0.7, k=5, α₁=0.9, α₂=0.1.
All three principled ingredients, no heuristic decay.

Queued last so per-ingredient results (071–073) arrive first; if a component underperforms
there, re-assembly is cheap (config-level).

**Run:** `bash run_setup.sh`, `python plot_compare.py` → `figures/exp075_final_combo.png`
(vs best posdisc λ0.7 k5 + GRPO). Metrics: `final_combo/gated`, `final_combo/mean_h`.

## Results

_(in progress)_
