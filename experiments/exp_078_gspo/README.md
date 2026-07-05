# exp_078 — GSPO baseline + our shaping on top (4 datasets)

GSPO (arXiv:2507.18071, Qwen team — the Qwen3 RL algorithm): sequence-level importance
sampling. TRL 0.23.1 native: `importance_sampling_level="sequence"`, paper clip range
`epsilon=3e-4, epsilon_high=4e-4`, `loss_type="grpo"` (with a sequence-level ratio the
per-token weights are constant, so per-sequence-mean aggregation reproduces the GSPO
objective). Everything else identical to the arc (Qwen3-4B-Base, ng=4, 300 steps, seed 3407).

- `gspo` — plain GRPOTrainer + GSPO config knobs
- `gspo_shaped` — our per-token shaping (gtpo_ema_flipped FIXED + pos_discount, λ0.7, k5)
  composed with the same GSPO config → tests algorithm-agnosticism (cf. exp_077 for DAPO)

## Run
```
bash run_setup.sh        # gspo ×4, then gspo_shaped ×4
python plot_compare.py   # figures/exp078_gspo_shaped.png (2×2 vs GRPO/posdisc refs)
python ../../skills/baseline_peak_table.py --dirs . \
  --baseline-suffix gspo --baseline-label GSPO \
  --ours-suffix gspo_shaped --ours-label "Ours (GSPO + shaping)"
```

## Results

_(in progress)_
