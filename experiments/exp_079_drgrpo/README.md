# exp_079 — Dr.GRPO baseline + our shaping on top (4 datasets)

Dr.GRPO (arXiv:2503.20783, "Understanding R1-Zero-Like Training"): unbiased GRPO —
constant-normalized token loss (`loss_type="dr_grpo"`) and NO std scaling of rewards
(`scale_rewards="none"`), removing the length and question-difficulty biases. Directly
relevant to our length-bias analysis (analysis/exp055-070_deep_analysis.md §2).
Everything else identical to the arc (Qwen3-4B-Base, ng=4, 300 steps, seed 3407).

- `drgrpo` — plain GRPOTrainer + Dr.GRPO config knobs
- `drgrpo_shaped` — our per-token shaping (gtpo_ema_flipped FIXED + pos_discount, λ0.7, k5)
  composed with the same config → algorithm-agnosticism test (cf. exp_077/078)

## Run
```
bash run_setup.sh        # drgrpo ×4, then drgrpo_shaped ×4
python plot_compare.py   # figures/exp079_drgrpo_shaped.png
python ../../skills/baseline_peak_table.py --dirs . \
  --baseline-suffix drgrpo --baseline-label "Dr.GRPO" \
  --ours-suffix drgrpo_shaped --ours-label "Ours (Dr.GRPO + shaping)"
```

## Results

_(in progress)_
