# exp_064 — GRPO per-position C / logprob profile (for adaptive pos_discount)

Diagnostic: run **plain GRPO** (loss unchanged) and log, per rollout token,
`C_{i,t} = -mean_topk log p` (confidence/peakedness) and `logprob_{i,t}` (of the
sampled token), accumulated by ABSOLUTE position into count/sum/sumsq for three
groups: all / correct / incorrect rollouts (correctness = exact boxed match).
Goal: see how C and logprob depend on generation position, to design a more
adaptive pos_discount (currently a fixed g(t)=τ/(τ+t)).

- `src/grpo_posstats_trainer.py` — GRPOPosStatsTrainer (observes in
  _generate_and_score, one extra no-grad forward for C+logprob; saves diag/posstats_<ds>.npz).
- accumulator CPU unit-tested (`tests/test_posstats.py`).
- `run_posstats.sh` — gsm8k + bigmath, 300 steps.
- `analyze_posstats.py` — figures/exp064_posstats_<ds>.png (C, logprob, coverage vs position).

Same hyperparameters as exp_058/062 (Qwen3-4B-Base, ng=4, bs=1, ga=4, 300 steps).
