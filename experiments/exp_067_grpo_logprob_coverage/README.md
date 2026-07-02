# exp_067 — GRPO rollout logprob coverage (for adaptive top-k / nucleus)

Dump, per rollout token, the sorted top-K (=128) logprobs of the policy
distribution during a plain GRPO run (loss unchanged) + the rollout token
sequences. Then inspect how much probability mass top-k covers, to design
adaptive top-k / top-p (nucleus) / adaptive-nucleus selection for C.

- `src/grpo_lpdump_trainer.py` — GRPOLpdumpTrainer; saves diag/lpdump_<ds>/step_*.npz
  (completion_ids, mask, topk_lp fp16 (G,T,128), sampled_lp). Every step, first 100.
- `view_logprob_coverage.py <ds>` — coverage(k) mean/min/max/percentiles, k-needed
  for top-p nucleus thresholds, fraction of sampled tokens within top-k;
  figures/exp067_coverage_<ds>.png. Coverage math CPU-verified.

Run: `python train.py --dataset <ds> --method grpo_lpdump`
