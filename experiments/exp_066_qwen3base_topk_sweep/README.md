# exp_066 — top_k sweep for confidence C (base pos_discount + FIXED λ=0.7)

Reflect on how C is computed: C_{i,t} = -mean over top-k logprobs. Sweep
top_k ∈ {5,10,20,40} on the prior-best setup (pos_discount + gtpo_ema_flipped
FIXED λ=0.7). k=20 is the default (= exp_063 posdisc_lam0.7, reused). Only k affects
the confidence granularity feeding EMA(C) in the flipped shaping.

12 new runs (k=5/10/40 × 4 datasets), overlay vs GRPO + k=20. Same hyperparameters
as exp_058/062/063. train.py gains --top_k override. run_topk.sh; chain_topk.sh
queues it after exp_065's batch (GPU busy).
