# exp_080 — Llama-3.2-3B-Instruct: GRPO vs Ours (per-token shaping)

Second base model for the paper (generality across architectures). Repeats the core
GRPO-vs-Ours comparison from the Qwen3-4B-Base study on **`unsloth/Llama-3.2-3B-Instruct`**
(the Llama model used in exp_001–016), **identical hyperparameters**: ng=4, bs=1, ga=4,
lr 5e-6 cosine, 300 steps, seed 3407, max_seq 4096, integer boxed reward, β=0 (no KL).

- `grpo` — plain GRPOTrainer baseline
- `ours` — our best per-token shaping: gtpo_ema_flipped (FIXED, group-visible) + pos_discount,
  EMA λ=0.7, C top-k=5, α₁=0.9, α₂=0.1

Datasets: gsm8k / math500 / bigmath / omnimath (all 4).

Note: the reward includes a `format_thinking` term (`<think>…</think>`); Llama-Instruct does
not emit those tags natively, so that component is ~constant across both runs — it cancels in
the GRPO-vs-Ours comparison (answer_boxed / answer_numeric carry the signal). Both arms share
the exact same reward, so the delta is a clean measure of the shaping's effect.

## Run
```
bash run_setup.sh        # grpo ×4, then ours ×4
python plot_compare.py   # figures/exp080_llama.png
python ../../skills/baseline_peak_table.py --dirs . \
  --baseline-suffix grpo --baseline-label GRPO \
  --ours-suffix ours --ours-label "Ours (GRPO + shaping)"
```

## Results

_(in progress)_
