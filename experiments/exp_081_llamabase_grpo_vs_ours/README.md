# exp_081 — Llama-3.2-3B **Base** (non-SFT): GRPO vs Ours (per-token shaping)

Rerun of exp_080 with the **base** (non-instruct) checkpoint `unsloth/Llama-3.2-3B`,
directly testing exp_080's hypothesis (3): the Qwen study used a BASE model
(Qwen3-4B-Base), so an SFT'd Instruct policy may explain why the shaping config failed to
transfer. Chat template borrowed from `unsloth/Llama-3.2-3B-Instruct` (same pattern as
Qwen3-4B-Base borrowing Qwen3-4B's template). All hyperparameters identical to
exp_080 / the Qwen study: ng=4, bs=1, ga=4, lr 5e-6 cosine, 300 steps, seed 3407,
max_seq 4096, integer boxed reward, β=0.

- `grpo` — plain GRPOTrainer baseline
- `ours` — gtpo_ema_flipped (FIXED) + pos_discount, EMA λ=0.7, C top-k=5 (no gate)

Datasets: gsm8k / math500 / bigmath / omnimath.

## Run
```
bash run_setup.sh        # grpo ×4, then ours ×4
python plot_compare.py   # figures/exp081_llama_base.png
python ../../skills/baseline_peak_table.py --dirs . \
  --baseline-suffix grpo --baseline-label GRPO \
  --ours-suffix ours --ours-label "Ours (GRPO + shaping)"
```

## Results

_(in progress)_
