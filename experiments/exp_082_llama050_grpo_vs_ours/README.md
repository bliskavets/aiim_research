# exp_082 — Llama-3.2-3B-Instruct on the exp_050 harness: GRPO vs Ours

exp_080 showed the Qwen harness (`<think>`+`\boxed{}` reward, 300 steps) does not fit
Llama; exp_081 showed Llama-3.2-3B **Base** cannot bootstrap at all (GRPO flat ~0). But
exp_050 proved Llama-3.2-3B-**Instruct** DOES learn Big-Math with a different harness.
This experiment redoes GRPO-vs-Ours on that proven harness:

- **Format:** custom tags `<start_working_out>…<end_working_out>` + `<SOLUTION>…</SOLUTION>`,
  taught explicitly in the system prompt.
- **Rewards (exp_050/exp_026 family):** format_exact (3.0) + format_approximate (±0.5/tag)
  + answer_exact (graded 3.0/1.5/1.0/0.5/−1.5) + answer_numeric (1.5/−0.5) — graded format
  signal Llama can climb (unlike the saturated format_thinking).
- **500 steps** (exp_050 shows Llama takes off at ~250–350), max_completion 2048, max_seq 2560.
- Model bf16 + LoRA r=64, ng=4, bs=1, ga=4, lr 5e-6 cosine, seed 3407 (as everywhere).

Methods:
- `grpo` — plain GRPOTrainer
- `ours` — gtpo_ema_flipped (FIXED, group-visible) + pos_discount, λ=0.7, k=5; tag-mask on
  the four custom tags (mechanism from exp_050, patterns swapped).

Datasets: gsm8k / math500 / bigmath / omnimath.

NOTE vs exp_050: exp_050's shaped curves predate the exp_058 FIX (B=1 degeneracy /
unsloth bypass) — only its GRPO control is trustworthy. Here the shaping is the FIXED
group-visible implementation.

## Run
```
bash run_setup.sh        # grpo ×4, then ours ×4 (500 steps each)
python plot_compare.py   # figures/exp082_llama050.png (metric: answer_exact)
python ../../skills/baseline_peak_table.py --dirs . \
  --baseline-suffix grpo --baseline-label GRPO \
  --ours-suffix ours --ours-label "Ours (GRPO + shaping)" \
  --metric reward_answer_exact/mean
```

## Results

_(in progress)_
