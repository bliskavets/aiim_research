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

## Results (500 steps) — GRPO learns on this harness; OURS ACTIVELY PREVENTS LEARNING

gsm8k reward components by phase (grpo → ours):

| window | fmt_approx (grpo) | ans_num (grpo) | len | fmt_approx (OURS) | ans_num (OURS) | len |
|---|---|---|---|---|---|---|
| 1–100   | −1.14 | +0.35 | 228 | −1.13 | +0.18 | 217 |
| 100–250 | +0.03 | +0.83 | 221 | **−2.01** | +0.01 | 161 |
| 250–400 | +0.43 | +0.93 | 241 | **−2.50** | +0.00 | 103 |
| 400–500 | +0.47 | +1.01 | 225 | **−2.46** | +0.00 | 84 |

- **GRPO**: the exp_050 harness works — format_approximate climbs −1.1→+0.5, answer_numeric
  →+1.0, format_exact beginning to appear (+0.07) at step 500 (mid-takeoff, consistent with
  exp_050's ~250–350 takeoff; a longer run would finish the climb). Same shape on the other
  datasets. answer_exact still ~0 at 500 (needs format_exact first).
- **Ours**: shaping ANTI-learns on all 4 datasets — format never acquired, answer_numeric
  dies, completions shrink to ~80–540 tokens of junk.

**Mechanism (the real cross-model finding).** Our polarity split is `adv > 0` — group-
RELATIVE. On Qwen (strong base, format saturates instantly) advantage variance ≈ answer
correctness, so O+ ≈ "correct" and the shaping semantics hold. On Llama during cold-start
NOTHING is correct; the graded format reward still creates advantage variance, so O+ =
"slightly-less-bad junk" — the 1/EMA(C) bonus then actively reinforces junk exploration
while O− punishes the model's fluent (peaked) instruct-style text. Result: fluency is
suppressed, format never forms. Tag-masking doesn't help because the damage is on content
tokens, not the tag tokens.

**Implication for the paper/method:** the shaping needs a CORRECTNESS-GROUNDED polarity
(e.g. O+ only if the rollout's answer reward > 0), or a shaping warm-up (plain GRPO until
format/answer signal exists, then enable shaping), or α₂ scaled by group success rate.
On Qwen these coincide with adv>0, which is why this never surfaced in exp_055–079.
