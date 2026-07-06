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

## Results (300 steps, L50 boxed / len) — NEGATIVE for direct transfer

| dataset | GRPO | Ours (posdisc λ0.7 k5, no gate) | Δ |
|---|---|---|---|
| gsm8k    | **+2.33** / 260 | +0.09 / 132 | **−2.24** |
| math500  | +0.33 / 643 | **+0.49** / 641 | +0.16 |
| bigmath  | **+0.43** / 629 | +0.25 / 505 | −0.18 |
| omnimath | −0.61 / 1104 | −0.65 / 687 | −0.04 |

**The tuned Qwen config does NOT transfer as-is to Llama-3.2-3B-Instruct.** gsm8k shows a
slow degradation (healthy +1.32 in steps 1–40, then declines to +0.09 with shrinking
completions 211→128) — NOT a length-farming collapse. Hypotheses: (1) no zero-variance gate
— Llama saturates gsm8k fast, ~33–45% of groups have std(R)=0 and the ungated shaping turns
them into penalty noise (exp_071 diagnosis); (2) C-scale miscalibration — Llama's logprob
distribution differs from Qwen's, so α₂=0.1/λ=0.7/k=5 are off-scale; (3) Instruct (SFT'd)
policy vs the Base policy used in the whole Qwen study. exp_081 tests (3) directly by
rerunning on Llama-3.2-3B **Base**.
