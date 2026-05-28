# exp_049 — GRPO vs three confidence/entropy-shaped candidates on Big-Math int-2000

Head-to-head of the GRPO baseline against the three reward-shaping setups that
worked in earlier experiments, all under one identical configuration so the only
moving part is the trainer.

## Methods

| key | candidate | source | shaping coefficients |
|-----|-----------|--------|----------------------|
| `grpo` | GRPO baseline | exp_014 / exp_027 | — |
| `grpo_s_entropy` | seq-level entropy weighting (orange curve, exp_005 dashboard) | exp_002 `grpo_s_trainer.py` | β1=1.0, β2=0.1 |
| `gtpo_conf` | per-token confidence bonus (green curve, exp_005 dashboard) | exp_005 `gtpo_conf_trainer.py` | α1=1.0, α2=0.1, top_k=20 |
| `gtpo_ema_flipped` | flipped EMA-confidence shaping | exp_026 `gtpo_ema_flipped_trainer.py` | α1=0.9, α2=0.1, λ=0.9, top_k=20 |

All three shaped methods split O+/O- on the sign of the standard GRPO advantage
(`reward_threshold=0.0`) — the exp_026 setting.

## Shared configuration

Training hyperparameters are taken verbatim from **exp_005** and applied
identically to all four runs:

| | value | source |
|---|---|---|
| model | meta-llama/Llama-3.2-3B-Instruct | exp_005 |
| LoRA | r=64, α=64, 7 target modules | exp_005 |
| learning rate | 5e-6, cosine, warmup 0.1, wd 0.1 | exp_005 |
| optimizer | adamw_8bit, max_grad_norm 1.0 | exp_005 |
| batch | bs=1 × grad_accum=4 × num_gen=4 (4 prompts → 16 seqs/step) | exp_005 |
| max_steps | 500 | exp_005 |
| seed | 3407 (LoRA init, dataset shuffle, GRPOConfig) | exp_005 |
| max_seq_length | 2560 (512 prompt + 2048 completion) | exp_027/028* |
| gpu_memory_utilization | 0.55 | exp_027/028* |

\* The two starred values are the only deviation from exp_005's 2048 / 0.9.
They are memory/length knobs, not optimization hyperparameters — raised so the
longer Big-Math completions are not clipped, matching the budget this dataset
was actually trained at in exp_027/028. Identical across all four runs.

## Dataset

`SynthLabsAI/Big-Math-RL-Verified`, integer-answer filter, first 2000 in
shuffled order (seed 3407) — identical to exp_027/028.

## Rewards

The exp_026 reward family — `reward_format_exact`, `reward_format_approximate`,
`reward_answer_exact`, `reward_answer_numeric` — adapted to Big-Math's
`problem`/`answer` fields and negative integers (exp_028 plumbing). Same scoring
tiers (exact 3.0 / strip 1.5 / within-10% 1.0 / within-20% 0.5 / wrong -1.5;
numeric ±1.5/-0.5).

## Hypothesis

On a real (non-GSM8K) math dataset under matched hyperparameters, at least one
of the three shaping methods that beat GRPO on GSM8K (exp_005 dashboard) also
beats the GRPO baseline here. Earlier evidence is mixed: per-token shaping
collapsed on MATH levels 3-5 (exp_043/044/047) while flipped EMA worked on
Big-Math at a different batch config (exp_028). This run isolates the trainer
as the single variable.

## Run

```bash
HF_TOKEN=... bash run_049.sh        # runs all 4 methods sequentially
python plot_metrics.py              # 4-way overlay + last-50 summary
pytest tests/ -q                    # shaping-math unit tests (no training)
```

## Results

| method | r@L50 | peak | answer_exact@L50 | format_exact@L50 | KL@L50 | steps |
|--------|-------|------|------------------|------------------|--------|-------|
| grpo | | | | | | |
| grpo_s_entropy | | | | | | |
| gtpo_conf | | | | | | |
| gtpo_ema_flipped | | | | | | |

_(filled after the run from `plot_metrics.py`)_
