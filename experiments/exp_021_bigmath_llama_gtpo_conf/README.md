# Experiment 021: GTPO-Conf (Confidence, No EMA) on Big-Math (Integer Filter)

## Status: Pending

## Method: GTPO-Conf (Confidence-Based Token-Level Shaping, No EMA)

GTPO-Conf replaces Shannon entropy with a confidence metric derived from the top-k log probabilities of the policy. Unlike GTPO-EMA (exp_018), there is no temporal smoothing — confidence is measured directly at each token position. It is adapted from exp_005.

### Confidence Metric

```
C_i,t = -mean_{v in top-k}(log pi(v | context_i,t))
```

- Small C → model is focused/certain (probability peaked on a few tokens)
- Large C → model is uncertain (probability spread across top-k tokens)

Key difference from Shannon entropy:
- Entropy uses ALL vocabulary tokens: `H = -sum_v pi(v) * log pi(v)`
- Confidence uses only top-k tokens: `C = -mean_{top-k} log pi(v)`
- Confidence is cheaper and more interpretable at inference

### Algorithm

For each sequence i and token position t:

1. Compute confidence: `C_i,t = -mean_{top-k}(log pi(v | context))`
2. Compress: `C_tilde_i,t = log(1 + C_i,t)`
3. Compute inverse-compressed: `I_i,t = log(1 + 1/(C_i,t + eps))`
4. Count active sequences: `d_t` (O+), `h_t` (O-)

For O+ tokens (correct sequences):
```
r_tilde_i,t = alpha1 * 1 + alpha2 * (C_tilde_i,t / sum_{k in O+, active} C_tilde_k,t) * d_t
```
High C (uncertain) → larger C_tilde → larger bonus (reward exploration on correct paths).

For O- tokens (incorrect sequences):
```
r_tilde_j,t = -(alpha1 * 1 + alpha2 * (I_j,t / sum_{k in O-, active} I_k,t) * h_t)
```
Low C (confident but wrong) → large 1/C → large I → large penalty.

5. Normalize per-class (O+ and O- separately) → advantages (B, T)

### Key Differences from exp_017 (Baseline GRPO)

| Aspect | exp_017 (GRPO) | exp_021 (GTPO-Conf) |
|--------|---------------|--------------------|
| Trainer | `GRPOTrainer` | `GTPOConfTrainer` |
| Advantage granularity | Sequence-level (B,) | Token-level (B, T) |
| Metric | None | Top-k confidence C_i,t |
| Temporal smoothing | None | None (direct, no EMA) |
| O+/O- shaping | No | Yes |
| Compression | None | log(1 + C) |
| Extra hyperparams | None | alpha1, alpha2, top_k |

### Key Differences from exp_018 (GTPO-EMA)

| Aspect | exp_018 (GTPO-EMA) | exp_021 (GTPO-Conf) |
|--------|-------------------|---------------------|
| Temporal smoothing | EMA (lambda=0.9) | None |
| Metric | EMA-smoothed C | Raw C |
| Extra hyperparams | lam | — |

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Model | meta-llama/Llama-3.2-3B-Instruct |
| LoRA rank | 64 |
| LoRA alpha | 64 |
| Max seq length | 4096 |
| Learning rate | 5e-6 |
| LR scheduler | cosine |
| Warmup ratio | 0.1 |
| Optimizer | adamw_8bit |
| Batch size (per device) | 4 |
| Gradient accumulation | 1 |
| num_generations | 16 |
| Max steps | 1000 |
| Max grad norm | 1.0 |
| bf16 | True |
| alpha1 | 1.0 |
| alpha2 | 0.1 |
| top_k | 20 |
| reward_threshold | 0.0 |
| Dataset | SynthLabsAI/Big-Math-RL-Verified (integer filter) |
| Output dir | /mnt/data/outputs/exp_021 |

## GTPO-Conf Formula

For O+ token (i, t):
```
r_tilde_i,t = alpha1 * 1 + alpha2 * (log(1+C_i,t) / sum_k log(1+C_k,t)) * d_t
```

For O- token (j, t):
```
r_tilde_j,t = -(alpha1 * 1 + alpha2 * (log(1+1/C_j,t) / sum_k log(1+1/C_k,t)) * h_t)
```

After normalization (Eq. 6 style, per class):
```
A_tilde_i,t = (r_tilde_i,t - mean(R_tilde+)) / std(R_tilde+)   for O+ tokens
A_tilde_j,t = (r_tilde_j,t - mean(R_tilde-)) / std(R_tilde-)   for O- tokens
```

Final: `token_advantages = adv_pos + adv_neg`

## Reward Functions

| Function | Max Score | Description |
|----------|-----------|-------------|
| `reward_format_exact` | 3.0 | Full regex match of reasoning + solution tags |
| `reward_format_approximate` | 2.0 | Partial tag presence check (0.5 per tag) |
| `reward_answer_exact` | 3.0 | Exact or near-exact answer match |
| `reward_answer_numeric` | 1.5 | Numeric value comparison |

## Files

| File | Description |
|------|-------------|
| `train.py` | Main training script |
| `requirements.txt` | Python dependencies |
| `src/__init__.py` | Package init |
| `src/confidence_utils.py` | Confidence metric, compression, GTPO-Conf reward shaping |
| `src/gtpo_conf_trainer.py` | Custom trainer: GTPOConfTrainer |
| `tests/test_exp021.py` | Unit tests for confidence utilities |
| `figures/` | Training curves (populated after run) |

## Pre-Training Checklist

- [ ] GPU available and CUDA accessible
- [ ] HF_TOKEN environment variable set
- [ ] `/mnt/data/outputs/exp_021` writable
- [ ] `pip install -r requirements.txt` executed
- [ ] Unit tests pass: `pytest tests/`
- [ ] Dataset filter verified (integer answers only)

## Running

```bash
cd /mnt/data/aiim_research/experiments/exp_021_bigmath_llama_gtpo_conf
pip install -r requirements.txt
pytest tests/ -v
HF_TOKEN=<token> python train.py
```

## Results

| Metric | Value |
|--------|-------|
| Final reward (step 1000) | TBD |
| Best reward | TBD |
| Format pass rate | TBD |
| Answer accuracy | TBD |
| vs exp_017 (GRPO baseline) | TBD |
| vs exp_018 (GTPO-EMA) | TBD |
| vs exp_020 (GTPO entropy) | TBD |

## Notes

- Confidence metric adapted from "Deep Think with Confidence" (Meta, arXiv:2508.15260): C = -mean_{top-k}(log pi).
- No EMA smoothing — provides a cleaner ablation vs exp_018 to measure whether the EMA temporal smoothing adds value.
- Confidence requires an extra forward pass to obtain full logits (not just selected-token logps), handled via `torch.no_grad()` in `_compute_loss`.
- Adapted from exp_005 for Big-Math integer-filtered dataset with 16 generations.
