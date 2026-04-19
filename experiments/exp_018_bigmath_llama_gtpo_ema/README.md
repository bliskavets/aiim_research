# Experiment 018: GTPO-EMA on Big-Math (Integer Filter)

## Status: Pending

## Method: GTPO-EMA

GTPO-EMA (Group Token Policy Optimization with Exponential Moving Average confidence) applies token-level reward shaping using an EMA-smoothed confidence signal. It is adapted from exp_010.

### Algorithm

At each training step, for each generated sequence i and token position t:

1. Compute per-token confidence: `C_i,t = -mean_{v in top-k}(log pi(v | context))`
2. Compute EMA along the sequence:
   - `EMA_i,0 = C_i,0`
   - `EMA_i,t = lambda * EMA_i,t-1 + (1 - lambda) * C_i,t`
3. Compute group-relative base advantage: `A_i = (r_i - mu) / sigma`
4. Split sequences into O+ (reward > threshold) and O-:
   - O+: `bonus_i,t = normalize_within_group(log(1 + EMA_i,t))` — reward exploration
   - O-: `bonus_i,t = -normalize_within_group(log(1 + 1/EMA_i,t))` — penalize overconfidence
5. Final token advantage: `adv_i,t = alpha1 * A_i + alpha2 * bonus_i,t`

### Key Differences from exp_017 (Baseline GRPO)

| Aspect | exp_017 (GRPO) | exp_018 (GTPO-EMA) |
|--------|---------------|-------------------|
| Trainer | `GRPOTrainer` | `GTPoEMATrainer` |
| Advantage granularity | Sequence-level (B,) | Token-level (B, T) |
| Confidence signal | None | EMA-smoothed top-k confidence |
| O+/O- shaping | No | Yes (via EMA bonus) |
| Extra hyperparams | None | alpha1, alpha2, top_k, lam |

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
| lam (EMA decay) | 0.9 |
| reward_threshold | 0.0 |
| Dataset | SynthLabsAI/Big-Math-RL-Verified (integer filter) |
| Output dir | /mnt/data/outputs/exp_018 |

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
| `src/ema_confidence_utils.py` | EMA confidence computation and GTPO-EMA advantage shaping |
| `src/gtpo_ema_trainer.py` | Custom trainer: GTPoEMATrainer |
| `tests/test_exp018.py` | Unit tests for EMA confidence utilities |
| `figures/` | Training curves (populated after run) |

## Pre-Training Checklist

- [ ] GPU available and CUDA accessible
- [ ] HF_TOKEN environment variable set for dataset + model download
- [ ] `/mnt/data/outputs/exp_018` writable
- [ ] `unsloth`, `trl>=0.15.0`, `transformers>=4.48.0` installed
- [ ] `pip install -r requirements.txt` executed
- [ ] Unit tests pass: `pytest tests/`
- [ ] Dataset filter verified (integer answers only)

## Running

```bash
cd /mnt/data/aiim_research/experiments/exp_018_bigmath_llama_gtpo_ema
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

## Notes

- EMA confidence key fix (from exp_010): token advantages computed from raw rewards, not pre-normalized advantages, to preserve the EMA signal.
- Adapted from exp_010 (EMA confidence fixed) for the Big-Math integer-filtered dataset with 16 generations (same setup as exp_017 baseline).
