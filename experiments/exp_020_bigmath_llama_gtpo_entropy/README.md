# Experiment 020: GTPO Per-Token Entropy on Big-Math (Integer Filter)

## Status: Pending

## Method: GTPO (Group Token Policy Optimization with Per-Token Entropy)

GTPO applies per-token reward shaping using Shannon entropy at each token position. It is adapted from exp_002. Unlike GRPO-S which operates at sequence level, GTPO shapes advantages at every token position using a per-token entropy bonus.

### Algorithm

For each sequence i and token position t:

1. Compute per-token Shannon entropy: `H_i,t = -sum_v pi(v) * log pi(v)`
2. Clip entropies: `H_i,t = clamp(H_i,t, eps_low, eps_high)`
3. Split into O+ (reward > threshold) and O-
4. Compute d_t = number of active O+ sequences at position t

For O+ tokens (Eq. 3):
```
r_tilde_i,t = alpha1 * 1 + alpha2 * (H_i,t / sum_{k in O+, active at t} H_k,t) * d_t
```
High entropy at position t → model was exploring → larger bonus for that token.

For O- tokens (Eq. 5):
```
r_tilde_j,t = -(alpha1 * 1 + alpha2 * (1/H_j,t / sum_{k in O-, active at t} 1/H_k,t) * h_t)
```
Low entropy (confident) but wrong → larger penalty for that token.

5. Normalize per-class (separately for O+ and O-) → advantages (B, T)
6. Apply PPO-clip with token-level IS weights

### Key Differences from exp_017 (Baseline GRPO)

| Aspect | exp_017 (GRPO) | exp_020 (GTPO) |
|--------|---------------|----------------|
| Trainer | `GRPOTrainer` | `GTPOTrainer` |
| Advantage granularity | Sequence-level (B,) | Token-level (B, T) |
| Loss normalization | Per-sequence | Per-token (DAPO-style) |
| Entropy signal | None | Per-token Shannon entropy |
| O+/O- shaping | No | Yes |
| d_t accounting | No | Yes (active seqs per position) |
| Extra hyperparams | None | alpha1, alpha2, eps_entropy bounds |

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
| eps_entropy_low | 0.2 |
| eps_entropy_high | 0.28 |
| reward_threshold | 0.0 |
| Dataset | SynthLabsAI/Big-Math-RL-Verified (integer filter) |
| Output dir | /mnt/data/outputs/exp_020 |

## GTPO Formula

For O+ token (i, t):
```
r_tilde_i,t = alpha1 * r_i + alpha2 * (H_i,t / sum_k H_k,t) * d_t
```

For O- token (j, t):
```
r_tilde_j,t = -(alpha1 * |r_j| + alpha2 * (1/H_j,t / sum_k 1/H_k,t) * h_t)
```

Where:
- `d_t` = number of O+ sequences with valid token at position t
- `h_t` = number of O- sequences with valid token at position t
- Advantages are then normalized separately over O+ and O- token sets (Eq. 6)

Final token advantage: `A_i,t = adv_pos_i,t + adv_neg_i,t`

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
| `src/entropy_utils.py` | Shannon entropy computation and GTPO/GRPO-S reward shaping |
| `src/gtpo_trainer.py` | Custom trainer: GTPOTrainer |
| `tests/test_exp020.py` | Unit tests for GTPO per-token rewards |
| `figures/` | Training curves (populated after run) |

## Pre-Training Checklist

- [ ] GPU available and CUDA accessible
- [ ] HF_TOKEN environment variable set
- [ ] `/mnt/data/outputs/exp_020` writable
- [ ] `pip install -r requirements.txt` executed
- [ ] Unit tests pass: `pytest tests/`
- [ ] Dataset filter verified (integer answers only)

## Running

```bash
cd /mnt/data/aiim_research/experiments/exp_020_bigmath_llama_gtpo_entropy
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
| vs exp_019 (GRPO-S) | TBD |

## Notes

- GTPO uses DAPO-style per-token loss normalization (divide by total tokens, not sequences), which better handles variable-length completions.
- The d_t active-sequence count ensures the entropy bonus scales properly when sequences end at different positions.
- Entropy recomputed from current model in `_compute_loss` (approximation for old policy entropy) to avoid Unsloth buffer splitting issues.
- Adapted from exp_002 for Big-Math integer-filtered dataset with 16 generations.
