# Experiment 019: GRPO-S Entropy on Big-Math (Integer Filter)

## Status: Pending

## Method: GRPO-S (Sequence-Level Entropy Shaping)

GRPO-S (Group Relative Policy Optimization - Sequence level) applies sequence-level reward shaping using mean entropy per sequence. It is adapted from exp_002.

### Algorithm

For each sequence i in the batch:

1. Compute per-token Shannon entropy: `H_i,t = -sum_v pi(v) * log pi(v)`
2. Clip entropies to [eps_low, eps_high] for stability
3. Compute mean entropy per sequence: `H_hat_i = (1/|o_i|) * sum_t H_i,t`
4. Split into O+ (reward > threshold) and O-:

For O+ sequences (Eq. 9 top):
```
r_hat_i = beta1 * 1 + beta2 * (H_hat_i / sum_k H_hat_k) * n
```
Higher mean entropy → model explored more → larger bonus.

For O- sequences (Eq. 9 bottom):
```
r_hat_j = -(beta1 * 1 + beta2 * (1/H_hat_j / sum_k 1/H_hat_{n+k}) * m)
```
Lower entropy (more confident but wrong) → larger penalty.

5. Normalize shaped rewards within group → advantages
6. Use sequence-level IS weight: `w_hat_i = (1/|o_i|) * sum_t (pi_theta / pi_theta_old)_t`

### Key Differences from exp_017 (Baseline GRPO)

| Aspect | exp_017 (GRPO) | exp_019 (GRPO-S) |
|--------|---------------|-----------------|
| Trainer | `GRPOTrainer` | `GRPOSTrainer` |
| IS weight | Token-level | Sequence-level (mean) |
| Reward shaping | None | Entropy-weighted sequence reward |
| O+/O- shaping | No | Yes (entropy bonus/penalty) |
| Extra hyperparams | None | beta1, beta2, eps_entropy bounds |

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
| beta1 | 1.0 |
| beta2 | 0.1 |
| eps_entropy_low | 0.2 |
| eps_entropy_high | 0.28 |
| reward_threshold | 0.0 |
| Dataset | SynthLabsAI/Big-Math-RL-Verified (integer filter) |
| Output dir | /mnt/data/outputs/exp_019 |

## GRPO-S Formula

For O+ sequences:
```
r_hat_i = beta1 + beta2 * (H_hat_i / sum_{k in O+} H_hat_k) * n
```

For O- sequences:
```
r_hat_j = -(beta1 + beta2 * (1/H_hat_j / sum_{k in O-} 1/H_hat_k) * m)
```

Where:
- `H_hat_i` = mean entropy of sequence i (clipped to [eps_low, eps_high])
- `n` = number of O+ sequences in the group
- `m` = number of O- sequences in the group
- `beta1, beta2` = shaping weights

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
| `src/entropy_utils.py` | Shannon entropy computation and GRPO-S/GTPO reward shaping |
| `src/grpo_s_trainer.py` | Custom trainer: GRPOSTrainer |
| `tests/test_exp019.py` | Unit tests for entropy utilities |
| `figures/` | Training curves (populated after run) |

## Pre-Training Checklist

- [ ] GPU available and CUDA accessible
- [ ] HF_TOKEN environment variable set
- [ ] `/mnt/data/outputs/exp_019` writable
- [ ] `pip install -r requirements.txt` executed
- [ ] Unit tests pass: `pytest tests/`
- [ ] Dataset filter verified (integer answers only)

## Running

```bash
cd /mnt/data/aiim_research/experiments/exp_019_bigmath_llama_grpos_entropy
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

- GRPO-S uses sequence-level IS weights rather than token-level, matching the sequence-level advantage formulation.
- Entropy clipping to [0.2, 0.28] prevents extreme bonus/penalty values from dominating training.
- Adapted from exp_002 for the Big-Math integer-filtered dataset with 16 generations.
