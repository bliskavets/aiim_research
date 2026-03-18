# Experiment 002: GTPO and GRPO-S Implementation Plan

## Overview
Replication of "GTPO and GRPO-S: Token and Sequence-Level Reward Shaping with Policy Entropy" (ICML 2026 submission).

## What We're Implementing

### GRPO (baseline, for reference)
Standard uniform advantage: every token in sequence i gets the same advantage Â_i = (r_i - mean(r)) / std(r).

### GTPO (Group Token Policy Optimization)
Token-level entropy-weighted reward shaping:
- Split sequences into O+ (correct) and O- (incorrect)
- **O+ tokens:** r̃_i,t = α₁·r_i + α₂·(H_i,t / Σ_k H_k,t) · d_t
  - H_i,t = token entropy at position t from old policy
  - d_t = number of successful sequences with length ≥ t
  - Rewards high-entropy tokens in correct sequences more
- **O- tokens:** r̃_j,t = α₁·(-1) + α₂·(1/H_j,t / Σ_k 1/H_{n+k,t}) · h_t · (-1)
  - h_t = number of unsuccessful sequences with length ≥ t
  - Penalizes low-entropy (confident) tokens in wrong sequences more
- Separate advantage normalization over all O+ tokens and O- tokens
- Loss: token-level PPO with normalization by total token count

### GRPO-S (Sequence-Level GRPO)
Sequence-level entropy-weighted reward shaping:
- Ĥ_i = average token entropy of sequence i
- **O+ sequences:** r̂_i = β₁·r_i + β₂·(Ĥ_i / Σ_k Ĥ_k) · n
- **O- sequences:** r̂_j = β₁·(-1) + β₂·(1/Ĥ_j / Σ_k 1/Ĥ_{n+k}) · m · (-1)
- Advantage normalized at sequence level (analogous to GRPO)
- Importance weight: sequence-level average of token IS weights
- Loss: sequence-level PPO (structure same as GRPO but with shaped rewards+weights)

**Hyperparameters (from paper):** α₁=β₁=1, α₂=β₂=0.1, ε_low=0.2, ε_high=0.28

---

## Where Changes Are Needed in TRL's GRPOTrainer

TRL's GRPOTrainer (`trl/trainer/grpo_trainer.py`) is the base. We need to override/patch:

### 1. `_generate_and_score_completions()` — reward shaping
**Current:** computes uniform `advantages = rewards - mean_grouped_rewards`, normalized by group std.
**Change:** after collecting rewards and entropies:
- Split sequences into O+/O- by reward sign
- Compute per-token or per-sequence entropy-weighted rewards
- Compute separate advantage normalization for O+ and O-

### 2. `_get_per_token_logps_and_entropies()` — entropy extraction
**Current:** already computes entropies (H_i,t from logits). ✅ Can reuse.
**Change needed:** entropies must be saved and passed through the pipeline to the reward-shaping step.
Currently `entropies` are computed in `_compute_loss()` but we need them *earlier*, during `_generate_and_score_completions()` (before advantages are computed), using the OLD policy.

### 3. `_compute_loss()` — loss computation
**For GTPO:** 
- Per-token advantages (shape: B×T) instead of per-sequence (shape: B)
- Token-level IS weights (already default)
- Normalization by total token count (use `loss_type="dapo"` style or custom)

**For GRPO-S:**
- Sequence-level advantages (shape: B)
- Sequence-level IS weights: mean of token-level weights per sequence
- Normalization by G sequences (same as GRPO)

---

## Implementation Strategy

### Approach: Subclass GRPOTrainer (no forking TRL)

We will **subclass** `GRPOTrainer` and override the minimum necessary methods:

```
src/
├── gtpo_trainer.py      # GTPOTrainer(GRPOTrainer) — overrides reward shaping + loss
├── grpo_s_trainer.py    # GRPOSTrainer(GRPOTrainer) — overrides reward shaping + loss
└── entropy_utils.py     # shared entropy computation utilities
```

**Why subclass (not patch):**
- TRL is installed system-wide in Docker, modifying it risks breaking exp_001
- Subclassing is clean, reproducible, and explicit about what changed
- We only need to override ~3 methods

### Key Technical Details

#### Getting entropies during generation phase
The OLD policy entropies H_i,t must be computed during generation (when old policy is active), not during the training forward pass. We'll override `_generate_and_score_completions()` to:
1. Call parent's generation
2. Run an extra forward pass with the OLD policy (vLLM mode) or reuse existing logps to compute entropies
3. Store entropies alongside `completion_ids` in the batch dict

#### Entropy computation from logits
TRL already has `entropy_from_logits()` utility — we'll reuse it.
H_i,t = -Σ_v π(v|context) log π(v|context) = -Σ_v softmax(logits)·log_softmax(logits)

#### Numerical stability
Per paper remark 2.1: add ε=1e-8 to denominators where entropy appears (division by Σ H_k,t or H_j,t directly).

#### Entropy clipping
Paper uses ε_low=0.2, ε_high=0.28 for entropy clipping to avoid extreme values.

---

## Libraries Needed

### No new external libraries required!
All dependencies already exist in the Docker container:

| Need | Available via |
|------|--------------|
| Token entropy H_i,t | `trl.trainer.utils.entropy_from_logits` ✅ |
| Per-token logprobs | `GRPOTrainer._get_per_token_logps_and_entropies()` ✅ |
| Reward shaping math | pure PyTorch ✅ |
| Training loop | `GRPOTrainer` (subclass) ✅ |

**No PYTHONPATH tricks needed** — all code lives in the experiment folder, imported directly.

---

## Files to Create

```
experiments/exp_002_gtpo_and_grpo_s/
├── PLAN.md                    ← this file
├── README.md                  ← filled after training
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── entropy_utils.py       ← entropy computation helpers
│   ├── gtpo_trainer.py        ← GTPOTrainer subclass
│   └── grpo_s_trainer.py      ← GRPOSTrainer subclass
├── tests/
│   ├── test_entropy_utils.py  ← unit tests for entropy math
│   ├── test_gtpo_shapes.py    ← test shaped rewards match paper formulas
│   └── test_grpo_s_shapes.py
├── train_gtpo.py              ← training script for GTPO
├── train_grpo_s.py            ← training script for GRPO-S
└── figures/                   ← generated after training
```

---

## Step-by-Step Execution Plan

1. **[DONE]** Read paper and analyze TRL source ✅
2. **[DONE]** Write this plan ✅
3. **[NEXT]** Implement `entropy_utils.py` + `gtpo_trainer.py` + `grpo_s_trainer.py`
4. **[NEXT]** Write mini-tests verifying:
   - Entropy shapes are correct
   - Reward mass conservation: Σ r̃_i,t = d_t (Proposition 2.2)
   - Advantages have correct shapes
   - IS weights for GRPO-S are sequence-level means
5. **[NEXT]** Run GTPO training (500 steps, same config as exp_001)
6. **[NEXT]** Run GRPO-S training (500 steps)
7. **[NEXT]** Log metrics, generate plots, push to GitHub

---

## Open Questions / Risks

1. **Entropy during vLLM generation:** In exp_001, model uses vLLM for fast inference. Entropies from vLLM are not directly accessible (it returns token IDs only, not logits). We'll need to run a separate forward pass through the HF model to get logits → entropies during the scoring phase. This is the same path TRL uses for `old_per_token_logps`. Should be fine.

2. **GTPO with per-token advantages:** TRL's current `advantages` tensor is shape `(B,)` — one value per sequence. GTPO needs `(B, T)`. We'll need to expand and handle masking carefully in `_compute_loss()`.

3. **Memory:** Per-token entropy storage for GTPO is an extra `(B, T)` float tensor per step. With B=4, T≤2048, this is negligible on A100 80GB.

4. **Binary rewards assumption:** Paper assumes r_i ∈ {0, 1} for O+/O- split. Our exp_001 reward functions return continuous values (e.g., -1.5 to +3.0). We'll threshold: O+ = sequences where total reward > 0, O- = sequences where total reward ≤ 0.

---

## Hyperparameters for Our Runs

Matching exp_001 base config + paper's entropy params:

| Parameter | Value |
|-----------|-------|
| Model | meta-llama/Llama-3.2-3B-Instruct |
| Dataset | openai/gsm8k |
| Max steps | 500 |
| α₁, β₁ | 1.0 |
| α₂, β₂ | 0.1 |
| ε_low (entropy clip) | 0.2 |
| ε_high (entropy clip) | 0.28 |
| Learning rate | 5e-6 |
| Batch size | 1 (grad accum=4) |
| Num generations | 4 |
