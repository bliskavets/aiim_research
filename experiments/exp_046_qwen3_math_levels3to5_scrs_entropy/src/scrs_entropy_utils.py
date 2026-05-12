"""
scrs_entropy_utils.py
---------------------
SCRS-Entropy: sequence-level Shannon entropy shaping for GRPO (exp_046).

Like SCRS (exp_045) but uses top-k approximate Shannon entropy H instead of
top-k confidence C as the sequence-level shaping scalar:

  H_i,t = -Σ_{v ∈ top-k} p̃_v log p̃_v  (renormalised top-k entropy, nats)
  High H → flat/uncertain; Low H → peaked/certain

  shaped_adv_i = grpo_adv_i + alpha2 * entropy_adj_i

  grpo_adv > 0 (correct):
    high H (uncertain correct) → larger shaped_adv  → amplify hard wins
    low  H (certain  correct)  → smaller shaped_adv → model was already confident

  grpo_adv < 0 (wrong):
    high H (uncertain wrong)   → less negative      → model was hedging, lighter push
    low  H (certain  wrong)    → more negative      → confident mistake, extra penalty

Sign is + alpha2 * entropy_adj because entropy ≈ inverse of confidence (exp_045).
"""

import torch

EPS = 1e-8


def compute_top_k_entropy(log_probs: torch.Tensor, top_k: int = 100) -> torch.Tensor:
    """
    Approximate per-token Shannon entropy using top-k renormalised probabilities.

        H_t ≈ -Σ_{v ∈ top-k} p̃_v log p̃_v  (nats)

    Args:
        log_probs: (B, T, V) log-softmax output (must be contiguous float tensor)
        top_k:     number of top tokens — higher is more accurate (default 100)

    Returns:
        entropy: (B, T) float32, H ≥ 0, detached, no grad
    """
    k = min(top_k, log_probs.shape[-1])
    topk_lp = torch.topk(log_probs, k, dim=-1).values   # (B, T, k)
    topk_p  = topk_lp.exp()                              # (B, T, k)
    sum_p   = topk_p.sum(dim=-1, keepdim=True).clamp(min=EPS)
    p_norm  = topk_p / sum_p                             # renormalise: sums to 1
    entropy = -(p_norm * torch.log(p_norm + EPS)).sum(dim=-1)  # (B, T)
    return entropy.detach()


def compute_scrs_entropy_advantages(
    grpo_advantages: torch.Tensor,
    entropy: torch.Tensor,
    completion_mask: torch.Tensor,
    alpha2: float = 0.1,
):
    """
    Compute SCRS-Entropy per-token advantages.

    Args:
        grpo_advantages: (B,) GRPO z-normalised sequence advantages
        entropy:         (B, T) per-token top-k entropy, detached
        completion_mask: (B, T) 1 for valid tokens
        alpha2:          shaping weight (0 = pure GRPO)

    Returns:
        token_advantages: (B, T) shaped advantages broadcast to tokens
        seq_entropy:      (B,)   mean entropy per rollout (for logging)
        entropy_adj:      (B,)   z-normalised adjustment (for logging)
    """
    total_tokens = completion_mask.sum(1).clamp(min=1.0)  # (B,)
    seq_entropy  = (entropy * completion_mask).sum(1) / total_tokens  # (B,)

    if seq_entropy.numel() > 1 and seq_entropy.std() > EPS:
        entropy_adj = (seq_entropy - seq_entropy.mean()) / (seq_entropy.std() + EPS)
    else:
        entropy_adj = torch.zeros_like(seq_entropy)

    # + alpha2 (not −) because high entropy ↔ low confidence
    shaped_adv = grpo_advantages + alpha2 * entropy_adj   # (B,)

    token_advantages = (
        shaped_adv.unsqueeze(1).expand_as(completion_mask.float()) * completion_mask
    )
    return token_advantages, seq_entropy, entropy_adj
