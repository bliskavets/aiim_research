"""
gtpo_conf_trainer.py
--------------------
GTPOConfTrainer: GTPO variant using confidence-based reward shaping
instead of entropy-based shaping (exp_005).

Confidence metric: C_i,t = -mean_{top-k}(log π(v | context))
"""

import torch
from .searchr1_trainer import SearchR1GRPOTrainer
from .confidence_utils import (
    confidence_from_logits,
    confidence_from_model_chunked,
    compute_gtpo_conf_rewards,
    EPS,
)
from .format_tag_mask import build_tag_mask, apply_tag_mask_to_token_advantages


class GTPOConfTrainer(SearchR1GRPOTrainer):
    """
    GTPO with confidence-based token-level reward shaping.

    Extra kwargs:
        alpha1 (float): base reward weight. Default 1.0
        alpha2 (float): confidence bonus weight. Default 0.1
        top_k  (int):   top-k tokens for confidence. Default 20
        reward_threshold (float): O+/O- split. Default 0.0
        format_tag_patterns (list[list[int]] | None): if provided, token-id
            subsequences for the format tags. On positions matching any of
            these patterns the per-token shaped advantage is replaced by
            the seq-level GRPO advantage (no shaping on tag tokens).
    """

    def __init__(self, *args, **kwargs):
        self.alpha1               = kwargs.pop("alpha1", 1.0)
        self.alpha2               = kwargs.pop("alpha2", 0.1)
        self.top_k                = kwargs.pop("top_k", 20)
        self.reward_threshold     = kwargs.pop("reward_threshold", 0.0)
        self.format_tag_patterns  = kwargs.pop("format_tag_patterns", None)
        self.conf_micro_bs        = kwargs.pop("conf_micro_bs", 2)
        super().__init__(*args, **kwargs)

    def _compute_loss(self, model, inputs):
        prompt_ids      = inputs["prompt_ids"]
        prompt_mask     = inputs["prompt_mask"]
        completion_ids  = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        seq_advantages  = inputs["advantages"]   # (B,) standard GRPO advantages

        input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Forward pass — get logits for confidence + logps for IS weights
        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=False,
        )

        # Compute confidence from logits via a separate forward pass.
        # Chunked over the batch dim (micro_bs) so the full (B, L, V) fp32
        # logits tensor over Qwen3's ~152k vocab is never materialized at
        # once — otherwise the second forward OOMs the backward on long
        # Search-R1 rollouts. Math is identical to a single full forward.
        confidence = confidence_from_model_chunked(
            model, input_ids, attention_mask, logits_to_keep,
            top_k=self.top_k,
            pass_logits_to_keep=("logits_to_keep" in self.model_kwarg_keys),
            micro_bs=self.conf_micro_bs,
        )  # (B, T)

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Compute GTPO-Conf per-token advantages
        adv_pos, adv_neg = compute_gtpo_conf_rewards(
            rewards          = seq_advantages,
            confidence       = confidence,
            completion_mask  = completion_mask,
            alpha1           = self.alpha1,
            alpha2           = self.alpha2,
            top_k            = self.top_k,
            reward_threshold = self.reward_threshold,
        )
        token_advantages = adv_pos + adv_neg   # (B, T)

        # Optional: replace per-token shaped advantage with seq-level advantage
        # on tokens that belong to structural format tags. This stops the
        # confidence-based shaping from distorting gradients on tag substrings
        # (where the model is highly peaked and the bonus/penalty would
        # otherwise dominate the format-learning signal).
        if self.format_tag_patterns:
            tag_mask = build_tag_mask(completion_ids, self.format_tag_patterns)
            token_advantages = apply_tag_mask_to_token_advantages(
                token_advantages, seq_advantages, tag_mask)

        # GTPO-Conf objective: normalize over total tokens
        per_token_loss1 = coef_1 * token_advantages
        per_token_loss2 = coef_2 * token_advantages
        per_token_loss  = -torch.min(per_token_loss1, per_token_loss2)

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )
            per_token_loss = per_token_loss + self.beta * per_token_kl

        total_tokens = completion_mask.sum().clamp(min=1.0)
        loss = (per_token_loss * completion_mask).sum() / total_tokens
        loss = loss / self.current_gradient_accumulation_steps

        # Logging
        mode = "train" if model.training else "eval"
        mean_conf = (confidence * completion_mask).sum() / total_tokens
        self._metrics[mode].setdefault("gtpo_conf/mean_confidence", []).append(
            self.accelerator.gather(mean_conf).mean().item()
        )
        mean_adv = (token_advantages * completion_mask).sum() / total_tokens
        self._metrics[mode].setdefault("gtpo_conf/mean_token_advantage", []).append(
            self.accelerator.gather(mean_adv).mean().item()
        )
        n_pos = (seq_advantages > self.reward_threshold).float().sum()
        n_neg = (seq_advantages <= self.reward_threshold).float().sum()
        self._metrics[mode].setdefault("gtpo_conf/frac_pos", []).append(
            (n_pos / (n_pos + n_neg + EPS)).item()
        )

        return loss
