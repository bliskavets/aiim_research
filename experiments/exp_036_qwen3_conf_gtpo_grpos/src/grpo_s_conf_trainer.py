"""
grpo_s_conf_trainer.py
----------------------
GRPOSConfTrainer: GRPO-S variant using confidence-based reward shaping (exp_005).
"""

import torch
from trl import GRPOTrainer
from .confidence_utils import confidence_from_logits, compute_grpo_s_conf_rewards, EPS


class GRPOSConfTrainer(GRPOTrainer):
    """
    GRPO-S with confidence-based sequence-level reward shaping.

    Extra kwargs:
        beta1 (float): base reward weight. Default 1.0
        beta2 (float): confidence bonus weight. Default 0.1
        top_k  (int):  top-k tokens for confidence. Default 20
        reward_threshold (float): O+/O- split. Default 0.0
    """

    def __init__(self, *args, **kwargs):
        self.beta1            = kwargs.pop("beta1", 1.0)
        self.beta2            = kwargs.pop("beta2", 0.1)
        self.top_k            = kwargs.pop("top_k", 20)
        self.reward_threshold = kwargs.pop("reward_threshold", 0.0)
        super().__init__(*args, **kwargs)

    def _compute_loss(self, model, inputs):
        prompt_ids      = inputs["prompt_ids"]
        prompt_mask     = inputs["prompt_mask"]
        completion_ids  = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        grpo_advantages = inputs["advantages"]   # (B,) standard GRPO

        input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=False,
        )

        # Get full logits for confidence computation
        with torch.no_grad():
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1
            raw_out = model(**model_inputs)
            logits = raw_out.logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            confidence = confidence_from_logits(logits, top_k=self.top_k)  # (B, T)

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        # Compute GRPO-S-Conf shaped sequence rewards
        shaped_rewards, seq_avg_conf = compute_grpo_s_conf_rewards(
            rewards          = grpo_advantages,
            confidence       = confidence,
            completion_mask  = completion_mask,
            beta1            = self.beta1,
            beta2            = self.beta2,
            reward_threshold = self.reward_threshold,
        )

        # Normalize within groups → sequence-level advantages
        B = shaped_rewards.shape[0]
        G = self.num_generations
        shaped_grouped = shaped_rewards.view(-1, G)
        mean_s = shaped_grouped.mean(dim=1, keepdim=True)
        std_s  = shaped_grouped.std(dim=1, keepdim=True).clamp(min=EPS)
        advantages = ((shaped_grouped - mean_s) / std_s).view(B)

        # Sequence-level IS weights: mean of token IS weights (Eq. 11 from GTPO paper)
        log_ratio      = per_token_logps - old_per_token_logps
        token_is       = torch.exp(log_ratio)
        seq_lengths    = completion_mask.sum(dim=1).clamp(min=1)
        seq_is_weight  = (token_is * completion_mask).sum(dim=1) / seq_lengths  # (B,)

        coef_1 = seq_is_weight
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        per_seq_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )
            per_seq_kl = (per_token_kl * completion_mask).sum(dim=1) / seq_lengths
            per_seq_loss = per_seq_loss + self.beta * per_seq_kl

        loss = per_seq_loss.mean()
        loss = loss / self.current_gradient_accumulation_steps

        # Logging
        mode = "train" if model.training else "eval"
        self._metrics[mode].setdefault("grpo_s_conf/mean_seq_confidence", []).append(
            self.accelerator.gather(seq_avg_conf).mean().item()
        )
        self._metrics[mode].setdefault("grpo_s_conf/mean_seq_is_weight", []).append(
            self.accelerator.gather(seq_is_weight).mean().item()
        )
        n_pos = (grpo_advantages > 0).float().sum()
        n_neg = (grpo_advantages <= 0).float().sum()
        self._metrics[mode].setdefault("grpo_s_conf/frac_pos", []).append(
            (n_pos / (n_pos + n_neg + EPS)).item()
        )

        return loss
