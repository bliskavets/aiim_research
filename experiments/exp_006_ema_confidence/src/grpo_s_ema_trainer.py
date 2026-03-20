"""
grpo_s_ema_trainer.py
---------------------
GRPO-S with last-EMA sequence-level confidence (exp_006).
"""
import torch
from trl import GRPOTrainer
from .ema_confidence_utils import confidence_from_logits, compute_grpo_s_ema_rewards, EPS


class GRPOSEMATrainer(GRPOTrainer):
    """
    Extra kwargs:
        beta1, beta2: reward shaping weights
        top_k: top-k for confidence (default 20)
        lam: EMA decay factor (default 0.9)
        reward_threshold: O+/O- split (default 0.0)
    """
    def __init__(self, *args, **kwargs):
        self.beta1            = kwargs.pop("beta1", 1.0)
        self.beta2            = kwargs.pop("beta2", 0.1)
        self.top_k            = kwargs.pop("top_k", 20)
        self.lam              = kwargs.pop("lam", 0.9)
        self.reward_threshold = kwargs.pop("reward_threshold", 0.0)
        super().__init__(*args, **kwargs)

    def _compute_loss(self, model, inputs):
        prompt_ids      = inputs["prompt_ids"]
        prompt_mask     = inputs["prompt_mask"]
        completion_ids  = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        grpo_advantages = inputs["advantages"]

        input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=False,
        )

        with torch.no_grad():
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1
            raw_out = model(**model_inputs)
            logits = raw_out.logits[:, :-1, :][:, -logits_to_keep:, :]
            confidence = confidence_from_logits(logits, top_k=self.top_k)

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        shaped_rewards, last_ema = compute_grpo_s_ema_rewards(
            rewards=grpo_advantages, confidence=confidence,
            completion_mask=completion_mask,
            beta1=self.beta1, beta2=self.beta2,
            lam=self.lam, reward_threshold=self.reward_threshold,
        )

        # Normalize within groups
        B = shaped_rewards.shape[0]
        G = self.num_generations
        shaped_grouped = shaped_rewards.view(-1, G)
        mean_s = shaped_grouped.mean(dim=1, keepdim=True)
        std_s  = shaped_grouped.std(dim=1, keepdim=True).clamp(min=EPS)
        advantages = ((shaped_grouped - mean_s) / std_s).view(B)

        # Sequence-level IS weights
        log_ratio     = per_token_logps - old_per_token_logps
        token_is      = torch.exp(log_ratio)
        seq_lengths   = completion_mask.sum(dim=1).clamp(min=1)
        seq_is_weight = (token_is * completion_mask).sum(dim=1) / seq_lengths

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

        loss = per_seq_loss.mean() / self.current_gradient_accumulation_steps

        mode = "train" if model.training else "eval"
        self._metrics[mode].setdefault("grpo_s_ema/mean_last_ema", []).append(
            self.accelerator.gather(last_ema).mean().item()
        )
        self._metrics[mode].setdefault("grpo_s_ema/mean_seq_is_weight", []).append(
            self.accelerator.gather(seq_is_weight).mean().item()
        )
        n_pos = (grpo_advantages > 0).float().sum()
        n_neg = (grpo_advantages <= 0).float().sum()
        self._metrics[mode].setdefault("grpo_s_ema/frac_pos", []).append(
            (n_pos / (n_pos + n_neg + EPS)).item()
        )

        return loss
