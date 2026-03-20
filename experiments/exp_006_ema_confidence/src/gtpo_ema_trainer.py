"""
gtpo_ema_trainer.py
-------------------
GTPO with EMA-smoothed per-token confidence (exp_006).
"""
import torch
from trl import GRPOTrainer
from .ema_confidence_utils import confidence_from_logits, compute_gtpo_ema_rewards, EPS


class GTPOEMATrainer(GRPOTrainer):
    """
    Extra kwargs:
        alpha1, alpha2: reward shaping weights
        top_k: top-k for confidence (default 20)
        lam: EMA decay factor (default 0.9)
        reward_threshold: O+/O- split (default 0.0)
    """
    def __init__(self, *args, **kwargs):
        self.alpha1           = kwargs.pop("alpha1", 1.0)
        self.alpha2           = kwargs.pop("alpha2", 0.1)
        self.top_k            = kwargs.pop("top_k", 20)
        self.lam              = kwargs.pop("lam", 0.9)
        self.reward_threshold = kwargs.pop("reward_threshold", 0.0)
        super().__init__(*args, **kwargs)

    def _compute_loss(self, model, inputs):
        prompt_ids      = inputs["prompt_ids"]
        prompt_mask     = inputs["prompt_mask"]
        completion_ids  = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        seq_advantages  = inputs["advantages"]

        input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=False,
        )

        # Get logits for confidence
        with torch.no_grad():
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1
            raw_out = model(**model_inputs)
            logits = raw_out.logits[:, :-1, :][:, -logits_to_keep:, :]
            confidence = confidence_from_logits(logits, top_k=self.top_k)

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        adv_pos, adv_neg = compute_gtpo_ema_rewards(
            rewards=seq_advantages, confidence=confidence,
            completion_mask=completion_mask,
            alpha1=self.alpha1, alpha2=self.alpha2,
            lam=self.lam, reward_threshold=self.reward_threshold,
        )
        token_advantages = adv_pos + adv_neg

        per_token_loss = -torch.min(coef_1 * token_advantages, coef_2 * token_advantages)

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
        self._metrics[mode].setdefault("gtpo_ema/mean_confidence", []).append(
            self.accelerator.gather(mean_conf).mean().item()
        )
        mean_adv = (token_advantages * completion_mask).sum() / total_tokens
        self._metrics[mode].setdefault("gtpo_ema/mean_token_advantage", []).append(
            self.accelerator.gather(mean_adv).mean().item()
        )
        n_pos = (seq_advantages > self.reward_threshold).float().sum()
        n_neg = (seq_advantages <= self.reward_threshold).float().sum()
        self._metrics[mode].setdefault("gtpo_ema/frac_pos", []).append(
            (n_pos / (n_pos + n_neg + EPS)).item()
        )

        return loss
