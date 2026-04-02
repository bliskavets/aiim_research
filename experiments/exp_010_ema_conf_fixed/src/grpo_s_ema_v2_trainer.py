"""
grpo_s_ema_v2_trainer.py
------------------------
Fixed GRPO-S-EMA trainer for exp_010.

Key fix: sequence advantages computed from raw rewards, EMA bonus
added on top without z-score that erases the signal.
"""
import torch
from trl import GRPOTrainer
from .ema_confidence_utils_v2 import confidence_from_logits, compute_grpo_s_ema_advantages, EPS


class GRPOSEMAv2Trainer(GRPOTrainer):
    """
    GRPO-S-EMA v2: sequence-level EMA confidence advantage shaping.

    Kwargs:
        beta1 (float): weight for base GRPO advantage (default 1.0)
        beta2 (float): weight for EMA confidence bonus (default 0.1)
        top_k (int):   top-k for confidence (default 20)
        lam   (float): EMA decay factor (default 0.9)
        reward_threshold (float): O+/O- split (default 0.0)
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

        # Raw rewards — key fix
        raw_rewards = inputs.get("rewards")
        if raw_rewards is not None and isinstance(raw_rewards, torch.Tensor):
            seq_rewards = raw_rewards
        else:
            seq_rewards = inputs["advantages"]

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

        token_advantages, last_ema = compute_grpo_s_ema_advantages(
            rewards=seq_rewards,
            confidence=confidence,
            completion_mask=completion_mask,
            beta1=self.beta1,
            beta2=self.beta2,
            lam=self.lam,
            reward_threshold=self.reward_threshold,
        )

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

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
        mean_last_ema = last_ema.mean()
        self._metrics[mode].setdefault("grpo_s_ema/mean_last_ema", []).append(
            self.accelerator.gather(mean_last_ema).mean().item()
        )
        mean_adv = (token_advantages * completion_mask).sum() / completion_mask.sum().clamp(min=1)
        self._metrics[mode].setdefault("grpo_s_ema/mean_token_advantage", []).append(
            self.accelerator.gather(mean_adv).mean().item()
        )
        n_pos = (seq_rewards > self.reward_threshold).float().sum()
        n_neg = (seq_rewards <= self.reward_threshold).float().sum()
        self._metrics[mode].setdefault("grpo_s_ema/frac_pos", []).append(
            (n_pos / (n_pos + n_neg + EPS)).item()
        )

        return loss
