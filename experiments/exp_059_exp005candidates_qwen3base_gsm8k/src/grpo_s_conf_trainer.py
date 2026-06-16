"""
grpo_s_conf_trainer.py (exp_059) — GRPO-S-Conf with exp_005's sequence-level
confidence shaping, wired through the FIXED injection framework.

Sequence-level: compute confidence-shaped sequence rewards (B,), re-normalize
within GRPO groups -> advantages (B,), and inject (1-D) into the compiled loss.
"""
import torch
from trl import GRPOTrainer

from .confidence_utils import confidence_from_logits, compute_grpo_s_conf_rewards, EPS
from .shaped_loss import forward_completion_logits, inject_advantages


class GRPOSConfTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        beta1 = kwargs.pop("beta1", 1.0)
        beta2 = kwargs.pop("beta2", 0.1)
        top_k = kwargs.pop("top_k", 20)
        reward_threshold = kwargs.pop("reward_threshold", 0.0)
        super().__init__(*args, **kwargs)
        self.beta1, self.beta2, self.top_k = beta1, beta2, top_k
        self.reward_threshold = reward_threshold

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")

        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        grpo_advantages = inputs["advantages"]
        input_ids = torch.cat([inputs["prompt_ids"], completion_ids], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            logits = forward_completion_logits(self, model, input_ids, attention_mask, logits_to_keep)
            confidence = confidence_from_logits(logits, top_k=self.top_k)   # (B, Lk)
        del logits

        shaped_rewards, seq_avg_conf = compute_grpo_s_conf_rewards(
            rewards=grpo_advantages, confidence=confidence, completion_mask=completion_mask,
            beta1=self.beta1, beta2=self.beta2, reward_threshold=self.reward_threshold,
        )
        # re-normalize within GRPO groups -> (B,) advantages
        G = self.num_generations
        sg = shaped_rewards.view(-1, G)
        mean = sg.mean(dim=1, keepdim=True)
        std = sg.std(dim=1, keepdim=True).clamp(min=EPS)
        advantages = ((sg - mean) / std).reshape(-1)

        mode = "train" if model.training else "eval"
        self._metrics[mode].setdefault("grpo_s_conf/mean_seq_confidence", []).append(
            self.accelerator.gather(seq_avg_conf).mean().item())
        self._metrics[mode].setdefault("grpo_s_conf/mean_shaped_advantage", []).append(
            self.accelerator.gather(advantages).mean().item())

        inputs = inject_advantages(inputs, advantages, logits_to_keep)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
