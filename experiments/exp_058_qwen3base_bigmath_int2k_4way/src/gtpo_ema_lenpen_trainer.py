"""
gtpo_ema_lenpen_trainer.py
--------------------------
exp_058 method #1 (v2): gtpo_ema_flipped + a GROUP-RELATIVE length penalty.

Why v2: gtpo_ema_flipped's shaping discards the reward MAGNITUDE (it uses the
seq advantage only for the O+/O- sign, then z-norms a confidence weighting). So
(a) a reward-level penalty barely matters (it only flips signs), and (b) an
ABSOLUTE per-sequence penalty on the shaped advantage (v1, alpha=0.0015..0.005)
failed to stop the length collapse. We instead apply a penalty that is
GROUP-CENTERED (relative within each group of num_generations completions):

    pen_i      = alpha_len * max(0, |o_i| - L)
    pen_rel_i  = pen_i - mean_{group}(pen)        # >0 if longer than group avg
    Ã_{i,t}   <- Ã_{i,t} - pen_rel_i              # shorter-than-avg -> boosted

so within each prompt's group, shorter completions get a higher shaped advantage
and longer ones lower — directly ranking "short" above "long". Because
compute_loss runs on B=1 microbatches (no group there), pen_rel is computed in
_generate_and_score_completions (full group available) and propagated through the
buffer to compute_loss as out["len_pen"].
"""
import torch
from trl import GRPOTrainer
from .ema_flipped_utils import (
    confidence_from_model_chunked, compute_ema_vectorized,
    compute_gtpo_ema_flipped_advantages, EPS,
)
from .format_tag_mask import build_tag_mask, apply_tag_mask_to_token_advantages
from .shaped_loss import inject_advantages


class GTPOEMAFlippedLenPenTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        alpha1               = kwargs.pop("alpha1", 0.9)
        alpha2               = kwargs.pop("alpha2", 0.1)
        lam                  = kwargs.pop("lam", 0.9)
        top_k                = kwargs.pop("top_k", 20)
        reward_threshold     = kwargs.pop("reward_threshold", 0.0)
        format_tag_patterns  = kwargs.pop("format_tag_patterns", None)
        conf_micro_bs        = kwargs.pop("conf_micro_bs", 2)
        alpha_len            = kwargs.pop("alpha_len", 0.005)
        length_L             = kwargs.pop("length_L", 1024)
        super().__init__(*args, **kwargs)
        self.alpha1, self.alpha2, self.lam, self.top_k = alpha1, alpha2, lam, top_k
        self.reward_threshold = reward_threshold
        self.format_tag_patterns = format_tag_patterns
        self.conf_micro_bs = conf_micro_bs
        self.alpha_len, self.length_L = alpha_len, length_L

    # group-relative length penalty, computed where the full group is available
    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        cm = out["completion_mask"]
        lengths = cm.sum(dim=1).float()                              # (B_gen,)
        pen = self.alpha_len * (lengths - self.length_L).clamp(min=0.0)
        ng = self.num_generations
        if pen.numel() % ng == 0:
            pen_rel = (pen.view(-1, ng) - pen.view(-1, ng).mean(dim=1, keepdim=True)).reshape(-1)
        else:                                                        # safety: center over batch
            pen_rel = pen - pen.mean()
        out["len_pen"] = pen_rel
        return out

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")
        completion_ids  = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        seq_advantages  = inputs["advantages"]
        input_ids      = torch.cat([inputs["prompt_ids"], completion_ids], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        confidence = confidence_from_model_chunked(
            model, input_ids, attention_mask, logits_to_keep, top_k=self.top_k,
            pass_logits_to_keep=("logits_to_keep" in self.model_kwarg_keys),
            micro_bs=self.conf_micro_bs)

        token_advantages = compute_gtpo_ema_flipped_advantages(
            rewards=seq_advantages, confidence=confidence, completion_mask=completion_mask,
            alpha1=self.alpha1, alpha2=self.alpha2, lam=self.lam,
            reward_threshold=self.reward_threshold)
        if self.format_tag_patterns:
            tag_mask = build_tag_mask(completion_ids, self.format_tag_patterns)
            token_advantages = apply_tag_mask_to_token_advantages(
                token_advantages, seq_advantages, tag_mask)

        # group-relative length penalty (propagated from generation)
        B = completion_mask.shape[0]
        pen_rel = inputs.get("len_pen")
        if pen_rel is None or pen_rel.shape[0] != B:
            pen_rel = torch.zeros(B, device=completion_mask.device, dtype=token_advantages.dtype)
            pen_present = 0.0
        else:
            pen_rel = pen_rel.to(token_advantages.dtype); pen_present = 1.0
        token_advantages = token_advantages - pen_rel.unsqueeze(1) * completion_mask

        # ── metrics ──
        mode = "train" if model.training else "eval"
        tot = completion_mask.sum().clamp(min=1.0)
        ema = compute_ema_vectorized(confidence, completion_mask, lam=self.lam)
        self._metrics[mode].setdefault("gtpo_ema_lenpen/mean_ema", []).append(
            self.accelerator.gather((ema * completion_mask).sum() / tot).mean().item())
        self._metrics[mode].setdefault("gtpo_ema_lenpen/mean_len", []).append(
            self.accelerator.gather(completion_mask.sum(dim=1).float().mean()).mean().item())
        self._metrics[mode].setdefault("gtpo_ema_lenpen/pen_rel_absmean", []).append(
            self.accelerator.gather(pen_rel.abs().mean()).mean().item())
        self._metrics[mode].setdefault("gtpo_ema_lenpen/pen_present", []).append(pen_present)

        inputs = inject_advantages(inputs, token_advantages, logits_to_keep)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
