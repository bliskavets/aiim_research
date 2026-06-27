"""
gtpo_ema_flipped_grop_trainer.py
--------------------------------
exp_058 — gtpo_ema_flipped + Group Relative Overlong Punishment (GROP), the
length-control heuristic from Appendix D of arXiv:2508.04349 (the GTPO/GRPO-S
paper). Per group of num_generations responses, classify the question by solve
rate frac = n_correct/G:
  - EASY   (frac >= gamma1)      -> penalize the CORRECT responses (knee L+ over
                                     correct lengths);
  - HARD   (frac <= 1-gamma1)    -> no penalty (preserve solving ability);
  - MEDIUM (otherwise)           -> knee L- over ALL G lengths; if n>m penalize
                                     correct, else penalize incorrect.
Penalty R(i) in [-0.5,0]: -0.5*(|o|-L)/L on the ramp L<=|o|<2L, -0.5 for |o|>=2L.
See src/adaptive_lenpen_utils.group_relative_overlong_punishment.

"Correct" = exact boxed answer match to the gold (terminal-reward correctness),
computed in _generate_and_score_completions where prompts/answers + the full group
are available; the per-response penalty is propagated to compute_loss via
out["len_pen"].

INJECTION-POINT NOTE: the paper adds R(i) to the REWARD. Our gtpo_ema_flipped
shaping discards the reward magnitude (it uses only the SIGN of the group-relative
reward for the O+/O- split — see DIAG_LENGTH_EXPLOSION.md), so a reward-level add
is ~no-op here. To preserve the paper's INTENT (actually control length) we
subtract the same R(i) from the shaped per-token advantage, consistent with the
working exp_058 length-penalty family.
"""
import torch
from trl import GRPOTrainer
from .ema_flipped_utils import (
    confidence_from_model_chunked, compute_ema_vectorized,
    compute_gtpo_ema_flipped_advantages, EPS,
)
from .format_tag_mask import build_tag_mask, apply_tag_mask_to_token_advantages
from .shaped_loss import inject_advantages
from .adaptive_lenpen_utils import group_relative_overlong_punishment


def _answers_equal(guess, gold):
    if guess is None:
        return False
    try:
        return float(str(guess).strip().replace(",", "")) == float(str(gold).strip().replace(",", ""))
    except (ValueError, TypeError):
        return False


class GTPOEMAFlippedGROPTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        alpha1               = kwargs.pop("alpha1", 0.9)
        alpha2               = kwargs.pop("alpha2", 0.1)
        lam                  = kwargs.pop("lam", 0.9)
        top_k                = kwargs.pop("top_k", 20)
        reward_threshold     = kwargs.pop("reward_threshold", 0.0)
        format_tag_patterns  = kwargs.pop("format_tag_patterns", None)
        conf_micro_bs        = kwargs.pop("conf_micro_bs", 2)
        gamma1               = kwargs.pop("gamma1", 0.75)
        answer_extractor     = kwargs.pop("answer_extractor", None)
        super().__init__(*args, **kwargs)
        self.alpha1, self.alpha2, self.lam, self.top_k = alpha1, alpha2, lam, top_k
        self.reward_threshold = reward_threshold
        self.format_tag_patterns = format_tag_patterns
        self.conf_micro_bs = conf_micro_bs
        self.gamma1 = gamma1
        self.answer_extractor = answer_extractor

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        cm = out["completion_mask"]
        cids = out["completion_ids"]
        device = cm.device
        n_comp = cm.shape[0]
        lengths = cm.sum(dim=1).float()
        # terminal-reward correctness: exact boxed match to gold
        correct = torch.zeros(n_comp, device=device)
        try:
            tok = self.processing_class
            for i in range(n_comp):
                ids_i = cids[i][cm[i].bool()] if cm[i].any() else cids[i][:0]
                text = tok.decode(ids_i, skip_special_tokens=False) if ids_i.numel() else ""
                gold = inputs[i].get("answer") if i < len(inputs) and isinstance(inputs[i], dict) else None
                guess = self.answer_extractor(text) if self.answer_extractor else None
                correct[i] = 1.0 if _answers_equal(guess, gold) else 0.0
        except Exception as e:
            print(f"[grop] correctness calc failed, no penalty this step: {e!r}", flush=True)
            out["len_pen"] = torch.zeros(n_comp, device=device)
            out["grop_regime"] = torch.zeros(n_comp, device=device)
            return out
        pen, regime = group_relative_overlong_punishment(
            lengths, correct, self.num_generations, gamma1=self.gamma1)
        out["len_pen"] = pen
        out["grop_regime"] = regime.float()
        out["grop_correct"] = correct
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

        # GROP penalty (R(i) <= 0) subtracted from the shaped advantage; propagated
        # from _generate_and_score (full group) since compute_loss runs on B=1.
        B = completion_mask.shape[0]
        pen = inputs.get("len_pen")
        if pen is None or pen.shape[0] != B:
            pen = torch.zeros(B, device=completion_mask.device, dtype=token_advantages.dtype)
            pen_present = 0.0
        else:
            pen = pen.to(token_advantages.dtype); pen_present = 1.0
        token_advantages = token_advantages - pen.unsqueeze(1) * completion_mask

        mode = "train" if model.training else "eval"
        self._metrics[mode].setdefault("gtpo_ema_flipped_grop/mean_len", []).append(
            self.accelerator.gather(completion_mask.sum(dim=1).float().mean()).mean().item())
        self._metrics[mode].setdefault("gtpo_ema_flipped_grop/pen_absmean", []).append(
            self.accelerator.gather(pen.abs().mean()).mean().item())
        self._metrics[mode].setdefault("gtpo_ema_flipped_grop/pen_present", []).append(pen_present)
        reg = inputs.get("grop_regime")
        if reg is not None and reg.shape[0] == B:
            r = reg.to(token_advantages.dtype)
            self._metrics[mode].setdefault("gtpo_ema_flipped_grop/frac_easy", []).append(
                self.accelerator.gather((r == 1).to(token_advantages.dtype).mean()).mean().item())
            self._metrics[mode].setdefault("gtpo_ema_flipped_grop/frac_hard", []).append(
                self.accelerator.gather((r == 0).to(token_advantages.dtype).mean()).mean().item())
            self._metrics[mode].setdefault("gtpo_ema_flipped_grop/frac_medium", []).append(
                self.accelerator.gather((r == 2).to(token_advantages.dtype).mean()).mean().item())
        cor = inputs.get("grop_correct")
        if cor is not None and cor.shape[0] == B:
            self._metrics[mode].setdefault("gtpo_ema_flipped_grop/frac_correct", []).append(
                self.accelerator.gather(cor.to(token_advantages.dtype).mean()).mean().item())

        inputs = inject_advantages(inputs, token_advantages, logits_to_keep)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
