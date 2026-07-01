"""
novel_trainers.py — exp_062 candidate trainers.

ALL candidates compute their shaped per-token advantage on the FULL group inside
_generate_and_score_completions (the FIXED pattern) and propagate the 2-D (G,Lk)
advantage to compute_loss for injection — never the degenerate B=1 recompute that
broke the original gtpo_ema_flipped. The group-visibility + propagation lives in
GroupShapedBase (one tested code path); each candidate only supplies the math via
_token_advantage(out).
"""
import torch
from trl import GRPOTrainer
from .ema_flipped_utils import confidence_from_model_chunked, compute_ema_vectorized
from .format_tag_mask import build_tag_mask, apply_tag_mask_to_token_advantages
from .shaped_loss import inject_advantages
from .novel_shaping import (
    flipped_advantages, apply_sign_gate, refdelta_advantages,
    position_discount, token_logprobs_chunked,
)

_PARAMS = ("alpha1", "alpha2", "lam", "top_k", "reward_threshold",
           "conf_micro_bs", "pos_tau")
_DEFAULTS = dict(alpha1=0.9, alpha2=0.1, lam=0.9, top_k=20,
                 reward_threshold=0.0, conf_micro_bs=1, pos_tau=1024.0)


class GroupShapedBase(GRPOTrainer):
    tag = "groupshaped"

    def __init__(self, *args, **kwargs):
        cfg = {p: kwargs.pop(p) for p in _PARAMS if p in kwargs}
        fmt = kwargs.pop("format_tag_patterns", None)
        super().__init__(*args, **kwargs)
        # set AFTER super().__init__ so GRPOTrainer can't clobber (e.g. top_k)
        for p in _PARAMS:
            setattr(self, p, cfg.get(p, _DEFAULTS[p]))
        self.format_tag_patterns = fmt
        self._warned = False

    # candidate math — returns (G, Lk) token advantage; runs with the FULL group
    def _token_advantage(self, out):
        raise NotImplementedError

    def _grid(self, out):
        cids = out["completion_ids"]; cm = out["completion_mask"]
        input_ids = torch.cat([out["prompt_ids"], cids], dim=1)
        attn = torch.cat([out["prompt_mask"], cm], dim=1)
        return cids, cm, input_ids, attn, cids.size(1)

    def _confidence(self, out):
        cids, cm, input_ids, attn, ltk = self._grid(out)
        return confidence_from_model_chunked(
            self.model, input_ids, attn, ltk, top_k=self.top_k,
            pass_logits_to_keep=("logits_to_keep" in self.model_kwarg_keys),
            micro_bs=self.conf_micro_bs)

    @torch.no_grad()
    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        try:
            ta = self._token_advantage(out)                      # (G, Lk)
            if self.format_tag_patterns:
                tag_mask = build_tag_mask(out["completion_ids"], self.format_tag_patterns)
                ta = apply_tag_mask_to_token_advantages(ta, out["advantages"], tag_mask)
            out["shaped_adv"] = ta.to(out["advantages"].dtype)
        except Exception as e:
            if not self._warned:
                print(f"[{self.tag}] shaping failed -> seq-adv fallback: {e!r}", flush=True)
                self._warned = True
        return out

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")
        cm = inputs["completion_mask"]
        ltk = inputs["completion_ids"].size(1)
        B, W = cm.shape
        shaped = inputs.get("shaped_adv")
        if shaped is not None and shaped.shape[0] == B and shaped.shape[1] == W:
            ta = shaped.float() * cm; used = 1.0
        else:
            ta = inputs["advantages"].unsqueeze(1).float() * cm; used = 0.0
        mode = "train" if model.training else "eval"
        tot = cm.sum().clamp(min=1.0)
        self._metrics[mode].setdefault(f"{self.tag}/used_group_shaped", []).append(used)
        self._metrics[mode].setdefault(f"{self.tag}/mean_token_advantage", []).append(
            self.accelerator.gather((ta * cm).sum() / tot).mean().item())
        inputs = inject_advantages(inputs, ta, ltk)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


class SignGateTrainer(GroupShapedBase):
    """FIXED gtpo_ema_flipped shaping, then sign-consistency gate vs GRPO advantage."""
    tag = "sign_gate"

    def _token_advantage(self, out):
        cids, cm, _, _, _ = self._grid(out)
        conf = self._confidence(out)
        ema = compute_ema_vectorized(conf, cm, lam=self.lam)
        shaped = flipped_advantages(out["advantages"], ema, cm, self.alpha1, self.alpha2,
                                    reward_threshold=self.reward_threshold)
        gated = apply_sign_gate(shaped, out["advantages"], cm)
        # metric: fraction of active tokens the gate left as shaped (vs reverted)
        mode = "train" if self.model.training else "eval"
        active = cm.bool()
        kept = ((torch.sign(shaped) == torch.sign(out["advantages"].unsqueeze(1) * cm)) & active)
        self._metrics[mode].setdefault("sign_gate/frac_kept", []).append(
            (kept.float().sum() / active.float().sum().clamp(min=1)).item())
        return gated


class PosDiscountTrainer(GroupShapedBase):
    """FIXED shaping with a gentle position discount g(t)=tau/(tau+t) on the bonus."""
    tag = "pos_discount"

    def _token_advantage(self, out):
        cids, cm, _, _, _ = self._grid(out)
        conf = self._confidence(out)
        ema = compute_ema_vectorized(conf, cm, lam=self.lam)
        T = cm.size(1)
        g = position_discount(T, self.pos_tau, cm.device).unsqueeze(0).expand(cm.size(0), T)
        return flipped_advantages(out["advantages"], ema, cm, self.alpha1, self.alpha2,
                                  reward_threshold=self.reward_threshold, bonus_mult=g)


class RawCTrainer(GroupShapedBase):
    """Same flipped formula but use raw C_{i,t} instead of EMA(C)."""
    tag = "raw_c"

    def _token_advantage(self, out):
        cids, cm, _, _, _ = self._grid(out)
        conf = self._confidence(out)                              # NO EMA
        return flipped_advantages(out["advantages"], conf, cm, self.alpha1, self.alpha2,
                                  reward_threshold=self.reward_threshold)


class RefDeltaTrainer(GroupShapedBase):
    """Reference-relative log-delta: credit ~ deviation from the frozen base
    (LoRA disabled). delta = logπ_θ(o_t) - logπ_base(o_t)."""
    tag = "ref_delta"

    def _token_advantage(self, out):
        cids, cm, input_ids, attn, ltk = self._grid(out)
        plk = ("logits_to_keep" in self.model_kwarg_keys)
        policy_lp = token_logprobs_chunked(self.model, input_ids, attn, ltk, cids,
                                           pass_logits_to_keep=plk, micro_bs=self.conf_micro_bs)
        with self.model.disable_adapter():
            ref_lp = token_logprobs_chunked(self.model, input_ids, attn, ltk, cids,
                                            pass_logits_to_keep=plk, micro_bs=self.conf_micro_bs)
        delta = (policy_lp - ref_lp) * cm
        mode = "train" if self.model.training else "eval"
        self._metrics[mode].setdefault("ref_delta/mean_abs_delta", []).append(
            (delta.abs().sum() / cm.sum().clamp(min=1)).item())
        return refdelta_advantages(out["advantages"], delta, cm, self.alpha1, self.alpha2,
                                   reward_threshold=self.reward_threshold)
