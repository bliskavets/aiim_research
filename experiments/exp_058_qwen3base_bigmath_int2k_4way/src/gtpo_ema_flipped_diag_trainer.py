"""
gtpo_ema_flipped_diag_trainer.py
--------------------------------
DIAGNOSTIC clone of gtpo_ema_flipped (bare) — identical math, plus heavy per-step
logging to understand WHY length explodes. The shaping/loss path is byte-for-byte
the same as GTPOEMAFlippedTrainer; we only ADD two JSONL writers:

  diag/diag_<tag>_gens.jsonl   (one record per generation step, full group)
     step, per-completion: token length, seq advantage (sign => O+/O-),
     boxed-correct (vs gold), and the completion text (head+tail).

  diag/diag_<tag>_shape.jsonl  (one record per compute_loss microbatch)
     step, per-completion: length, seq_adv, is_pos, sum/mean shaped token-adv,
     and token-advantage / EMA / confidence binned by RELATIVE position (10 bins)
     split by O+/O-. This is what reveals whether later tokens (or longer
     completions) are systematically rewarded -> a length incentive.
"""
import os, json
import torch
from trl import GRPOTrainer
from .ema_flipped_utils import (
    confidence_from_model_chunked, compute_ema_vectorized,
    compute_gtpo_ema_flipped_advantages, EPS,
)
from .format_tag_mask import build_tag_mask, apply_tag_mask_to_token_advantages
from .shaped_loss import inject_advantages

_NBINS = 10


def _answers_equal(guess, gold):
    if guess is None:
        return False
    try:
        return float(str(guess).strip().replace(",", "")) == float(str(gold).strip().replace(",", ""))
    except (ValueError, TypeError):
        return False


def _pos_binned(values, mask, nbins=_NBINS):
    """Mean of `values` over valid tokens, bucketed by RELATIVE position in the
    completion (0..1 -> nbins). values,mask: (B,Lk). Returns (nbins,) list with
    NaN for empty bins (averaged across all rows)."""
    acc = [0.0] * nbins
    cnt = [0] * nbins
    B = values.shape[0]
    for i in range(B):
        idx = mask[i].nonzero(as_tuple=True)[0]
        L = idx.numel()
        if L == 0:
            continue
        v = values[i, idx].detach().float().tolist()
        for k in range(L):
            b = min(int(k / L * nbins), nbins - 1)
            acc[b] += v[k]; cnt[b] += 1
    return [(acc[b] / cnt[b]) if cnt[b] else float("nan") for b in range(nbins)]


class GTPOEMAFlippedDiagTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        alpha1               = kwargs.pop("alpha1", 0.9)
        alpha2               = kwargs.pop("alpha2", 0.1)
        lam                  = kwargs.pop("lam", 0.9)
        top_k                = kwargs.pop("top_k", 20)
        reward_threshold     = kwargs.pop("reward_threshold", 0.0)
        format_tag_patterns  = kwargs.pop("format_tag_patterns", None)
        conf_micro_bs        = kwargs.pop("conf_micro_bs", 2)
        answer_extractor     = kwargs.pop("answer_extractor", None)
        diag_dir             = kwargs.pop("diag_dir", "diag")
        diag_tag             = kwargs.pop("diag_tag", "gtpo_ema_flipped")
        super().__init__(*args, **kwargs)
        self.alpha1, self.alpha2, self.lam, self.top_k = alpha1, alpha2, lam, top_k
        self.reward_threshold = reward_threshold
        self.format_tag_patterns = format_tag_patterns
        self.conf_micro_bs = conf_micro_bs
        self.answer_extractor = answer_extractor
        os.makedirs(diag_dir, exist_ok=True)
        self._gens_path  = os.path.join(diag_dir, f"diag_{diag_tag}_gens.jsonl")
        self._shape_path = os.path.join(diag_dir, f"diag_{diag_tag}_shape.jsonl")

    def _append(self, path, rec):
        try:
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:
            print(f"[diag] write failed {path}: {e!r}", flush=True)

    # ── log the full group's generations (text, length, reward, correctness) ──
    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        try:
            cm = out["completion_mask"]
            cids = out["completion_ids"]
            adv = out["advantages"].detach().float().reshape(-1).tolist()
            lens = cm.sum(dim=1).long().tolist()
            tok = self.processing_class
            recs = []
            for i in range(cids.shape[0]):
                L = int(cm[i].sum().item())
                ids_i = cids[i][cm[i].bool()] if cm[i].any() else cids[i][:0]
                text = tok.decode(ids_i, skip_special_tokens=False) if ids_i.numel() else ""
                gold = inputs[i].get("answer") if i < len(inputs) and isinstance(inputs[i], dict) else None
                guess = self.answer_extractor(text) if self.answer_extractor else None
                recs.append({
                    "len": L,
                    "seq_adv": round(adv[i], 4) if i < len(adv) else None,
                    "boxed": guess,
                    "boxed_correct": bool(_answers_equal(guess, gold)),
                    "head": text[:400],
                    "tail": text[-400:],
                })
            self._append(self._gens_path,
                         {"step": int(self.state.global_step), "n": len(recs), "gens": recs})
        except Exception as e:
            print(f"[diag] gens log skipped: {e!r}", flush=True)
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

        # ── standard presence metrics ──
        mode = "train" if model.training else "eval"
        ema = compute_ema_vectorized(confidence, completion_mask, lam=self.lam)
        tot = completion_mask.sum().clamp(min=1.0)
        self._metrics[mode].setdefault("gtpo_ema_flipped/mean_ema", []).append(
            self.accelerator.gather((ema * completion_mask).sum() / tot).mean().item())
        self._metrics[mode].setdefault("gtpo_ema_flipped/mean_token_advantage", []).append(
            self.accelerator.gather((token_advantages * completion_mask).sum() / tot).mean().item())

        # ── DIAGNOSTIC shape log ──
        try:
            is_pos = (seq_advantages > self.reward_threshold)
            pm = completion_mask * is_pos.float().unsqueeze(1)
            nm = completion_mask * (~is_pos).float().unsqueeze(1)
            lens = completion_mask.sum(dim=1).float()
            sum_adv = (token_advantages * completion_mask).sum(dim=1)
            mean_adv = sum_adv / lens.clamp(min=1.0)
            rec = {
                "step": int(self.state.global_step),
                "len": lens.long().tolist(),
                "seq_adv": [round(x, 4) for x in seq_advantages.detach().float().reshape(-1).tolist()],
                "is_pos": is_pos.detach().reshape(-1).long().tolist(),
                "sum_tok_adv": [round(x, 4) for x in sum_adv.detach().float().tolist()],
                "mean_tok_adv": [round(x, 4) for x in mean_adv.detach().float().tolist()],
                # per relative-position bins (split by polarity) — the key view
                "tok_adv_bins_pos": _pos_binned(token_advantages, pm),
                "tok_adv_bins_neg": _pos_binned(token_advantages, nm),
                "ema_bins_pos":     _pos_binned(ema, pm),
                "ema_bins_neg":     _pos_binned(ema, nm),
                "conf_bins":        _pos_binned(confidence, completion_mask),
            }
            self._append(self._shape_path, rec)
        except Exception as e:
            print(f"[diag] shape log skipped: {e!r}", flush=True)

        inputs = inject_advantages(inputs, token_advantages, logits_to_keep)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
