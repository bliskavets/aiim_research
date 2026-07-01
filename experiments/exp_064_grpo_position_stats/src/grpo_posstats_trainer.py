"""
grpo_posstats_trainer.py — plain GRPO with per-position logging of C_{i,t}
(confidence = -mean_topk log p) and logprob_{i,t} (of the sampled token), to study
how they depend on generation position (for designing an adaptive pos_discount).

The loss/training is UNCHANGED plain GRPO — we only observe in
_generate_and_score_completions (full group, policy θ_old). Running per-absolute-
position accumulators (count, sum, sumsq) are kept for three groups: all rollouts,
correct rollouts, incorrect rollouts (correctness = exact boxed match). Saved to an
.npz periodically + at the end; analyze_posstats.py turns it into mean±std vs pos.
"""
import os
import numpy as np
import torch
from trl import GRPOTrainer
from .ema_flipped_utils import confidence_from_logits


def _answers_equal(guess, gold):
    if guess is None:
        return False
    try:
        return float(str(guess).strip().replace(",", "")) == float(str(gold).strip().replace(",", ""))
    except (ValueError, TypeError):
        return False


@torch.no_grad()
def confidence_and_logprob_chunked(model, input_ids, attention_mask, logits_to_keep,
                                   target_ids, top_k=20, pass_logits_to_keep=False, micro_bs=1):
    """One forward per micro-batch -> (C, lp), both (B, logits_to_keep).
    C = -mean_topk log p (confidence/peakedness); lp = log p of the sampled token."""
    B = input_ids.size(0)
    Cs, Ls = [], []
    for s in range(0, B, micro_bs):
        e = min(s + micro_bs, B)
        mi = {"input_ids": input_ids[s:e], "attention_mask": attention_mask[s:e]}
        if pass_logits_to_keep:
            mi["logits_to_keep"] = logits_to_keep + 1
        logits = model(**mi).logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :].float()
        lp = torch.log_softmax(logits, dim=-1)
        k = min(top_k, logits.size(-1))
        C = -lp.topk(k, dim=-1).values.mean(dim=-1)
        tok_lp = lp.gather(-1, target_ids[s:e].unsqueeze(-1)).squeeze(-1)
        Cs.append(C); Ls.append(tok_lp)
        del logits, lp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(Cs, dim=0), torch.cat(Ls, dim=0)


def new_accumulator(max_pos):
    z = lambda: np.zeros(max_pos, dtype=np.float64)
    return {g: {"n": z(), "C": z(), "C2": z(), "lp": z(), "lp2": z()}
            for g in ("all", "correct", "incorrect")}


def accumulate_posstats(acc, C, lp, mask, correct):
    """Add a batch to the per-position accumulators. C, lp, mask: (B, T) numpy;
    correct: (B,) 0/1. Position = column index (completions are right-padded, so
    valid tokens occupy the first L_i columns)."""
    T = C.shape[1]
    m = mask.astype(np.float64)
    Cm, Lm = C * m, lp * m
    def add(group, sel):
        ms = m * sel[:, None]
        acc[group]["n"][:T] += ms.sum(0)
        acc[group]["C"][:T] += (C * ms).sum(0)
        acc[group]["C2"][:T] += (C * C * ms).sum(0)
        acc[group]["lp"][:T] += (lp * ms).sum(0)
        acc[group]["lp2"][:T] += (lp * lp * ms).sum(0)
    ones = np.ones(C.shape[0])
    add("all", ones)
    add("correct", correct.astype(np.float64))
    add("incorrect", 1.0 - correct.astype(np.float64))
    return acc


class GRPOPosStatsTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        top_k = kwargs.pop("top_k", 20)
        conf_micro_bs = kwargs.pop("conf_micro_bs", 1)
        answer_extractor = kwargs.pop("answer_extractor", None)
        stats_path = kwargs.pop("stats_path", "diag/grpo_posstats.npz")
        max_pos = kwargs.pop("max_pos", 3584)
        super().__init__(*args, **kwargs)
        self.top_k = top_k
        self.conf_micro_bs = conf_micro_bs
        self.answer_extractor = answer_extractor
        self.stats_path = stats_path
        self.max_pos = max_pos
        self.acc = new_accumulator(max_pos)
        self._gen_calls = 0
        self._save_every = 20
        self._warned = False
        os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)

    def _save(self):
        flat = {f"{g}_{k}": self.acc[g][k] for g in self.acc for k in self.acc[g]}
        flat["gen_calls"] = np.array([self._gen_calls])
        np.savez(self.stats_path, **flat)

    @torch.no_grad()
    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)   # plain GRPO scoring
        try:
            cids = out["completion_ids"]; cm = out["completion_mask"]
            input_ids = torch.cat([out["prompt_ids"], cids], dim=1)
            attn = torch.cat([out["prompt_mask"], cm], dim=1)
            ltk = cids.size(1)
            C, lp = confidence_and_logprob_chunked(
                self.model, input_ids, attn, ltk, cids, top_k=self.top_k,
                pass_logits_to_keep=("logits_to_keep" in self.model_kwarg_keys),
                micro_bs=self.conf_micro_bs)
            tok = self.processing_class
            correct = np.zeros(cids.shape[0])
            for i in range(cids.shape[0]):
                ids_i = cids[i][cm[i].bool()] if cm[i].any() else cids[i][:0]
                text = tok.decode(ids_i, skip_special_tokens=False) if ids_i.numel() else ""
                gold = inputs[i].get("answer") if i < len(inputs) and isinstance(inputs[i], dict) else None
                guess = self.answer_extractor(text) if self.answer_extractor else None
                correct[i] = 1.0 if _answers_equal(guess, gold) else 0.0
            accumulate_posstats(self.acc, C.float().cpu().numpy(), lp.float().cpu().numpy(),
                                cm.cpu().numpy(), correct)
            self._gen_calls += 1
            if self._gen_calls % self._save_every == 0:
                self._save()
        except Exception as e:
            if not self._warned:
                print(f"[posstats] logging skipped: {e!r}", flush=True); self._warned = True
        return out

    def train(self, *a, **k):
        r = super().train(*a, **k)
        self._save()
        return r
