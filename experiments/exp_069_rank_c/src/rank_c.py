"""
rank_c.py — adaptive top-k for the confidence signal C, driven by the RANK of the
actually-sampled token in the model's (descending-logprob) distribution.

Per token t let r_t = 1-indexed rank of the sampled token = #{tokens with logprob
strictly greater than the sampled token's} + 1. Then
    k_t = clamp(r_t, min_k, cap)          # e.g. sampled = argmax -> r=1 -> k=1
    C_it = -(1/k_t) Σ_{j<k_t} logπ_(j)    # mean of the top-k_t log-probs (sorted desc)

Rationale (vs nucleus_C / exp_068): here k is small EXACTLY on the confident tokens
(model sampled its argmax, r=1) where -log p_max is a clean sharp signal, and only
grows (up to `cap`) when the model actually sampled from the tail. Unlike nucleus,
k does NOT inflate on flat-but-argmax-picked positions.
"""
import torch


def rank_C(sorted_logprobs, sampled_logprob, ranks, cap=5, min_k=1):
    """Pure/tensor form for unit testing.
    sorted_logprobs: (..., N) top-N log-probs sorted DESC (N >= cap).
    sampled_logprob: (...,) logprob of the realized token (unused for C; kept for API symmetry).
    ranks:           (...,) 1-indexed rank of the realized token.
    Returns (C (...,), k (...,))."""
    lp = sorted_logprobs.float()
    N = lp.shape[-1]
    k = ranks.clamp(min=min_k, max=min(cap, N))
    ar = torch.arange(N, device=lp.device)
    mask = (ar.expand_as(lp) < k.unsqueeze(-1)).float()      # keep first k
    C = -(lp * mask).sum(dim=-1) / k.to(lp.dtype).clamp(min=1)
    return C, k


@torch.no_grad()
def rank_C_from_model_chunked(model, input_ids, attention_mask, completion_ids,
                              logits_to_keep, cap=5, min_k=1,
                              pass_logits_to_keep=False, micro_bs=1):
    """One forward per micro-batch -> (C (G,T), k (G,T)).
    Rank of the sampled token computed over the FULL vocab (strictly-greater count);
    C uses the top-`cap` log-probs (cap is the max k we ever need)."""
    B = input_ids.size(0)
    Cs, Ks = [], []
    for s in range(0, B, micro_bs):
        e = min(s + micro_bs, B)
        mi = {"input_ids": input_ids[s:e], "attention_mask": attention_mask[s:e]}
        if pass_logits_to_keep:
            mi["logits_to_keep"] = logits_to_keep + 1
        logits = model(**mi).logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :].float()          # (b, T, V)
        lp = torch.log_softmax(logits, dim=-1)
        tok = completion_ids[s:e].unsqueeze(-1)                   # (b, T, 1)
        samp_lp = lp.gather(-1, tok).squeeze(-1)                  # (b, T)
        ranks = (lp > samp_lp.unsqueeze(-1)).sum(dim=-1) + 1      # (b, T) 1-indexed
        kcap = min(cap, lp.shape[-1])
        top_lp = lp.topk(kcap, dim=-1).values                    # (b, T, cap) sorted desc
        C, k = rank_C(top_lp, samp_lp, ranks, cap=cap, min_k=min_k)
        Cs.append(C); Ks.append(k.float())
        del logits, lp, top_lp, samp_lp, ranks
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(Cs, 0), torch.cat(Ks, 0)
