"""
nucleus_c.py — dynamic (top-p / nucleus) k for the confidence signal C.

Per token: choose the nucleus size n = #{ leading tokens whose cumulative
PROBABILITY mass ≤ top_p } (i.e. largest n with Σ_{j<n} p_(j) ≤ top_p, and adding
the next token would exceed top_p), clamped to ≥ min_k. Then
    C_it = −(1/n) Σ_{j<n} logπ_(j)
i.e. we use PROBABILITIES only to pick n, but compute C from the LOG-probs (same
signal as the fixed top-k C, just with a per-token adaptive k).
"""
import torch
EPS = 1e-8


def nucleus_C(sorted_logprobs, top_p, min_k=1):
    """sorted_logprobs: (..., N) top-N log-probs sorted DESC. Returns (C (...,), n (...,)).
    Probabilities used ONLY to determine n; C computed from the log-probs."""
    lp = sorted_logprobs.float()
    p = torch.exp(lp)                                  # (..., N) prob (sorted desc)
    cum = torch.cumsum(p, dim=-1)                      # (..., N)
    N = lp.shape[-1]
    n = (cum <= top_p).sum(dim=-1)                     # count of prefix sums ≤ top_p
    n = n.clamp(min=min_k, max=N)                      # min_k guard; cap at N
    ar = torch.arange(N, device=lp.device)
    mask = (ar.expand_as(lp) < n.unsqueeze(-1)).float()  # keep first n
    C = -(lp * mask).sum(dim=-1) / n.to(lp.dtype).clamp(min=1)
    return C, n


@torch.no_grad()
def nucleus_C_from_model_chunked(model, input_ids, attention_mask, logits_to_keep,
                                 top_p, min_k=1, cap=256, pass_logits_to_keep=False, micro_bs=1):
    """One forward per micro-batch -> (C (G,T), n (G,T)). Takes top-`cap` logprobs
    (enough to reach top_p for the vast majority of tokens; flat tokens cap at `cap`)."""
    B = input_ids.size(0)
    Cs, Ns = [], []
    for s in range(0, B, micro_bs):
        e = min(s + micro_bs, B)
        mi = {"input_ids": input_ids[s:e], "attention_mask": attention_mask[s:e]}
        if pass_logits_to_keep:
            mi["logits_to_keep"] = logits_to_keep + 1
        logits = model(**mi).logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :].float()
        lp = torch.log_softmax(logits, dim=-1)
        k = min(cap, lp.shape[-1])
        top_lp = lp.topk(k, dim=-1).values           # (b, T, cap) sorted desc
        C, n = nucleus_C(top_lp, top_p, min_k)
        Cs.append(C); Ns.append(n.float())
        del logits, lp, top_lp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(Cs, 0), torch.cat(Ns, 0)
