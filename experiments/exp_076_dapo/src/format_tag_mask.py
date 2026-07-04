"""
format_tag_mask.py
------------------
Build a (B, T) boolean mask over a batch of completion token sequences,
marking positions that belong to one of the structural format tags
(<start_working_out>, <end_working_out>, <SOLUTION>, </SOLUTION>).

Used by shaping trainers (GRPO-S, GTPO-Conf, GTPO-EMA-Flipped) in exp_050:
on tag positions, the per-token shaped advantage is replaced by the
seq-level GRPO advantage, so the shaping does not "rewrite" the gradient
on tokens that are pure format control. Content tokens stay shaped as usual.

Pattern matching is done by exact token-id subsequence match. The model
produces tag strings verbatim, so the tokenisation is consistent across
all completions and we don't need to re-decode/re-tokenise per step.
"""
from typing import Iterable, List, Sequence

import torch


def encode_tag_patterns(tokenizer, tag_strings: Iterable[str]) -> List[List[int]]:
    """Tokenize each tag string into a list of token-id sequences.

    Returns a list of patterns — one pattern per tag. add_special_tokens=False
    so we get just the tag's own tokens, no BOS/EOS contamination.

    For each tag we also try the "leading space" variant ` <tag>` because
    BPE often merges a leading space with `<`. We keep deduped non-empty
    patterns.
    """
    patterns: List[List[int]] = []
    seen = set()
    for tag in tag_strings:
        for variant in (tag, " " + tag):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if not ids:
                continue
            key = tuple(ids)
            if key in seen:
                continue
            seen.add(key)
            patterns.append(list(ids))
    return patterns


def build_tag_mask(completion_ids: torch.Tensor,
                   patterns: Sequence[Sequence[int]]) -> torch.Tensor:
    """Return a (B, T) bool tensor — True on positions inside any tag pattern.

    Slides each pattern across the completion ids and marks the L-position
    window True wherever the subsequence matches exactly. Overlapping
    patterns are unioned.
    """
    B, T = completion_ids.shape
    mask = torch.zeros((B, T), dtype=torch.bool, device=completion_ids.device)
    ids = completion_ids.detach().cpu().tolist()
    for b in range(B):
        row = ids[b]
        for pat in patterns:
            L = len(pat)
            if L == 0 or L > T:
                continue
            for i in range(T - L + 1):
                if row[i:i + L] == pat:
                    mask[b, i:i + L] = True
    return mask


def apply_tag_mask_to_token_advantages(token_advantages: torch.Tensor,
                                       seq_advantages: torch.Tensor,
                                       tag_mask: torch.Tensor) -> torch.Tensor:
    """On tag positions, replace per-token shaped advantage with seq-level adv.

    token_advantages: (B, T) — shaped per-token advantages
    seq_advantages:   (B,)   — standard GRPO seq-level advantages
    tag_mask:         (B, T) — True where token belongs to a format tag

    Returns (B, T) where tag tokens get the unshaped seq-level advantage and
    content tokens keep their shaped advantage.
    """
    seq_bcast = seq_advantages.view(-1, 1).expand_as(token_advantages)
    return torch.where(tag_mask, seq_bcast, token_advantages)
