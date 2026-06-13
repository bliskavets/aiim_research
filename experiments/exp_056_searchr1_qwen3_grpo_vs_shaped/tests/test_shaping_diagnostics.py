"""
test_shaping_diagnostics.py — regression tests pinning the EMPIRICAL behaviour of
the gtpo_ema_flipped shaping, written after a "why does it look like grpo?" audit.

These document what the shaping actually does (not what the prose claims):
  1. the tag mask is precise (single special-token ids) and covers <5% of a
     realistic completion — it does NOT mask away most tokens;
  2. the shaped per-token advantage IS applied (materially != grpo broadcast);
  3. BUT the per-polarity z-norm centers each group at ~0 and washes out
     alpha1/alpha2, so the sequence-reward magnitude is discarded and the
     shaped advantage is ~uncorrelated with the GRPO seq advantage.
"""
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.ema_flipped_utils import compute_gtpo_ema_flipped_advantages
from src.format_tag_mask import (
    encode_tag_patterns, build_tag_mask, apply_tag_mask_to_token_advantages,
)


def _batch(G=8, T=200, conf_std=1.0, seed=1):
    g = torch.Generator().manual_seed(seed)
    rewards = torch.randn(G, generator=g)
    seq_adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    conf = (6.0 + conf_std * torch.randn(G, T, generator=g)).clamp(min=0.1)
    return seq_adv, conf, torch.ones(G, T)


def test_shaping_is_applied_not_a_noop():
    """Shaped advantage must differ materially from the grpo broadcast."""
    seq_adv, conf, mask = _batch()
    shaped = compute_gtpo_ema_flipped_advantages(seq_adv, conf, mask, 0.9, 0.1, 0.9)
    grpo = seq_adv.view(-1, 1).expand_as(shaped)
    rel = (shaped - grpo).norm() / (grpo.norm() + 1e-8)
    assert rel > 0.5, f"shaping barely changes advantage (rel={rel:.3f}) — possible no-op"
    within = shaped.std(dim=1).mean()
    assert within > 0.1, "shaped advantage is uniform within a sequence (no per-token spread)"


def test_alpha_is_washed_out_by_znorm():
    """alpha1/alpha2 must NOT survive the per-polarity z-norm (documents the bug)."""
    seq_adv, conf, mask = _batch()
    a = compute_gtpo_ema_flipped_advantages(seq_adv, conf, mask, 0.9, 0.1, 0.9)
    b = compute_gtpo_ema_flipped_advantages(seq_adv, conf, mask, 0.1, 0.9, 0.9)
    assert (a - b).abs().max() < 1e-3, \
        "alpha1/alpha2 affect the output — z-norm not washing them out as observed"


def test_each_polarity_centered_near_zero():
    """z-norm centers O+ and O- pools at ~0 — the seq-reward magnitude is gone."""
    seq_adv, conf, mask = _batch()
    shaped = compute_gtpo_ema_flipped_advantages(seq_adv, conf, mask, 0.9, 0.1, 0.9)
    is_pos = seq_adv > 0
    assert shaped[is_pos].mean().abs() < 0.05, "O+ pool not centered at 0"
    assert shaped[~is_pos].mean().abs() < 0.05, "O- pool not centered at 0"


def test_shaped_weakly_correlated_with_reward():
    """Shaped advantage is ~uncorrelated with the GRPO seq advantage."""
    seq_adv, conf, mask = _batch()
    shaped = compute_gtpo_ema_flipped_advantages(seq_adv, conf, mask, 0.9, 0.1, 0.9)
    grpo = seq_adv.view(-1, 1).expand_as(shaped)
    s, gv = shaped[mask.bool()], grpo[mask.bool()]
    corr = torch.corrcoef(torch.stack([s, gv]))[0, 1].abs()
    assert corr < 0.3, f"shaped adv unexpectedly tracks reward (corr={corr:.3f})"


def test_tag_mask_is_precise_and_small():
    """Tag patterns are single special-token ids; mask covers a few tokens only."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    pats = encode_tag_patterns(tok, ["<think>", "</think>", "<|im_start|>", "<|im_end|>"])
    # the bare-tag patterns must each be a single token id
    singles = [p for p in pats if len(p) == 1]
    assert len(singles) == 4, f"expected 4 single-token tag patterns, got {singles}"
    content = "Let me solve. x equals five plus three so the answer is eight. " * 8
    ids = tok.encode("<think>" + content + "</think> So \\boxed{8}.<|im_end|>",
                     add_special_tokens=False)
    t = torch.tensor([ids])
    mask = build_tag_mask(t, pats)
    frac = mask.float().mean().item()
    assert frac < 0.05, f"tag mask covers {100*frac:.1f}% of tokens — far too much"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
