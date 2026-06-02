"""
test_format_tag_mask.py — minimal correctness check for the tag-mask logic.

We do not depend on a real HF tokenizer here. Instead we fake a tiny
"tokeniser" — a Python dict where each character is its own token id —
and verify build_tag_mask + apply_tag_mask_to_token_advantages on a
hand-rolled batch.
"""
import sys
import os

import torch

# allow `python -m pytest` from the experiment dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.format_tag_mask import (
    build_tag_mask,
    apply_tag_mask_to_token_advantages,
)


def char_ids(s, vocab):
    return [vocab[c] for c in s]


def test_build_tag_mask_marks_only_tag_positions():
    # vocab: each char -> unique id
    chars = "abcdef<>/SOL"
    vocab = {c: i for i, c in enumerate(chars)}

    # tag = "<SOL>", tokens = [vocab[c] for c in "<SOL>"]
    pat_open  = char_ids("<SOL>", vocab)
    pat_close = char_ids("</SOL>", vocab)
    patterns = [pat_open, pat_close]

    # completion: "ab<SOL>cd</SOL>ef"  -> 17 tokens
    completion = "ab<SOL>cd</SOL>ef"
    ids = torch.tensor([char_ids(completion, vocab)], dtype=torch.long)

    mask = build_tag_mask(ids, patterns)

    # positions for "<SOL>" -> 2..6, for "</SOL>" -> 9..14
    expected = torch.zeros_like(ids, dtype=torch.bool)
    expected[0, 2:7]  = True   # <SOL>
    expected[0, 9:15] = True   # </SOL>

    assert mask.shape == ids.shape
    assert torch.equal(mask, expected), (
        f"\nmask:     {mask.int().tolist()}\nexpected: {expected.int().tolist()}")


def test_apply_tag_mask_replaces_token_adv_with_seq_adv_on_tag_positions():
    B, T = 2, 6
    token_adv = torch.arange(B * T, dtype=torch.float32).view(B, T)  # 0..11
    seq_adv = torch.tensor([-1.0, +1.0])
    mask = torch.tensor([
        [False, False, True,  True,  False, False],
        [True,  False, False, False, True,  True ],
    ])
    out = apply_tag_mask_to_token_advantages(token_adv, seq_adv, mask)

    # row 0: tag at positions 2,3 → -1.0, others keep 0,1,4,5
    assert out[0].tolist() == [0.0, 1.0, -1.0, -1.0, 4.0, 5.0]
    # row 1: tag at positions 0,4,5 → +1.0, others keep 7,8,9
    assert out[1].tolist() == [+1.0, 7.0, 8.0, 9.0, +1.0, +1.0]


def test_empty_patterns_yields_zero_mask():
    ids = torch.zeros((1, 8), dtype=torch.long)
    mask = build_tag_mask(ids, [])
    assert mask.sum().item() == 0


def test_pattern_longer_than_completion_is_no_op():
    ids = torch.zeros((1, 3), dtype=torch.long)
    patterns = [[1, 2, 3, 4, 5]]  # longer than completion
    mask = build_tag_mask(ids, patterns)
    assert mask.sum().item() == 0


if __name__ == "__main__":
    test_build_tag_mask_marks_only_tag_positions(); print("  ✓ build_tag_mask marks tag positions")
    test_apply_tag_mask_replaces_token_adv_with_seq_adv_on_tag_positions(); print("  ✓ apply_tag_mask replaces on tag positions")
    test_empty_patterns_yields_zero_mask(); print("  ✓ empty patterns -> zero mask")
    test_pattern_longer_than_completion_is_no_op(); print("  ✓ pattern longer than completion -> no-op")
    print("all tag-mask tests passed")
