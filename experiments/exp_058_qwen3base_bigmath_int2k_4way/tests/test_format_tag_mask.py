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


# ─── exp_055: Qwen3 native special tokens only (no custom format tags) ───────
#
# Mask covers 4 Qwen3-native single special tokens. These tests guard against
# accidentally re-introducing custom tags (which conflict with Qwen3's
# enable_thinking=True default and caused the 76% clipping seen in exp_053).

TAGS_EXPECTED = ["<think>", "</think>", "<|im_start|>", "<|im_end|>"]


class _CharTokenizer:
    """Tiny char-level "tokenizer" — enough surface to drive encode_tag_patterns
    for unit tests without loading a real HF model."""
    def __init__(self):
        chars = list(set(" abcdefghijklmnopqrstuvwxyz<>/_|isart"))
        self.vocab = {c: i for i, c in enumerate(sorted(chars))}

    def encode(self, s, add_special_tokens=False):
        return [self.vocab[c] for c in s if c in self.vocab]

    def decode(self, ids):
        inv = {i: c for c, i in self.vocab.items()}
        return "".join(inv.get(i, "?") for i in ids)


def test_encode_tag_patterns_produces_unique_nonempty_patterns_for_all_four_native_tags():
    from src.format_tag_mask import encode_tag_patterns
    tok = _CharTokenizer()
    pats = encode_tag_patterns(tok, TAGS_EXPECTED)
    assert len(pats) >= len(TAGS_EXPECTED), \
        f"expected ≥{len(TAGS_EXPECTED)} patterns, got {len(pats)}"
    assert all(len(p) > 0 for p in pats), "every pattern must be non-empty"
    seen = set()
    for p in pats:
        key = tuple(p)
        assert key not in seen, f"duplicate pattern {p}"
        seen.add(key)
    decoded = [tok.decode(p).strip() for p in pats]
    for tag in TAGS_EXPECTED:
        assert tag in decoded, f"no pattern decodes back to {tag!r} (got {decoded})"


def test_build_tag_mask_detects_all_four_native_tags_in_a_mixed_completion():
    from src.format_tag_mask import encode_tag_patterns, build_tag_mask
    tok = _CharTokenizer()
    pats = encode_tag_patterns(tok, TAGS_EXPECTED)
    completion = "<|im_start|>foo<think>bar</think>baz<|im_end|>"
    ids = torch.tensor([tok.encode(completion)], dtype=torch.long)
    mask = build_tag_mask(ids, pats)
    inv = {i: c for c, i in tok.vocab.items()}
    masked_chars = "".join(inv[ids[0, i].item()] for i in range(ids.shape[1]) if mask[0, i])
    for tag in TAGS_EXPECTED:
        assert tag in masked_chars, (
            f"tag {tag!r} not fully covered by mask; got masked-chars={masked_chars!r}")


def test_train_script_passes_qwen3_native_tag_strings_to_encode_tag_patterns():
    """Static AST check: train.py for exp_055 must call encode_tag_patterns
    with EXACTLY the 4 Qwen3-native tags — no custom format tag references."""
    import ast, pathlib

    src = pathlib.Path(__file__).resolve().parent.parent / "train.py"
    tree = ast.parse(src.read_text())
    found = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "encode_tag_patterns"):
            # the second arg is the list of tag strings/names
            tags_arg = node.args[1] if len(node.args) >= 2 else None
            assert isinstance(tags_arg, ast.List), (
                f"encode_tag_patterns second arg must be a literal list, "
                f"got {ast.dump(tags_arg)}")
            literals = []
            for el in tags_arg.elts:
                if isinstance(el, ast.Constant) and isinstance(el.value, str):
                    literals.append(el.value)
                elif isinstance(el, ast.Name):
                    literals.append(el.id)
            # exp_055 must use ONLY Qwen3 native tags. assert presence...
            for tag in ("<think>", "</think>", "<|im_start|>", "<|im_end|>"):
                assert tag in literals, f"missing native tag {tag!r} in mask"
            # ...and assert ABSENCE of the old custom-tag identifiers
            for old in ("REASONING_START", "REASONING_END",
                        "SOLUTION_START", "SOLUTION_END"):
                assert old not in literals, (
                    f"old custom tag {old} should not be in exp_055 mask; "
                    f"got literals={literals}")
            found = True
            break
    assert found, "no encode_tag_patterns(...) call found in train.py"


def test_train_script_does_not_reference_custom_format_tag_identifiers():
    """Static AST check: train.py for exp_055 must NOT define REASONING_START
    et al as module-level constants (would indicate we still carry exp_053
    custom-format machinery)."""
    import ast, pathlib
    src = pathlib.Path(__file__).resolve().parent.parent / "train.py"
    tree = ast.parse(src.read_text())
    module_level_names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    module_level_names.add(tgt.id)
    for forbidden in ("REASONING_START", "REASONING_END",
                      "SOLUTION_START", "SOLUTION_END"):
        assert forbidden not in module_level_names, (
            f"{forbidden} still defined at module level in exp_055 train.py")


if __name__ == "__main__":
    test_build_tag_mask_marks_only_tag_positions(); print("  ✓ build_tag_mask marks tag positions")
    test_apply_tag_mask_replaces_token_adv_with_seq_adv_on_tag_positions(); print("  ✓ apply_tag_mask replaces on tag positions")
    test_empty_patterns_yields_zero_mask(); print("  ✓ empty patterns -> zero mask")
    test_pattern_longer_than_completion_is_no_op(); print("  ✓ pattern longer than completion -> no-op")
    test_encode_tag_patterns_produces_unique_nonempty_patterns_for_all_four_native_tags(); print("  ✓ encode_tag_patterns covers 4 Qwen3-native tags")
    test_build_tag_mask_detects_all_four_native_tags_in_a_mixed_completion(); print("  ✓ build_tag_mask catches all 4 native tags")
    test_train_script_passes_qwen3_native_tag_strings_to_encode_tag_patterns(); print("  ✓ train.py wires the 4 native tags only")
    test_train_script_does_not_reference_custom_format_tag_identifiers(); print("  ✓ train.py drops the exp_053 custom-format constants")
    test_encode_tag_patterns_produces_unique_nonempty_patterns_for_all_six_tags(); print("  ✓ encode_tag_patterns covers all 6 special tags")
    test_build_tag_mask_detects_all_six_tags_in_a_mixed_completion(); print("  ✓ build_tag_mask catches all 6 tags in a mixed completion")
    test_train_script_passes_six_special_tag_strings_to_encode_tag_patterns(); print("  ✓ train.py wires the 6 special tags into encode_tag_patterns")
    print("all tag-mask tests passed")
