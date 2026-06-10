"""
test_em_score.py — exact-match scoring matches Search-R1 official semantics.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.em_score import extract_answer, em_match, reward_em, reward_subem


def test_extract_answer_returns_last():
    text = "blah <answer> first </answer> blah <answer> SECOND </answer>"
    assert extract_answer(text) == "SECOND"


def test_extract_answer_none_when_no_tag():
    assert extract_answer("no answer tag here") is None


def test_em_match_normalization():
    # case is lowered, whitespace collapsed
    assert em_match("  The   Quick   brown   fox  ", "the quick brown fox")
    assert em_match("Paris", "PARIS")
    # articles dropped on both sides
    assert em_match("A book", "the book")
    # punctuation is stripped without replacement (SQuAD-style — matches original Search-R1)
    assert em_match("brown,fox", "brownfox")
    # punctuation differences still cause mismatch when they bridge words
    assert not em_match("brown,fox", "brown fox")


def test_em_match_list_of_golds():
    assert em_match("Paris", ["London", "PARIS", "Berlin"])
    assert not em_match("Madrid", ["London", "PARIS"])


def test_reward_em_paths():
    # no answer tag -> format_score (default 0)
    assert reward_em("no tag", "x") == 0.0
    assert reward_em("no tag", "x", format_score=-0.5) == -0.5
    # correct EM
    assert reward_em("<answer>Paris</answer>", "paris") == 1.0
    # wrong answer extracted
    assert reward_em("<answer>Madrid</answer>", "paris") == 0.0
    # custom em_score / wrong_score
    assert reward_em("<answer>x</answer>", ["x", "y"], em_score=3.0) == 3.0


def test_reward_subem_substring_semantics():
    # substring of prediction counts
    assert reward_subem("<answer>The Eiffel Tower is in Paris.</answer>", "paris") == 1.0
    # full EM required for em path
    assert reward_em("<answer>The Eiffel Tower is in Paris.</answer>", "paris") == 0.0


if __name__ == "__main__":
    for name in dir():
        if name.startswith("test_"):
            globals()[name]()
            print(f"  ok  {name}")
    print("all em_score tests passed")
