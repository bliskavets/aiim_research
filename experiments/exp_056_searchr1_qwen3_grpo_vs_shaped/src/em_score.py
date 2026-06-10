"""
em_score.py — Exact-Match scoring for Search-R1 style QA answers.

Ported from the official Search-R1 verl/utils/reward_score/qa_em.py with the
same normalization rules used by the original benchmarks:

  1. lowercase
  2. strip punctuation
  3. drop articles {a, an, the}
  4. collapse whitespace

The reward_em function checks normalized exact match between the model's
\\boxed{<answer>...</answer>} extraction and the gold answer.
"""
from __future__ import annotations

import re
import string
from typing import Iterable, List, Optional, Union

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, drop articles, collapse whitespace."""
    if text is None:
        return ""
    text = text.lower()
    # strip punctuation
    text = "".join(c for c in text if c not in string.punctuation)
    # drop articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # collapse whitespace
    text = " ".join(text.split())
    return text


def extract_answer(completion: str) -> Optional[str]:
    """Return the content of the LAST <answer>...</answer> tag, or None.

    Matches the Search-R1 convention of taking the last answer if there are
    multiple (the model is supposed to emit at most one but may produce more).
    """
    matches = _ANSWER_RE.findall(completion)
    if not matches:
        return None
    return matches[-1].strip()


def em_match(pred: Optional[str], gold: Union[str, Iterable[str]]) -> bool:
    """Normalized exact match. `gold` can be a single string or list of acceptable
    answers (any one matching is enough)."""
    if pred is None:
        return False
    pred_n = _normalize(pred)
    if isinstance(gold, str):
        return pred_n == _normalize(gold)
    return any(pred_n == _normalize(g) for g in gold)


def subem_match(pred: Optional[str], gold: Union[str, Iterable[str]]) -> bool:
    """Normalized substring match — gold is a substring of prediction."""
    if pred is None:
        return False
    pred_n = _normalize(pred)
    if isinstance(gold, str):
        return _normalize(gold) in pred_n
    return any(_normalize(g) in pred_n for g in gold)


def reward_em(completion: str, gold: Union[str, Iterable[str]],
              format_score: float = 0.0, em_score: float = 1.0,
              wrong_score: float = 0.0) -> float:
    """Reward used by Search-R1: extract <answer>, normalize, return:
        em_score        — if exact match
        wrong_score     — if extracted but no match
        format_score    — if no <answer> tag at all
    Default values mirror the official paper.
    """
    pred = extract_answer(completion)
    if pred is None:
        return format_score
    return em_score if em_match(pred, gold) else wrong_score


def reward_subem(completion: str, gold: Union[str, Iterable[str]],
                 format_score: float = 0.0, sub_score: float = 1.0,
                 wrong_score: float = 0.0) -> float:
    """Looser variant: substring match instead of full EM."""
    pred = extract_answer(completion)
    if pred is None:
        return format_score
    return sub_score if subem_match(pred, gold) else wrong_score
