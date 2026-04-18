"""
Unit tests for exp_017 — no GPU, no model loading, no training.
Tests cover:
  - Dataset integer filtering logic
  - Reward functions (with mock completions)
  - Format regex patterns
  - Answer normalization
  - Numeric extraction
"""

import re
import sys
import os
import pytest

# ── constants duplicated from train.py (no import to avoid side-effects) ─────
REASONING_START = "<start_working_out>"
REASONING_END   = "<end_working_out>"
SOLUTION_START  = "<SOLUTION>"
SOLUTION_END    = "</SOLUTION>"

SYSTEM_PROMPT = (
    f"You are given a problem.\n"
    f"Think about the problem and provide your working out.\n"
    f"Place it between {REASONING_START} and {REASONING_END}.\n"
    f"Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"
)

match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{REASONING_START}.+?{REASONING_END}.*?"
    rf"{SOLUTION_START}(.+?){SOLUTION_END}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

match_numbers = re.compile(
    SOLUTION_START + r".*?([-\d\.,]+)",
    flags=re.MULTILINE | re.DOTALL,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (mirror of train.py functions, no imports needed)
# ─────────────────────────────────────────────────────────────────────────────

def is_integer_answer(example: dict) -> bool:
    raw = str(example.get("answer", "")).strip().replace(",", "")
    try:
        return float(raw) == int(float(raw))
    except (ValueError, OverflowError):
        return False


def normalize_integer(raw: str) -> str:
    return str(int(float(raw.strip().replace(",", ""))))


def reward_format_exact(completions, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        scores.append(3.0 if match_format.search(response) is not None else 0.0)
    return scores


def reward_format_approximate(completions, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 0.0
        score += 0.5 if response.count(REASONING_START) == 1 else -1.0
        score += 0.5 if response.count(REASONING_END)   == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_START)  == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_END)    == 1 else -1.0
        scores.append(score)
    return scores


def reward_answer_exact(prompts, completions, answer, **kwargs):
    responses = [c[0]["content"] for c in completions]
    extracted = [
        m.group(1) if (m := match_format.search(r)) is not None else None
        for r in responses
    ]
    scores = []
    for guess, true_answer in zip(extracted, answer):
        if guess is None:
            scores.append(0.0)
            continue
        if guess == true_answer:
            scores.append(3.0)
        elif guess.strip() == true_answer.strip():
            scores.append(1.5)
        else:
            try:
                ratio = float(guess.replace(",", "")) / float(true_answer)
                if   0.9 <= ratio <= 1.1: scores.append(1.0)
                elif 0.8 <= ratio <= 1.2: scores.append(0.5)
                else:                     scores.append(-1.5)
            except (ValueError, ZeroDivisionError):
                scores.append(-1.5)
    return scores


def reward_answer_numeric(prompts, completions, answer, **kwargs):
    responses = [c[0]["content"] for c in completions]
    extracted = [
        m.group(1) if (m := match_numbers.search(r)) is not None else None
        for r in responses
    ]
    scores = []
    for guess, true_answer in zip(extracted, answer):
        if guess is None:
            scores.append(0.0)
            continue
        try:
            guess_val = float(guess.strip().replace(",", ""))
            true_val  = float(true_answer.strip())
            scores.append(1.5 if guess_val == true_val else -0.5)
        except (ValueError, AttributeError):
            scores.append(0.0)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_completion(text: str):
    """Wrap text in the completion format expected by reward functions."""
    return [{"role": "assistant", "content": text}]


def perfect_response(answer: str) -> str:
    return (
        f"{REASONING_START}Step 1: add numbers. Step 2: profit.{REASONING_END}"
        f"{SOLUTION_START}{answer}{SOLUTION_END}"
    )


def no_format_response(answer: str) -> str:
    return f"The answer is {answer}."


def wrong_format_response() -> str:
    """Has reasoning but no solution tags."""
    return f"{REASONING_START}I think the answer is 42.{REASONING_END}"


def duplicate_tags_response(answer: str) -> str:
    return (
        f"{REASONING_START}step1{REASONING_END}"
        f"{REASONING_START}step2{REASONING_END}"
        f"{SOLUTION_START}{answer}{SOLUTION_END}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: integer filter
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegerFilter:
    def test_clean_integer(self):
        assert is_integer_answer({"answer": "42"}) is True

    def test_float_integer(self):
        assert is_integer_answer({"answer": "42.0"}) is True

    def test_negative_integer(self):
        assert is_integer_answer({"answer": "-7"}) is True

    def test_float_with_fraction(self):
        assert is_integer_answer({"answer": "3.14"}) is False

    def test_non_numeric_string(self):
        assert is_integer_answer({"answer": "x+1"}) is False

    def test_empty_answer(self):
        assert is_integer_answer({"answer": ""}) is False

    def test_comma_formatted_integer(self):
        assert is_integer_answer({"answer": "1,024"}) is True

    def test_zero(self):
        assert is_integer_answer({"answer": "0"}) is True

    def test_large_integer(self):
        assert is_integer_answer({"answer": "1000000"}) is True


class TestNormalizeInteger:
    def test_plain(self):
        assert normalize_integer("42") == "42"

    def test_float_form(self):
        assert normalize_integer("42.0") == "42"

    def test_negative(self):
        assert normalize_integer("-7") == "-7"

    def test_comma(self):
        assert normalize_integer("1,024") == "1024"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: format regex
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatRegex:
    def test_perfect_response_matches(self):
        assert match_format.search(perfect_response("42")) is not None

    def test_no_format_no_match(self):
        assert match_format.search(no_format_response("42")) is None

    def test_partial_format_no_match(self):
        assert match_format.search(wrong_format_response()) is None

    def test_extracts_answer_correctly(self):
        m = match_format.search(perfect_response("123"))
        assert m is not None
        assert m.group(1).strip() == "123"

    def test_multiline_reasoning(self):
        response = (
            f"{REASONING_START}\nLine1\nLine2\nLine3\n{REASONING_END}"
            f"\n{SOLUTION_START}99{SOLUTION_END}"
        )
        m = match_format.search(response)
        assert m is not None

    def test_whitespace_around_response(self):
        response = "\n\n" + perfect_response("7") + "\n\n"
        assert match_format.search(response) is not None


# ─────────────────────────────────────────────────────────────────────────────
# Tests: reward_format_exact
# ─────────────────────────────────────────────────────────────────────────────

class TestRewardFormatExact:
    def test_perfect_gets_3(self):
        completions = [make_completion(perfect_response("42"))]
        assert reward_format_exact(completions) == [3.0]

    def test_no_format_gets_0(self):
        completions = [make_completion(no_format_response("42"))]
        assert reward_format_exact(completions) == [0.0]

    def test_batch(self):
        completions = [
            make_completion(perfect_response("1")),
            make_completion(no_format_response("2")),
            make_completion(perfect_response("3")),
        ]
        scores = reward_format_exact(completions)
        assert scores == [3.0, 0.0, 3.0]


# ─────────────────────────────────────────────────────────────────────────────
# Tests: reward_format_approximate
# ─────────────────────────────────────────────────────────────────────────────

class TestRewardFormatApproximate:
    def test_perfect_gets_2(self):
        completions = [make_completion(perfect_response("42"))]
        scores = reward_format_approximate(completions)
        assert scores == [2.0]

    def test_empty_response_gets_minus4(self):
        completions = [make_completion("")]
        scores = reward_format_approximate(completions)
        assert scores == [-4.0]

    def test_duplicate_reasoning_start_penalized(self):
        completions = [make_completion(duplicate_tags_response("42"))]
        scores = reward_format_approximate(completions)
        # REASONING_START appears twice → -1.0 for that tag
        assert scores[0] < 2.0

    def test_single_tag_present(self):
        response = f"{REASONING_START}some reasoning{REASONING_END}"
        completions = [make_completion(response)]
        scores = reward_format_approximate(completions)
        # 2 tags correct (+0.5 each), 2 missing (-1.0 each)
        assert scores[0] == pytest.approx(1.0 - 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: reward_answer_exact
# ─────────────────────────────────────────────────────────────────────────────

class TestRewardAnswerExact:
    def _run(self, response_text: str, true_answer: str):
        completions = [make_completion(response_text)]
        prompts = [[{"role": "user", "content": "Q"}]]
        return reward_answer_exact(prompts, completions, [true_answer])[0]

    def test_exact_match_gets_3(self):
        assert self._run(perfect_response("42"), "42") == 3.0

    def test_strip_match_gets_1p5(self):
        response = (
            f"{REASONING_START}work{REASONING_END}"
            f"{SOLUTION_START} 42 {SOLUTION_END}"
        )
        assert self._run(response, "42") == pytest.approx(1.5)

    def test_within_10pct_gets_1(self):
        # 100 vs 105 → ratio = 1.05, within 10%
        assert self._run(perfect_response("105"), "100") == pytest.approx(1.0)

    def test_within_20pct_gets_0p5(self):
        # 100 vs 115 → ratio = 1.15, within 20% but not 10%
        assert self._run(perfect_response("115"), "100") == pytest.approx(0.5)

    def test_wrong_answer_gets_minus1p5(self):
        assert self._run(perfect_response("999"), "42") == pytest.approx(-1.5)

    def test_no_format_gets_0(self):
        assert self._run(no_format_response("42"), "42") == 0.0

    def test_zero_denominator_gets_minus1p5(self):
        # true_answer = 0 → ZeroDivisionError → -1.5
        assert self._run(perfect_response("5"), "0") == pytest.approx(-1.5)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: reward_answer_numeric
# ─────────────────────────────────────────────────────────────────────────────

class TestRewardAnswerNumeric:
    def _run(self, response_text: str, true_answer: str):
        completions = [make_completion(response_text)]
        prompts = [[{"role": "user", "content": "Q"}]]
        return reward_answer_numeric(prompts, completions, [true_answer])[0]

    def test_correct_integer_gets_1p5(self):
        assert self._run(perfect_response("42"), "42") == pytest.approx(1.5)

    def test_wrong_integer_gets_minus0p5(self):
        assert self._run(perfect_response("99"), "42") == pytest.approx(-0.5)

    def test_no_format_gets_0(self):
        assert self._run(no_format_response("42"), "42") == 0.0

    def test_comma_formatted_number(self):
        response = (
            f"{REASONING_START}work{REASONING_END}"
            f"{SOLUTION_START}1,024{SOLUTION_END}"
        )
        assert self._run(response, "1024") == pytest.approx(1.5)

    def test_negative_answer(self):
        assert self._run(perfect_response("-7"), "-7") == pytest.approx(1.5)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: full reward pipeline consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestRewardPipelineConsistency:
    def test_perfect_response_maximum_total(self):
        """Perfect response should get max total reward across all functions."""
        completions = [make_completion(perfect_response("42"))]
        prompts = [[{"role": "user", "content": "Q"}]]
        answer = ["42"]

        r1 = reward_format_exact(completions)[0]
        r2 = reward_format_approximate(completions)[0]
        r3 = reward_answer_exact(prompts, completions, answer)[0]
        r4 = reward_answer_numeric(prompts, completions, answer)[0]

        assert r1 == 3.0
        assert r2 == 2.0
        assert r3 == 3.0
        assert r4 == 1.5
        assert r1 + r2 + r3 + r4 == pytest.approx(9.5)

    def test_empty_completion_minimum_total(self):
        """Empty response should get near-minimum reward."""
        completions = [make_completion("")]
        prompts = [[{"role": "user", "content": "Q"}]]
        answer = ["42"]

        r1 = reward_format_exact(completions)[0]
        r2 = reward_format_approximate(completions)[0]
        r3 = reward_answer_exact(prompts, completions, answer)[0]
        r4 = reward_answer_numeric(prompts, completions, answer)[0]

        assert r1 == 0.0
        assert r2 == -4.0
        assert r3 == 0.0
        assert r4 == 0.0

    def test_batch_returns_correct_length(self):
        """Reward functions return exactly one score per completion."""
        batch_size = 16  # matches num_generations
        completions = [make_completion(perfect_response(str(i))) for i in range(batch_size)]
        prompts = [[{"role": "user", "content": "Q"}]] * batch_size
        answer = [str(i) for i in range(batch_size)]

        assert len(reward_format_exact(completions)) == batch_size
        assert len(reward_format_approximate(completions)) == batch_size
        assert len(reward_answer_exact(prompts, completions, answer)) == batch_size
        assert len(reward_answer_numeric(prompts, completions, answer)) == batch_size
