"""Shared helper utilities for evaluation scripts."""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

import datetime as _dt
import json
import os
import re
from pathlib import Path

import diskcache
from tenacity import retry

from openai import OpenAI
from pydantic import BaseModel, Field


# ===== Math equality judge =====


class ExpressionsAreEqualSchema(BaseModel):
    answers_are_equivalent: bool = Field(
        ..., description="answers are equivalent"
    )


_MODEL_ID = os.getenv("MATH_CRITIC_MODEL", "o3")
_openai_client = OpenAI()

_CACHE_DIR = Path(os.getenv("DISKCACHE_DIR", "experiments/diskcache"))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_disk_cache = diskcache.Cache(str(_CACHE_DIR))


_CRITIC_PROMPT_PREFIX = """
You are a Fields Medalist mathematician. You are given a question and two answers to it.
You need to find out whether the two answers are equivalent or not.

Remember: you must ignore any non-essential differences in answers.
Also note: if the two answers have different complexity levels then they're different.

Examples:
`pi / 3` is equivalent to `(pi * 1) / 3.0`
`x ^ 2 + y` is not equivalent to `x * 2 + y`
`(3,pi/2)` is equivalent to `\\left( 3, \\frac{\\pi}{2} \\right)`
"""

_CRITIC_PROMPT_SUFFIX = """
You must return json containing single key:
    - answers_are_equivalent: bool # True or False

Question:
{task}

Answer 1:
```
{answer1}
```

Answer 2:
```
{answer2}
```
"""


@retry
def _critic_get_response(prompt: str) -> dict:
    try:
        return _disk_cache[prompt]
    except KeyError:
        pass

    response = _openai_client.responses.parse(
        model=_MODEL_ID,
        input=[{"role": "user", "content": prompt}],
        text_format=ExpressionsAreEqualSchema,
    )
    result: dict = response.output_parsed.model_dump()
    _disk_cache[prompt] = result
    return result


def parse_answer(answer: str) -> str:
    """Extract boxed answer from a LaTeX math string."""
    try:
        last_boxed_idx = answer.rfind('boxed{')
        if last_boxed_idx == -1:
            return None
        suffix = answer[last_boxed_idx + len('boxed'):]
        idx = suffix.index('{')
        stack = 1
        pos = idx + 1
        while stack != 0 and pos < len(suffix):
            if suffix[pos] == '{':
                stack += 1
            elif suffix[pos] == '}':
                stack -= 1
            pos += 1
        answer = suffix[idx + 1 : pos - 1]
    except Exception:
        print('Answer not parsed:', answer)
    return answer


@lru_cache(maxsize=100000)
def equations_are_equal_new(task: str, gt_answer: str, answer: str) -> int:
    """Return 1 if the model answer equals the ground truth, else 0."""
    if 'boxed{' not in answer:
        return 0
    try:
        last_boxed_idx = answer.rfind('boxed{')
        if last_boxed_idx == -1:
            print(f'gt_answer: {gt_answer}, parsed answer: --not found--, response: False')
            return 0
        suffix = answer[last_boxed_idx + len('boxed'):]
        idx = suffix.index('{')
        stack = 1
        pos = idx + 1
        while stack != 0 and pos < len(suffix):
            if suffix[pos] == '{':
                stack += 1
            elif suffix[pos] == '}':
                stack -= 1
            pos += 1
        answer = suffix[idx + 1 : pos - 1]
    except Exception:
        print('Answer not parsed:', answer)

    prompt = _CRITIC_PROMPT_PREFIX + _CRITIC_PROMPT_SUFFIX.format(
        task=task,
        answer1=gt_answer,
        answer2=answer,
    )
    response = _critic_get_response(prompt)
    print(f'gt_answer: {gt_answer}, parsed answer: {answer}, response: {response["answers_are_equivalent"]}')
    return int(response.get('answers_are_equivalent') is True)


# ===== Generic helpers =====


def ensure_output_dir(output_path: Optional[str]) -> Path:
    """Create and return the output directory.

    If output_path is None, create experiments/outputs/<timestamp>.
    """
    if output_path:
        out_dir = Path(output_path)
    else:
        timestamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = Path("experiments") / "outputs" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_prompt(problem: str, prompt_template: Optional[Callable[[str], str] | str]) -> str:
    """Apply prompt_template to problem.

    - If prompt_template is None: append suffix "Return the final answer within \\boxed{...}".
    - If it's a callable: return prompt_template(problem).
    - If it's a string and contains "{query}": return prompt_template.format(query=problem).
      Otherwise, concatenate problem + "\\n" + prompt_template.
    """
    default_suffix = "Return the final answer within \\boxed{...}"
    if prompt_template is None:
        return f"{problem}\n{default_suffix}"
    if callable(prompt_template):
        return prompt_template(problem)
    if isinstance(prompt_template, str):
        if "{query}" in prompt_template:
            return prompt_template.format(query=problem)
        return f"{problem}\n{prompt_template}"
    raise TypeError("prompt_template must be None, callable, or str")


def append_jsonl(record: dict | list[dict], path: Path) -> None:
    """Append a record or list of records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fout:
        if isinstance(record, list):
            for rec in record:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
