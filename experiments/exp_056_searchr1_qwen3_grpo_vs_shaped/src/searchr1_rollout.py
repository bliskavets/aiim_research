"""
searchr1_rollout.py — multi-turn rollout for Search-R1 style training.

The model generates an interleaved sequence:

  <think>...</think>
  <search>query</search>
  -- retrieval is injected --
  <information>doc1 doc2 doc3</information>
  <think>...</think>
  <search>another query</search>
  -- retrieval is injected --
  <information>...</information>
  ...
  <answer>final answer</answer>

We chain vLLM `generate` calls, stopping at `</search>` and `</answer>`:

  - On `</search>`: extract the query, hit the retriever, format the docs
    inside an `<information>...</information>` block, append, generate again.
  - On `</answer>`: rollout is done.
  - On hitting `max_turns` or running out of completion budget: stop.

For each rollout we return:
    completion_text  — full rollout text (no prompt)
    token_ids        — full token-id stream for the completion
    model_mask       — 1 for tokens the model generated, 0 for tokens
                       we injected from retrieval. The trainer uses this
                       as `completion_mask` so loss/shaping only acts on
                       model-generated tokens.

This module is decoupled from any specific trainer / engine. Callers pass:
    generate_fn(prompts: list[str], sampling_params: dict) -> list[GenerationResult]
    tokenizer
    retriever
    config (max_turns, topk, max_total_tokens, etc.)

so the same rollout works under TRL+unsloth, raw vLLM, or a stub for tests.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .retriever import Retriever, format_information_block

_SEARCH_BLOCK_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_STOP_SEARCH = "</search>"
_STOP_ANSWER = "</answer>"


@dataclass
class GenerationResult:
    """Minimal abstraction over a single sampled completion.

    The `text` field is the generated text (without the prompt) and
    `token_ids` is the matching list of token ids. Any backend that returns
    these two fields will work.
    """
    text: str
    token_ids: List[int]
    finish_reason: Optional[str] = None   # 'stop', 'length', 'eos', ...
    stopped_at: Optional[str] = None      # which stop string was hit


@dataclass
class RolloutConfig:
    max_turns: int = 4
    topk: int = 3
    max_completion_tokens: int = 4096   # total budget across all turns
    per_turn_max_tokens: int = 1024     # cap one generate() call
    temperature: float = 0.7
    top_p: float = 0.95
    seed: int = 3407


@dataclass
class RolloutTrace:
    """The trainer-facing artefact of one rollout."""
    completion_text: str
    token_ids: List[int]
    model_mask: List[int]            # 1 = model, 0 = retrieval injection
    n_turns: int                     # number of model generation passes
    n_searches: int                  # how many <search>...</search> blocks fired
    finish_reason: str               # 'answer' | 'truncated' | 'max_turns'
    queries: List[str] = field(default_factory=list)


GenerateFn = Callable[[List[str], dict], List[GenerationResult]]


def extract_last_search_query(text: str) -> Optional[str]:
    """Return the LAST <search>...</search> query, stripped, or None."""
    matches = _SEARCH_BLOCK_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def run_rollouts(
    prompts: Sequence[str],
    generate_fn: GenerateFn,
    encode_fn: Callable[[str], List[int]],
    retriever: Retriever,
    cfg: RolloutConfig,
) -> List[RolloutTrace]:
    """Run one Search-R1 multi-turn rollout per prompt.

    Returns a `RolloutTrace` per prompt, in the same order.
    `encode_fn(text) -> token_ids` is needed to tokenize injected
    <information> blocks so model_mask aligns with token_ids.
    """
    traces: List[RolloutTrace] = []
    for prompt in prompts:
        traces.append(_run_one(prompt, generate_fn, encode_fn, retriever, cfg))
    return traces


def _run_one(prompt: str,
             generate_fn: GenerateFn,
             encode_fn: Callable[[str], List[int]],
             retriever: Retriever,
             cfg: RolloutConfig) -> RolloutTrace:
    completion_text = ""
    token_ids: List[int] = []
    model_mask: List[int] = []
    queries: List[str] = []
    budget_left = cfg.max_completion_tokens
    finish_reason = "max_turns"

    for turn in range(cfg.max_turns):
        if budget_left <= 0:
            finish_reason = "truncated"
            break
        sp = {
            "max_tokens": min(cfg.per_turn_max_tokens, budget_left),
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "stop": [_STOP_SEARCH, _STOP_ANSWER],
            "include_stop_str_in_output": True,
            "seed": cfg.seed,
        }
        outs = generate_fn([prompt + completion_text], sp)
        out = outs[0]
        gen_text = out.text
        gen_ids = list(out.token_ids)

        completion_text += gen_text
        token_ids.extend(gen_ids)
        model_mask.extend([1] * len(gen_ids))
        budget_left -= len(gen_ids)

        # decide what happens next
        if _STOP_ANSWER in gen_text:
            finish_reason = "answer"
            break
        if _STOP_SEARCH in gen_text:
            query = extract_last_search_query(gen_text) or ""
            queries.append(query)
            docs = retriever.retrieve(query, topk=cfg.topk)
            info_block = format_information_block(docs)
            info_ids = encode_fn(info_block)
            # bound the injected length so we don't blow the budget on a
            # single retrieval result
            if len(info_ids) > budget_left:
                info_ids = info_ids[:max(0, budget_left)]
                info_block = ""   # we still append text, but trimmed: see below
            completion_text += info_block
            token_ids.extend(info_ids)
            model_mask.extend([0] * len(info_ids))
            budget_left -= len(info_ids)
            continue

        # no stop string but we got a completion — generation ran out of
        # max_tokens within a turn. Stop the rollout, treat as truncated.
        finish_reason = "truncated"
        break

    return RolloutTrace(
        completion_text=completion_text,
        token_ids=token_ids,
        model_mask=model_mask,
        n_turns=turn + 1 if 'turn' in locals() else 0,
        n_searches=len(queries),
        finish_reason=finish_reason,
        queries=queries,
    )
