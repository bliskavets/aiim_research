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
    seed: Optional[int] = None   # MUST be None for GRPO: a fixed seed makes
                                 # vLLM produce IDENTICAL completions for the
                                 # num_generations copies of a prompt, killing
                                 # within-group variance → zero advantage →
                                 # zero gradient (observed in the first 100
                                 # steps: frac_reward_zero_std=1.0 every step,
                                 # completions min==max==mean length).


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
    """Sequential reference implementation — one rollout at a time.

    Kept for the scripted-LLM unit tests (which feed GenerationResults in a
    fixed per-call order). For training use `run_rollouts_batched`, which is
    orders of magnitude faster because it generates all still-active rollouts
    of the group in a single vLLM call per turn and batch-retrieves all
    queries in one server round-trip.
    """
    traces: List[RolloutTrace] = []
    for prompt in prompts:
        traces.append(_run_one(prompt, generate_fn, encode_fn, retriever, cfg))
    return traces


def run_rollouts_batched(
    prompts: Sequence[str],
    generate_fn: GenerateFn,
    encode_fn: Callable[[str], List[int]],
    retriever: Retriever,
    cfg: RolloutConfig,
) -> List[RolloutTrace]:
    """Batched multi-turn rollout.

    Per turn:
      1. Collect every rollout that is still active (not finished, has budget).
      2. ONE generate_fn call for all their current (prompt+completion) strings
         — vLLM parallelises these internally.
      3. Split the active rollouts into those that emitted </search> (need
         retrieval) vs </answer>/truncated (done).
      4. ONE retriever.retrieve_batch() call for all search queries.
      5. Append the per-rollout <information> blocks, decrement budgets.
    Repeat up to cfg.max_turns. Returns one RolloutTrace per prompt, in order.
    """
    n = len(prompts)
    state = [
        {
            "prompt": p,
            "completion": "",
            "token_ids": [],
            "model_mask": [],
            "queries": [],
            "budget": cfg.max_completion_tokens,
            "done": False,
            "finish_reason": "max_turns",
            "n_turns": 0,
        }
        for p in prompts
    ]

    for _turn in range(cfg.max_turns):
        active_idx = [i for i in range(n) if not state[i]["done"] and state[i]["budget"] > 0]
        # mark out-of-budget-but-not-done as truncated
        for i in range(n):
            if not state[i]["done"] and state[i]["budget"] <= 0:
                state[i]["done"] = True
                state[i]["finish_reason"] = "truncated"
        if not active_idx:
            break

        # 1 generate call for all active rollouts
        batch_prompts = [state[i]["prompt"] + state[i]["completion"] for i in active_idx]
        per_turn_max = min(cfg.per_turn_max_tokens,
                           max(1, min(state[i]["budget"] for i in active_idx)))
        sp = {
            "max_tokens": per_turn_max,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "stop": [_STOP_SEARCH, _STOP_ANSWER],
            "include_stop_str_in_output": True,
            "seed": cfg.seed,
        }
        outs = generate_fn(batch_prompts, sp)

        # 2 apply generations, decide who searches
        search_slots: List[int] = []
        search_queries: List[str] = []
        for slot, i in enumerate(active_idx):
            out = outs[slot]
            gen_text = out.text
            gen_ids = list(out.token_ids)
            st = state[i]
            st["completion"] += gen_text
            st["token_ids"].extend(gen_ids)
            st["model_mask"].extend([1] * len(gen_ids))
            st["budget"] -= len(gen_ids)
            st["n_turns"] += 1

            if _STOP_ANSWER in gen_text:
                st["done"] = True
                st["finish_reason"] = "answer"
            elif _STOP_SEARCH in gen_text:
                q = extract_last_search_query(gen_text) or ""
                st["queries"].append(q)
                search_slots.append(i)
                search_queries.append(q)
            else:
                # ran out of per-turn tokens without a stop string
                st["done"] = True
                st["finish_reason"] = "truncated"

        # 3 one batched retrieval for everyone who searched
        if search_queries:
            docs_batch = retriever.retrieve_batch(search_queries, topk=cfg.topk)
            for i, docs in zip(search_slots, docs_batch):
                st = state[i]
                info_block = format_information_block(docs)
                info_ids = encode_fn(info_block)
                if len(info_ids) > st["budget"]:
                    info_ids = info_ids[:max(0, st["budget"])]
                st["completion"] += info_block
                st["token_ids"].extend(info_ids)
                st["model_mask"].extend([0] * len(info_ids))
                st["budget"] -= len(info_ids)

    return [
        RolloutTrace(
            completion_text=st["completion"],
            token_ids=st["token_ids"],
            model_mask=st["model_mask"],
            n_turns=st["n_turns"],
            n_searches=len(st["queries"]),
            finish_reason=st["finish_reason"],
            queries=st["queries"],
        )
        for st in state
    ]


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
