"""
test_searchr1_rollout.py — multi-turn rollout against a scripted fake LLM.

We replay deterministic scripts to verify:
  - answer path: model emits <answer> immediately, rollout stops
  - search path: model emits <search>, retriever returns docs, second turn
    sees the <information> block and emits <answer>
  - max_turns cap: model keeps searching, rollout caps at max_turns
  - mask: model-generated tokens get mask=1, injected <information> tokens get mask=0
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retriever import StubRetriever
from src.searchr1_rollout import (
    RolloutConfig, GenerationResult, run_rollouts, extract_last_search_query,
)


# Char-level "tokenizer" for tests
def encode_fn(text: str):
    return list(text.encode("utf-8"))


def scripted_generator(scripts):
    """Each element of `scripts` is a list of GenerationResult, consumed in order."""
    state = {"idx": 0}

    def _gen(prompts, sp):
        out = []
        for _ in prompts:
            if state["idx"] >= len(scripts):
                # ran out — return EOS-ish
                out.append(GenerationResult(text="", token_ids=[], finish_reason="eos"))
            else:
                out.append(scripts[state["idx"]])
                state["idx"] += 1
        return out

    return _gen


def test_answer_path_one_turn():
    gen = scripted_generator([
        GenerationResult(
            text="<think>think</think><answer>Paris</answer>",
            token_ids=encode_fn("<think>think</think><answer>Paris</answer>"),
            finish_reason="stop", stopped_at="</answer>",
        )
    ])
    cfg = RolloutConfig(max_turns=4)
    traces = run_rollouts(["Q: Where is Eiffel? "], gen, encode_fn, StubRetriever(), cfg)
    t = traces[0]
    assert t.finish_reason == "answer"
    assert t.n_searches == 0
    assert "<answer>Paris</answer>" in t.completion_text
    # all model tokens, no injected
    assert all(m == 1 for m in t.model_mask)


def test_search_then_answer():
    s1 = "<think>need to look up</think><search>eiffel tower</search>"
    s2 = "<think>got it</think><answer>Paris</answer>"
    gen = scripted_generator([
        GenerationResult(text=s1, token_ids=encode_fn(s1),
                         finish_reason="stop", stopped_at="</search>"),
        GenerationResult(text=s2, token_ids=encode_fn(s2),
                         finish_reason="stop", stopped_at="</answer>"),
    ])
    cfg = RolloutConfig(max_turns=4)
    traces = run_rollouts(["Q: Where is Eiffel? "], gen, encode_fn, StubRetriever(), cfg)
    t = traces[0]
    assert t.finish_reason == "answer"
    assert t.n_searches == 1
    assert t.queries == ["eiffel tower"]
    # the <information> block is injected -> some mask=0 tokens
    assert any(m == 0 for m in t.model_mask)
    # but mask=1 tokens exist too (the model turns)
    assert any(m == 1 for m in t.model_mask)
    # injected mask=0 segment is contiguous and matches the info block
    # find first 0 and last 0
    zeros = [i for i, m in enumerate(t.model_mask) if m == 0]
    assert zeros == list(range(zeros[0], zeros[-1] + 1)), "mask=0 block should be contiguous"


def test_max_turns_cap():
    # model keeps searching, never answers
    search = "<think>still searching</think><search>more</search>"
    scripts = [
        GenerationResult(text=search, token_ids=encode_fn(search),
                         finish_reason="stop", stopped_at="</search>")
        for _ in range(10)
    ]
    gen = scripted_generator(scripts)
    cfg = RolloutConfig(max_turns=3)
    traces = run_rollouts(["Q: ..."], gen, encode_fn, StubRetriever(), cfg)
    t = traces[0]
    assert t.finish_reason == "max_turns"
    assert t.n_searches == 3


def test_extract_last_search_query():
    text = "blah <search>first q</search> blah <search>second q</search> blah"
    assert extract_last_search_query(text) == "second q"
    assert extract_last_search_query("no tag") is None


def test_token_id_count_matches_mask_length():
    s1 = "<think>x</think><search>q</search>"
    s2 = "<answer>A</answer>"
    gen = scripted_generator([
        GenerationResult(text=s1, token_ids=encode_fn(s1), finish_reason="stop"),
        GenerationResult(text=s2, token_ids=encode_fn(s2), finish_reason="stop"),
    ])
    cfg = RolloutConfig(max_turns=4)
    traces = run_rollouts(["Q?"], gen, encode_fn, StubRetriever(), cfg)
    t = traces[0]
    assert len(t.token_ids) == len(t.model_mask)


if __name__ == "__main__":
    for name in [n for n in dir() if n.startswith("test_")]:
        globals()[name]()
        print(f"  ok  {name}")
    print("all rollout tests passed")
