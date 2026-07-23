"""Unit tests for SAGE scoring primitives (no server or GPU required)."""
from sage.solver import (
    get_contrastive_score,
    find_opening_tag,
    find_logprobs,
    form_best_and_worst_groups_relaxed,
    get_verified_group,
)


def test_contrastive_score_sign():
    lp = {"yes": -0.1, "no": -2.0}
    # positive when the "yes" label carries more probability mass than "no"
    assert get_contrastive_score(lp, ["yes"], ["no"]) > 0
    assert get_contrastive_score(lp, ["no"], ["yes"]) < 0


def test_contrastive_score_missing_labels():
    # missing labels fall back to the minimum observed logprob (never crashes)
    assert get_contrastive_score({"maybe": -1.0}, ["yes"], ["no"]) == 0.0
    assert get_contrastive_score(None, ["yes"], ["no"]) < -1e9


def test_find_tag_span_and_logprobs():
    tokens = ["<verdict", ">", "correct", "</verdict", ">"]
    (o0, o1), (c0, c1) = find_opening_tag(tokens, "<verdict>", "</verdict>")
    assert (o0, o1, c0, c1) == (0, 1, 3, 4)
    lps = [{}, {}, {"correct": -0.2, "incorrect": -1.5}, {}, {}]
    matched = find_logprobs(lps, tokens, o1 + 1, c0)
    assert matched is not None and matched["correct"] > matched["incorrect"]


def test_group_formation_orders_by_score():
    answers = [{"answer": f"a{i}", "final_llm_judge_score": s} for i, s in enumerate([0.1, 0.9, 0.5, -0.3])]
    vg = get_verified_group(answers, configurations={})
    groups = form_best_and_worst_groups_relaxed(vg, configurations={}, k_best=2, k_worst=1, m_min=1)
    assert groups["best"][0]["final_llm_judge_score"] == 0.9
    assert groups["worst"][0]["final_llm_judge_score"] == -0.3


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
