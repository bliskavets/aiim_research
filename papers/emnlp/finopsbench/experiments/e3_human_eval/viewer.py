"""Generic single-annotator viewer for the E3 human-evaluation sets.

Handles two record types via the per-record ``task`` field:
  * ``v1_answer``   -- is the trace's final answer correct for the query?
  * ``v2_validity`` -- is this v2 environment a valid, solvable example whose
                       gold answer correctly answers the question?

Labels are written back into the JSONL file in place. Automatic-scorer
verdicts (for v1) are hidden behind an expander so the label is not anchored.

Run:
    DATA=data/sample_v1_judge.jsonl streamlit run viewer.py \
        --server.port 8788 --server.address 0.0.0.0 --server.headless true
"""

import json
import os
import tempfile
from pathlib import Path

import streamlit as st

DATA_PATH = Path(os.environ.get("DATA", "data/sample_v1_judge.jsonl"))
if not DATA_PATH.is_absolute():
    DATA_PATH = Path(__file__).parent / DATA_PATH

st.set_page_config(page_title=f"E3 annotation — {DATA_PATH.name}", layout="wide")


def load():
    return [json.loads(l) for l in DATA_PATH.open() if l.strip()]


def save(cases):
    fd, tmp = tempfile.mkstemp(dir=DATA_PATH.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    os.replace(tmp, DATA_PATH)


if "cases" not in st.session_state:
    st.session_state.cases = load()
if "idx" not in st.session_state:
    st.session_state.idx = next(
        (i for i, c in enumerate(st.session_state.cases) if c.get("human_label") is None), 0)

cases = st.session_state.cases
n = len(cases)
labelled = sum(c.get("human_label") is not None for c in cases)
task = cases[0].get("task", "v1_answer")


def set_label(value):
    cases[st.session_state.idx]["human_label"] = value
    save(cases)
    st.session_state.idx = next(
        (i for i in range(st.session_state.idx + 1, n) if cases[i].get("human_label") is None),
        min(st.session_state.idx + 1, n - 1))


with st.sidebar:
    st.markdown(f"### {DATA_PATH.name}")
    st.markdown(f"**Progress: {labelled}/{n}**")
    st.progress(labelled / n)
    st.session_state.idx = st.number_input("Case #", 0, n - 1, st.session_state.idx, 1)
    if task == "v1_answer":
        st.caption("**Is the trace's final answer a correct answer to the query, "
                   "given the reference answer?** Judge content, not formatting.")
    else:
        st.caption("**Is this a valid, solvable example whose gold answer correctly "
                   "answers the question?** Check the reference plan computes the gold "
                   "from the tools and that the question is well-posed.")
    rem = [i for i, c in enumerate(cases) if c.get("human_label") is None]
    st.caption(f"Unlabelled: {len(rem)}" + (f" (next #{rem[0]})" if rem else " — done!"))

case = cases[st.session_state.idx]
st.markdown(f"#### Case {st.session_state.idx} / {n - 1}  ·  task = `{task}`")
if case.get("human_label") is not None:
    st.info(f"Current label: **{case['human_label']}**")

if task == "v1_answer":
    st.markdown("**Query**")
    st.markdown(f"> {case['query']}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Reference (expected) answer**")
        st.text_area("g", case["gold"], height=300, label_visibility="collapsed", disabled=True)
    with c2:
        st.markdown("**Trace final answer**")
        st.text_area("a", case["answer"], height=300, label_visibility="collapsed", disabled=True)
    b1, b2, b3, b4 = st.columns(4)
    b1.button("✅ Correct", on_click=set_label, args=(True,), use_container_width=True)
    b2.button("❌ Incorrect", on_click=set_label, args=(False,), use_container_width=True)
    b3.button("🤷 Unclear", on_click=set_label, args=("unclear",), use_container_width=True)
    b4.button("Skip →", on_click=lambda: st.session_state.update(idx=min(st.session_state.idx + 1, n - 1)),
              use_container_width=True)
    with st.expander("Automatic-scorer verdicts (open AFTER labelling)"):
        st.write({"numeric_match": case["numeric_match"], "judge_correct": case["judge_correct"]})
else:
    top1, top2 = st.columns(2)
    with top1:
        st.markdown("**Agentic question (v2, rephrased)**")
        st.markdown(f"> {case['question']}")
        st.markdown(f"**v2 gold answer:** `{case['gold']}`")
    with top2:
        if case.get("finqa_question"):
            flag = case.get("finqa_answer_matches_gold")
            badge = "✅ matches gold" if flag else "⚠️ differs from gold"
            st.markdown(f"**Original FinQA question** ({case.get('finqa_id','?')})")
            st.markdown(f"> {case['finqa_question']}")
            st.markdown(f"**FinQA answer:** `{case.get('finqa_answer','')}`  ·  {badge}")
        else:
            st.caption("Original FinQA item not resolved for this example.")

    b1, b2, b3, b4 = st.columns(4)
    b1.button("✅ Valid", on_click=set_label, args=(True,), use_container_width=True)
    b2.button("❌ Invalid", on_click=set_label, args=(False,), use_container_width=True)
    b3.button("🤷 Unclear", on_click=set_label, args=("unclear",), use_container_width=True)
    b4.button("Skip →", on_click=lambda: st.session_state.update(idx=min(st.session_state.idx + 1, n - 1)),
              use_container_width=True)
    st.caption("Valid = the reference plan computes the gold from the tools AND the "
               "agentic question is a faithful, well-posed transformation of the FinQA item.")

    if case.get("finqa_table_md"):
        st.markdown("**Original FinQA table**")
        st.markdown(case["finqa_table_md"])
    ft1, ft2 = st.columns(2)
    with ft1:
        with st.expander("FinQA pre-text (narrative before the table)"):
            st.text(case.get("finqa_pre_text", "") or "—")
    with ft2:
        with st.expander("FinQA post-text (narrative after the table)"):
            st.text(case.get("finqa_post_text", "") or "—")

    col_t, col_p = st.columns(2)
    with col_t:
        st.markdown(f"**Tools ({len(case.get('tool_names', []))}): full definitions**")
        st.code(case.get("tool_source", "\n".join(case.get("tool_names", []))), language="python")
    with col_p:
        st.markdown("**Reference plan (should compute the gold from the tools)**")
        st.code(case.get("reference_plan", ""), language="python")
