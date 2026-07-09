"""Annotation viewer for the E2 judge-vs-deterministic disagreement cases.

Blind annotation: the two scorers' verdicts are hidden behind an expander so
the human label is not anchored. Labels are written back into the JSONL file
in place (atomic rewrite) under the ``human_label`` key.

Run:
    streamlit run annotation_viewer.py --server.port 8787 --server.address 0.0.0.0 --server.headless true
"""

import json
import os
import tempfile
from pathlib import Path

import streamlit as st

DATA_PATH = Path(__file__).parent / "results" / "disagreements_for_human_annotation.jsonl"

st.set_page_config(page_title="FinOpsBench E2 annotation", layout="wide")


def load_cases() -> list[dict]:
    return [json.loads(line) for line in DATA_PATH.open() if line.strip()]


def save_cases(cases: list[dict]) -> None:
    fd, tmp = tempfile.mkstemp(dir=DATA_PATH.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    os.replace(tmp, DATA_PATH)


if "cases" not in st.session_state:
    st.session_state.cases = load_cases()
if "idx" not in st.session_state:
    # jump to the first unlabelled case
    st.session_state.idx = next(
        (i for i, c in enumerate(st.session_state.cases) if c.get("human_label") is None),
        0,
    )

cases = st.session_state.cases
n = len(cases)
labelled = sum(c.get("human_label") is not None for c in cases)


def set_label(value):
    cases[st.session_state.idx]["human_label"] = value
    save_cases(cases)
    # advance to the next unlabelled case, else next case
    nxt = next(
        (i for i in range(st.session_state.idx + 1, n) if cases[i].get("human_label") is None),
        min(st.session_state.idx + 1, n - 1),
    )
    st.session_state.idx = nxt


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown(f"### Progress: {labelled}/{n}")
    st.progress(labelled / n)
    st.session_state.idx = st.number_input(
        "Case #", min_value=0, max_value=n - 1, value=st.session_state.idx, step=1
    )
    st.caption(
        "Question: **is the trace's final answer a correct answer to the query, "
        "given the reference answer?** Judge content, not formatting."
    )
    remaining = [i for i, c in enumerate(cases) if c.get("human_label") is None]
    if remaining:
        st.caption(f"Unlabelled: {len(remaining)} (next: #{remaining[0]})")
    else:
        st.success("All cases labelled — thank you!")

case = cases[st.session_state.idx]

# ---------------- main ----------------
st.markdown(f"#### Case {st.session_state.idx} / {n - 1}")
current = case.get("human_label")
if current is not None:
    st.info(f"Current label: **{current}**")

st.markdown("**Query**")
st.markdown(f"> {case['query']}")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Reference (expected) answer**")
    st.text_area("gold", case["gold"], height=320, label_visibility="collapsed", disabled=True)
with col2:
    st.markdown("**Trace final answer (to be judged)**")
    st.text_area("answer", case["answer"], height=320, label_visibility="collapsed", disabled=True)

b1, b2, b3, b4 = st.columns([1, 1, 1, 1])
b1.button("✅ Correct", on_click=set_label, args=(True,), use_container_width=True)
b2.button("❌ Incorrect", on_click=set_label, args=(False,), use_container_width=True)
b3.button("🤷 Unclear", on_click=set_label, args=("unclear",), use_container_width=True)
b4.button("Skip →", on_click=lambda: st.session_state.update(idx=min(st.session_state.idx + 1, n - 1)),
          use_container_width=True)

with st.expander("Scorer verdicts (open only AFTER labelling — blind annotation)"):
    st.write({
        "numeric_match": case["numeric_match"],
        "judge_correct": case["judge_correct"],
        "disagreement_type": case["type"],
    })
