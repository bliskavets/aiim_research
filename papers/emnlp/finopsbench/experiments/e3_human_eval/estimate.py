"""Combine E2 + E3 labels into the E3 headline numbers.

v1 judge accuracy: stratified over the scalar subset (362 items) using
  - stratum "contested" (93): all labelled in E2 (disagreements file)
  - stratum "non-contested" (269): sampled and labelled in E3 (sample_v1_judge)
Judge accuracy = P(judge verdict == human label), estimated per stratum and
weighted by stratum size. Also reports the answer-correctness validity rate
(fraction of scalar-subset trace answers the human judged correct).

v2 validity: fraction of the v2 sample the human judged valid.

Usage: python estimate.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
E2_CONTESTED = HERE.parent / "e2_judge_agreement" / "results" / "disagreements_for_human_annotation.jsonl"
V1_NONCONT = HERE / "data" / "sample_v1_judge.jsonl"
V2_VALIDITY = HERE / "data" / "sample_v2_validity.jsonl"

# scalar-subset stratum sizes (from the E2 run: 93 contested + 269 non-contested)
N_CONTESTED, N_NONCONTESTED = 93, 269
N_SCALAR = N_CONTESTED + N_NONCONTESTED


def load(path):
    return [json.loads(l) for l in path.open() if l.strip()] if path.exists() else []


def frac(rows, agree_key=None):
    """If agree_key is None: fraction where human_label is True.
    Else: fraction where record[agree_key] == human_label (agreement)."""
    lab = [r for r in rows if isinstance(r.get("human_label"), bool)]
    if not lab:
        return None, 0
    if agree_key is None:
        return sum(r["human_label"] for r in lab) / len(lab), len(lab)
    return sum(r[agree_key] == r["human_label"] for r in lab) / len(lab), len(lab)


def main() -> None:
    contested = load(E2_CONTESTED)
    noncont = load(V1_NONCONT)

    # --- judge accuracy per stratum (judge_correct vs human_label) ---
    acc_c, n_c = frac(contested, "judge_correct")
    acc_nc, n_nc = frac(noncont, "judge_correct")

    out = {"v1_judge_accuracy": {}, "v1_answer_validity": {}, "v2_validity": {}}

    if acc_c is not None and acc_nc is not None:
        judge_acc = (N_CONTESTED * acc_c + N_NONCONTESTED * acc_nc) / N_SCALAR
        out["v1_judge_accuracy"] = {
            "contested": {"n_labelled": n_c, "judge_acc": round(acc_c, 3), "stratum_size": N_CONTESTED},
            "non_contested": {"n_labelled": n_nc, "judge_acc": round(acc_nc, 3), "stratum_size": N_NONCONTESTED},
            "stratified_judge_accuracy_scalar_subset": round(judge_acc, 3),
        }

    # --- answer-correctness validity of trace answers (human says correct) ---
    val_c, _ = frac(contested)
    val_nc, _ = frac(noncont)
    if val_c is not None and val_nc is not None:
        validity = (N_CONTESTED * val_c + N_NONCONTESTED * val_nc) / N_SCALAR
        out["v1_answer_validity"] = {
            "stratified_trace_answer_correct_rate_scalar_subset": round(validity, 3)
        }

    # --- v2 holistic validity ---
    v2 = load(V2_VALIDITY)
    val_v2, n_v2 = frac(v2)
    if val_v2 is not None:
        out["v2_validity"] = {"n_labelled": n_v2, "valid_rate": round(val_v2, 3)}

    print(json.dumps(out, indent=2))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "e3_summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
