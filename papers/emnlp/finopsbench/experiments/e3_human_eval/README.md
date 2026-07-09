# E3 — Human evaluation (single expert annotator)

**Claim addressed:** "Include a human evaluation study for both benchmark
versions ... 200–300 examples independently verified" (Reviewer PVoW);
also PVoW-2 (LLM-judge ↔ human agreement) and R3-1 (machine-verifiable ground truth).

Single domain-expert annotator. With one annotator, inter-annotator κ is not
available; we instead report **human ↔ automatic-scorer agreement**, which is
exactly what PVoW-2 asks for. Maximally reuses the E2 annotation.

## Design

**(b) Evaluation-judge accuracy — v1, stratified over the scalar subset (362).**
- *contested* stratum (93 items, judge ≠ numeric matcher): fully labelled in E2
  (`../e2_judge_agreement/results/disagreements_for_human_annotation.jsonl`).
- *non-contested* stratum (269 items, judge = numeric): random sample of 80
  labelled here (`data/sample_v1_judge.jsonl`).
- `estimate.py` weights the two strata by size → unbiased judge accuracy over
  the scalar subset (rather than the E2 worst-case 82.6% on contested only).

**(a) Dataset validity — v2 (the version with no prior human check).**
- Random sample of 100 v2 environments (`data/sample_v2_validity.jsonl`):
  holistic valid/invalid on whether the reference plan computes the gold answer
  from the tools and the question is well-posed.
- The v1 answer-correctness validity rate falls out of the (b) strata for free.

## Annotate

```bash
python build_samples.py                       # (re)build the two sample files
DATA=data/sample_v1_judge.jsonl    streamlit run viewer.py --server.port 8788 --server.address 0.0.0.0 --server.headless true
DATA=data/sample_v2_validity.jsonl streamlit run viewer.py --server.port 8789 --server.address 0.0.0.0 --server.headless true
python estimate.py                             # combine E2 + E3 -> results/e3_summary.json
```
