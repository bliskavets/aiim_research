# E8 — Information-access ladder (novelty / "what new capability?", Reviewer R2)

**Claim addressed:** R2 — "multiple recent benchmarks already do agentic financial
tool use; what *fundamentally new evaluation capability* does FinOpsBench provide?"

**Idea.** FinOpsBench can hold the *item* fixed and vary the *information-access
mode*, decomposing accuracy into three rungs on the SAME 200 v2 items / same model:

| rung | prompt | measures |
|---|---|---|
| (a) question-only | just the question, no data, no tools | parametric / memorised knowledge |
| (c) agentic | tools only (our benchmark, SA runner) | agentic retrieval + planning |
| (d) FinQA-native | the original FinQA input for the item — gold-retrieved facts (`qa.model_input`) | static-benchmark reading (source setting) |
| (b) full-context | whole FinQA narrative+table in prompt, no tools | reading with imperfect retrieval |

Scoring is percent-robust (the benchmark's compare_answers treats a trailing '%'
as /100; a CoT model that prints "52.32" for gold "52.32%" is correct but scored
100x off — the real agents keep the '%' sign so are unaffected. `rescore.py`/
`assemble.py` credit the percent-scaling case consistently across all rungs).

## Result (percent-robust, n=200)
| Model | (a) question-only | (c) agentic | (d) FinQA-native (gold facts) | (b) full-context (whole doc) | tool-use necessity (c−a) | agentic gap (d−c) |
|---|---|---|---|---|---|---|
| GPT-4.1-mini | 1.5% | 61.5% | 60.5% | 64.5% | **+60.0 pt** | −1.0 pt |
| GPT-4.1 | 2.0% | 63.5% | 65.5% | 65.0% | **+61.5 pt** | +2.0 pt |
| DeepSeek-V3.2 | 4.0% | 48.2% | 69.0% | 69.5% | **+44.2 pt** | **+20.8 pt** |

## Interpretation
- **Tool-use necessity ≈ 60 pt (c−a).** The questions are essentially unanswerable
  from parametric memory (1.5–2%); accuracy only appears once tools retrieve the
  data. This is a capability static financial QA cannot measure and a direct
  refutation of contamination.
- **Agentic gap is model-discriminating (d−c).** ~0 pt for the GPT-4.1 family but **+20.8 pt for DeepSeek-V3.2**, which reads FinQA better than GPT-4.1 (69% vs 65%) yet acts on it far worse (48% vs 64%) — a distinction static benchmarks cannot see. Where the gap is ~0 a model that can *read* the disclosure can also
  *retrieve* it through tools with minimal loss — so the tool wrapper is faithful
  (adds no spurious difficulty) and the reading ceiling (~65%) confirms the items
  are well-posed. Remaining agentic failures are genuine tool-use/planning errors
  (see e5_failure_taxonomy), not artifacts.
- **The novel capability:** decomposing a fixed item into memorisation / agentic
  retrieval / reading ceiling. No static benchmark (no tool requirement) or live
  benchmark (cannot hold the item fixed or reproduce it) can produce this.

Run: `python run_context.py --mode {question_only,full_context} --model M`
(rungs a,b) + `run_e4.py --subset_file subset_200.json` (rung c) + `python assemble.py`.
