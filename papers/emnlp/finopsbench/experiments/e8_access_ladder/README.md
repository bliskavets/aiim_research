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
| Model | question-only | agentic (leaky) | **agentic (clean)** | FinQA-native | full-context | **agentic gap (clean)** | n |
|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 2.5% | 66.5% | **69.9%** | 64.5% | 66.5% | **-5.4** | 103* |
| Claude-Sonnet-4.5 | 1.5% | 69.2% | **68.6%** | 68.5% | 69.5% | **-0.1** | 156† |
| GPT-4.1 | 2.0% | 63.5% | **66.0%** | 65.5% | 65.0% | **-0.5** | 200 |
| Claude-Haiku-4.5 | 0.5% | 67.5% | **65.5%** | 67.0% | 69.5% | **+1.5** | 200 |
| Qwen3-235B-A22B | 2.5% | 65.0% | **65.0%** | 65.0% | 68.0% | **+0.0** | 200 |
| GPT-4.1-mini | 1.5% | 61.5% | **60.0%** | 60.5% | 64.5% | **+0.5** | 200 |
| DeepSeek-V4-Flash | 2.5% | 71.0% | **54.3%** | 68.0% | 71.0% | **+13.7** | 162* |
| DeepSeek-V3.2 | 4.0% | 48.2% | **38.6%** | 69.0% | 69.5% | **+30.4** | 158* |
| Llama-3.3-70B | 3.0% | 29.9% | **19.8%** | 57.0% | 59.0% | **+37.2** | 106* |

_Agentic column re-run on leak-cleaned prompts (see e11_prompt_leak_audit); `agentic (leaky)` shown for reference. Cleaning is model-dependent: strong tool-users unchanged, read-well-act-poorly models (DeepSeek-V3.2/V4-Flash, Llama-3.3-70B) drop sharply → larger agentic gaps._

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
