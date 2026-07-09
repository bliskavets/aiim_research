# E10 — Cross-benchmark comparison vs an open competitor (Reviewer R2)

**Claim addressed:** R2 — existing finance benchmarks already do agentic tool use;
what does FinOpsBench add? We run the SAME model on an external, open competitor
benchmark and on FinOpsBench to show empirically what the competitors do not test.

**Competitor:** TAT-QA (NExT++, open) — a static financial table+text QA benchmark
explicitly cited in the paper's Related Work. We evaluate its arithmetic questions in
the standard reading setting (table + paragraphs in the prompt, no tools), scored with
the benchmark's own percent-robust numeric comparator. (FinGAIA / FinAgentBench /
Finance Agent Benchmark are agentic but ship as bespoke live-retrieval harnesses; the
general open agentic benchmark τ-bench is retail/airline, domain-mismatched — so TAT-QA
is the cleanest apples-to-apples external finance comparison.)

## Result — same model (gpt-4.1-mini), 200 items each
| Benchmark | Setting | Accuracy |
|---|---|---|
| **TAT-QA** (external, static finance QA) | reading: table+text in prompt, no tools | **89.0%** |
| FinOpsBench-v2 | reading: full-context, no tools | 64.5% |
| FinOpsBench-v2 | **agentic: tools only** | 61.5% |
| FinOpsBench-v2 | closed-book: no data, no tools | **1.5%** |

## Takeaway
The same model that answers an external static finance benchmark at **89%** — pure
reading comprehension — **collapses to 1.5%** on FinOpsBench without tools, and only
recovers (to ~62%) once it uses tools to retrieve the data. Static finance benchmarks
(TAT-QA, FinQA, TAT-QA-like) measure reading over provided context; FinOpsBench measures
the tool-use/retrieval-planning capability they structurally cannot test. This is the
"fundamentally new evaluation capability" the reviewer asked about, shown against a real
competitor rather than argued.

Run: `python run_tatqa.py --model openai/gpt-4.1-mini --n 200`
