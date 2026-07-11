# Reviewer PVoW

Thank you for the detailed, actionable review; we ran the studies you asked for, and will fix the minor typos in the camera-ready.

**Human validation and LLM-judge reliability.** We ran a human evaluation on both halves with a domain-knowledgeable judge.

| Half | check | agreement with the human |
|---|---|---|
| v1 | automatic judge vs human | 85.1%, Cohen's κ = 0.67 |
| v2 | reference plan reproduces gold (execution) | 178/200, 89% |

v1 needs a semantic judge because most answers are free-form analyst deliverables (invoice-exception lists, per-supplier variance tables), not single numbers a string or numeric match can grade.

**Releasing the pipeline prompts.** The prompts were already in the release; the paper link now points to the current code, and a top-level PROMPTS.md maps every pipeline stage to its exact prompt (the v1 stages and final filter, the v1 judge prompt, the full v2 environment generator).

**Benchmark diversity and failures.** Appendices C, D and G already give per-item statistics, example queries and the v1 category distribution. Beyond them, v1 has 742 distinct roles, no duplicate queries, and analytic reference SQL:

| v1 SQL surface | share of items |
|---|---|
| JOIN | 70% |
| aggregate | 35% |
| subquery | 22% |
| two or more JOINs | 33% |

v2 environments average five tool calls among nine tools with distractors, and the halves cover complementary concepts (v1 accounts payable 52%, v2 statement ratios 77%). Difficulty scales with required tool-chain depth:

| tool-chain depth | 1-3 | 4-5 | 6-7 | 8+ |
|---|---|---|---|---|
| pooled agentic accuracy | 61.6% | 61.1% | 57.0% | 45.8% |

A failure taxonomy over 779 traces (eight categories) shows the profile shift:

| Model | half | malformed args | incomplete retrieval | wrong-tool | calc | round-limit |
|---|---|---|---|---|---|---|
| GPT-4.1 | v1 | 42% | 33% | 4% | 5% | 3% |
| Claude-Sonnet-4.5 | v2 | 12% | 15% | 20% | 17% | 7% |
| DeepSeek-V3 | v2 | 7% | 16% | 23% | 14% | 25% |

v1 failures are semantic (wrong predicate, incomplete retrieval); v2 shifts to wrong-tool selection under distractors, and open models exhaust their step budget far more often.

**Construction cost.** Fully automated, no paid annotation; the cost is API usage, measured by replaying each stage: v1 about $450 (10000 to 5979 items, $0.075 each), v2 about $340 (1247 to 1108, $0.307), roughly $790 total, API-only with no GPU. The single H100 serves open models only at evaluation. Wall-clock is about 24 hours (v1) and 5 (v2), run 8-way parallel.

**Bias from proprietary models.** A cross-vendor judge panel, different models for generation and judging, execution-based v2 ground truth, and a human in the loop during development already limit single-vendor influence. And if v1 favoured its own generator (GPT-4.1-mini) that model should top v1; instead it is the lowest-scoring frontier model there (61.5% against GPT-5's 68.9%), so the pipeline gives its generator no advantage.

# Reviewer 6zfv

Thank you for the questions on scope, novelty and diagnostics; we answer each with direct measurements.

**Fundamental capability advanced.** FinOpsBench isolates capabilities a QA dataset does not: planning under partial observability (the data is not in context, so the model must probe the schema, plan retrieval and aggregate), writing real analytic SQL (a JOIN in 70% of items), and turning an open-ended request into a multi-step analysis while ignoring distractors. These are generic tool-using-agent skills that carry over beyond finance, which here only supplies verifiable semantics and executable ground truth.

**Novelty versus recent agentic-finance benchmarks.** Recent benchmarks buy realism with live tools, useful but not reproducible or controllable. FinOpsBench is hermetic and executable: items rerun identically, answers are scored against an executable plan rather than a rubric, and difficulty is a released knob. Even the strongest agent stays below 70% on a uniform subset, so the benchmark is far from saturated (gpt-oss-120b 69.9%, Claude-Sonnet-4.5 68.6%, Qwen3-235B 65.0%, GPT-4.1-mini 60.0%, DeepSeek-V4-Flash 54.3%, Llama-3.3-70B 19.8%). The same model reads static finance QA off the prompt but cannot operate ours without tools:

| Same model (GPT-4.1-mini) | accuracy |
|---|---|
| TAT-QA, reading | 89% |
| FinQA, reading | 67% |
| FinOpsBench-v2, no tools | 1.5% |
| FinOpsBench-v2, agentic | 61.5% |

Difficulty is also controllable: pooled accuracy falls from 61.6% at shallow depth to 45.8% at eight or more tool calls.

**Diagnostics beyond final-answer accuracy.** Scoring is calibrated first (judge matches a domain human 85.1%, κ 0.67; v2 execution-based). On top, a 779-trace, eight-category failure taxonomy with process metrics separates models at equal accuracy:

| Model | half | malformed args | incomplete retrieval | wrong-tool | round-limit |
|---|---|---|---|---|---|
| GPT-4.1 | v1 | 42% | 33% | 4% | 3% |
| Claude-Sonnet-4.5 | v2 | 12% | 15% | 20% | 7% |
| DeepSeek-V3 | v2 | 7% | 16% | 23% | 25% |

v1 errors are semantic (wrong predicate, incomplete retrieval), v2 shifts to wrong-tool selection under distractors, and open models hit their step limit far more often.

**Dependence on LLM-generated data and judgments.** The dependence is asymmetric: v2 questions are human-authored (FinQA) and validated by execution, not judgement; where judgement is used it tracks a human (85.1%, κ 0.67). The released set is filtered, not raw: execution checks and a cross-vendor panel discard about 40% of v1 (10000 to 5979) and 11% of v2 (1247 to 1108) candidates. The two halves agree within 2.6 points per model, and we release the construction code so the generator and judge models can be swapped.

# Reviewer j7in

Thank you for the detailed review; we address the design questions with direct evidence and clarify the harness and model points.

**v1 and machine-verifiable ground truth.** Every v1 item carries a hard expected answer, created with the data and enforced by execution-based validation plus an answer-consistency filter; the panel is an extra gate, not a replacement. Deterministic matching is well-defined for only 4.4% of answers (the rest are free-form analyst outputs). On the contested cases a domain human sides with the judge on 82.6% (κ 0.64), and overall the judge matches the human on 85.1% (κ 0.67). v2 is scored fully deterministically.

**v2 derived from FinQA.** Deriving v2 from FinQA is deliberate: we hold the question content fixed and change only the access mode, from reading to tool use, which isolates the agentic component (static FinQA reaches about 80-85%, yet the best agent reaches only about 69% on the same questions here). The native, business-driven tasks are v1's role: 5979 analyst tasks across payables aging, reconciliation, variance and revenue recognition. Together the halves give breadth and controlled verifiability.

**Missing top agent models and finance LLMs.** Claude Code, Codex and OpenCode are products, not base models; the base models behind them (Claude, GPT-5, GPT-4.1) are already evaluated. Scoring through a product harness measures its scaffolding, not the model, and is not reproducible: even switching from native to ReAct moves v1 accuracy by up to 6.4 points. The paper already includes frontier models (GPT-5, o4-mini), and we added more vendors. Open finance models are text continued-pretrains without reliable tool-calling.

**"Outdated smolagents".** smolagents is a current, actively maintained library, chosen because it is minimal; v1 does not use it at all (a native loop and a ReAct variant over the model API), so it cannot affect more than half the benchmark. Rankings also hold across two protocols and two stacks, with per-model accuracy agreeing across the two versions within 2.6 points.

**Data-contamination risk for v2.** We test this three ways. Closed-book (the prompt but no tools) is flat at about 14% while agentic reaches 53 to 68%, so memorization does not supply the answer:

| Model | closed-book | agentic |
|---|---|---|
| GPT-5-mini | 14.7% | 67.5% |
| GPT-4.1 | 13.3% | 60.6% |
| Qwen3-30B-A3B | 13.8% | 53.0% | On a 200-item subset accuracy rises monotonically with access, the opposite of recall:

| Model | question-only | tools | gold facts in-context |
|---|---|---|---|
| DeepSeek-V4-Flash | 2.5% | 54.3% | 68.0% |
| DeepSeek-V3.2 | 4.0% | 38.6% | 69.0% |
| Llama-3.3-70B | 3.0% | 19.8% | 57.0% |

And half the benchmark, v1's 5979 items, is freshly generated and never published, yet gives the same per-model ranking (within 2.6 points).
