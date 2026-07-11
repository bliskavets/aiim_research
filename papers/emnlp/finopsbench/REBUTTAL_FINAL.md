# Reviewer PVoW

Thank you for the careful, actionable review. We ran the studies you asked for and answer each point below; the minor typos will be fixed in the camera-ready.

### Human validation and LLM-judge reliability
We ran a human evaluation on both halves with a domain-knowledgeable judge. On v1 the automatic judge agrees with the human on 85.1% of items (Cohen's κ = 0.67); on v2, scoring is execution-based and 178 of 200 sampled environments reproduce their gold answer (89%). v1 needs a semantic judge because most of its answers are free-form analyst deliverables, such as invoice-exception lists and variance tables, not single numbers a string or numeric match can grade.

### Releasing the pipeline prompts
The prompts were already part of the release. The paper link now points to the current code, and a top-level PROMPTS.md maps every pipeline stage to its exact prompt: the v1 stages and final filter, the v1 judge prompt, and the full v2 environment generator.

### Benchmark diversity and failure analysis
Appendices C, D and G report per-item statistics, example queries and the v1 category distribution. Beyond them, v1 has 742 distinct roles, no duplicate queries, and reference SQL using a JOIN in 70% of items; v2 environments average five tool calls among nine tools with distractors, and the two halves cover complementary concepts. A taxonomy over 779 failing traces shows v1 errors are semantic, a wrong predicate or incomplete retrieval, while v2 shifts to wrong-tool selection under distractors.

### Construction cost, compute, and runtime
Construction is fully automated with no paid annotation; the cost is API usage, measured by replaying each stage. v1 cost about $450 (10000 candidates to 5979 final, $0.075 each) and v2 about $340 (1247 to 1108, $0.307), roughly $790 total, API-only with no GPU. The single H100 serves open models only at evaluation. Wall-clock is about 24 hours for v1 and 5 for v2, run 8-way parallel.

### Potential bias from proprietary models
Several choices limit single-vendor influence: a cross-vendor judge panel, different models for generation and judging, execution-based v2 ground truth, and a human in the loop during development. Empirically, if v1 favoured its own generator, GPT-4.1-mini, that model should top v1; instead it is the lowest-scoring frontier model there (61.5% against GPT-5's 68.9%), so the pipeline gives its generator no advantage.

# Reviewer 6zfv

Thank you for the questions on scope, novelty and diagnostics. We answer each with direct measurements.

### Fundamental NLP capability advanced
FinOpsBench isolates capabilities a QA dataset does not: planning under partial observability, since the data is not in context and the model must probe the schema, plan a retrieval path and aggregate; writing real analytic SQL, with a JOIN in 70% of items; and turning an open-ended request into a multi-step analysis while ignoring distractors. These are generic tool-using-agent skills that carry over beyond finance, which here only supplies verifiable semantics and executable ground truth.

### Novelty versus recent agentic-finance benchmarks
Recent finance-agent benchmarks buy realism with live tools, which is useful but not reproducible or controllable. FinOpsBench is hermetic and executable: items rerun identically, answers are scored against an executable plan rather than a rubric, and difficulty is a released knob. On the same model, static finance QA is largely read off the prompt (TAT-QA 89%, FinQA 67%), yet FinOpsBench-v2 scores 1.5% without tools and only recovers with tool use, which those benchmarks cannot test.

### Diagnostics beyond final-answer accuracy
We report validated diagnostics beyond accuracy. Scoring is calibrated first: the judge matches a domain human on 85.1% of v1 items (κ 0.67), and v2 is execution-based. On top, a failure taxonomy over 779 traces in eight categories, with process metrics, separates models at equal accuracy: v1 errors are semantic (wrong predicate, incomplete retrieval), v2 moves to wrong-tool selection under distractors, and open models hit their step limit far more often.

### Dependence on LLM-generated data and judgments
The dependence is asymmetric: v2 questions are human-authored (FinQA) and validated by execution, not judgement. Where judgement is used it tracks a human (85.1%, κ 0.67). The released set is filtered, not raw generation: execution checks and a cross-vendor panel discard about 40% of v1 and 11% of v2 candidates. The two halves agree within 2.6 points per model, and we release the construction code so the generator and judge models can be swapped.

# Reviewer j7in

Thank you for the detailed review. We address the design questions with direct evidence and clarify the harness and model points.

### v1 and machine-verifiable ground truth
Every v1 item carries a hard expected answer, created with the data and enforced by execution-based validation and an answer-consistency filter; the panel is an extra gate, not a replacement. Deterministic matching is well-defined for only 4.4% of answers, the rest being free-form analyst outputs. On that scalar subset a domain human sides with the judge on 82.6% of contested cases (κ 0.64), and overall the judge matches the human on 85.1% (κ 0.67). v2 is scored fully deterministically.

### v2 derived from FinQA: "monotonous," "artificially added multi-hop"
Deriving v2 from FinQA is deliberate. We hold the question content fixed and change only the access mode, from reading to tool use, which isolates the agentic component: reading static FinQA reaches about 80-85%, yet the best agent reaches only about 69% on the same questions here. The native, business-driven tasks are v1's job, with 5979 analyst tasks across payables aging, reconciliation, variance and revenue recognition. Together the halves give breadth and controlled verifiability.

### Missing top agent models and finance-specialized LLMs
Claude Code, Codex and OpenCode are products, not base models; the base models behind them, Claude, GPT-5 and GPT-4.1, are already evaluated. Scoring through a product harness measures its scaffolding, not the model, and is not reproducible: even switching from native to ReAct moves v1 accuracy by up to 6.4 points. The paper already includes frontier models like GPT-5 and o4-mini, and we added more vendors. Open finance models are text continued-pretrains without reliable tool-calling.

### "Outdated smolagents" / framework noise
smolagents is a current, actively maintained library, chosen because it is minimal. v1 does not use it at all, running a native loop and a ReAct variant over the model API, so it cannot affect more than half the benchmark. Framework noise is also checked empirically: rankings hold across two protocols and two stacks, and per-model accuracy agrees across the two versions within 2.6 points.

### Data-contamination risk for v2
We test this three ways. Closed-book, with the prompt but no tools, is flat at about 14% across models while agentic reaches 53 to 68%, so memorization does not supply the answer. On a 200-item subset accuracy rises monotonically with access: question-only 2-4%, tools 20-54%, gold facts in context 57-69%, the opposite of recall. And half the benchmark, v1's 5979 items, is freshly generated and never published, yet gives the same per-model ranking within 2.6 points.
