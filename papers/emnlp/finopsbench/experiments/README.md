# FinOpsBench — Rebuttal Experiments

Experiment code for the EMNLP rebuttal (submission 5243). Each subfolder is
self-contained: a runner, raw results (`results/*.jsonl`), and a summary.
The benchmark source is used read-only (`--benchmark_root`, default
`/tmp/FinOpsBench` — a checkout of the FinOpsBench release repo); no
experiment code lives in the benchmark repository.

| Folder | Question answered | Reviewers |
|---|---|---|
| `e1_closed_book/` | Does FinQA contamination provide an answer pathway for v2? | R3 (contamination) |
| `e2_judge_agreement/` | How does the v1 LLM judge relate to deterministic numeric scoring? | PVoW (judge validity), R3 (ground truth) |

All API calls go through OpenRouter (`OPENROUTER_API_KEY`).
