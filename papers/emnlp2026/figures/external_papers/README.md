# External-paper figures

Replications of the FinOpsBench accuracy-vs-size analysis on other
financial agent benchmarks, for comparison.

## `fig_financeagentbench_size_scatter.{png,pdf}`

**Source paper.** "Finance Agent Benchmark: Benchmarking LLMs on
Real-world Financial Research Tasks", arXiv:2508.00828.

**Source data.** Table 2 of the paper (Class-Balanced Accuracy
column — the metric the paper recommends as most representative).
All 22 LLM rows; the human "Expert" baseline row is excluded.

**Size estimates.** Open-weight models (LLaMA 3.3 70B, LLaMA 4
Scout / Maverick, Mistral Small 3.1, Command A) use vendor-reported
total parameters. The 17 proprietary models use Li (2026), IKP
(arXiv:2604.24827) — direct quotes from the main estimates table on
page 13 where available; the rest are derived by inverting the
paper's log-linear calibration (slope 0.147/decade, intercept 13.3)
on the model's IKP raw score from the extended table on pages 50–52.
3× CI bars shown for all estimated sizes.

**Finding.** A clear log-linear trend with **slope ~15.5 pp/decade**
of parameters — roughly **2×** steeper than what we observed on
FinOpsBench (~8 pp/decade). Reasoning models (green) consistently
sit above non-reasoning models (blue) at comparable size. The most
visible outliers are:

- **LLaMA 4 Maverick (400B, 3.1%)** — the paper's own analysis flags
  it as misusing the tool interface (hallucinating documents).
- **o1 (3.5T, 21.4%)** — far below the trend line; an older reasoning
  generation outperformed by smaller, newer reasoners (o4 Mini, Grok
  3 Mini Reason.).
- **Claude 3.7 Sonnet (676B, 44.3%)** — far *above* the trend line;
  the only non-reasoning model in the top tier.

The regression line is fit through **all 22 models** (the user did
not ask to exclude outliers on this dataset).

## `fig_finagentbench_size_scatter.{png,pdf}`

**Source paper.** Lee et al., "FinAgentBench: A Benchmark Dataset for
Agentic Retrieval in Financial Question Answering",
arXiv:2508.14052v3.

**Source data.** Tables 1 (Document Ranking) and 2 (Chunk Ranking) of
the paper. Only three models are evaluated: GPT-o3, Claude-Opus-4, and
Claude-Sonnet-4 — all proprietary, no open-weight points.

**Size estimates** come from Li (2026), "Incompressible Knowledge
Probes" (arXiv:2604.24827):

| Model | Size (B) | Source |
|---|---:|---|
| GPT-o3 | ~3,000 | IKP main table p.13 (64.4% → 3.0T) |
| Claude-Opus-4 | ~1,400 | IKP main table p.13 (59.7% → 1.4T) |
| Claude-Sonnet-4 | ~237 | IKP extended table p.52 (IKP 0.482, calibrated via 14.7 pp/decade slope → 237B) |

3× CI horizontal bars are drawn on every marker, matching the IKP
calibration spread.

## Finding

The size-accuracy trend on FinAgentBench is **opposite** to the one we
see on FinOpsBench (`figures/fig_accuracy_vs_size.png`):

- On **both** FinAgentBench tables, **Claude-Sonnet-4 (smallest) wins
  or ties on every metric**. GPT-o3 (largest, ~12× more params) is the
  worst on most metrics.
- This is a robust "training matters more than size" signal at the
  proprietary frontier, even sharper than what we observed on
  FinOpsBench. It also suggests that ~8 pp/decade scaling found on
  FinOpsBench is a property of *the FinOpsBench task* — it does **not**
  generalize across financial-agent benchmarks.

## Reproduce

```bash
cd papers/emnlp2026/figures/external_papers
python3 make_finagentbench_size_scatter.py
```
