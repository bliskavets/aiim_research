# External-paper figures

Replications of the FinOpsBench accuracy-vs-size analysis on other
financial agent benchmarks, for comparison.

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
