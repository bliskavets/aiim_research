# E6 — Quantitative diversity statistics

**Claim tested:** "Analyze benchmark diversity more quantitatively. Statistics
on reasoning operations, SQL complexity, tool-chain depth, numerical
operations, financial concepts, and template diversity would strengthen the
benchmark description" (Reviewer PVoW); "query types are monotonous"
(Reviewer R3, about v2).

Pure offline analysis of the released benchmark data; no API calls.

Headline numbers (`results/diversity_summary.json`):

- **v1** (8,233-item pool): 742 distinct user roles; zero duplicate queries;
  distinct 3-gram ratio 0.52. SQL surface of reference solutions: 70% of
  examples require a JOIN, 42% ORDER BY, 35% aggregates, 31% GROUP BY,
  22% subqueries, 19% date functions, 9% CASE, 7% HAVING.
- **v2** (1,301 environments on disk): reference plans make a median of
  5 tool calls (p90 = 7, max 15) against a median of 9 available tools
  (core + partial + distractor). Operation mix: aggregation 51%,
  difference/YoY 41%, ratio 32%, average 11%, percent change 11%.

Run: `python compute_diversity.py [--benchmark_root /tmp/FinOpsBench]`
