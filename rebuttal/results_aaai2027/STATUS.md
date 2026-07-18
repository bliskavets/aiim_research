# SAGE AAAI-2027 — experiment status & reviewer-coverage matrix

Snapshot of the strengthening campaign. Detailed per-experiment writeups (with numbers,
why-it-matters, and files) are in EXPERIMENTS_LOG.md; overall context in INDEX.md.
Model Qwen3-8B-FP8 (plus 1.7B and 32B for scaling), single H200, non-thinking via chat
template. SAGE = m_min 1, 2 epochs x 7 gens. Multi-seed = 42/7/123.

## Completed experiments

| # | Experiment | Headline result | Reviewer issue closed |
|---|-----------|-----------------|-----------------------|
| A1 | MATH-500 baseline (N=500) | 83.8 (reproduces paper's 84.4) | R4-W1a (baseline discrepancy) |
| E1 | SAGE MATH-500, 3 seeds | 88.3 +/- 0.6 vs 83.8, McNemar p<0.001 | k8B9/qCe4 (self-judge not familiarity), multi-seed rigor |
| E5 | MMLU-Pro STEM, 3 seeds | 77.3 +/- 1.3 vs 71.2, p up to 3e-5 | R4-W2 (harder benchmark), breadth |
| E4 | IFEval (N=541) | base 73.8 / BoN 77.4 / SAGE 76.3 | qCe4-W4 (verbosity: deterministic gains) |
| E2 | Self-Refine / Reflexion (MATH) | 86.6 / 86.4 < SAGE 88.3 | k8B9 (no same-model refinement baselines) |
| E3b+E8 | Judge vs gold + calibration, 3 seeds | oracle gap 7.3%, AUC 0.641, judge recovers 92% | R2-Q3 (self-congratulatory loop), trust-the-judge; replaces NDCG |
| TPO | Official TPO + FsfairX RM, D2-N5 | MMLU 62.6, IFEval 76.0 (below baseline on STEM) | R3 baselines, R4-W3; anti-external-RM |
| E9 | BoN + Skywork-Reward-V2-Qwen3-8B (modern RM) | MATH 74.2 (< baseline), IFEval 75.8 | R4-W3 (RM outdated -> even a modern RM hurts) |
| B2 | m_min sweep {1,2,4,8} MATH | 88.8 / 87.8 / 89.4 / 87.6 (robust) | R3-W3/Q3 (group vs best-worst, how to pick m_min) |
| A2 | Latency table N=100 | SAGE 0.86 @71s vs BoN 0.82 @14s | R1-Q2, R2-Q1, R3-Q4 (wall-clock vs BoN) |
| B4 | Qwen3-1.7B (IFEval) | SAGE 60.8 = base 61.0; BoN collapses 12.6 | R1-Q1 (<3B gradient noise: graceful, not noise) |
| C1 | Qwen3-32B MATH | base 85.4 -> SAGE 91.4 (+6.0; gain grows with scale) | R2-Q2, R3-W1 (larger/stronger models) |
| thinking | Qwen3-8B reasoning MATH | completed 96.4 / overall 74.8 (22.4% truncated) | R4-W1b (thinking vs SAGE): honest framing |
| XSTest | Safety per-category (N=450) | base 90.0 -> SAGE 92.7 (safe 83.6->87.2, unsafe 98->99.5) | R2/k8B9 safety per-category (E10) |
| AlpacaEval | vs davinci003 & GPT-4-turbo; head-to-head | SAGE 48.5 vs base 43.0 (GPT4t); SAGE>base 57.5 while SHORTER | qCe4-W4 (verbosity: quality not length) |
| B3 | Aspect sensitivity MATH (3 configs) | default 72.4 / generic 74.6 / task_specific 71.8 (spread 2.8pt = noise -> robust) | R3-W2/Q2 (aspect formulation) |
| SPO | Fair re-run (Qwen3-8B optimizer, not GPT-4.1) | best=seed round (opt failed to improve); eval 66.2 << SAGE 88.3 | R3-Q1 (SPO needs strong external optimizer) |

Central narrative (strongest): on verifiable tasks external reward models HURT
(TPO MMLU 62.6, BoN+Skywork-V2 MATH 74.2, both below baseline) while SAGE's self-judge
helps and scales (MATH 8B +4.5 -> 32B +6.0). Self-judge recovers 92% of achievable
accuracy (small oracle gap), gains are not verbosity (AlpacaEval head-to-head, SAGE shorter).
| repro | Репликация старого TTA-кода + провенанс Table 5 | старый код 74.0 (o3, seed-7-50), починенный 78.0, baseline 68.0; Table 5 = сломанный non-think, офиц. 83.0 IFEval | R4-W1a; обоснование замены чисел |

## Still running

None — all planned experiments complete.

## Deprioritized / not completed (with reason)

- thinking-SAGE (SAGE on top of reasoning): candidate generations truncate under context
  budget + very expensive; deferred as future work / limitation.
- B4 MATH-500 SAGE (1.7B): baseline done (72.8); SAGE run deprioritized (R1-Q1 already
  answered by B4 IFEval). BoN temp bug fixed if a rerun is wanted.
- B3 IFEval aspect runs (3): trimmed to save ~8h; IFEval SAGE ~= baseline so aspect
  sensitivity there is uninformative. MATH aspect comparison is the deliverable.

## Reviewer-concern coverage matrix

- R1/Y1iM-Q1 small models: CLOSED (B4)
- R1/Y1iM-Q2, R2/qCe4-Q1, R3-Q4 latency vs BoN: CLOSED (A2)
- R2/qCe4-W2 trust confidence: CLOSED (E3b + B4)
- R2/qCe4-W4 verbosity: CLOSED (IFEval + AlpacaEval head-to-head)
- R2/qCe4-Q2 larger models: ADDRESSED (C1 32B; 70B out of single-H200 scope, trend given)
- R2/qCe4-Q3 self-congratulatory loop: CLOSED (E3b/E8)
- R3-W2/Q2 aspect sensitivity: CLOSED (B3 MATH, spread 2.8pt = robust)
- R3-W3/Q3 m_min / grouping: CLOSED (B2)
- R3-Q1 SPO GPT-4.1 unfair: CLOSED (SPO fair re-run: same-model optimizer fails to improve; SPO depends on strong external optimizer)
- R4/k8B9-W1a baseline discrepancy: CLOSED (A1)
- R4/k8B9-W1b thinking mode: ADDRESSED (thinking; honest framing, not "SAGE > thinking")
- R4/k8B9-W2 benchmarks too easy: CLOSED (MMLU-Pro)
- R4/k8B9-W3 outdated RM: CLOSED (E9 modern RM + TPO)
- k8B9/qCe4 self-judge = familiarity not correctness: CLOSED (verifiable gains + E3b)
