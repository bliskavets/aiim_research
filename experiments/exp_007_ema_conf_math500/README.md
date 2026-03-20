# Experiment 007: EMA Confidence GTPO/GRPO-S on MATH-500

## Overview
= exp_004 setup (MATH-500, <think>/<answer>, answer bonus=5.0)
+ exp_006 methods (EMA-smoothed confidence, λ=0.9, top_k=20)

## Results vs MATH-500 baselines

| Method | Step 1 | Step 250 | Step 500 | Peak | @Step | Format@500 | KL@500 |
|--------|--------|----------|----------|------|-------|------------|--------|
| GRPO (exp003) | 0.75 | 7.88 | 0.62 | 10.00 | 150 | 0.75 | 0.0453 |
| GTPO entropy (exp004) | 0.75 | 5.50 | 2.38 | 10.00 | 155 | 2.25 | 0.0035 |
| GRPO-S entropy (exp004) | 0.75 | 6.38 | 2.38 | 10.00 | 136 | 2.25 | 0.0041 |
| **GTPO-EMA (exp007)** | **1.12** | 7.25 | 2.38 | **10.00** | **97** | 2.25 | 0.0101 |
| **GRPO-S-EMA (exp007)** | **1.12** | **8.50** | 2.38 | **10.00** | **104** | 2.25 | 0.0359 |

## Key Observations
- EMA методы достигают пика **раньше** (step ~100 vs ~150 у GRPO)
- GRPO-S-EMA показывает лучший step 250 reward (8.50) среди всех MATH-500 методов
- Все методы сходятся к ~2.38 к шагу 500 — collapse на малом датасете (500 примеров, 1 эпоха)
- EMA confidence стартует лучше (step1=1.12 vs 0.75) — более стабильный сигнал с первых шагов
