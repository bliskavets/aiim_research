# Experiment 006: EMA-Smoothed Confidence GTPO and GRPO-S on GSM8K

## Overview
Modified versions of exp_005 replacing raw confidence C_i,t with its
**Exponential Moving Average** (λ=0.9):

```
EMA_i,t = λ · EMA_i,t-1 + (1-λ) · C_i,t
```

- **GTPO-EMA:** uses EMA_i,t at each token position (same structure as GTPO-Conf but smoothed)
- **GRPO-S-EMA:** uses EMA_i,T (LAST value) per sequence — captures confidence TREND at end of generation

## Config
Same as exp_005 + EMA params:
- **λ (lam):** 0.9
- **top_k:** 20
- **α₁=β₁=1.0, α₂=β₂=0.1**
- **GSM8K, Llama-3.2-3B, 500 steps, A100 80GB**

## Results (GSM8K)

| Method | Step 1 | Step 250 | Step 500 | Peak | @Step | Format@500 | KL@500 |
|--------|--------|----------|----------|------|-------|------------|--------|
| GRPO baseline | -1.375 | 2.000 | **3.000** | 8.375 | 169 | **3.00** | 0.0855 |
| GRPO + seq-level entropy (exp002) | -1.000 | 1.500 | **3.000** | 9.500 | 294 | **3.00** | 0.1172 |
| GTPO-Conf (exp005) | -1.000 | 2.875 | 2.375 | 9.500 | 268 | 2.25 | 0.0691 |
| **GTPO-EMA (exp006)** | -1.375 | **3.500** | **3.000** | **9.500** | 204 | **3.00** | 0.1035 |
| **GRPO-S-EMA (exp006)** | -1.375 | 2.250 | **3.000** | **9.500** | 231 | **3.00** | 0.1191 |

## Key Observations

**GTPO-EMA — лучший результат среди всех token-level методов:**
- Единственный token-level метод достигший Step 500 reward=3.0 И format=3.0
- Пик 9.5 достигнут раньше (шаг 204 vs 268 у GTPO-Conf) — EMA ускоряет обучение
- Step 250 reward=3.5 — самый высокий на этом шаге среди всех методов

**GRPO-S-EMA — успех после провала GRPO-S-Conf:**
- GRPO-S-Conf (exp005) полностью упал (reward=0 к шагу 500)
- GRPO-S-EMA восстановился: reward=3.0, format=3.0 к шагу 500
- Last-EMA сигнал стабильнее среднего confidence — ключевое улучшение

**Сравнение с exp_005:**
- GTPO: raw conf (2.375) → EMA conf (3.000) ✅
- GRPO-S: raw conf (0.000) → last EMA (3.000) ✅✅

**EMA smoothing работает:** сглаживание шума в confidence по ходу последовательности даёт более стабильный градиентный сигнал.

## Files
- `src/ema_confidence_utils.py` — EMA computation + reward shaping
- `src/gtpo_ema_trainer.py` — GTPOEMATrainer
- `src/grpo_s_ema_trainer.py` — GRPOSEMATrainer
- `tests/test_ema_utils.py` — 13/13 unit tests passing
- `train_gtpo_ema.log` / `train_grpo_s_ema.log` — training logs
