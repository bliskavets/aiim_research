# Scaling Laws

## Основные концепты

### Chinchilla Scaling Laws (Hoffmann et al., 2022)
Оптимальное соотношение параметров N и токенов D:
```
Оптимально: D ≈ 20 * N
Т.е. для модели 7B параметров: ~140B токенов оптимально
```

**Chinchilla закон loss:**
```
L(N, D) = E + A/N^α + B/D^β
где α ≈ 0.34, β ≈ 0.28
```

### Kaplan et al. (OpenAI, 2020)
- Первые scaling laws для LLM
- Loss убывает как степенная функция N, D, C
- Ошибка: переоценивали важность N vs D

---

## Вычисление числа параметров

### Базовая формула для Transformer
```python
# Параметры в одном transformer block:
attention = 4 * d_model^2  # Q, K, V, O projection (при d_k = d_model/H)
ffn = 8 * d_model^2         # 2 матрицы FFN (скрытый размер 4*d_model)
layer_norm = 4 * d_model     # 2 слоя нормализации

block_params = 12 * d_model^2

# Полная модель:
total = n_layers * 12 * d_model^2 + vocab_size * d_model * 2  # эмбеддинги
```

**Практические примеры:**
- GPT-3 175B: 96 layers, d_model=12288 → 96 * 12 * 12288² ≈ 174B (без vocab)
- LLaMA 7B: 32 layers, d_model=4096 → 32 * 12 * 4096² ≈ 6.4B

### Compute (FLOPs на токен)
```
FLOPs ≈ 6 * N * D  (для training: 2 forward + 4 backward)
FLOPs/token ≈ 6 * N
```

---

## Chinchilla vs "compute-optimal"

### До Chinchilla (Kaplan)
- Тренировать большую модель, мало токенов
- GPT-3: 175B параметров, 300B токенов (недообученная)

### После Chinchilla
- Меньше параметров, больше токенов = тот же compute = лучше
- Chinchilla (70B параметров, 1.4T токенов) > Gopher (280B, 300B токенов)

### "Over-training" тренд (LLaMA 1, Mistral)
- LLaMA 7B обучен на 1T+ токенов (>>chinchilla optimal)
- Цель: хороший inference-compute, не оптимальный train-compute
- Мотивация: inference дороже в production, чем training

---

## MoE и Scaling Laws

- MoE модели эффективнее по FLOPs: при тех же вычислениях больше параметров
- Switch Transformer: scaling MoE следует похожим законам
- Mixtral 8x7B ≈ активных параметров 12-14B, но качество как у ~40B dense

---

## Типичные вопросы

### Q: Сколько GPU нужно для тренировки LLM?
**A:** FLOPs = 6*N*D. GPU время = FLOPs / (GPU_FLOPS * utilization). Для LLaMA 7B на 1T токенов: 6 * 7e9 * 1e12 = 4.2e22 FLOPS. A100 даёт ~300 TFLOPS при bf16 → 4.2e22 / 3e14 ≈ 1.4e8 GPU секунд ≈ 1.6 GPU лет ≈ ~1700 A100 за 1 месяц.

### Q: Зачем тренировать "сверх Chinchilla"?
**A:** Chinchilla оптимизирует training compute. Но inference стоит дороже в production (много запросов). Меньшая но хорошо обученная модель имеет тот же inference compute что крупная недообученная, при меньшем memory footprint.

---

## Частые ошибки
- ❌ Думать что больше параметров = всегда лучше
- ❌ Не знать что Chinchilla опровергла Kaplan по поводу оптимального N/D баланса
- ❌ Путать training FLOPs и inference FLOPs
