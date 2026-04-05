# Mixture of Experts (MoE)

## Основная идея

Dense модель: **каждый токен** проходит через **все FFN параметры**

MoE модель: **каждый токен** проходит через **только K из N экспертов** (обычно K=2 из N=8 или N=64)

```
Dense: output = FFN(x)  — все параметры активны
MoE:   output = Σ g_i(x) * Expert_i(x)  — только top-K экспертов активны
где g = softmax(Router(x))[:, top_k]  — sparse gating
```

### Структура
```
Transformer Block:
  - Multi-Head Attention  (одинаково для dense и MoE)
  - FFN → заменяется на MoE Layer:
      - Router (маленькая линейная сеть)
      - N экспертов (каждый = обычный FFN)
      - Top-K selection
```

---

## Ключевые параметры

- **N** — число экспертов (обычно 8, 16, 64, 256)
- **K** — активных экспертов на токен (обычно 2)
- **Expert capacity** — максимум токенов на эксперта в батче (capacity factor × batch_size/N)

**Пример:** Mixtral 8x7B
- 8 экспертов, каждый ~7B параметров по FFN
- Top-2 activation → активных параметров ~14B из ~46B
- Сравним с активированием: как 14B dense, качество ~как 70B

---

## Routing (Gating)

### Standard Top-K Routing
```python
logits = x @ W_router  # [batch, n_experts]
gates = softmax(logits)
top_k_gates, top_k_idx = topk(gates, k=K)
top_k_gates = top_k_gates / top_k_gates.sum()  # renormalize
```

### Load Balancing Problem
Без специального лосса: все токены идут к 1-2 "популярным" экспертам → остальные не обучаются → **expert collapse**

**Auxiliary Loss (Switch Transformer):**
```
L_balance = alpha * N * Σ_i (f_i * P_i)
где f_i = доля токенов, routing к эксперту i
    P_i = средняя вероятность routing к эксперту i
```

**DeepSeek MoE innovations:**
- Fine-grained experts (много мелких вместо нескольких крупных)
- Shared experts (всегда активны) + routed experts
- Device-limited routing (каждый токен только к экспертам на M устройствах)

---

## Обучение MoE

### Параллелизм
- **Expert Parallelism**: разные эксперты на разных GPU
- Проблема: нужен **All-to-All communication** — токены отправляются к GPU с нужным экспертом

```
Usual:       GPU1 → attention → GPU1 (tensor parallel)
MoE routing: GPU1 → token x → GPU3 (где эксперт 3) → result → GPU1
```

- **All-to-All** — дорогостоящая операция, растёт с числом экспертов и GPU

### Token Dropping
- При переполнении capacity buffer: токены, которым "некуда идти", дропаются
- Dropout + capacity factor управляют этим компромиссом

---

## Inference с MoE

### Отличие от обучения
- **Обучение:** forward через **все** эксперты (backward нужен для всех через autograd)
  - Нет! На самом деле: вычисляем только top-K, остальные получают нулевые градиенты
  - Gradient только у активированных экспертов
- **Inference:** только top-K экспертов — меньше FLOPS

### Сервинг MoE
- Все веса нужно держать в памяти (даже неактивных экспертов)
- При малом batch size: слабое использование GPU (sparse activation)
- **Expert offloading**: неактивные эксперты на CPU/NVMe, загружаем по требованию
- Latency ↑ из-за CPU↔GPU transfer, но throughput может быть OK

---

## Знания в FFN

### "FFN как key-value memory" (Geva et al., 2021)
- Первая матрица FFN (W1) работает как "keys" — паттерны активации
- Вторая матрица (W2) как "values" — что выдавать при активации ключа
- Квантование FFN → потеря фактуальных знаний
- Квантование Attention → потеря связности/структуры, но фактуальность сохраняется

### Проверка знаний
- Causal tracing (ROME): отключать слои по одному, смотреть на изменение вероятности конкретного факта
- Нейроны памяти: некоторые FFN-нейроны специализированы на конкретных фактах
- Knowledge editing (ROME/MEMIT): целевое изменение фактов через хирургическую правку FFN

---

## Scaling Laws для MoE

- При одинаковом числе **активных параметров** MoE ≈ Dense по качеству inference
- При одинаковом **числе полных параметров** MoE лучше Dense (больше экспертов = больше ёмкость)
- **Training FLOPs** для MoE ≈ Dense с размером = активным параметрам × данные
- Mixtral 8x7B: ~14B активных = FLOPs как у 14B dense, качество как у ~40-70B dense

---

## Типичные вопросы на интервью

### Q: В чём главное преимущество MoE перед dense?
**A:** При том же числе активных параметров (= те же FLOPs на токен) MoE имеет больше параметров в целом → больше ёмкость для знаний → лучше качество. По сути: дешевле inference при том же качестве.

### Q: Какая главная проблема при обучении MoE?
**A:** Load balancing — без специального лосса одни эксперты перегружены, другие не учатся. Решение: auxiliary loss + capacity factor + token dropping.

### Q: Почему MoE сложнее сервировать чем dense?
**A:** (1) Все параметры в памяти даже при малом использовании, (2) All-to-All communication для expert routing, (3) При малом batch size низкая утилизация GPU из-за sparse activation.

### Q: Как FFN хранит фактуальные знания?
**A:** FFN работает как key-value memory: первый слой активирует паттерны (ключи), второй выдаёт ассоциированные значения. Квантование FFN → фактуальные ошибки. Проверяется causal tracing.

---

## Частые ошибки
- ❌ Думать что при inference нужны все эксперты (нет, только top-K)
- ❌ Путать total params и active params у MoE
- ❌ Не знать про load balancing loss
- ❌ Путать expert parallelism (all-to-all) и tensor parallelism (all-reduce)
