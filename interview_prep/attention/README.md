# Attention Mechanisms

## Multi-Head Self-Attention (базовый)

### Формулы
```
Q = X @ W_Q,  K = X @ W_K,  V = X @ W_V
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V

MultiHead(Q,K,V) = Concat(head_1, ..., head_h) @ W_O
head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
```

**Зачем делить на sqrt(d_k)?**
- При больших d_k dot products имеют большую дисперсию → softmax уходит в saturated region → градиенты vanish
- Деление нормирует дисперсию к ~1

### Causal (decoder) attention
- Маска: заполняем -inf над главной диагональю QK^T (верхний треугольник)
- После softmax: token i может attend только к позициям j ≤ i
- При обучении: **один forward pass** обрабатывает всю последовательность параллельно (teacher forcing)
- При inference: авторегрессивная генерация, токен за токеном

### KV Cache при inference
- Сохраняем K и V для всех предыдущих токенов
- При генерации нового токена: только вычисляем Q для нового токена, K/V берём из кэша
- Memory: O(L * d * layers) — растёт линейно с длиной контекста

---

## Эволюция архитектур

### Multi-Query Attention (MQA, Shazeer 2019)
- Все головы Q отдельные, но K и V **общие для всех голов** (один набор K, один V)
- Снижение KV cache: в H раз (где H — число голов)
- Небольшое падение качества

### Grouped Query Attention (GQA, Ainslie et al. 2023)
- G групп: каждая группа из H/G голов Q делит одну пару K, V
- MHA (H=G) → GQA (1 < G < H) → MQA (G=1)
- LLaMA 2/3, Mistral, Gemma используют GQA
- Лучший компромисс качество/скорость чем MQA

### Multi-head Latent Attention (MLA, DeepSeek-V2)
- Вместо H пар K/V хранит **один низкоразмерный latent вектор** c (c << d*H)
- K, V разворачиваются из c через матрицы проекций при inference
- KV cache = только c (маленький), вычисление KK/VV online
- Сокращение KV cache в ~13.5x при DeepSeek-V2 (d=2048→c=512)

---

## Flash Attention (Dao et al. 2022)

### Проблема стандартного attention
- QK^T матрица N×N — квадратичная по памяти: O(N²) для N = seq_len
- Нужно записать в HBM (медленная глобальная память GPU) и прочитать обратно
- Bottleneck: HBM bandwidth, не FLOPS

### Иерархия памяти GPU
```
SRAM (регистры + L1 cache):   ~20 MB,   быстрая (~19 TB/s)
HBM (High Bandwidth Memory):  ~80 GB,   медленнее (~2 TB/s)
CPU RAM / NVMe:                терабайты, очень медленно
```

### Flash Attention решение: Tiling + Online Softmax
1. Разбиваем Q, K, V на блоки (tiles), помещающиеся в SRAM
2. Для каждого блока вычисляем частичный attention **не материализуя полную N×N матрицу**
3. Используем **online softmax trick**: накапливаем `max` и `sum` для правильной нормализации
4. Записываем в HBM только финальный результат O (не промежуточные N×N матрицы)

**Результат:**
- Memory: O(N) вместо O(N²)
- Speed: 2-4x быстрее за счёт снижения HBM read/writes
- Exact attention (не приближение)

### Flash Attention 2 (2023)
- Улучшена параллелизация по batch и head dimensions
- Оптимизированы warps для снижения sync overhead
- 2x быстрее FA1

### Flash Attention 3 (2024)
- Использует H100 Tensor Core асинхронные операции
- WGMMA (warpgroup matrix multiply) + асинхронный softmax

---

## KV Cache Management

### Ring Attention
- Для очень длинных контекстов (миллионы токенов)
- KV cache разбивается на блоки, каждый GPU держит свой блок
- Q с одного GPU "проходит кольцо" через все GPU, собирая attention от всех KV блоков
- Параллелизм по sequence length

### KV Cache Eviction (SnapKV, StreamingLLM)
- Не все KV нужны: часть можно удалить без потери качества
- StreamingLLM: sink attention (первые токены) + sliding window
- SnapKV: обнаружение важных KV по attention patterns

### Paged Attention (vLLM)
- KV cache хранится не contiguously, а в страницах (как OS paging)
- Устраняет фрагментацию памяти при variable-length sequences

---

## Типичные вопросы на интервью

### Q: Почему attention O(N²) и как это решают?
**A:** QK^T матрица N×N. Решения: (1) Flash Attention — IO-efficient без приближений, (2) Linear Attention — аппроксимация через kernel trick O(N), (3) GQA/MQA — снижают KV cache, (4) Sparse attention (Longformer, BigBird) — только локальные + глобальные attention.

### Q: Объясни Flash Attention простыми словами
**A:** Стандартный attention записывает N×N матрицу в медленную GPU память (HBM) и читает обратно — это медленно. Flash Attention считает всё блоками в быстрой SRAM, используя онлайн-нормализацию softmax. Матрица N×N никогда не записывается в HBM → меньше IO → быстрее.

### Q: Что такое GQA и зачем?
**A:** В MHA каждая голова имеет свои K и V → большой KV cache. В GQA несколько голов Q делят одну пару K/V → cache уменьшается в G раз. Незначительное падение качества при существенном ускорении inference.

### Q: В чём суть MLA (Multi-head Latent Attention)?
**A:** Вместо хранения K, V для каждой головы хранится маленький latent вектор c. K и V разворачиваются из c при inference. KV cache уменьшается радикально (например, в 13x), но нужно немного compute для развёртки.

---

## Частые ошибки
- ❌ Путать HBM и SRAM в контексте Flash Attention (HBM = медленная, SRAM = быстрая)
- ❌ Думать что Flash Attention = аппроксимация (нет, exact attention)
- ❌ Забывать про sqrt(d_k) нормировку и не знать зачем она
- ❌ Не знать разницу MHA → GQA → MQA → MLA
