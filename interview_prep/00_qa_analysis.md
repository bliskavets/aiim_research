# Анализ мок-интервью: AI Researcher @ Nebius
> Интервьюер: Alexander Golubev (RL для LLM agents, SWE-rebench, DAPO)

---

## Вопрос: Что такое токенизаторы и зачем они нужны?

**Ответ кандидата:** Токенизаторы разбивают текст на числовые представления. Нужны для уменьшения размера словаря по сравнению с word-level подходом. Упомянул BPE: начинаем с символов, итеративно объединяем наиболее частые пары, добавляем в словарь. Упомянул несколько типов: BPE, character-level, и др.

**Оценка:** ⚠️

**Что дополнить:**
- Не объяснил разницу между BPE, WordPiece, Unigram LM (SentencePiece), TikToken
- Не упомянул, что BPE работает на уровне байт (Byte-level BPE) в современных моделях (GPT-2+)
- Не объяснил ключевую проблему: OOV (out-of-vocabulary) → в byte-level BPE это решается автоматически
- Не упомянул `[UNK]` токен vs. fallback на байты

---

## Вопрос: Почему GPT-4 токенизирует тот же текст в меньшее количество токенов, чем Qwen?

**Ответ кандидата:** Разные обучающие корпуса → разные merge rules → разные словари. Если токенизатор обучен на английском, английские слова будут целыми токенами. Китайский токенизатор будет плохо покрывать английский и наоборот. Упомянул Unicode vs. ASCII (многобайтность CJK символов).

**Оценка:** ⚠️

**Что дополнить:**
- Ключевой ответ: **размер словаря (vocabulary size)** — при большем vocab model покрывает больший контекст одним токеном
- **Алгоритм разбиения** — greedy longest match vs. другие стратегии дают разное количество токенов на одном словаре
- Unicode: CJK символы = 3 байта → при byte-level BPE нужно больше merge операций для покрытия; английские ASCII символы = 1 байт → быстрее объединяются в слова
- Следствие: **Fertility** (токенов на слово) выше для языков с большими Unicode символами при том же vocab size

---

## Вопрос: Опишите шаги обучения BPE токенизатора

**Ответ кандидата:** Начинаем с Unicode/ASCII символов. Вычисляем частоту всех пар соседних токенов. Берем самую частую пару, добавляем как новый токен в словарь. Повторяем до достижения нужного размера словаря.

**Оценка:** ✅

**Что дополнить:**
- Корректно в целом
- Не упомянул, что после merge нужно **обновить** все вхождения этой пары в корпусе (не просто добавить в словарь)
- Не упомянул pre-tokenization (split по пробелам/регулярным выражениям) перед BPE — это важно для GPT-2 TikToken
- Стоит упомянуть отличие от **WordPiece** (maximizes likelihood вместо frequency)

---

## Вопрос: Что такое позиционные кодировки и как они эволюционировали?

**Ответ кандидата:** Без позиционных кодировок attention не учитывает порядок токенов (всё работает как bag-of-words). Исходно: sinusoidal (BERT) и learned (GPT). Позже пришли к RoPE — ротационные кодировки через комплексные числа, пары измерений вращаются на разные углы, что кодирует относительные позиции. Упомянул ALiBi.

**Оценка:** ✅

**Что дополнить:**
- Не объяснил точную формулу RoPE: q' = q * e^(iθm), k' = k * e^(iθn), тогда dot product = f(m-n)
- Не упомянул **relative position bias** (T5) как промежуточный шаг
- Хорошо бы добавить: RoPE хорош тем, что attention score зависит **только от разности позиций** (m-n), а не от абсолютных

---

## Вопрос: Почему без позиционных кодировок attention работает как bag-of-words?

**Ответ кандидата:** Если переставить токены местами, веса внимания не изменятся (при отсутствии позиционного сигнала). Привел пример с "комаром" — меняем порядок слов в русском предложении, смысл меняется, но без pos encoding модель этого не заметит.

**Оценка:** ✅

**Что дополнить:**
- Математически: attention score = softmax(QK^T/√d)V. Если Q и K не содержат позиционной информации, то permutation Q и K даст permuted результат V — т.е. выход просто permuted, а не изменён. Модель "не видит" что что-то поменялось.
- Хорошо объяснить это одним предложением: "Self-attention — это взвешенная сумма значений, инвариантная к перестановке, если нет позиционных кодировок"

---

## Вопрос: Как работают sinusoidal позиционные кодировки?

**Ответ кандидата:** Для каждой позиции и каждого измерения вектора — отдельное значение sin/cos с разными частотами. Формулы не вспомнил точно, но правильно упомянул разные частоты для разных измерений.

**Оценка:** ⚠️

**Что дополнить:**
- Формула: PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
- Низкие i → высокая частота (быстро меняются с позицией) → кодируют локальные различия
- Высокие i → низкая частота → кодируют глобальные позиции
- Ключевое свойство: PE(pos+k) можно выразить через линейную функцию от PE(pos) → позволяет модели обобщаться на разные относительные позиции

---

## Вопрос: Context length extension — NTK scaling и другие методы

**Ответ кандидата:** Описал NTK scaling — масштабирование базы углов в RoPE. Если обучали на 4K контексте, можно масштабировать для работы на 16K без дообучения. Потеря: локальная чувствительность (близкие токены хуже различаются). Описал практику: обучение начинают с короткого контекста, в последних шагах увеличивают.

**Оценка:** ⚠️

**Что дополнить:**
- NTK scaling: base = base * (L_new/L_train)^(d/(d-2)), где d — размерность RoPE
- Ключевая интуиция: мы "сжимаем" углы так, чтобы позиции в новом диапазоне [0, L_new] отображались на тот же угловой диапазон, что [0, L_train]
- NTK работает без fine-tuning, но "Dynamic NTK" (применять NTK только к длинным последовательностям) работает лучше
- Упомянул бы YaRN отдельно (не просто NTK, а нелинейное масштабирование разных частот)

---

## Вопрос: YaRN — что это и как работает?

**Ответ кандидата:** YaRN использует нелинейное масштабирование углов, сохраняя локальные отношения (близкие позиции почти не масштабируются), дальние — масштабируются сильнее. Есть параметры β и нелинейная функция f. Аналогия: как если растягивать карту, сохраняя детали в нужных областях.

**Оценка:** ⚠️

**Что дополнить:**
- Формально: YaRN разделяет измерения на три группы — low frequency (интерполяция), high frequency (экстраполяция), средние (смесь через β)
- Параметр α определяет порог между high/mid/low frequency
- Параметр β — плавный переход (lerp)
- Дополнительно: YaRN включает temperature scaling для нормализации attention distribution при длинных контекстах

---

## Вопрос: Разница между T5 (encoder-decoder) и GPT-2 (decoder-only)

**Ответ кандидата:** T5 — encoder-decoder, GPT-2 — decoder-only. T5 обрабатывает весь вход через encoder (bidirectional attention), decoder генерирует с cross-attention к encoder states. GPT-2 — stack decoder layers с causal mask.

**Оценка:** ✅

**Что дополнить:**
- Важное следствие: encoder видит весь контекст двунаправленно (лучше для понимания), decoder — только предыдущие токены (лучше для генерации)
- T5 использует relative position bias вместо абсолютных positional encodings
- GPT использует causal (upper triangular) mask в attention
- Для задач извлечения/классификации encoder лучше; для генерации decoder-only или encoder-decoder

---

## Вопрос: Как работает multi-head self-attention во время обучения (forward pass)?

**Ответ кандидата:** Матрица эмбеддингов → три проекции (Q, K, V) → Q*K^T → attention scores → softmax → взвешенная сумма V. Упомянул, что нужен один forward pass для NTP loss, но при memory constraints — gradient checkpointing.

**Оценка:** ✅

**Что дополнить:**
- Стоит упомянуть masking: padding mask + causal mask
- Масштабирование на √d_k важно: без него градиенты vanish через softmax при больших d_k
- Multi-head: H голов = H параллельных attention, concat + projection W_O

---

## Вопрос: Что изменилось в attention за последние годы?

**Ответ кандидата:** Multi-Latent Attention (MLA), Flash Attention, pre-layer normalization, RMSNorm, GQA (grouped query attention), MQA (multi-query attention), KV cache eviction (SnapKV), Ring Attention, Linear Attention (Kimi).

**Оценка:** ✅

**Что дополнить:**
- Хорошо структурировать по категориям: эффективность памяти (GQA/MQA/MLA), скорость вычислений (Flash Attention), длинный контекст (Ring Attention, Linear Attention)
- MLA (DeepSeek): хранит low-rank compressed KV (latent vector), разворачивает при inference → меньше KV cache при том же качестве

---

## Вопрос: Как работает Flash Attention?

**Ответ кандидата:** Использует иерархию GPU памяти (SRAM/L2/HBM). Не материализует полную matрицу softmax(QK^T). Все вычисления в быстрой памяти (назвал HBM ошибочно, но потом поправился — нужен SRAM). Уменьшает memory IO и latency.

**Оценка:** ⚠️

**Что дополнить:**
- Уровни памяти GPU: **SRAM** (самая быстрая, ~20MB) → **HBM** (High Bandwidth Memory, ~80GB, медленнее) → CPU RAM
- Flash Attention: tiling — разбиваем Q, K, V на блоки, обрабатываем в SRAM, используем online softmax trick для накопления результата без материализации полной N×N матрицы
- Результат: memory O(N) вместо O(N²), и быстрее за счёт снижения HBM reads/writes
- Flash Attention 2: улучшила параллелизм по batch и head dimension

---

## Вопрос: Mixture of Experts — что это и зачем?

**Ответ кандидата:** Идея Хинтона: разные сети решают разные задачи. В LLM — FFN слои заменены на набор экспертов с gating network. Router (gating) решает, какие эксперты активировать для каждого токена. Экспертизация расширяет модель в ширину. Inference — только топ-K экспертов.

**Оценка:** ⚠️

**Что дополнить:**
- Не упомянул проблему **load balancing** — без специального лосса все токены идут к одним экспертам (collapse)
- Auxiliary loss для балансировки: importance loss + load loss
- **Expert parallelism** — эксперты распределяются по GPU, нужен all-to-all communication
- Token dropping при переполнении capacity

---

## Вопрос: Как FFN хранит фактуальные знания? Как это проверить?

**Ответ кандидата:** Квантование FFN слоёв → падение фактуальной точности, галлюцинации. Attention слои — меньше влияют на факты, больше на структуру. Можно также менять размер FFN hidden dimension и смотреть на изменения.

**Оценка:** ⚠️

**Что дополнить:**
- Ключевые работы: "Locating and Editing Factual Associations in GPT" (ROME), "Knowledge Neurons in Pretrained Transformers"
- FFN работают как "key-value memories": ключ (первая матрица) активирует, значение (вторая матрица) вспоминает
- Методы локализации: causal tracing, activation patching, knowledge editing (ROME/MEMIT)

---

## Вопрос: MoE при сервинге vs. dense модели — разница?

**Ответ кандидата:** MoE хранит все параметры в памяти, но активирует только часть при inference. Expert offloading на CPU. All-reduce коммуникации между GPU растут с количеством экспертов. Современные фреймворки (vLLM, TensorRT) поддерживают оба типа, но с нюансами оптимизаций.

**Оценка:** ⚠️

**Что дополнить:**
- Ключевое: MoE имеет **higher memory footprint** но **same FLOPs per token** (как dense модель с размером одного эксперта × top-k)
- All-to-all коммуникация (expert parallelism) vs all-reduce (tensor parallelism) — разные паттерны
- При serving: MoE-модели хуже используют GPU при маленьком batch size (sparse activation = idle cores)
- Expert caching strategies для уменьшения latency

---

## Вопрос: Эволюция RL алгоритмов для LLM post-training

**Ответ кандидата:** SFT → RLHF с PPO (2017, OpenAI, actor-critic) → rejection sampling → GRPO (DeepSeek). Упомянул Bradley-Terry objective для reward model. PPO имеет critic instability проблемы. GRPO убрал critic, стабилизировал через group-based advantage.

**Оценка:** ⚠️

**Что дополнить:**
- Правильный порядок: SFT → RLHF (PPO, 2022 InstructGPT) → DPO (2023) → GRPO/DAPO (2024-2025)
- Rejection sampling fine-tuning (ReST, STaR) — важный промежуточный метод
- DPO: offline, без reward model, оптимизирует preference напрямую через Bradley-Terry
- GRPO: group sampling, advantage = (r - mean(r)) / std(r) в группе

---

## Вопрос: Проблемы PPO для LLM

**Ответ кандидата:** Нестабильность из-за critic (value model), reward hacking, проблема шума в Монте-Карло оценках advantage, GAE для стабилизации. Проблема при sparse rewards.

**Оценка:** ✅

**Что дополнить:**
- KL divergence constraint между current и reference policy — предотвращает reward hacking
- Critic обучается параллельно с actor → train instability, особенно при длинных rollouts
- Memory overhead: нужно держать 4 модели (actor, critic, ref, reward)
- Sparse reward = высокая variance в advantage estimates

---

## Вопрос: Когда PPO хуже простого SFT?

**Ответ кандидата:** При маленьком датасете SFT (LoRA/Ptuning) лучше. При хорошо определённых задачах с чёткими данными — SFT достаточно. PPO нужен для творческих/неоднозначных задач.

**Оценка:** ⚠️

**Что дополнить:**
- Конкретный канонический случай: **distribution collapse** — PPO может найти способы fool reward model и деградировать
- **Overthinking/verbosity**: модель учится генерировать длинные ответы ради более высокого reward
- При **dense rewards** (каждый токен получает сигнал) PPO работает лучше; при sparse (только финальный reward) — хуже

---

## Вопрос: Где ещё применим PPO в 2026?

**Ответ кандидата:** Для safety/alignment. Выбор между PPO и DPO/GRPO трудный — зависит от задачи. DPO стабильнее, GRPO для reasoning. Нужно анализировать конкретный случай.

**Оценка:** ⚠️

**Что дополнить:**
- PPO остаётся актуальным для **online RL с верификатором** (code execution, math checking)
- Для задач, где reward model нестабилен, PPO с KL constraint лучше удерживает качество
- PPO с process reward model (PRM) вместо outcome reward model (ORM) — активная область

---

## Вопрос: Проблемы GRPO (DPO?)

**Ответ кандидата:** Дисбаланс длин последовательностей (длинные и короткие получают разный градиентный сигнал). Проблема language collapse (DeepSeek думал на неизвестном языке). Нужен balance позитивных/негативных примеров в батче.

**Оценка:** ✅

**Что дополнить:**
- DAPO решает length bias через **token-level policy gradient** вместо sequence-level
- Clip-higher trick в DAPO: asymmetric clipping для стимуляции exploration
- Entropy bonus для предотвращения mode collapse
- Dynamic sampling: фильтрация промптов, где все N ответов одинаково правильны/неправильны (нет контраста)

---

## Вопрос: Pipeline для post-training code-assistant модели с нуля

**Ответ кандидата:** Анализ требований (code review в CI/CD, diverse coding tasks). Начал описывать pipeline, но транскрипт обрывается.

**Оценка:** ⚠️

**Что дополнить:**
- Полный pipeline:
  1. **Data curation**: code repos (GitHub), synthetic data (self-play, execution feedback)
  2. **SFT**: на curated code instructions (CodeAlpaca, OSS-Instruct, Magicoder)
  3. **Reward modeling** или **verifiable rewards** (unit tests, compilation)
  4. **RL**: GRPO/DAPO с execution-based reward → именно это исследует Golubev
  5. **Evaluation**: HumanEval, MBPP, SWE-bench, LiveCodeBench
- Ключевые инсайты для Golubev: SWE-bench требует agent-level RL (multi-step), а не просто one-shot generation
