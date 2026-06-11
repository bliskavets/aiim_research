---
marp: true
title: RL-методы пост-тренинга LLM — GRPO и CISPO
author: ""
paginate: true
math: katex
theme: default
style: |
  section {
    font-size: 26px;
    padding: 48px 64px;
  }
  h1 { color: #1a3c6e; }
  h2 { color: #1a3c6e; }
  table { font-size: 22px; }
  code { font-size: 0.85em; }
  .small { font-size: 20px; color: #555; }
  .tag { color: #b5651d; font-weight: 600; }
---

<!-- _paginate: false -->

# RL-методы пост-тренинга LLM

## Сравнение **GRPO** и **CISPO**

<br>

Policy-gradient методы для обучения reasoning-моделей с подкреплением

<span class="small">Источники: DeepSeekMath (GRPO), MiniMax-M1 (CISPO), ms-swift docs</span>

---

## Зачем нужны эти методы

- После SFT модель дообучают с RL на **проверяемых наградах** (математика, код, reasoning).
- Базовый PPO требует отдельную **value-модель (critic)** — дорого и нестабильно на длинных цепочках рассуждений.
- Идея семейства методов: убрать critic, оценивать действия **относительно группы** сэмплов.

<br>

**Эволюция:** PPO → **GRPO** → DAPO → **CISPO**

> Каждый шаг борется со своей проблемой: стоимость critic'а, обрезка градиентов, потеря редких токенов.

---

## GRPO — идея

**Group Relative Policy Optimization** (DeepSeekMath, 2024)

- На каждый промпт $q$ сэмплируем **группу** из $G$ ответов $\{o_1, \dots, o_G\}$.
- Награды $r_i$ нормируем внутри группы → получаем advantage **без critic'а**:

$$
\hat{A}_{i,t} = \frac{r_i - \operatorname{mean}(\mathbf{r})}{\operatorname{std}(\mathbf{r})}
$$

- Группа сама себе служит **baseline** → меньше дисперсия, нет отдельной value-сети.
- Обновление политики — по схеме PPO с **клиппингом ratio** + KL-штраф к референсной модели.

---

## GRPO — целевая функция

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\min\Big( r_{i,t}\,\hat{A}_{i,t},\ \operatorname{clip}(r_{i,t},\, 1-\varepsilon,\, 1+\varepsilon)\,\hat{A}_{i,t} \Big)\Bigg]
- \beta\, D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})
$$

где importance-ratio

$$
r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}\mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q, o_{i,<t})}
$$

- $\min(\cdot,\operatorname{clip}(\cdot))$ — **trust region**: ограничивает шаг обновления политики.
- $\beta D_{\text{KL}}$ — удерживает модель рядом с референсной.

---

## GRPO — слабое место

Клиппинг применяется к **ratio самого токена**. Если для токена ratio выходит за $[1-\varepsilon,\,1+\varepsilon]$ → его **градиент зануляется**.

<br>

🔴 Проблема для reasoning:

- Редкие, но **критичные «развилочные» токены** — *«However», «Wait», «Aha», «Recheck»* — имеют низкую вероятность.
- У них большой ratio → они первыми попадают под обрезку и **выпадают из обучения**.
- Именно эти токены меняют ход рассуждения. Их потеря → «логическая путаница» на длинных цепочках.

> GRPO стабилизирует обучение ценой выбрасывания самых информативных токенов.

---

## CISPO — идея

**Clipped IS-weight Policy Optimization** (MiniMax-M1, 2025)

Ключевая смена точки клиппинга:

| | Что клиппируется |
|---|---|
| GRPO / PPO | обновление **токена** (через $\min$/clip ratio) |
| **CISPO** | сам **importance-sampling вес**, а не токен |

- Клип кладётся **внутрь stop-gradient** → вес ограничен по величине, но **градиент по $\log\pi_\theta$ течёт для каждого токена**.
- Отказ от trust-region-обрезки: **ни один токен не выбрасывается**, включая редкие reasoning-токены.
- Энтропия всё равно остаётся в разумных пределах → стабильное исследование.

---

## CISPO — целевая функция

$$
\mathcal{J}_{\text{CISPO}}(\theta) = \mathbb{E}\Bigg[\frac{1}{\sum_i |o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}
\operatorname{sg}\!\big(\hat{r}_{i,t}(\theta)\big)\,\hat{A}_{i,t}\,\log \pi_\theta(o_{i,t}\mid q, o_{i,<t})\Bigg]
$$

с **клиппированным IS-весом**

$$
\hat{r}_{i,t}(\theta) = \operatorname{clip}\big( r_{i,t}(\theta),\ 1-\varepsilon^{\text{IS}}_{\text{low}},\ 1+\varepsilon^{\text{IS}}_{\text{high}} \big)
$$

- $\operatorname{sg}(\cdot)$ — **stop-gradient** (`.detach()`): вес становится бескградиентным множителем.
- $\varepsilon^{\text{IS}}_{\text{high}}$ берут **большим** → больше свободы для апдейтов редких токенов.
- Градиент токена ≠ 0 никогда → редкие «fork»-токены сохраняются.

---

## CISPO — псевдокод

```python
log_ratio        = per_token_logps - old_per_token_logps
importance_weights = torch.exp(log_ratio)              # r = π_θ / π_old
clamped_ratios   = torch.clamp(importance_weights,
                               max=epsilon_high).detach()  # sg(clip(r))
per_token_loss   = -clamped_ratios * advantages.unsqueeze(1) * per_token_logps
```

<span class="small">Клип ограничивает только <b>множитель</b>; <code>log π_θ</code> всегда остаётся обучаемым — поэтому ни один токен не «выпадает».</span>

---

## GRPO vs CISPO — сводка

| | **GRPO** | **CISPO** |
|---|---|---|
| Critic / value-сеть | нет (group baseline) | нет (group baseline) |
| Advantage | нормировка по группе | нормировка по группе |
| Что клиппируется | ratio **токена** (trust region) | **IS-вес** под stop-gradient |
| Судьба редких токенов | градиент **зануляется** | **сохраняются** все |
| KL-штраф | да, $\beta D_{KL}$ | опционально / убран |
| Риск | потеря reasoning-токенов | нужен подбор $\varepsilon^{IS}_{high}$ |

---

## CISPO — результаты (MiniMax-M1)

- 📈 **Превосходит GRPO и DAPO** по качеству на reasoning-бенчмарках.
- ⚡ Достигает уровня **DAPO за ~50% шагов** обучения (≈2× эффективность).
- 🎯 Более гладкие траектории обучения, выше sample-efficiency.
- 🧠 Особенно выигрывает на **длинных цепочках рассуждений**, где важны редкие токены.

> Главный вывод: проблема была не в размере шага, а в том, **какие токены вообще участвуют в градиенте**.

---

## GSPO — идея

**Group Sequence Policy Optimization** (Qwen, 2025)

Диагноз авторов: нестабильность GRPO — в **некорректном применении importance-веса на уровне токена**. Шум растёт с длиной ответа и усиливается клиппингом.

🔑 Решение: перенести всё на **уровень последовательности**, а не токена.

- Importance-ratio, клиппинг, награда и оптимизация — **по всему ответу целиком**.
- Ratio **нормируется по длине** (степень $1/|o_i|$) → гасит экспоненциальный разброс.
- Особенно стабилизирует **RL для MoE-моделей**. Лёг в основу улучшений **Qwen3**.

---

## GSPO — целевая функция

Sequence-level importance ratio с нормировкой по длине:

$$
s_i(\theta) = \left( \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)} \right)^{1/|o_i|}
$$

PPO-style клиппинг, но **по всей последовательности**:

$$
\mathcal{J}_{\text{GSPO}}(\theta) = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}
\min\Big( s_i(\theta)\,\hat{A}_i,\ \operatorname{clip}(s_i(\theta),\, 1-\varepsilon,\, 1+\varepsilon)\,\hat{A}_i \Big)\Bigg]
$$

- Advantage $\hat{A}_i$ — **один на весь ответ** (та же group-нормировка, что и в GRPO).
- Парадокс: GSPO обрезает **больше** токенов, но всё равно **обходит GRPO** → токен-уровневые градиенты GRPO просто более шумные.

---

## Tree-GRPO — идея

**Tree-based GRPO** (*Tree Search for LLM Agent RL*, 2025)

Заточен под **агентов** и многошаговый reasoning (multi-hop QA, tool-use).

- Роллауты сэмплируются не линейно, а **деревом**: узел = полный шаг взаимодействия агента, ветви = точки расхождения решений.
- **Общие префиксы переиспользуются** → за тот же бюджет токенов/tool-call'ов больше роллаутов.
- Из дерева строятся **пошаговые (process) сигналы** даже при наличии только финальной (outcome) награды.

---

## Tree-GRPO — advantage

Относительные advantage считаются на **двух уровнях**:

| Уровень | Что сравнивается |
|---|---|
| **Intra-tree** | ветви внутри одного дерева (общий префикс) |
| **Inter-tree** | разные деревья в группе (как в обычном GRPO) |

- Intra-tree объектив **эквивалентен step-level preference learning** (DPO-подобный сигнал на шагах).
- Иерархическое распространение награды по дереву → **дешевле** и точнее на длинных агентных траекториях.

---

<!-- Раздел-задел под будущие методы. Дублируйте блок ниже на каждый новый метод. -->

## Другие методы — *(задел)*

Место под следующие методы для сравнения:

- **PPO** — базовый actor-critic, отправная точка.
- **DAPO** — Decoupled clip + Dynamic sampling (развитие GRPO).

<span class="small">Слайды добавим на следующем шаге.</span>

---

## Семейство методов — где «живёт» клиппинг

| Метод | Уровень ratio | Что клиппируется | Фокус |
|---|---|---|---|
| **GRPO** | токен | ratio токена (trust region) | базовый group-RL |
| **CISPO** | токен | IS-вес под stop-grad | сохранить редкие токены |
| **GSPO** | **последовательность** | seq-ratio (норм. по длине) | стабильность, MoE |
| **Tree-GRPO** | токен/шаг | как GRPO, но по дереву | агенты, process-награда |

---

## Источники

- DeepSeekMath: *GRPO* — arXiv:2402.03300
- MiniMax-M1: *Scaling Test-Time Compute Efficiently* (CISPO) — arXiv:2506.13585
- Qwen: *Group Sequence Policy Optimization* (GSPO) — arXiv:2507.18071
- *Tree Search for LLM Agent RL* (Tree-GRPO) — arXiv:2509.21240
- ms-swift docs — *Clipped Importance Sampling Policy Optimization (CISPO)*
- EmergentMind — обзорные статьи GRPO / CISPO / GSPO / Tree-GRPO
