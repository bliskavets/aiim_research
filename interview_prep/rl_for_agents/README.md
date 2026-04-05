# RL для LLM Agents (SWE-bench, Code Agents)

## Контекст: почему это важно для интервью с Golubev

Alexander Golubev исследует:
- **SWE-rebench**: переработанный benchmark для software engineering tasks
- **DAPO**: применение к code agents
- **RL для LLM agents**: multi-step tasks с execution feedback

---

## Отличие Agent RL от One-Shot RL

### One-shot RL (например, code generation)
- Промпт → один ответ → reward (pass/fail тесты)
- GRPO, DAPO работают хорошо
- Короткий horizon

### Agent RL (SWE-bench style)
- Промпт → последовательность действий: [read file] → [edit] → [run tests] → [debug] → ...
- **Long-horizon**: десятки шагов
- **Sparse reward**: только финальный результат (тест pass/fail)
- **Partial observability**: агент видит только часть репозитория
- **Tool use**: bash, file editor, search

---

## SWE-bench

### Что это?
- GitHub issues из реальных open-source репозиториев (Django, scikit-learn, etc.)
- Задача: дан issue → сделать изменение в коде, которое его решает
- Оценка: patch проходит тест suite (unit tests)

### Метрики
- **Resolved rate**: % задач решённых корректно
- GPT-4 (2023): ~1.7% → Claude 3.5 Sonnet (2024): ~49% → лучшие агенты 2025: >50%

### SWE-bench Verified / SWE-rebench
- Часть задач в оригинальном SWE-bench имеют проблемы с тестами
- SWE-bench Verified: отфильтрованные задачи (500 задач, человечески верифицированные)
- **SWE-rebench**: переработка для более чистой оценки (Golubev et al.)

---

## RL Framework для Code Agents

### Reward Design
**Outcome-based (ORM):**
- +1 если все тесты прошли, 0 иначе
- Sparse → высокая variance
- Простота, нет reward hacking через reward model

**Process-based (PRM):**
- Награда на каждом шаге (правильно прочитал нужный файл? правильно идентифицировал баг?)
- Требует human annotation или proxy metrics

**Execution feedback:**
- Компиляция: +0.1
- Тесты: +r за каждый прошедший тест
- Синтаксис: -0.5 за syntax error

### Алгоритмы для агентов

**REINFORCE:**
- Простейший policy gradient
- Gradient: ∇J = E[G_t * ∇log π(a_t|s_t)]
- Высокая variance, но просто и работает

**PPO для агентов:**
- Каждый шаг — отдельный action
- GAE для advantage по шагам
- Проблема: длинные episodes = много памяти для rollouts

**GRPO для агентов:**
- G параллельных эпизодов из одного промпта
- Advantage нормируется внутри группы
- Работает для code gen, но сложнее для multi-step

### Вызовы multi-step RL
1. **Credit assignment**: какое действие привело к успеху?
2. **Exploration**: агент застревает в локальных оптимумах (открывает одни и те же файлы)
3. **Distribution shift**: при обучении агент видит свои ошибки, при inference — нет
4. **Reward sparsity**: репозиторий из 1000 файлов, нужно найти 2 строки

---

## Архитектуры Code Agents

### ReAct (Yao et al., 2022)
```
Thought: [рассуждение]
Action: bash("grep -r 'bug_function' src/")
Observation: [результат bash]
Thought: ...
```
- Interleaving reasoning и actions
- Базовая архитектура для большинства code agents

### SWE-agent (Yang et al., 2024)
- Специальный Agent-Computer Interface (ACI)
- Custom tools: file viewer, search, edit (с linting)
- Важность: хорошие tools >>> лучший LLM

### Moatless Tools
- Минимальный набор инструментов для SWE-bench
- Показал что простота tool design важна

---

## Ключевые инсайты для позиции

### "Training SWE-agents with RL" (2508.03501)
Что, скорее всего, обсуждается в статье:
- Как применить RL к multi-step code editing
- Curriculum learning: начинать с простых задач
- Reward shaping для file navigation
- Comparison с imitation learning (SFT на human trajectories)

### Практические приёмы
- **Curriculum**: сначала обучать на одно-файловых задачах, потом multi-file
- **Replay buffer**: сохранять успешные траектории для off-policy learning
- **Tool call reward**: небольшой reward за полезные инструменты
- **Format reward**: штраф за неправильный формат вызовов инструментов

---

## Типичные вопросы на интервью (с Golubev)

### Q: Как применить GRPO/DAPO к multi-step agent?
**A:** Основная проблема — credit assignment в длинных эпизодах. Подходы: (1) treat entire trajectory как один "ответ" с финальным reward, (2) промежуточные rewards за отдельные шаги, (3) Monte Carlo rollouts с group normalization. GRPO применим, если treat trajectory как sequence.

### Q: Что такое SWE-bench и почему он важен?
**A:** Real-world software engineering benchmark: GitHub issues → patch. Измеряет способность агента понимать codebase, найти баг, исправить. Более реалистичен чем HumanEval (синтетические задачи). Resolved rate = % решённых задач.

### Q: Какой reward function использовать для code agent RL?
**A:** Идеально — execution-based: запустить тесты, +1 за прошедший. Можно добавить process rewards: компиляция, частичное прохождение тестов. Reward shaping для навигации рискован (reward hacking).

### Q: Как избежать reward hacking в code agent?
**A:** (1) Использовать held-out тесты (агент не видит их во время обучения), (2) diversity bonus (разные подходы к решению), (3) KL constraint от reference policy, (4) отдельный verifier.

---

## Частые ошибки
- ❌ Думать что HumanEval ≈ SWE-bench (это принципиально разные задачи)
- ❌ Применять GRPO напрямую к multi-step без адаптации
- ❌ Не учитывать credit assignment при sparse reward
- ❌ Забывать про exploration в большом codebase
