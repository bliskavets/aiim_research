# Недостающие эксперименты для AAAI-27 (инструкции для Claude Opus 4.8)

Контекст: статья `papers/aaai27/aaai27_submission/main.tex`. Ишьюсы — `rebuttal/reviewer_pass2_issues.md`.
Журнал прошлых замеров — `rebuttal/results_aaai2027/EXPERIMENTS_LOG.md` (обязателен к прочтению: записи 19–32
описывают починку весов, движки, судью). Все прогоны логируй туда же (следующая запись — 33) и клади
компактные по-задачные артефакты в `rebuttal/results_aaai2027/per_problem_repro/`.

Общая инфраструктура:
- vLLM-сервер Qwen3-8B-FP8: `/root/aiim/.venv/bin/vllm serve Qwen/Qwen3-8B-FP8 --host 0.0.0.0 --port 9090 --max-model-len 32768 --gpu-memory-utilization 0.85` (веса починены 19.07 — НЕ перекачивать).
- «Движок статьи» (orig): env `SAGE_ORIG_NOTHINK=1`, код `/root/aiim_2/TTA/experiments/exp_10/math500/test_new_method.py`; канонический non-thinking: харнесс `/root/aiim_2/redflag_rerun/`.
- Семантический судья: env `MATH_EXACT_MATCH=0`, `OPENAI_BASE_URL=https://openrouter.ai/api/v1`, `MATH_CRITIC_MODEL=openai/gpt-4.1-mini`, ключ OpenRouter в `repro_out/night_queue.sh`.
- ЗАПРЕТ: не убивать процессы по паттерну (pkill/pgrep) — только по явному числовому PID из вывода ps.
- Правило отбора: результат идёт в статью только если он НЕ противоречит основным цифрам статьи в худшую сторону; иначе — только в лог.

## 1. [КРИТ, issue 9] AlpacaEval-2 официальным пайплайном
Цель: защитимые LC-числа вместо непроверяемых 35.4/63.0 (наш сабсет-замер N=200 дал base 43.0 → SAGE 48.5 — с колонкой статьи не бьётся).
Как: поставить пакет `alpaca_eval` в ОТДЕЛЬНЫЙ venv (не трогать /root/aiim_2/venv!). Прогнать Qwen3-8B base и SAGE-ответы
(генерация нашим харнессом, все 805 инструкций), судья и референс — дефолт лидерборда (weighted_alpaca_eval_gpt4_turbo).
Ответы SAGE брать канонический non-thinking (`redflag_rerun`, alpaca-конфиг из кампании).
Успех: SAGE > base по LC. Тогда заменить AlpacaEval-колонку честными числами (жёлтым) и убрать footnote о несравнимости.
Провал/нет бюджета: оставить текущий protocol-footnote (уже в статье) — ничего не менять.

## 2. [КРИТ, issue 1] Провенанс Table 4 (held-out subset seed=42)
Цель: подтвердить или заменить строки tab:small-large-models (8B baseline 0.61 / SAGE 0.97; 1.7B ряды BoN N=21/70).
Как: выяснить, какой сабсет даёт baseline 0.61 при seed=42 (проверить selection в старом коде TTA exp_10; ноутбуки в репо TTA).
Если сабсет восстановим — перегнать 8B baseline и SAGE на нём orig-движком (50 задач ≈ 1.5 ч). BoN-ряды 1.7B (FsfairX и
Skywork-V2-Llama-3.1-8B, N=21/70) — дороже (~4-6 ч, RM на GPU рядом, gpu-mem-util сервера снизить до 0.55).
Успех: числа ±2-3 пт от таблицы — добавить в caption протокол. Провал: заменить строки перемеренными числами (жёлтым).

## 3. [СРЕДН, issue 12 / R3-Q1] SPO same-model оптимизатор на здоровых весах
Прошлый ран (лог, запись 18) шёл на порченых весах — вывод «best=seed» надо перепроверить.
Как: официальный SPO (XiangJinyu/SPO), optimizer/evaluator/executor = Qwen3-8B через наш vLLM, max-rounds 6 на MATH,
затем eval лучшего промпта на MATH-500 (судья 4.1-mini). ~4-6 ч.
Успех (для статьи): SPO не улучшает seed-промпт → короткая жёлтая фраза в аппендикс («SPO's gains depend on a strong
external optimizer; with the same base model as optimizer, prompt search did not beat the seed prompt in six rounds»),
это защищает низкие SPO-числа Table 1. Если SPO улучшит — в лог, в статью не носить.

## 4. [СРЕДН, issue 16] Перенос порога 75% на независимый датасет
Цель: снять претензию «порог тюнился на eval-задаче».
Как: NDCG-свип порога X∈{50,65,75,85,95} на 50 задачах AGIEval-Math (или GSM8K-hard) на iteration-0 кандидатах,
аналогично Appendix F (скрипт свипа искать в TTA exp_10; вход — judge-скоры 21 кандидата).
Успех: оптимум ~75 → заменить оговорку в Appendix F на «optimum transfers to an out-of-domain math set». Провал: оставить
текущую формулировку («selected once, reused unchanged») — она уже в статье.

## 5. [БЛОКЕР ДЛЯ 32B] Починка весов Qwen3-32B-FP8
`fix_32b.sh` готов (4 битых шарда; см. запись 20 в логе). Требует ~60 ГБ скачивания. После починки: перегнать
tab:qwen32b (baseline 0.80 / SAGE 0.97, сабсет seed=42) для валидации. Если числа не подтвердятся —
пометить таблицу как измеренную до инцидента и решение принимает автор.

## 6. [ОПЦ] Мульти-сид для group-size фигуры (fig:group-size)
РИСК: сид-разброс сабсета ±3.9 может сломать монотонность 82→86→92 — обсудить с автором ДО запуска.
Если запускать: сиды 42/123 для gs=1/2/default на `test_new_method_gs.py` (6 ранов × ~1.5 ч), рисовать error bars.

Приоритет: 1 → 2 → 3 → 4 (5 и 6 только по команде автора).
