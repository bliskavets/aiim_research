# Ревьюерский проход №2 (полное чтение статьи, 2026-07-20)

Текущая версия main.tex (rev 3b, с blue/green/yellow правками). Проверены данные
конкурентов по первоисточникам: TPO (arXiv:2501.12895, ICML 2025), SPO
(arXiv:2502.06855, EMNLP 2025 Findings), Skywork-Reward-V2 (arXiv:2507.01352),
MathArena aime_2026 (HF, 30 задач), Qwen3 tech report, AlpacaEval-2 leaderboard.
Пометки: [CARRY] = из прошлого аудита, до сих пор не исправлено; [NEW] = найдено
в этом проходе. Уровни: КРИТ / СРЕДН / МЕЛК.

## A. Числовые несостыковки внутри статьи

1. [CARRY][КРИТ] Table 1 (base 84.4, BoN 85.4, SAGE 92.0, full-500) vs Table 4
   (baseline 0.61, BoN 0.72, SAGE 0.97, «held-out subset seed=42»). Разрыв базы
   23 пт на подмножестве той же выборки — первое, что спросит ревьюер.
2. [CARRY][СРЕДН] Figure 3 (headline): SAGE+RM = 89.9; Table 2 = 89.8.
3. [CARRY][СРЕДН] Caption Table 1 описывает «Top: SAGE with RM… Middle… Bottom…»,
   а блоков в таблице два. qCe4 указывал дословно.
4. [CARRY][СРЕДН] Table 2: у Llama XSTest идентичные 95.1/95.1 (SAGE и SAGE+RM);
   AlpacaEval raw почти равны (74.9/74.7) при LC-разрыве 6.2 пт — необъяснённая
   аномалия длин между вариантами одного метода.
5. [NEW][СРЕДН] Yellow group-size (текст 4.4 + Figure 9): дефолт на графике
   называется «3», а метод/алгоритм описывают дефолт как 4 good / 3 bad
   (и «median size of each group is two to three»). Что именно отложено по оси —
   не определено. Плюс опасное совпадение: 92.0 на 50-задачном сабсете == 92.0
   хедлайна на полных 500 — ревьюер может заподозрить, что хедлайн тоже сабсетный.
   Нужно: определить ось (например, «cap on the worst-group size, default 3»),
   и в подписи фигуры оставить явное «50-problem subset» (есть) + НЕ давать
   поводов смешивать (можно добавить «headline full-set number is reported in
   Table 1» в caption или сместить шкалу).
6. [CARRY][СРЕДН] Таблица N,T-абляции (Appendix G): (7,3)=90.9 на сабсете <
   (7,2)=92.0 на полных 500 — больше вычислений дают меньше; тренд не бьётся.
7. [NEW][МЕЛК] Fig. 4 caption: «RM score dips at epoch 2 then rises at epoch 3»,
   но сетап = 2 optimization-эпохи (плюс начальная генерация). Индексация эпох
   на осях не определена (qCe4 просил) — при 2 эпохах «epoch 3» выглядит ошибкой.

## B. Сверка с внешними источниками (конкуренты и модели)

8. [CARRY][КРИТ] Table 5: IFEval baseline 42.0% vs официальный non-thinking
   Qwen3-8B = 83.0 (tech report, Table 18); MMLU-Pro STEM baseline 48.0 vs наш
   same-setup замер 71.2-74.0. Проверяется ревьюером за минуты; приросты «+28/+20»
   — артефакт (доказано red-flag rerun'ом: сломанный движок даёт 37.3/64.7).
9. [CARRY][КРИТ] AlpacaEval колонка: Llama base LC 35.37 vs официальный лидерборд
   22.9 (та же модель, тот же протокол gpt4_turbo); SAGE LC 62.99 > GPT-4o (57.5).
   Протокол (версия бенча, референс, судья) в статье не описан вовсе.
10. [NEW][СРЕДН] «Qwen3.5-4B-Thinking» (Table 3 и текст) — на HF модель называется
    Qwen/Qwen3.5-4B (hybrid reasoning). Точное имя чекпойнта надо привести к
    официальному, иначе выглядит как несуществующая модель.
11. [NEW][МЕЛК] «AIME 2026 … with fewer than 100 instances» — в MathArena/aime_2026
    ровно 30 задач. Писать точно («30 problems»), иначе смотрится как незнание
    собственного бенчмарка.
12. [NEW][МЕЛК] Наши TPO-описания сверены с оригиналом (ICML 2025): механизм
    best-vs-worst через RM — корректно; SPO (EMNLP 2025 Findings) — корректно
    процитирован, но наши SPO-числа (9.68/7.65 win rate) аномально низки на фоне
    их заявленных результатов; текущее объяснение (OOD-калибровка на 5 примерах)
    ревьюеру, знакомому со SPO, покажется недостаточным — нужен конфиг в аппендикс.
13. [CARRY][МЕЛК] Llama3-8B vs Llama-3.1-8B: скрипты кампании — llama31_8b;
    официальный MATH-500 Llama-3.1-8B-Instruct = 54.8 (tech report Table 18) при
    наших 45.6. Назвать чекпойнт точно.

## C. Методологические претензии, которые зададут снова

14. [CARRY][СРЕДН] m_min: в тексте «minimum group size», в формулах — верхний cap
    (top min(|G_pre|, m_min)). qCe4 просил определить; так и не исправлено.
15. [NEW][МЕЛК] «Group formation proceeds in two stages» — далее перечислены ТРИ
    шага (partition / ranking-truncation / fallback).
16. [CARRY][СРЕДН] Порог 75% в judge-промпте подобран sweep'ом по NDCG на сабсете
    самой MATH-500 (Appendix F) — тюнинг на eval-задаче; перевыбрать на
    AGIEval-Math или явно оговорить.
17. [NEW][МЕЛК] Yellow oracle-фраза «across three seeds on MATH-500» — единственное
    место с мульти-сидом; ревьюер спросит, почему главная таблица single-seed,
    а вспомогательный анализ — три сида.
18. [CARRY][МЕЛК] «calibrated contrastive score» в абстракте/интро/методе: слово
    «calibrated» осталось, ranking-caveat добавлен только в Method. Абстракт/интро
    стоит смягчить (например «contrastive log-probability margin»).

## D. Текст, грамматика, оформление

19. [NEW][МЕЛК] Битые/корявые конструкции: «difficult when methods that compare
    only the single best and worst candidates …, which is outlier-sensitive»
    (сломанный синтаксис); «produce a more confident estimates»; «lower judging
    temperatures improves … higher generation temperatures … maximizes»
    (согласование); «comparing only extremes candidates»; «These rollouts are
    then to be scored».
20. [CARRY][МЕЛК] Опечатки, на которые qCe4 указал дословно: `</asnwer>` (Fig. 11),
    «Optimizational epoch» (2 раза в Appendix B).
21. [CARRY][МЕЛК] Разная точность чисел в Table 1 (62.99 / 50.12 / 9.68 vs 74.9 /
    57.8) — унифицировать до одного знака.
22. [NEW][МЕЛК] Args → в оригинале пишется ARGS (Khanov et al., ICLR 2024).
23. [CARRY][МЕЛК] Протокол AlpacaEval, судья GPT-4.1 в head-to-head (green) и
    N=200 сабсет — параметры head-to-head не названы в тексте (сколько пар, какой
    référence); одна короткая скобка решит вопрос.

## Приоритет исправлений (когда дадут команду)

КРИТ: 1, 8, 9 — лечатся только заменой чисел Table 1/4/5 и AlpacaEval-колонки
(либо снятием) — решение за автором.
СРЕДН (можно исправить сразу, не трогая хедлайн): 2 (перерендер фигуры 89.9→89.8),
3 (caption), 4 (сноска/пояснение), 5 (определить ось фигуры 9), 6 (снять или
пересчитать N,T-таблицу), 7 (подписать оси/индексацию эпох), 10 (имя модели),
14 (определить m_min), 16 (оговорка про порог), 18 (смягчить «calibrated»).
МЕЛК: 11-13, 15, 17, 19-23 — правки текста на полчаса.
