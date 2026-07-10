=================

Reviewer PVoW

We would like to thank the reviewer for ...

> Heavy dependence on LLM-generated data without human validation This is the paper's largest weakness. Almost every stage of FinOpsBench-v1—including query generation, schema generation, data generation, agent traces, and quality control—is performed by LLMs. No human evaluation is conducted to verify whether generated financial scenarios are realistic, whether reasoning traces are correct, or whether the LLM judges make reliable decisions. Consequently, benchmark quality ultimately depends on the behavior of a small collection of proprietary models rather than independent validation.
> LLM-as-judge validation is insufficiently justified Although the authors employ a committee of three LLM judges, there is no measurement of agreement with human annotators or any estimate of judge accuracy. Majority voting among LLMs cannot substitute for demonstrating that the judgments are actually correct.
> Include a human evaluation study for both benchmark versions. Even a random sample of 200–300 examples independently verified by financial experts or trained annotators would substantially increase confidence in the benchmark.
> Report agreement between LLM judges and human evaluators (e.g., Cohen's κ or percentage agreement). This is especially important because the benchmark relies extensively on LLM judgments.


спасибо ревьюеру за это ценное замечание. мы отобрали порядка 200 примеров из подмножества v2 и порядка 200 из подмножества v1 и разметили его при помощи a human judge with knowledge of the domain.

мы посчитали метрики согласованности human judge с оценками судей на подмножестве v1 и валидность примеров в подмножестве v2 и получили... что говорит ... следовательно ...



> Limited transparency of dataset construction The paper describes the nine pipeline stages but does not release or describe the actual prompts used in these stages. This significantly limits reproducibility because prompt design likely has a major influence on the generated benchmark.
> Release all prompts used throughout the nine-stage pipelines, including prompts for query generation, schema generation, data generation, feedback reconciliation, and system prompt construction. These are essential for reproducibility.

Мы добавили недостающий код по закреплённой в статье ссылке... там есть промпты. 


> Evaluation methodology is relatively weak FinOpsBench-v1 evaluation itself relies on another LLM judge rather than deterministic correctness whenever possible. This introduces another layer of uncertainty since evaluation and dataset generation both depend heavily on LLM judgments.

оценка экспертами...

оценка на расходящихся примерах...


> Analyze benchmark diversity more quantitatively. Statistics on reasoning operations, SQL complexity, tool-chain depth, numerical operations, financial concepts, and template diversity would strengthen the benchmark description.
> Provide qualitative examples of common model failures beyond overall accuracy, including tool misuse, reasoning mistakes, planning failures, and financial misunderstandings.


спасибо ревьюеру за это замечание. мы уже выписали анализ нашего бенчмарка в пунктах C,D и G, где описали распределения заданий по категориям, распределение длин цепочек вызовов тулов, assistant tables, длин промптов заданий, total data rows per example. 
в дополнений к этому мы прикладываем следующие данные, покрывающие запрос ревьюера:

...


> Report annotation or generation costs, computational resources, and runtime required to construct the benchmark.

мы хотим поблагодарить ревьюера за это замечание, важное для понимания стоимости генерации подобного бенчмарка, напрямую влияющую на расширяемость бенчмарка.
...


> Discuss potential biases introduced by using proprietary models throughout the generation and validation pipeline.

...



=================
Reviewer 6zfv

мы бы хотели поблагодарить ревьюера за 

> While the benchmark targets agentic financial analysis, it remains unclear what fundamental NLP capability it advances beyond a domain-specific evaluation resource.

спасибо ревьюеру за это замечание...
мы сконструировали бенчмарк так что он тестирует не только умение модели разбираться в финансовых отчётах, но и базовые возможности тулколлинга (v2 часть требует от модели вызывать несколько тулл коллов), умение писать запросы и код (v1), способность транслировать неявные запросы в конкретные цепочки действий по извлечению и интерпретации данных. в качестве подкрепления мы прикладываем примеры запросов и трейсов модели на v2 и на v1.
... здесь приложи по одному примеру задача - трейс - финальный ответ - grounf_truth от модели - выбирай наиболее сложные примеры с большим количеством шагов.


это показывает что наша модель должна не только уметь понимать запрос пользователя в финансовой сфере, но и грамотно определить интент пользователя и сгенерировать последовательность шагов по достижению цели, что является базовым требованием к NLP модели лежащей в основе агентской системы...



> The paper argues that existing financial benchmarks do not adequately evaluate agentic tool use, but multiple recent benchmarks have already moved in this direction. It remains somewhat unclear what fundamentally new evaluation capability FinOpsBench provides.

...



> Although the benchmark is positioned as a diagnostic evaluation of agentic reasoning, the reported analyses are primarily based on final-answer accuracy. More fine-grained diagnostic metrics or failure analyses would better demonstrate that the benchmark provides insights beyond conventional benchmark evaluation.

спасибо ревьюеру за это замечание...

мы бы хотели уточнить тут что наш бенчмарк состоит специально из двух частей: v1 - части где модель не должна дать ответ на общий запрос, который далеко не всегда верифицируем простой exact match проверкой. мы приводим несколько примеров таких запросов ниже:

... приведи примеры тут запросов и golden ответов...

также мы провели замер agreement тут с human judge with knowledge of the domain. и получили ...


в дополнение к этому вторая половина нашего бенчмарка v2 состоит из примеров в которых модель в конце как раз должна дать верифицируемый ответ, что можно проверить уже конкретной exact match проверкой. мы также проводим анализ чистоты бенчмарка, попросив human judge with knowledge of the domain провалидировать случайно выбранное моножество из 200 примеров из этой части. как показали результаты ...

также следуя запросу ревьюера мы провели детальный анализ failure cases и публикуем его ниже...


> FinOpsBench-v1 is entirely LLM-generated, while FinOpsBench-v2 is largely constructed by transforming FinQA into tool-use environments. Although the authors employ a three-judge panel and execution-based validation, the final benchmark quality still depends substantially on LLM-generated queries, schemas, data, and judgments.

мы понимаем беспокойство ревьюера тут, поэтому как упомянуто в ответе к предыдущему пункту мы провели проверку с привлечением human judge with knowledge of the domain и, как показали резулдьтаты.




===============
Reviewer j7in

мы бы хотели поблагодарить ревьюера за ...

> v1 lacks machine-verifiable hard ground truth; fully relies on LLM panel judges, leading to subjective, biased evaluation results.

Чтобы адресовать консерн ревьюера мы отобрали порядка 200 примеров из подмножества v2 и порядка 200 из подмножества v1 и разметили его при помощи a human judge with knowledge of the domain...

как показали замеры согласованности...

...
следовательно...


> v2 is built entirely on FinQA, which was not designed for agent tool workflows; query types are monotonous and fail to integrate deep financial domain knowledge.
> v2 inherits FinQA’s simple numerical questions, with artificially added multi-hop tool logic rather than native business-driven agent tasks.


Спасибо ревьюеру за это меткое замечание. Мы должны заметить тут что наш бенчмарк как раз состоит из двух взаимодополняющих частей, одна из которых (v1) нацелена на оценку более общих (native) финансовых задач, требующая предоставить общий ответ с суждением по финансовой тематике с привлечением вызовов тулов, который не обязательно должен быть числом, а должен быть выводом или финансовым отчётом. примеры задач и golden ответов мы соответственно приводим ниже:

в то время как вторая адресует проблему проверяемости ответа и потому построена на основе верифицируемых задач как раз чтобы исключить риск неточностей в проверке ответа, требуя от агента совершить несколько вызовов тулов для получения нужной информации и правильно проигнорировать тулы дистракторы и лишнюю информацию. примеры подобных действий агента мы соответственно приводим ниже:
... приведи тут пример задачи из v2 длинного агентского трейса с большим вызовов тулов, дистракторов, провежуточных мыслей и тд.


> Experiment evaluation is incomplete: missing top agent/code frontier models (Claude Code, Codex, OpenCode); baselines only cover tiny open-source models without mainstream finance-specialized LLMs.

мы благодарим ревьюера за данный вопрос. мы хотим заметить также что в наших замерах были использованы не только open source llms, а были использованы llm от openai ...
также мы должны подчеркнуть что claude code и codex являются скорее агентами со специально написанной обвязкой, первоочередная цель которой писать код, которая требует доплнительного протокола настройки им доступа к тулам нашего бенчмарка.
... 

чтобы адресовать консерн ревьюера мы дополнительно проводим эксперименты на SOTA закрытых моделях таких как
Claude-... Gemini-... GPT-...

и на опенсорсных finance llms:

результаты представлены в таблицах ниже.
...


> Adopts outdated smolagents as agent harness, which may introduce framework noise and interfere with reliable tool-use performance measurement.

....



> High risk of data contamination for v2, as core questions come from widely publicized FinQA training corpus.

мы понимаем консерн ревьюера в данном вопросе. тем не менее как мы отмечаем в таблице 3 качество даже больших моделей на нашем бенчмарке не превышает 70 процентов в то время как заявляемое качество текущих топовых моделей порядка 80+% 
что говорит о том что качество на половине v2 нашего бенчмарка зависит не только от знаний модели выученных при тренировке но и умения правильно запланировать шаги по извлечению информации и вызвать нужные инструменты. В дополнение к данному утверждению мы приводим результаты следующего эксперимента где мы дали модели работать в трёх сеттингах:

...

мы наблюдаем что...

...



