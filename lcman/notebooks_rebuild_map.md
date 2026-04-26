# Карта разобранных notebook-файлов для пересборки на `raw_data/`

## Назначение файла

Этот документ фиксирует не старые численные результаты, а **цели, исследовательскую логику и структуру** трёх существующих notebook-файлов:

1. `01_eda.ipynb`
2. `02_multimodal_pipeline_baseline.ipynb`
3. `03_lightweight_multimodal_system.ipynb`

Старые notebook были написаны под неверный датасет (`log_csv/`, `kpi_csv/`, `outputs/final_demo/...`). Их нельзя переносить как код и нельзя переносить их численные выводы. Но можно и нужно сохранить их исследовательскую траекторию: от EDA, через baseline-контракт данных, к лёгкой мультимодальной модели.

Новая точка входа: `raw_data/`.

---

## 0. Что важно в новом датасете `raw_data/`

Новая структура отличается от старой. Теперь данные уже организованы вокруг workload/scenario:

```text
raw_data/<workload>/
  bench.log
  mapred_logs/<scenario>.log
  parsed_log/
    labeled.csv
    normal.csv
    tasks.csv
  kpis/<scenario>/
    instruments/<host>/
      cpustat
      iostat
      jvms
      memstat
      netstat
      vmstat
    jobhistory/stdout
    conf/
    gc_summary/
```

Ключевое отличие от старой постановки:

- в старом notebook приходилось **восстанавливать связь logs ↔ KPI** эвристически;
- в `raw_data/` связь во многом уже задана путём: `workload + scenario`;
- поэтому в новых notebook нужно начинать не с глобального `drain_labeled.csv`, а с сущности **run/scenario**: `(workload, scenario)`.

Предварительно замеченные особенности:

- большинство workload имеют `parsed_log/labeled.csv` с колонками `Timestamp, Template, Content, Component, Label`;
- `nweight/parsed_log/` содержит `normal.csv` и `tasks.csv`, но не содержит `labeled.csv`, это надо явно обработать в inventory;
- KPI-файлы не CSV с запятыми, а текстовые таблицы с пробельным разделителем и иногда повторяющимися строками заголовка;
- сценарии представлены как `normal1`, `normal2`, ..., `cpu_stress`, `datanode_killed`, `datanode_suspended`, и т.д.;
- raw log-файлы в `mapred_logs/` дают сильный provenance-сигнал, который старый notebook пытался реконструировать.

---

## 1. Общая исследовательская траектория `01 -> 02 -> 03`

Старые notebook задают правильную последовательность ВКР:

| Этап | Главный вопрос | Выход этапа | Что переносим |
|---|---|---|---|
| `01_eda.ipynb` | Как устроены логи и KPI, где ограничения данных? | Проверяемые гипотезы и правила подготовки | структуру EDA, локальные выводы, анализ coverage/provenance |
| `02_multimodal_pipeline_baseline.ipynb` | Как собрать честный baseline-контур log+KPI? | Multi-rate датасет, dual-target разметка, smoke-модели | data contract, anti-leakage split, availability-only контроль |
| `03_lightweight_multimodal_system.ipynb` | Даёт ли лёгкая missing-aware модель методическое усиление baseline? | LCMAN-lite, сравнение с baseline, model card, inference-store | компактную архитектуру, coverage-aware оценку, model card |

Главная мысль всей цепочки: **не строить большую инженерную систему**, а сделать исследовательские notebook, где читатель без чтения кода понимает ход анализа, локальные выводы и ограничения.

---

# 2. Notebook `01_eda.ipynb`: карта EDA

## 2.1. Исходная цель notebook

Старый notebook ставит цель: описать исходные логовые и KPI-данные, их структуру, временную организацию, качество, распределения и содержательные особенности. Результат должен быть не моделью, а аналитическим каркасом для подготовки данных.

## 2.2. Основные исследовательские вопросы

Смысловые вопросы `01`:

1. Какова структура источников и базовое качество данных?
2. Как распределён логовый поток во времени?
3. Каков дисбаланс нормальных и аномальных событий?
4. Как распределены логовые шаблоны?
5. Есть ли у аномальных эпизодов интерпретируемые семантические маркеры?
6. Как устроена KPI-модальность как самостоятельный источник?
7. Какое временное разрешение KPI сохраняет предсбойную динамику?
8. Насколько доступность KPI связана с положительным классом?
9. Можно ли строго связать log episodes и KPI-сценарии?
10. Какие гипотезы для оконной агрегации следуют из EDA?

## 2.3. Структура старого notebook

| Блок | Что делает | Локальный смысл |
|---|---|---|
| 1. Паспорт источников и контроль качества | считает объёмы, поля, пропуски, дубликаты, нарушения времени | перед агрегацией источники надо понимать как разные модальности |
| 2. Временной профиль логов и баланс классов | строит интенсивность логов, считает долю положительного класса на разных уровнях | это задача редких событий, метрики должны быть устойчивы к дисбалансу |
| 3. Распределение логовых шаблонов | анализирует `DrainTemplate`/шаблоны, top-k coverage, длины текстов | есть частотное ядро и редкий хвост, нужны интерпретируемые текстовые признаки |
| 4. Качественный просмотр сообщений | показывает top-шаблоны нормы/аномалии и фрагменты около отказа | аномалии имеют повторяющиеся семантические маркеры |
| 5. KPI-модальность | описывает сценарии, длительности, метрики, серии, масштабы значений | KPI нельзя грубо усреднять, нужна series-aware нормализация |
| 5.1. Временное разрешение KPI | сравнивает 10s/30s/1min/5min окна | короткие KPI-окна лучше сохраняют динамику отказа |
| 5.2. KPI-покрытие на шкале логов | измеряет `kpi_available` и связь с Label=1 | доступность KPI может стать shortcut-сигналом |
| 6. Рабочие гипотезы | превращает наблюдения EDA в правила следующего notebook | EDA заканчивается проверяемыми решениями для pipeline |
| 8. Reconstruction tournament | сравнивает стратегии восстановления log episodes | strict pairing возможен только при явном provenance |
| Итог EDA | перечисляет ограничения и правила | следующий шаг — оконная агрегация с контролем масштаба, coverage и pairing |

## 2.4. Что нужно сохранить при пересборке на `raw_data/`

Сохранить:

- формат: каждый значимый результат сопровождается markdown-блоком `Что показано / Что наблюдается / Локальный вывод / Почему важно для ВКР`;
- анализ двух модальностей как разных источников: logs и KPI;
- обязательный контроль качества времени, дублей и пропусков;
- анализ дисбаланса классов и редкости отказов;
- анализ логовых шаблонов и компонент;
- отдельный KPI-блок: host, metric, series, duration, resolution, missingness;
- проверку, не является ли `kpi_available` shortcut-сигналом;
- provenance/pairing audit.

Изменить:

- заменить старые `LOG_PATH = log_csv/drain_labeled.csv` и `KPI_ROOT = kpi_csv` на обход `raw_data/<workload>/...`;
- вместо одного глобального лога строить inventory по `(workload, scenario)`;
- вместо `DrainTemplate` использовать `Template` из `parsed_log/labeled.csv`, а при необходимости `EventTemplate` из `normal.csv/tasks.csv`;
- не делать вид, что связь logs↔KPI потеряна: в `raw_data/` она часто задана путём `workload/scenario`;
- блок reconstruction tournament превратить в **pairing/provenance audit**: проверить совпадение сценариев в `mapred_logs/`, `parsed_log/`, `kpis/`, а не заново угадывать пары.

## 2.5. Новая структура `01` под `raw_data/`

Рекомендуемая структура нового EDA:

1. **Inventory датасета**
   - workload count;
   - scenarios per workload;
   - наличие `bench.log`, `mapred_logs`, `parsed_log/labeled.csv`, `kpis`;
   - список исключений: например, workload без `labeled.csv`.

2. **Паспорт логов**
   - строки в `parsed_log/labeled.csv` по workload/scenario;
   - поля `Timestamp, Template, Content, Component, Label`;
   - пропуски, дубликаты, монотонность времени;
   - отличие `labeled.csv` от `normal.csv/tasks.csv`.

3. **Логовая динамика**
   - интенсивность по времени;
   - число событий по workload/scenario;
   - доля `Label=1`;
   - сравнение нормальных и отказных сценариев.

4. **Шаблоны и компоненты логов**
   - top templates;
   - top components;
   - coverage top-k;
   - отличия normal/failure сценариев;
   - качественные фрагменты из `mapred_logs/<scenario>.log`.

5. **KPI inventory**
   - структура `kpis/<scenario>/instruments/<host>/<metric>`;
   - какие hosts и metrics есть;
   - длительность KPI по scenario;
   - частота измерений;
   - повторяющиеся заголовки и правила парсинга.

6. **KPI statistics**
   - series-aware статистики по `host + metric + series`;
   - масштабы, нули, пропуски;
   - необходимость host/metric/series-specific нормализации.

7. **Multi-rate feasibility**
   - сколько KPI-точек дают окна 10s/30s/1min/5min;
   - какие окна логов разумны для macro-контекста;
   - оценка `kpi_available` и `has_both_modalities`.

8. **Pairing/provenance audit**
   - совпадает ли набор scenario в `mapred_logs` и `kpis`;
   - можно ли считать пару `(workload, scenario)` strict;
   - какие runs требуют fallback/исключения.

9. **Итоговые гипотезы для `02`**
   - macro log context;
   - micro KPI dynamics;
   - dual target;
   - hard negatives;
   - group-aware split по workload/scenario/failure family.

---

# 3. Notebook `02_multimodal_pipeline_baseline.ipynb`: карта baseline pipeline

## 3.1. Исходная цель notebook

Цель `02` — собрать и проверить инженерный baseline-контур мультимодальной подготовки данных. Это не notebook про максимальное качество модели. Его задача — доказать, что датасет собран воспроизводимо, leakage контролируется, а связка log+KPI работает хотя бы в smoke-режиме.

## 3.2. Основные вопросы `02`

1. Какие наблюдения из EDA становятся обязательными правилами pipeline?
2. Как построить multi-rate схему: macro-контекст логов + micro-динамика KPI/logs?
3. Как ввести `y_warning` и `y_risk` без leakage?
4. Как контролировать shortcut через `kpi_available`?
5. Работают ли минимальные модели: `availability-only`, `log-only`, `kpi-only`, `fused`?

## 3.3. Структура старого notebook

| Блок | Что делает | Локальный смысл |
|---|---|---|
| 1. EDA -> pipeline rules | связывает наблюдения EDA с правилами pipeline | baseline не набор техник, а следствие свойств данных |
| 2. Обработка логов | очистка, сортировка, macro windows, базовые логовые признаки | логовая ось становится устойчивой основой датасета |
| 3. KPI series-aware | сканирует KPI-каталог, определяет fail_start, series профили | KPI сохраняются как самостоятельная модальность |
| 4. Multi-rate alignment | подбирает `tolerance`, фиксирует macro=5min, micro=10s/30s | мультимодальность измеряется через coverage, а не декларируется |
| 5. Сборка датасета | строит micro-точки, macro-контекст, KPI features, dual targets | появляется data contract для `03` |
| 6. Smoke-модели | обучает availability/log/kpi/fused logistic baselines | проверяет работоспособность контура и shortcut |
| 7. Честный итог | фиксирует ограничения coverage и split | `02` — baseline protocol, а не финальная модель |

## 3.4. Что нужно сохранить при пересборке

Сохранить:

- таблицу `EDA observation -> pipeline rule -> mitigated risk`;
- multi-rate идею: более длинный log macro context и более короткая micro dynamics;
- dual targets:
  - `y_warning` — предсбойный интервал;
  - `y_risk`/`y_anomaly` — ближний к отказу интервал;
- `availability-only` как обязательную контрольную модель;
- coverage-срезы `kpi_available=1` и `has_both_modalities=1`;
- hard negatives: окна с KPI, но без warning/risk;
- group-aware split;
- финальный data contract для `03`.

Изменить:

- больше не выводить log runs из одного глобального потока через label transitions;
- базовой единицей сделать `run_id = <workload>/<scenario>`;
- pairing делать по пути `raw_data/<workload>/mapred_logs/<scenario>.log` и `raw_data/<workload>/kpis/<scenario>/...`;
- `tolerance` нужен не для угадывания пары log↔KPI, а для выравнивания временных шкал внутри уже известного run;
- KPI loader должен читать instrument-файлы с пробельным разделителем и повторяющимися headers;
- split должен учитывать не только `episode_id`, но и риск leakage между одинаковыми failure types в разных workloads.

## 3.5. Новый data contract для `03`

`02` должен сохранить таблицу, которую `03` будет читать без legacy fallback. Минимальный контракт:

| Группа | Колонки |
|---|---|
| Идентификаторы | `workload`, `scenario`, `run_id`, `group_id`, `failure_family`, `scenario_type` |
| Время | `window_ts`, `fail_start`, `minutes_before_fail`, `macro_window`, `micro_window`, `micro_window_sec` |
| Цели | `state`, `y_warning`, `y_risk`, `y_anomaly` |
| Логи | `log_text`, `log_available`, `macro_log_count`, `macro_unique_templates`, `macro_error_count`, `micro_log_count`, `micro_unique_templates`, `micro_error_count`, `micro_log_rate`, `micro_error_rate` |
| KPI агрегаты | `kpi_available`, `kpi_series_available`, `kpi_feature_count`, `kpi_avail_ratio`, `kpi_recency_sec`, `kpi_last`, `kpi_mean`, `kpi_delta_*`, `kpi_slope_*`, `kpi_roll_mean_*` |
| KPI tensor columns | series-aware признаки вида `<host>__<metric>__<series>__<stat>` |
| Coverage/leakage | `has_both_modalities`, `hard_negative`, `split_group` |

Рекомендуемые артефакты `02`:

```text
outputs/raw_multirate/aligned_multirate_windows.csv
outputs/raw_multirate/baseline_summary_metrics.json
outputs/raw_multirate/data_contract_report.csv
outputs/raw_multirate/split_report.csv
```

## 3.6. Новая структура `02` под `raw_data/`

1. **Правила pipeline из нового EDA**
   - таблица наблюдение -> правило -> риск.

2. **Log loader**
   - читает `parsed_log/labeled.csv`;
   - сохраняет `workload`, `scenario`, `Template`, `Content`, `Component`, `Label`;
   - при необходимости подтягивает raw `mapred_logs/<scenario>.log` для `log_text` и provenance.

3. **KPI loader**
   - читает `kpis/<scenario>/instruments/<host>/<metric>`;
   - парсит пробельные таблицы;
   - удаляет повторные headers;
   - формирует long table: `workload, scenario, host, metric, series, timestamp, value`.

4. **Run-level alignment**
   - `run_id = workload/scenario`;
   - scenario type: `normal` или `failure`;
   - `fail_start`: из `Label=1`, события failure, scenario boundary или явно описанного правила;
   - без `fail_start` run не использовать для warning/risk обучения, но можно использовать как normal/hard negative.

5. **Multi-rate windowing**
   - macro log windows: например 1min/5min;
   - micro windows: 10s/30s;
   - признаки логов и KPI собираются отдельно и затем соединяются по `run_id + window_ts`.

6. **Dual-target разметка**
   - `y_warning`: предсбойный горизонт;
   - `y_risk`: ближний горизонт;
   - normal windows: фоновые/далёкие окна;
   - все границы должны быть явно описаны markdown-текстом.

7. **Anti-leakage split**
   - минимум split по `run_id`;
   - лучше: split по `workload` или `failure_family`, если данных достаточно;
   - обязательный отчёт, какие группы попали в train/test.

8. **Smoke baselines**
   - `availability-only`;
   - `log-only`;
   - `kpi-only`;
   - `fused`;
   - primary metric: PR-AUC по `y_warning`;
   - secondary: PR-AUC по `y_risk`;
   - обязательные срезы coverage.

9. **Финальный вывод `02`**
   - что контур собирается;
   - где coverage ограничивает выводы;
   - какой data contract передан в `03`.

---

# 4. Notebook `03_lightweight_multimodal_system.ipynb`: карта LCMAN-lite

## 4.1. Исходная цель notebook

Цель `03` — не строить «зоопарк» моделей, а методически усилить baseline из `02`: оставить lightweight stack, добавить раздельные ветки для log/KPI, gated missing-aware fusion, dual-target обучение и coverage-aware оценку.

## 4.2. Структура старого notebook

| Блок | Что делает | Локальный смысл |
|---|---|---|
| 0. Связь `01 -> 02 -> 03` | фиксирует критерии готовности модели | модель не меняет постановку, а усиливает слой над контрактом `02` |
| 1. Data contract harmonization | подстраивает старые/новые колонки и fallback | в новой версии нужен строгий контракт без legacy fallback |
| 2. Входы модели | log embeddings, log_macro, log_micro, kpi_tensor, kpi_micro, masks | модель получает multi-rate и missing-aware входы |
| 3. Архитектура | Log macro branch, Log micro branch, KPI 1D-CNN, gated fusion, dual heads | компактная, но содержательно мультимодальная архитектура |
| 4. Training protocol | repeated group-aware split, modality dropout, early stopping | честная оценка без leakage |
| 5. Сравнение с baseline | availability/log/kpi/fused/LCMAN-lite, PR-AUC, coverage slices | качество интерпретируется только с учётом coverage |
| 6. Анализ поведения | графики, modality slices, demo cases, gate, KPI drivers | связывает метрики с интерпретируемыми примерами |
| 7. Inference mode | store-based inference без переобучения | демонстрационный слой для ВКР |
| 8. Model card | назначение, данные, архитектура, метрики, ограничения | итоговый исследовательский артефакт |
| 9. Финальный вывод | роль LCMAN-lite в ВКР | развитие лучше направлять в данные/coverage, а не в усложнение архитектуры |

## 4.3. Что нужно сохранить

Сохранить:

- постановку `03` как методическое усиление `02`, а не отдельный эксперимент;
- baseline-набор для сравнения: `availability-only`, `log-only`, `kpi-only`, `fused`, `LCMAN-lite`;
- primary target: `y_warning`;
- secondary target: `y_risk`;
- modality masks: `log_available`, `kpi_available`, `has_both_modalities`, `kpi_avail_ratio`;
- gated fusion с fallback-правилами:
  - нет KPI -> log-driven;
  - нет логов -> KPI-driven;
  - обе модальности -> learned gate;
- modality dropout;
- repeated group-aware evaluation;
- model card и inference-store как финальные артефакты.

Изменить:

- убрать legacy fallback на `outputs/final_demo/scored_windows_reference.csv`;
- читать только новый контракт из `outputs/raw_multirate/aligned_multirate_windows.csv`;
- не выводить и не проверять HF token в notebook-тексте;
- использовать безопасный encoder fallback: локальный/кэшированный embedding или hashing;
- все результаты пересчитать на новом датасете, старые метрики не переносить.

## 4.4. Новая структура `03` под `raw_data/`

1. **Постановка**
   - коротко повторить: `01` выявил свойства данных, `02` собрал контракт, `03` проверяет lightweight fusion.

2. **Strict data contract load**
   - если нет обязательных колонок — ошибка, а не silent fallback;
   - вывести паспорт контракта: rows, runs, groups, target balance, coverage.

3. **Input tensors**
   - `log_embedding` из `log_text`;
   - `log_macro`;
   - `log_micro`;
   - `kpi_tensor` по series-aware колонкам;
   - `kpi_micro` агрегаты;
   - masks.

4. **Architecture card before training**
   - таблица блоков модели;
   - схема потоков;
   - объяснение, почему модель lightweight.

5. **Training and evaluation**
   - repeated group-aware split;
   - train balancing;
   - modality dropout;
   - early stopping;
   - metrics: PR-AUC, ROC-AUC, F1, precision, recall;
   - target: `y_warning`; secondary: `y_risk`.

6. **Coverage-aware comparison**
   - all;
   - `kpi_available=1`;
   - `has_both_modalities=1`;
   - normal/failure-family slices, если хватает данных.

7. **Interpretability cases**
   - показать 2-3 окна;
   - `warning_score`, `risk_score`, `pred_state`;
   - gate;
   - available modalities;
   - top KPI drivers;
   - log excerpt.

8. **Inference store**
   - сохранить predictions table;
   - функция lookup по `(run_id, window_ts)`;
   - без повторного обучения.

9. **Model card**
   - данные;
   - архитектура;
   - протокол;
   - метрики;
   - ограничения;
   - вывод для ВКР.

---

# 5. Сквозные методологические правила для всех новых notebook

## 5.1. Формат markdown после результатов

Каждый важный вывод должен быть понятен без чтения кода. Рекомендуемый шаблон:

```markdown
### Интерпретация результата

- **Что показано:** ...
- **Что наблюдается:** ...
- **Локальный вывод:** ...
- **Почему это важно для ВКР:** ...
```

Для `03` можно использовать более короткий вариант:

```markdown
### Локальный вывод
...
```

## 5.2. Что нельзя переносить из старых notebook

Не переносить:

- численные результаты старого датасета;
- пути `log_csv/`, `kpi_csv/`, `outputs/final_demo/...`;
- выводы о количестве шаблонов, сценариев, PR-AUC и coverage;
- старую идею, что provenance обязательно потерян;
- HF token checks и любые потенциально чувствительные выводы;
- сложную инженерную обвязку, если она не помогает исследовательскому выводу.

## 5.3. Что обязательно сохранить

Сохранить:

- исследовательскую причинность: `свойство данных -> правило pipeline -> ограничение интерпретации`;
- multi-rate постановку;
- dual-target постановку;
- availability-only контроль;
- coverage-aware метрики;
- anti-leakage split;
- model card;
- честную формулировку ограничений.

---

# 6. Предлагаемый порядок пересборки

## Шаг 1. Новый `01_eda_raw_data.ipynb`

Цель: понять новый датасет и сформировать правила для `02`.

Минимальный результат:

- inventory `raw_data/`;
- log quality report;
- KPI quality report;
- pairing/provenance audit;
- обоснование временных окон;
- рабочие гипотезы для baseline.

## Шаг 2. Новый `02_multirate_baseline_raw_data.ipynb`

Цель: собрать reproducible data contract и smoke baseline.

Минимальный результат:

- `aligned_multirate_windows.csv`;
- baseline metrics;
- split report;
- coverage report;
- честный вывод о границах мультимодальности.

## Шаг 3. Новый `03_lcman_lite_raw_data.ipynb`

Цель: проверить lightweight missing-aware fusion поверх контракта `02`.

Минимальный результат:

- LCMAN-lite vs baselines;
- coverage-aware метрики;
- interpretability cases;
- inference store;
- model card.

---

# 7. Короткая карта: что преследует каждый notebook

| Notebook | Что преследует | Не преследует |
|---|---|---|
| `01` | понять данные, найти ограничения, сформировать гипотезы | обучать модель, доказывать качество |
| `02` | собрать честный baseline-контур и контракт данных | максимизировать метрики |
| `03` | проверить лёгкое методическое усиление fusion-механизма | строить production-систему или model zoo |

Главный итог разбора: старые notebook ценны не кодом и не числами, а **исследовательским сценарием**. Новая версия должна заново пройти тот же путь на `raw_data/`, но использовать естественный provenance нового датасета: `workload/scenario`.
