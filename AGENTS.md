# AGENTS.md

## Назначение проекта

Финальный контур ВКР: воспроизводимая AIOps-система раннего предупреждения на логах и KPI.

- источник данных: только `raw_data/`;
- контракт: `dashboard.pipeline.build_contract`;
- baseline: `dashboard.training.baselines`;
- финальная модель: `dashboard.training.final_model`;
- демонстрация: `dashboard/app.py`.

Ключевая идея: это не бинарный оракул сбоя, а лабораторная система раннего предупреждения с честным разделением:

```text
score модели != состояние alert-layer != факт incident
```

## Финальный маршрут

```powershell
python -m dashboard.pipeline.build_contract --raw-data raw_data

# 02_baseline_pipeline.ipynb обучает baseline
# 03_fused_model.ipynb обучает финальную Efficient HADES модель и экспортирует bundle

streamlit run dashboard/app.py
```

Если запускать обучение без notebook:

```powershell
python -m dashboard.training.baselines `
  --contract-path dashboard/artifacts/contract/strict_intensity_contract.csv.gz `
  --output-dir dashboard/artifacts/reports/protocol_fixed_baselines `
  --split-protocol stratified_by_workload `
  --targets default `
  --seed 43 `
  --validation-seed 43 `
  --fp-budget 0.10

python -m dashboard.training.final_model `
  --run-id final_protocol_fixed_hades `
  --device cuda `
  --run-root dashboard/artifacts/model_runs/protocol_fixed_hades `
  --contract-path dashboard/artifacts/contract/strict_intensity_contract.csv.gz `
  --model-out-dir dashboard/artifacts/model/protocol_fixed_efficient_hades `
  --bundle-out-dir dashboard/artifacts/bundle `
  --split-protocol stratified_by_workload `
  --variant modern_metric `
  --seed 43 `
  --monitor-target horizon_60s `
  --alert-target horizon_60s `
  --normalization workload_norm `
  --metric-feature-mode raw_plus_deviation `
  --epochs 30 `
  --batch-size 128 `
  --quantization dynamic

python -m dashboard.export.bundle --mode protocol_fixed
```

## Активная структура

Финальные notebooks:

1. `01_eda.ipynb` - данные, strict contract, workload/domain shift, targets.
2. `02_baseline_pipeline.ipynb` - protocol-fixed baseline и контрольные метрики.
3. `03_fused_model.ipynb` - финальная Efficient HADES модель, обучение, сравнение, export.

Активные модули `dashboard/` сгруппированы по слоям:

- `pipeline/build_contract.py` - сборка strict contract из `raw_data`;
- `training/baselines.py` - protocol-fixed baseline;
- `training/final_model.py` - обучение и promote финальной модели;
- `modeling/protocol_fixed_hades.py` - Efficient HADES training core;
- `modeling/hades_blocks.py` - encoder/runtime blocks, ONNX/INT8 helpers;
- `export/bundle.py` - единый dashboard bundle exporter;
- `runtime/contract.py`, `runtime/inference.py`, `runtime/view_model.py` - runtime-модель данных;
- `ui/components.py` - компоненты интерфейса;
- `app.py` - Streamlit entrypoint.

Старые hypothesis/smoke/legacy файлы хранятся в `.research_archive/`.

## Данные и артефакты

Contract:

- `dashboard/artifacts/contract/strict_intensity_contract.csv.gz`
- `dashboard/artifacts/contract/strict_contract_summary.csv`

Baseline:

- `dashboard/artifacts/reports/protocol_fixed_baselines/baseline_protocol_metrics.csv`
- `dashboard/artifacts/reports/protocol_fixed_baselines/baseline_protocol_predictions.csv.gz`
- `dashboard/artifacts/reports/protocol_fixed_baselines/baseline_protocol_workload_metrics.csv`
- `dashboard/artifacts/reports/protocol_fixed_baselines/baseline_protocol_manifest.json`

Финальная модель:

- `dashboard/artifacts/model_runs/protocol_fixed_hades/final_protocol_fixed_hades/`
- promoted artifacts: `dashboard/artifacts/model/protocol_fixed_efficient_hades/`

Dashboard bundle:

- `dashboard/artifacts/bundle/replay_bundle.csv.gz`
- `dashboard/artifacts/bundle/dashboard_replay_store.csv.gz`
- `dashboard/artifacts/bundle/dashboard_model_report.json`
- `dashboard/artifacts/bundle/alert_events.csv`
- `dashboard/artifacts/bundle/baseline_comparison.csv`
- `dashboard/artifacts/bundle/metrics_summary.csv`
- `dashboard/artifacts/bundle/model_card_product.md`
- `dashboard/artifacts/bundle/bundle_quality_report.json`

## Победная постановка

Основная финальная конфигурация:

- architecture: Protocol-Fixed Efficient HADES;
- split: `stratified_by_workload`;
- normalization: `workload_norm`;
- metric features: `raw_plus_deviation`;
- target design: multi-head;
- обязательные targets:
  - `y_warning`;
  - `y_anomaly`;
  - `y_risk`;
  - `horizon_30s`;
  - `horizon_60s`;
  - `horizon_120s`;
  - `horizon_300s`.

Интерпретация горизонтов:

- `horizon_300s` - лучший демонстрационный early-warning горизонт;
- `horizon_120s` - основной рабочий early-warning горизонт;
- `horizon_60s` - более ранний, но шумный alert-канал;
- `horizon_30s` / `y_risk` - отдельная short-risk задача.

## Правила notebooks

Ноутбуки `01/02/03` должны читаться как цельная исследовательская линия, а не как набор служебных ячеек. В тексте нужно показывать ход мысли автора: что проверяется, почему это важно для AIOps, что стало видно на данных и какое решение после этого переносится в pipeline.

Не использовать видимый шаблон вроде `Цель раздела -> код -> график -> локальный вывод` как повторяющуюся разметку. Эта логика остается внутренней, но на странице должна выглядеть как нормальное рассуждение студента, который последовательно строит работу.

Требования:

- русские подписи графиков и таблиц;
- единые термины: `Норма`, `Предупреждение`, `Риск`;
- тяжелая логика живет в `dashboard/*.py`, notebooks запускают, объясняют и интерпретируют;
- `02` обучает baseline и проверяет работоспособность выбранного pipeline;
- `03` обучает финальную модель, сравнивает ее с baseline и экспортирует bundle.

### Стандарт оформления 01/02/03

Правила графиков:

- один практический вопрос -> один график;
- график должен помогать принять решение по данным, pipeline, модели или dashboard;
- один и тот же смысл не дублировать в другом разделе;
- подписи, легенды и оси на русском, кроме общепринятых метрик (`F1`, `AP`, `ROC-AUC`);
- графики должны идти по логике рассказа: данные -> target -> split/preprocessing -> baseline -> финальная модель -> эксплуатационный вывод.

Правила текста:

- писать связными абзацами, без карточного шаблона `что увидели / что приняли / что поменяли`;
- объяснять не только результат, но и почему он важен для следующего шага;
- в `01_eda.ipynb` не описывать будущие решения как уже принятые: EDA формулирует наблюдения и вопросы для проверки в `02`, а не оправдывает заранее выбранный pipeline;
- не использовать формулировки вида "исправили прошлую версию";
- избегать канцелярита, пустых выводов и фраз, которые можно поставить под любой график;
- после важных чисел или графиков обязательно писать, что это значит для раннего предупреждения, FP/FN, split, preprocessing или выбора модели.

## Dashboard

`dashboard/app.py` работает по `dashboard/artifacts/bundle/`.

Семантика:

- `model_score_raw` / `alert_score` - score модели, не вероятность сбоя;
- `model_score_rank` - диагностический ранг;
- `risk_score_raw` - отдельный short-risk канал;
- `alert_state` - решение alert-layer;
- `incident_marker` - факт события, не предсказание.

UI-компоненты не менять без отдельной причины. Разрешены только изменения, которые подключают финальный bundle или исправляют некорректные команды запуска.

## ONNX/INT8

ONNX/INT8 - runtime-опция для слабого CPU-only железа. Она не является основной метрикой качества.

В export path зарезервированы:

- `model.pt`;
- `model.onnx`;
- `model.int8.onnx`.

Если INT8 отсутствует, dashboard продолжает работать от exported replay bundle.

## Критерий готовности

Система готова к показу, если:

1. `01 -> 02 -> 03` выполняются через `Run All`;
2. `dashboard/artifacts/bundle/*` формируется без ручных правок;
3. Streamlit работает в `model_scoring`;
4. в replay видны успехи и ошибки alert-layer;
5. ограничения модели явно описаны в `03_fused_model.ipynb` и `model_card_product.md`.
