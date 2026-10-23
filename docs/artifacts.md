# Карта артефактов

В репозитории сохранены данные для демонстрации, подготовленный контракт и результат выбранного обучения. Ниже перечислены существующие файлы; промежуточные выходы, которые ещё предстоит сгенерировать, отмечены отдельно.

## Демонстрация

Стандартный каталог интерфейса — [dashboard/artifacts/bundle](../dashboard/artifacts/bundle).

| Файл | Для чего нужен |
| --- | --- |
| [dashboard_model_report.json](../dashboard/artifacts/bundle/dashboard_model_report.json) | Описание выбранного запуска, каналов и runtime-артефактов |
| [replay_bundle.csv.gz](../dashboard/artifacts/bundle/replay_bundle.csv.gz) | Сохранённые окна с оценками для воспроизведения |
| [dashboard_replay_store.csv.gz](../dashboard/artifacts/bundle/dashboard_replay_store.csv.gz) | Дополнительное представление сохранённых сценариев |
| [baseline_comparison.csv](../dashboard/artifacts/bundle/baseline_comparison.csv) | Таблица сравнения с простыми моделями |
| [metrics_summary.csv](../dashboard/artifacts/bundle/metrics_summary.csv) | Сводка тревог на уровне запусков |
| [alert_events.csv](../dashboard/artifacts/bundle/alert_events.csv) | Сохранённые события тревог |
| [bundle_quality_report.json](../dashboard/artifacts/bundle/bundle_quality_report.json) | Проверки и свойства собранного bundle |
| [model_card_product.md](../dashboard/artifacts/bundle/model_card_product.md) | Карточка модели; отдельные пояснения о runtime могут относиться к более раннему состоянию проекта |

В `runtime/inference.py` минимальный компактный bundle определяется отчётом модели, replay-данными и сравнением с baseline. Веса не требуются для чтения сохранённых оценок. Live-режим дополнительно проверяет checkpoint и подготовленный срез в каталоге модели.

## Выбранная модель

Каталог: [protocol_fixed_efficient_hades](../dashboard/artifacts/model/protocol_fixed_efficient_hades).

| Группа | Файлы |
| --- | --- |
| Веса и экспорт | [model.pt](../dashboard/artifacts/model/protocol_fixed_efficient_hades/model.pt), [model.onnx](../dashboard/artifacts/model/protocol_fixed_efficient_hades/model.onnx), [model.int8.onnx](../dashboard/artifacts/model/protocol_fixed_efficient_hades/model.int8.onnx) |
| Входы для live-демонстрации | [live_inference_slice.npz](../dashboard/artifacts/model/protocol_fixed_efficient_hades/live_inference_slice.npz), [индекс окон](../dashboard/artifacts/model/protocol_fixed_efficient_hades/live_inference_index.csv.gz), [manifest среза](../dashboard/artifacts/model/protocol_fixed_efficient_hades/live_inference_manifest.json) |
| Конфигурация | [run_config.json](../dashboard/artifacts/model/protocol_fixed_efficient_hades/run_config.json), [feature_manifest.json](../dashboard/artifacts/model/protocol_fixed_efficient_hades/feature_manifest.json) |
| Происхождение обучения | [train_manifest.json](../dashboard/artifacts/model/protocol_fixed_efficient_hades/train_manifest.json), [final_training_manifest.json](../dashboard/artifacts/model/protocol_fixed_efficient_hades/final_training_manifest.json), [final_train_command.json](../dashboard/artifacts/model/protocol_fixed_efficient_hades/final_train_command.json) |
| Оконные метрики | [comparison_metrics.csv](../dashboard/artifacts/model/protocol_fixed_efficient_hades/comparison_metrics.csv), [metrics_summary.json](../dashboard/artifacts/model/protocol_fixed_efficient_hades/metrics_summary.json) |
| Разрезы оценки | [workload_metrics.csv](../dashboard/artifacts/model/protocol_fixed_efficient_hades/workload_metrics.csv), [split_summary.csv](../dashboard/artifacts/model/protocol_fixed_efficient_hades/split_summary.csv), [alert_level_metrics.csv](../dashboard/artifacts/model/protocol_fixed_efficient_hades/alert_level_metrics.csv) |
| Экспорт и время выполнения | [latency_and_export_report.json](../dashboard/artifacts/model/protocol_fixed_efficient_hades/latency_and_export_report.json) |
| Описание эксперимента | [EXPERIMENT_CARD.md](../dashboard/artifacts/model/protocol_fixed_efficient_hades/EXPERIMENT_CARD.md), [model_card_product.md](../dashboard/artifacts/model/protocol_fixed_efficient_hades/model_card_product.md) |
| Кеш текстовых представлений | [cache](../dashboard/artifacts/model/protocol_fixed_efficient_hades/cache) |

В этом же каталоге находятся копии отчёта модели, replay-данных, сравнения с baseline и отчёта качества bundle. [protocol_fixed_model_manifest.json](../dashboard/artifacts/model/protocol_fixed_efficient_hades/protocol_fixed_model_manifest.json) содержит описание выбранной модели.

В [model_runs](../dashboard/artifacts/model_runs) сохранены дополнительные ONNX-экспорты из каталога эксперимента. Рабочие имена для демонстрации — `model.pt`, `model.onnx` и `model.int8.onnx` в каталоге выбранной модели выше; названия файлов в `model_runs` отличаются.

## Данные и отчёты

| Артефакт | Содержание |
| --- | --- |
| [strict_intensity_contract.csv.gz](../dashboard/artifacts/contract/strict_intensity_contract.csv.gz) | Подготовленная таблица окон, признаков, разбиения и целевых значений |
| [strict_contract_summary.csv](../dashboard/artifacts/contract/strict_contract_summary.csv) | Размер контракта и доступность модальностей |
| [01_eda_summary.json](../dashboard/artifacts/reports/01_eda_summary.json) | Краткая сводка EDA |
| [final_data_pipeline_report.json](../dashboard/artifacts/reports/final_data_pipeline_report.json) | Отчёт подготовки данных |
| [baseline_summary_metrics.json](../dashboard/artifacts/reports/baseline_summary_metrics.json) | Сохранённая сводка baseline |
| [baseline_comparison_compact.csv](../dashboard/artifacts/reports/baseline_comparison_compact.csv) | Компактное сравнение простых моделей |
| [baseline_predictions.csv.gz](../dashboard/artifacts/reports/baseline_predictions.csv.gz) | Сохранённые предсказания baseline |

Отчёты могут относиться к разным этапам исследования. Сравнивать числа напрямую следует только при совпадении split, target и политики порога. Таблица в основном README относится к выбранному запуску из `model/protocol_fixed_efficient_hades`.

## Что не входит в текущий снимок

- Исходный `raw_data/` с полными журналами и KPI. Готовый контракт не заменяет исходники для восстановления всех временных сегментов и логовых последовательностей.
- Каталог `reports/protocol_fixed_baselines`, который создаётся командой `python -m dashboard.training.baselines`. Ноутбуки `02` и `03` обращаются к этим промежуточным файлам.
- Полный каталог обучения со всеми промежуточными checkpoint и predictions. В частности, `test_predictions.csv.gz` и `validation_predictions.csv.gz`, упомянутые в карточке эксперимента, в репозитории отсутствуют.
- Зафиксированное окружение исходного эксперимента.

Каталог `dashboard/artifacts` указан в `.gitignore`, но перечисленные файлы уже отслеживаются Git и входят в клон. Новые результаты в игнорируемых каталогах автоматически в Git не попадут.

[Вернуться к README](../README.md)
