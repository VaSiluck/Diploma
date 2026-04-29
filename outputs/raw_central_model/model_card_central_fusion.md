# Карточка модели central fusion

## Назначение
Модель используется как исследовательский scorer для оценки состояний `Предупреждение` и `Риск` по multimodal telemetry: логам, KPI и признакам доступности модальностей.

## Данные и split
- Контракт: `outputs\raw_multirate\aligned_multirate_windows.csv`.
- Train / validation / test: 5949 / 1721 / 2567 окон.
- Test workloads: aggregation, join, rf, svd.
- Split является workload-holdout; результаты нельзя трактовать как online-production benchmark.

## Обучение
- Лучшая эпоха: 6.
- Threshold warning: 0.500.
- Threshold risk: 0.345.
- Число обучаемых параметров: 101,584.

## Сравнение с baseline
- Риск: AP=0.3566; лучший baseline `log-only` AP=0.4312; delta=-0.0746 (ниже лучшего baseline).
- Предупреждение: AP=0.4732; лучший baseline `fused` AP=0.5513; delta=-0.0781 (ниже лучшего baseline).

## Inference boundary
Модель принимает не raw_data напрямую, а aligned multi-rate contract, собранный теми же правилами, что outputs/raw_multirate/aligned_multirate_windows.csv.

## Ограничения
- Targets унаследованы от proxy-разметки notebook 02.
- Failure anchors могут включать run_end_proxy, поэтому время отказа не всегда является истинной меткой события.
- Признаки доступности модальностей могут выступать shortcut-сигналом.
- Для финальной оценки желательны repeated seeds и отдельная проверка калибровки порогов.
