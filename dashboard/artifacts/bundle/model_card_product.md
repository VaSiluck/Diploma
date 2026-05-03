# Финальная модель dashboard

**Финальная модель:** мультимодальная strict-intensity fusion.

Модель использует логи, KPI и маски доступности модальностей. Основной score для дашборда — `model_score_raw` на основе `warning_intensity_score`; диагностический strict risk-channel — `risk_score_raw` / `state_score`.

## Alert-layer

- Порог наблюдения: `0.3879`
- Порог тревоги: `0.6156`
- Подтверждение тревоги: `2` окна подряд

## Run-level replay metrics

- Failure-run recall: `0.100`
- Precision по run-alert: `0.500`
- False-alert normal rate: `0.120`
- Alert density: `0.059`
- Median lead time, сек: `56.0`

## Ограничения

Это лабораторный replay-прототип. Он не является production-ready системой и не должен трактоваться как идеальный бинарный оракул аварий. Его назначение — честно показать динамику раннего предупреждения, вклад модальностей и ошибки alert-layer.
