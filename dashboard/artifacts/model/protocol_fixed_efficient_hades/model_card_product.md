# Финальная модель dashboard

**Модель:** Protocol-Fixed Efficient HADES.

Архитектурный путь сохраняет HADES-like смысл: log sequence modeling, metric time-series segment modeling, cross-modal fusion и multi-head early-warning targets.

## Основные каналы

- `model_score_raw` = `horizon_300s`: главный демонстрационный early-warning score.
- `model_score_120s`: основной рабочий early-warning канал.
- `model_score_60s`: агрессивный ранний канал, ожидаемо шумнее.
- `risk_score_30s`: отдельный short-horizon risk канал.

## Test metrics выбранного checkpoint

- `horizon_30s`: F1=0.392, Precision=0.333, Recall=0.476, AP=0.318, ROC-AUC=0.738
- `horizon_60s`: F1=0.489, Precision=0.449, Recall=0.537, AP=0.506, ROC-AUC=0.702
- `horizon_120s`: F1=0.579, Precision=0.670, Recall=0.510, AP=0.706, ROC-AUC=0.687
- `horizon_300s`: F1=0.682, Precision=0.785, Recall=0.603, AP=0.832, ROC-AUC=0.695

## Runtime

- Torch checkpoint: `available`
- ONNX FP32: `reserved_not_exported`
- ONNX INT8: `reserved_not_quantized`

ONNX/INT8 оставлены как совместимый runtime-слот для слабых CPU-only систем. До переноса полного checkpoint из обучающей машины dashboard работает от сохраненного replay bundle, а не от online forward pass.

## Ограничения

Score модели не является вероятностью сбоя. Alert-layer отделен от фактического incident marker. Пороги выбраны по validation и не подбираются по test.
