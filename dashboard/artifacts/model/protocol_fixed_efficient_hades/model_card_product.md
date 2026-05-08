# Финальная модель dashboard

**Модель:** Облегченный мультимодальный контур предупреждения.

Архитектурный путь сохраняет HADES-like смысл: log sequence modeling, metric time-series segment modeling, cross-modal fusion и multi-head early-warning targets. Название подчеркивает прикладной и облегченный характер реализации, а не заявку на новую фундаментальную архитектуру.

## Основные каналы

- `model_score_raw` = `horizon_300s`: главный демонстрационный early-warning score.
- `model_score_120s`: основной рабочий early-warning канал.
- `model_score_60s`: агрессивный ранний канал, ожидаемо шумнее.
- `risk_score_30s`: отдельный short-horizon risk канал.

## Test metrics выбранного checkpoint

- `horizon_30s`: F1=0.260, Precision=0.351, Recall=0.206, AP=0.331, ROC-AUC=0.730
- `horizon_60s`: F1=0.506, Precision=0.450, Recall=0.578, AP=0.510, ROC-AUC=0.705
- `horizon_120s`: F1=0.658, Precision=0.628, Recall=0.691, AP=0.698, ROC-AUC=0.679
- `horizon_300s`: F1=0.743, Precision=0.777, Recall=0.712, AP=0.832, ROC-AUC=0.697

## Runtime

- Torch checkpoint: `available`
- ONNX FP32: `available`
- ONNX INT8: `available`

ONNX/INT8 оставлены как совместимый runtime-слот для слабых CPU-only систем. До переноса полного checkpoint из обучающей машины dashboard работает от сохраненного replay bundle, а не от online forward pass.

## Ограничения

Score модели не является вероятностью сбоя. Alert-layer отделен от фактического incident marker. Пороги выбраны по validation и не подбираются по test.
