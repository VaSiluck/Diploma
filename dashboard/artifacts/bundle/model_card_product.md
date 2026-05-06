# Финальная модель dashboard

**Модель:** Protocol-Fixed Efficient HADES.

Архитектурный путь сохраняет HADES-like смысл: log sequence modeling, metric time-series segment modeling, cross-modal fusion и multi-head early-warning targets.

## Основные каналы

- `model_score_raw` = `horizon_300s`: главный демонстрационный early-warning score.
- `model_score_120s`: основной рабочий early-warning канал.
- `model_score_60s`: агрессивный ранний канал, ожидаемо шумнее.
- `risk_score_30s`: отдельный short-horizon risk канал.

## Test metrics выбранного checkpoint

- `horizon_30s`: F1=0.306, Precision=0.245, Recall=0.407, AP=0.280, ROC-AUC=0.765
- `horizon_60s`: F1=0.531, Precision=0.455, Recall=0.637, AP=0.490, ROC-AUC=0.742
- `horizon_120s`: F1=0.710, Precision=0.638, Recall=0.801, AP=0.703, ROC-AUC=0.756
- `horizon_300s`: F1=0.829, Precision=0.713, Recall=0.991, AP=0.835, ROC-AUC=0.768

## Runtime

- Torch checkpoint: `missing_checkpoint_copy_required`
- ONNX FP32: `reserved_not_exported`
- ONNX INT8: `reserved_not_quantized`

ONNX/INT8 оставлены как совместимый runtime-слот для слабых CPU-only систем. До переноса полного checkpoint из обучающей машины dashboard работает от сохраненного replay bundle, а не от online forward pass.

## Ограничения

Score модели не является вероятностью сбоя. Alert-layer отделен от фактического incident marker. Пороги выбраны по validation и не подбираются по test.
