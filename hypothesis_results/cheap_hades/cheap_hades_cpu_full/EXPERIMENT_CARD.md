# Cheap HADES Hypothesis: `cheap_hades_cpu_full`

Purpose: test a CPU-only HADES-like multimodal detector without online LLM or large Transformer inference.

Online path: Drain-like template parser -> template_id sequence -> nn.Embedding -> pooling/Conv1D; KPI window statistics -> normalization -> tiny MLP; cheap fusion -> sigmoid anomaly score.

This is not the final production training contour and does not replace `dashboard.train_multimodal_fusion`.

## Primary test metrics

| source | model | seed | target | average_precision | roc_auc | precision | recall | f1 | false_positives | missed_anomalies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cheap_hades | metric_only | 42 | y_anomaly | 0.4960 | 0.7324 | 0.3476 | 1.0000 | 0.5158 | 428 | 0 |
| cheap_hades | metric_only | 43 | y_anomaly | 0.4260 | 0.6821 | 0.3383 | 0.9868 | 0.5039 | 440 | 3 |
| cheap_hades | metric_only | 44 | y_anomaly | 0.5354 | 0.7697 | 0.3492 | 0.9956 | 0.5171 | 423 | 1 |
| cheap_hades | log_only | 42 | y_anomaly | 0.4128 | 0.6324 | 0.3444 | 1.0000 | 0.5124 | 434 | 0 |
| cheap_hades | log_only | 43 | y_anomaly | 0.4104 | 0.6277 | 0.3299 | 0.9956 | 0.4956 | 461 | 1 |
| cheap_hades | log_only | 44 | y_anomaly | 0.3879 | 0.5819 | 0.3319 | 1.0000 | 0.4984 | 459 | 0 |
| cheap_hades | concat | 42 | y_anomaly | 0.5247 | 0.7372 | 0.3486 | 1.0000 | 0.5170 | 426 | 0 |
| cheap_hades | concat | 43 | y_anomaly | 0.4577 | 0.6864 | 0.3403 | 1.0000 | 0.5078 | 442 | 0 |
| cheap_hades | concat | 44 | y_anomaly | 0.4623 | 0.7062 | 0.3418 | 1.0000 | 0.5095 | 439 | 0 |
| cheap_hades | product_diff | 42 | y_anomaly | 0.4002 | 0.6155 | 0.3436 | 0.9781 | 0.5086 | 426 | 5 |
| cheap_hades | product_diff | 43 | y_anomaly | 0.6018 | 0.6918 | 0.3358 | 1.0000 | 0.5028 | 451 | 0 |
| cheap_hades | product_diff | 44 | y_anomaly | 0.4470 | 0.6769 | 0.3418 | 1.0000 | 0.5095 | 439 | 0 |
| cheap_hades | gated | 42 | y_anomaly | 0.4935 | 0.6641 | 0.3408 | 1.0000 | 0.5084 | 441 | 0 |
| cheap_hades | gated | 43 | y_anomaly | 0.5463 | 0.7699 | 0.3418 | 1.0000 | 0.5095 | 439 | 0 |
| cheap_hades | gated | 44 | y_anomaly | 0.5642 | 0.7380 | 0.3358 | 1.0000 | 0.5028 | 451 | 0 |
| existing_baseline_predictions | log-only | None | y_anomaly | 0.4146 | 0.6627 | 0.4135 | 0.9430 | 0.5749 | 305 | 13 |
| existing_baseline_predictions | kpi-only | None | y_anomaly | 0.4306 | 0.6261 | 0.3792 | 0.7982 | 0.5141 | 298 | 46 |
| existing_baseline_predictions | availability-only | None | y_anomaly | 0.3544 | 0.5572 | 0.3354 | 0.9518 | 0.4960 | 430 | 11 |
| existing_baseline_predictions | fused | None | y_anomaly | 0.4451 | 0.6413 | 0.3743 | 0.8289 | 0.5157 | 316 | 39 |
| existing_final_model | final_multimodal:final_multimodal_strict_intensity_gap10 | None | y_anomaly | 0.4620 | 0.6927 | 0.3378 | 0.9956 | 0.5044 | 445 | 1 |

## Files

- `comparison_metrics.csv`: per-seed validation/test metrics for cheap variants and imported existing models.
- `comparison_metrics_aggregated.csv`: mean/std by model and split.
- `test_predictions.csv.gz`, `validation_predictions.csv.gz`: cheap model scores.
- `feature_manifest.json`, `preprocessor_state.json`: reproducible feature contract for this hypothesis.
- `exports/`: ONNX and INT8 ONNX for the best validation-F1 cheap variant when export succeeds.