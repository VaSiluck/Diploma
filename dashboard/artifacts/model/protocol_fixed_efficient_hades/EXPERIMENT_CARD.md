# Облегченный мультимодальный контур предупреждения: `final_protocol_fixed_hades`

This experiment keeps the HADES-like architecture and changes the data protocol first:

- split comparison: current, stratified-by-workload, leave-one-workload-out;
- metric normalization: `workload_norm`;
- metric features: `raw_plus_deviation`;
- separate heads for warning/risk/anomaly and horizons 30s/60s/120s/300s;
- alert-level evaluation with consecutive windows and cooldown.

## Best Test Rows By Target

| split_protocol | variant | seed | target | threshold_policy | average_precision | roc_auc | precision | recall | f1 | false_positives | missed_anomalies | torch_latency_ms_per_sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stratified_by_workload | efficient_hades | 43 | horizon_120s | validation_best_f1 | 0.6984 | 0.6793 | 0.6281 | 0.6911 | 0.6581 | 106 | 80 | 0.0350 |
| stratified_by_workload | efficient_hades | 43 | horizon_120s | validation_fp_budget_0.1 | 0.6984 | 0.6793 | 0.7067 | 0.4093 | 0.5183 | 44 | 153 | 0.0350 |
| stratified_by_workload | efficient_hades | 43 | horizon_300s | validation_best_f1 | 0.8320 | 0.6974 | 0.7774 | 0.7123 | 0.7434 | 73 | 103 | 0.0350 |
| stratified_by_workload | efficient_hades | 43 | horizon_300s | validation_fp_budget_0.1 | 0.8320 | 0.6974 | 0.8152 | 0.4190 | 0.5535 | 34 | 208 | 0.0350 |
| stratified_by_workload | efficient_hades | 43 | horizon_30s | validation_fp_budget_0.1 | 0.3312 | 0.7295 | 0.3412 | 0.4603 | 0.3919 | 56 | 34 | 0.0350 |
| stratified_by_workload | efficient_hades | 43 | horizon_30s | validation_best_f1 | 0.3312 | 0.7295 | 0.3514 | 0.2063 | 0.2600 | 24 | 50 | 0.0350 |
| stratified_by_workload | efficient_hades | 43 | horizon_60s | validation_best_f1 | 0.5098 | 0.7046 | 0.4497 | 0.5782 | 0.5060 | 104 | 62 | 0.0350 |
| stratified_by_workload | efficient_hades | 43 | horizon_60s | validation_fp_budget_0.1 | 0.5098 | 0.7046 | 0.5224 | 0.4762 | 0.4982 | 64 | 77 | 0.0350 |
| stratified_by_workload | efficient_hades | 43 | y_anomaly | validation_best_f1 | 0.7707 | 0.6746 | 0.7165 | 0.5844 | 0.6437 | 74 | 133 | 0.0350 |
| stratified_by_workload | efficient_hades | 43 | y_anomaly | validation_fp_budget_0.1 | 0.7707 | 0.6746 | 0.7546 | 0.3844 | 0.5093 | 40 | 197 | 0.0350 |
| stratified_by_workload | efficient_hades | 43 | y_risk | validation_fp_budget_0.1 | 0.3020 | 0.7275 | 0.3250 | 0.4127 | 0.3636 | 54 | 37 | 0.0350 |
| stratified_by_workload | efficient_hades | 43 | y_risk | validation_best_f1 | 0.3020 | 0.7275 | 0.3830 | 0.2857 | 0.3273 | 29 | 45 | 0.0350 |
| stratified_by_workload | efficient_hades | 43 | y_warning | validation_best_f1 | 0.7721 | 0.6751 | 0.7186 | 0.5906 | 0.6484 | 74 | 131 | 0.0350 |
| stratified_by_workload | efficient_hades | 43 | y_warning | validation_fp_budget_0.1 | 0.7721 | 0.6751 | 0.7593 | 0.3844 | 0.5104 | 39 | 197 | 0.0350 |

## Alert-Level Metrics

| split_protocol | variant | seed | target | threshold | consecutive_windows | cooldown_windows | alert_events | false_alerts | detected_incidents | missed_incidents |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stratified_by_workload | efficient_hades | 43 | horizon_60s | 0.6688 | 1 | 0 | 189 | 55 | 22 | 9 |
| stratified_by_workload | efficient_hades | 43 | horizon_60s | 0.6688 | 1 | 3 | 62 | 18 | 22 | 9 |
| stratified_by_workload | efficient_hades | 43 | horizon_60s | 0.6688 | 1 | 6 | 40 | 11 | 15 | 16 |
| stratified_by_workload | efficient_hades | 43 | horizon_60s | 0.6688 | 2 | 0 | 84 | 24 | 21 | 10 |
| stratified_by_workload | efficient_hades | 43 | horizon_60s | 0.6688 | 2 | 3 | 45 | 13 | 20 | 11 |
| stratified_by_workload | efficient_hades | 43 | horizon_60s | 0.6688 | 2 | 6 | 35 | 10 | 15 | 16 |
| stratified_by_workload | efficient_hades | 43 | horizon_60s | 0.6688 | 3 | 0 | 50 | 15 | 17 | 14 |
| stratified_by_workload | efficient_hades | 43 | horizon_60s | 0.6688 | 3 | 3 | 34 | 10 | 16 | 15 |
| stratified_by_workload | efficient_hades | 43 | horizon_60s | 0.6688 | 3 | 6 | 31 | 10 | 15 | 16 |

## Files

- `comparison_metrics.csv`
- `test_predictions.csv.gz`
- `validation_predictions.csv.gz`
- `workload_metrics.csv`
- `alert_level_metrics.csv`
- `feature_manifest.json`
- `train_manifest.json`
- `latency_and_export_report.json`