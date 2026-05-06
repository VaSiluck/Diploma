# Protocol-Fixed Efficient HADES: `final_protocol_fixed_hades`

This experiment keeps the HADES-like architecture and changes the data protocol first:

- split comparison: current, stratified-by-workload, leave-one-workload-out;
- metric normalization: `workload_norm`;
- metric features: `raw_plus_deviation`;
- separate heads for warning/risk/anomaly and horizons 30s/60s/120s/300s;
- alert-level evaluation with consecutive windows and cooldown.

## Best Test Rows By Target

| split_protocol | variant | seed | target | threshold_policy | average_precision | roc_auc | precision | recall | f1 | false_positives | missed_anomalies | torch_latency_ms_per_sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stratified_by_workload | modern_metric | 43 | horizon_120s | validation_best_f1 | 0.7064 | 0.6867 | 0.6701 | 0.5097 | 0.5789 | 65 | 127 | 0.0247 |
| stratified_by_workload | modern_metric | 43 | horizon_120s | validation_fp_budget_0.1 | 0.7064 | 0.6867 | 0.6951 | 0.4402 | 0.5390 | 50 | 145 | 0.0247 |
| stratified_by_workload | modern_metric | 43 | horizon_300s | validation_best_f1 | 0.8318 | 0.6946 | 0.7855 | 0.6034 | 0.6825 | 59 | 142 | 0.0247 |
| stratified_by_workload | modern_metric | 43 | horizon_300s | validation_fp_budget_0.1 | 0.8318 | 0.6946 | 0.8433 | 0.3156 | 0.4593 | 21 | 245 | 0.0247 |
| stratified_by_workload | modern_metric | 43 | horizon_30s | validation_fp_budget_0.1 | 0.3182 | 0.7379 | 0.3299 | 0.5079 | 0.4000 | 65 | 31 | 0.0247 |
| stratified_by_workload | modern_metric | 43 | horizon_30s | validation_best_f1 | 0.3182 | 0.7379 | 0.3333 | 0.4762 | 0.3922 | 60 | 33 | 0.0247 |
| stratified_by_workload | modern_metric | 43 | horizon_60s | validation_best_f1 | 0.5061 | 0.7020 | 0.4489 | 0.5374 | 0.4892 | 97 | 68 | 0.0247 |
| stratified_by_workload | modern_metric | 43 | horizon_60s | validation_fp_budget_0.1 | 0.5061 | 0.7020 | 0.5078 | 0.4422 | 0.4727 | 63 | 82 | 0.0247 |
| stratified_by_workload | modern_metric | 43 | y_anomaly | validation_best_f1 | 0.7771 | 0.6810 | 0.7346 | 0.5969 | 0.6586 | 69 | 129 | 0.0247 |
| stratified_by_workload | modern_metric | 43 | y_anomaly | validation_fp_budget_0.1 | 0.7771 | 0.6810 | 0.7655 | 0.3469 | 0.4774 | 34 | 209 | 0.0247 |
| stratified_by_workload | modern_metric | 43 | y_risk | validation_fp_budget_0.1 | 0.3099 | 0.7355 | 0.3333 | 0.5238 | 0.4074 | 66 | 30 | 0.0247 |
| stratified_by_workload | modern_metric | 43 | y_risk | validation_best_f1 | 0.3099 | 0.7355 | 0.3295 | 0.4603 | 0.3841 | 59 | 34 | 0.0247 |
| stratified_by_workload | modern_metric | 43 | y_warning | validation_best_f1 | 0.7772 | 0.6793 | 0.7333 | 0.6188 | 0.6712 | 72 | 122 | 0.0247 |
| stratified_by_workload | modern_metric | 43 | y_warning | validation_fp_budget_0.1 | 0.7772 | 0.6793 | 0.7737 | 0.3312 | 0.4639 | 31 | 214 | 0.0247 |

## Alert-Level Metrics

| split_protocol | variant | seed | target | threshold | consecutive_windows | cooldown_windows | alert_events | false_alerts | detected_incidents | missed_incidents |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stratified_by_workload | modern_metric | 43 | horizon_60s | 0.6941 | 1 | 0 | 176 | 54 | 20 | 11 |
| stratified_by_workload | modern_metric | 43 | horizon_60s | 0.6941 | 1 | 3 | 57 | 18 | 20 | 11 |
| stratified_by_workload | modern_metric | 43 | horizon_60s | 0.6941 | 1 | 6 | 40 | 13 | 13 | 18 |
| stratified_by_workload | modern_metric | 43 | horizon_60s | 0.6941 | 2 | 0 | 77 | 23 | 19 | 12 |
| stratified_by_workload | modern_metric | 43 | horizon_60s | 0.6941 | 2 | 3 | 41 | 12 | 19 | 12 |
| stratified_by_workload | modern_metric | 43 | horizon_60s | 0.6941 | 2 | 6 | 32 | 10 | 13 | 18 |
| stratified_by_workload | modern_metric | 43 | horizon_60s | 0.6941 | 3 | 0 | 45 | 12 | 17 | 14 |
| stratified_by_workload | modern_metric | 43 | horizon_60s | 0.6941 | 3 | 3 | 30 | 8 | 15 | 16 |
| stratified_by_workload | modern_metric | 43 | horizon_60s | 0.6941 | 3 | 6 | 27 | 8 | 14 | 17 |

## Files

- `comparison_metrics.csv`
- `test_predictions.csv.gz`
- `validation_predictions.csv.gz`
- `workload_metrics.csv`
- `alert_level_metrics.csv`
- `feature_manifest.json`
- `train_manifest.json`
- `latency_and_export_report.json`