# Protocol-Fixed Efficient HADES: `protocol_fixed_cuda_h60`

This experiment keeps the HADES-like architecture and changes the data protocol first:

- split comparison: current, stratified-by-workload, leave-one-workload-out;
- metric normalization: `workload_norm`;
- metric features: `raw_plus_deviation`;
- separate heads for warning/risk/anomaly and horizons 30s/60s/120s/300s;
- alert-level evaluation with consecutive windows and cooldown.

## Best Test Rows By Target

| split_protocol | variant | seed | target | threshold_policy | average_precision | roc_auc | precision | recall | f1 | false_positives | missed_anomalies | torch_latency_ms_per_sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stratified_by_workload | modern_metric | 43 | horizon_120s | validation_best_f1 | 0.7032 | 0.7563 | 0.6379 | 0.8009 | 0.7102 | 105 | 46 | 0.0251 |
| stratified_by_workload | efficient_hades | 43 | horizon_120s | validation_best_f1 | 0.7245 | 0.7587 | 0.6254 | 0.7662 | 0.6887 | 106 | 54 | 0.0272 |
| stratified_by_workload | modern_metric | 44 | horizon_120s | validation_best_f1 | 0.6882 | 0.7390 | 0.6613 | 0.7100 | 0.6848 | 84 | 67 | 0.0278 |
| stratified_by_workload | efficient_hades | 44 | horizon_120s | validation_best_f1 | 0.7113 | 0.7461 | 0.6559 | 0.7013 | 0.6778 | 85 | 69 | 0.0299 |
| stratified_by_workload | efficient_hades | 42 | horizon_120s | validation_best_f1 | 0.6953 | 0.7176 | 0.5745 | 0.8009 | 0.6691 | 137 | 46 | 0.0322 |
| stratified_by_workload | modern_metric | 43 | horizon_300s | validation_best_f1 | 0.8354 | 0.7678 | 0.7133 | 0.9907 | 0.8295 | 129 | 3 | 0.0251 |
| stratified_by_workload | modern_metric | 44 | horizon_300s | validation_best_f1 | 0.8101 | 0.7548 | 0.7867 | 0.8426 | 0.8137 | 74 | 51 | 0.0278 |
| stratified_by_workload | efficient_hades | 43 | horizon_300s | validation_best_f1 | 0.8536 | 0.7695 | 0.7337 | 0.9012 | 0.8089 | 106 | 32 | 0.0272 |
| stratified_by_workload | modern_metric | 42 | horizon_300s | validation_best_f1 | 0.8381 | 0.7446 | 0.7493 | 0.8765 | 0.8080 | 95 | 40 | 0.0245 |
| stratified_by_workload | efficient_hades | 42 | horizon_300s | validation_best_f1 | 0.8502 | 0.7685 | 0.7520 | 0.8704 | 0.8069 | 93 | 42 | 0.0322 |
| stratified_by_workload | efficient_hades | 44 | horizon_30s | validation_fp_budget_0.1 | 0.2834 | 0.7930 | 0.3659 | 0.5085 | 0.4255 | 52 | 29 | 0.0299 |
| stratified_by_workload | efficient_hades | 43 | horizon_30s | validation_best_f1 | 0.2889 | 0.7690 | 0.2946 | 0.5593 | 0.3860 | 79 | 26 | 0.0272 |
| stratified_by_workload | efficient_hades | 44 | horizon_30s | validation_best_f1 | 0.2834 | 0.7930 | 0.2883 | 0.5424 | 0.3765 | 79 | 27 | 0.0299 |
| stratified_by_workload | modern_metric | 44 | horizon_30s | validation_best_f1 | 0.2779 | 0.7574 | 0.2970 | 0.5085 | 0.3750 | 71 | 29 | 0.0278 |
| leave_one_workload_out:aggregation | modern_metric | 42 | horizon_30s | validation_fp_budget_0.1 | 0.1409 | 0.8150 | 0.2222 | 1.0000 | 0.3636 | 21 | 0 | 0.0304 |
| leave_one_workload_out:aggregation | modern_metric | 43 | horizon_60s | validation_best_f1 | 0.5071 | 0.9167 | 0.5263 | 0.6667 | 0.5882 | 9 | 5 | 0.0283 |
| leave_one_workload_out:aggregation | modern_metric | 42 | horizon_60s | validation_best_f1 | 0.3745 | 0.8840 | 0.3947 | 1.0000 | 0.5660 | 23 | 0 | 0.0304 |
| leave_one_workload_out:aggregation | modern_metric | 44 | horizon_60s | validation_best_f1 | 0.8029 | 0.9400 | 0.4583 | 0.7333 | 0.5641 | 13 | 4 | 0.0248 |
| leave_one_workload_out:aggregation | modern_metric | 44 | horizon_60s | validation_fp_budget_0.1 | 0.8029 | 0.9400 | 0.4762 | 0.6667 | 0.5556 | 11 | 5 | 0.0248 |
| stratified_by_workload | efficient_hades | 44 | horizon_60s | validation_best_f1 | 0.5283 | 0.7631 | 0.4754 | 0.6444 | 0.5472 | 96 | 48 | 0.0299 |
| stratified_by_workload | modern_metric | 43 | y_anomaly | validation_best_f1 | 0.7823 | 0.7628 | 0.6783 | 0.8785 | 0.7655 | 120 | 35 | 0.0251 |
| stratified_by_workload | efficient_hades | 44 | y_anomaly | validation_best_f1 | 0.8082 | 0.7677 | 0.6747 | 0.8715 | 0.7606 | 121 | 37 | 0.0299 |
| stratified_by_workload | efficient_hades | 43 | y_anomaly | validation_best_f1 | 0.8127 | 0.7736 | 0.7267 | 0.7847 | 0.7546 | 85 | 62 | 0.0272 |
| stratified_by_workload | modern_metric | 44 | y_anomaly | validation_best_f1 | 0.7678 | 0.7539 | 0.7562 | 0.7431 | 0.7496 | 69 | 74 | 0.0278 |
| stratified_by_workload | efficient_hades | 42 | y_anomaly | validation_best_f1 | 0.7987 | 0.7441 | 0.6480 | 0.8819 | 0.7471 | 138 | 34 | 0.0322 |
| stratified_by_workload | efficient_hades | 44 | y_risk | validation_fp_budget_0.1 | 0.2814 | 0.7929 | 0.3659 | 0.5085 | 0.4255 | 52 | 29 | 0.0299 |
| leave_one_workload_out:aggregation | efficient_hades | 44 | y_risk | validation_best_f1 | 0.2460 | 0.7951 | 0.5000 | 0.3333 | 0.4000 | 2 | 4 | 0.0330 |
| stratified_by_workload | efficient_hades | 43 | y_risk | validation_best_f1 | 0.2825 | 0.7706 | 0.2917 | 0.5932 | 0.3911 | 85 | 24 | 0.0272 |
| stratified_by_workload | modern_metric | 44 | y_risk | validation_best_f1 | 0.2692 | 0.7566 | 0.2941 | 0.5085 | 0.3727 | 72 | 29 | 0.0278 |
| stratified_by_workload | efficient_hades | 44 | y_risk | validation_best_f1 | 0.2814 | 0.7929 | 0.2593 | 0.5932 | 0.3608 | 100 | 24 | 0.0299 |
| stratified_by_workload | modern_metric | 43 | y_warning | validation_best_f1 | 0.7883 | 0.7683 | 0.6848 | 0.8750 | 0.7683 | 116 | 36 | 0.0251 |
| stratified_by_workload | efficient_hades | 43 | y_warning | validation_best_f1 | 0.8237 | 0.7832 | 0.7325 | 0.7986 | 0.7641 | 84 | 58 | 0.0272 |
| stratified_by_workload | efficient_hades | 44 | y_warning | validation_best_f1 | 0.8149 | 0.7725 | 0.6747 | 0.8715 | 0.7606 | 121 | 37 | 0.0299 |
| stratified_by_workload | modern_metric | 44 | y_warning | validation_best_f1 | 0.7749 | 0.7570 | 0.7544 | 0.7465 | 0.7504 | 70 | 73 | 0.0278 |
| stratified_by_workload | efficient_hades | 42 | y_warning | validation_best_f1 | 0.7907 | 0.7385 | 0.6544 | 0.8611 | 0.7436 | 131 | 40 | 0.0322 |

## Alert-Level Metrics

| split_protocol | variant | seed | target | threshold | consecutive_windows | cooldown_windows | alert_events | false_alerts | detected_incidents | missed_incidents |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current | efficient_hades | 42 | horizon_60s | 0.5315 | 1 | 0 | 310 | 190 | 27 | 3 |
| current | efficient_hades | 42 | horizon_60s | 0.5315 | 1 | 3 | 107 | 63 | 26 | 4 |
| current | efficient_hades | 42 | horizon_60s | 0.5315 | 1 | 6 | 78 | 45 | 21 | 9 |
| current | efficient_hades | 42 | horizon_60s | 0.5315 | 2 | 0 | 132 | 82 | 17 | 13 |
| current | efficient_hades | 42 | horizon_60s | 0.5315 | 2 | 3 | 70 | 43 | 17 | 13 |
| current | efficient_hades | 42 | horizon_60s | 0.5315 | 2 | 6 | 52 | 30 | 16 | 14 |
| current | efficient_hades | 42 | horizon_60s | 0.5315 | 3 | 0 | 83 | 54 | 16 | 14 |
| current | efficient_hades | 42 | horizon_60s | 0.5315 | 3 | 3 | 54 | 34 | 13 | 17 |
| current | efficient_hades | 42 | horizon_60s | 0.5315 | 3 | 6 | 46 | 27 | 15 | 15 |
| current | efficient_hades | 43 | horizon_60s | 0.9475 | 1 | 0 | 55 | 27 | 6 | 24 |
| current | efficient_hades | 43 | horizon_60s | 0.9475 | 1 | 3 | 27 | 16 | 5 | 25 |
| current | efficient_hades | 43 | horizon_60s | 0.9475 | 1 | 6 | 23 | 15 | 3 | 27 |
| current | efficient_hades | 43 | horizon_60s | 0.9475 | 2 | 0 | 18 | 7 | 3 | 27 |
| current | efficient_hades | 43 | horizon_60s | 0.9475 | 2 | 3 | 11 | 5 | 3 | 27 |
| current | efficient_hades | 43 | horizon_60s | 0.9475 | 2 | 6 | 10 | 5 | 2 | 28 |
| current | efficient_hades | 43 | horizon_60s | 0.9475 | 3 | 0 | 10 | 4 | 3 | 27 |
| current | efficient_hades | 43 | horizon_60s | 0.9475 | 3 | 3 | 8 | 4 | 3 | 27 |
| current | efficient_hades | 43 | horizon_60s | 0.9475 | 3 | 6 | 7 | 4 | 2 | 28 |
| current | efficient_hades | 44 | horizon_60s | 0.7310 | 1 | 0 | 286 | 215 | 23 | 7 |
| current | efficient_hades | 44 | horizon_60s | 0.7310 | 1 | 3 | 102 | 69 | 23 | 7 |
| current | efficient_hades | 44 | horizon_60s | 0.7310 | 1 | 6 | 74 | 48 | 19 | 11 |
| current | efficient_hades | 44 | horizon_60s | 0.7310 | 2 | 0 | 122 | 96 | 8 | 22 |
| current | efficient_hades | 44 | horizon_60s | 0.7310 | 2 | 3 | 58 | 45 | 8 | 22 |
| current | efficient_hades | 44 | horizon_60s | 0.7310 | 2 | 6 | 43 | 33 | 6 | 24 |
| current | efficient_hades | 44 | horizon_60s | 0.7310 | 3 | 0 | 77 | 61 | 7 | 23 |
| current | efficient_hades | 44 | horizon_60s | 0.7310 | 3 | 3 | 44 | 35 | 7 | 23 |
| current | efficient_hades | 44 | horizon_60s | 0.7310 | 3 | 6 | 33 | 25 | 6 | 24 |
| current | modern_metric | 42 | horizon_60s | 0.9213 | 1 | 0 | 112 | 73 | 8 | 22 |
| current | modern_metric | 42 | horizon_60s | 0.9213 | 1 | 3 | 39 | 25 | 8 | 22 |
| current | modern_metric | 42 | horizon_60s | 0.9213 | 1 | 6 | 28 | 18 | 6 | 24 |
| current | modern_metric | 42 | horizon_60s | 0.9213 | 2 | 0 | 49 | 32 | 5 | 25 |
| current | modern_metric | 42 | horizon_60s | 0.9213 | 2 | 3 | 27 | 18 | 5 | 25 |
| current | modern_metric | 42 | horizon_60s | 0.9213 | 2 | 6 | 22 | 15 | 4 | 26 |
| current | modern_metric | 42 | horizon_60s | 0.9213 | 3 | 0 | 28 | 19 | 3 | 27 |
| current | modern_metric | 42 | horizon_60s | 0.9213 | 3 | 3 | 18 | 13 | 3 | 27 |
| current | modern_metric | 42 | horizon_60s | 0.9213 | 3 | 6 | 16 | 12 | 2 | 28 |
| current | modern_metric | 43 | horizon_60s | 0.7084 | 1 | 0 | 182 | 139 | 8 | 22 |
| current | modern_metric | 43 | horizon_60s | 0.7084 | 1 | 3 | 58 | 43 | 7 | 23 |
| current | modern_metric | 43 | horizon_60s | 0.7084 | 1 | 6 | 39 | 29 | 4 | 26 |
| current | modern_metric | 43 | horizon_60s | 0.7084 | 2 | 0 | 83 | 65 | 4 | 26 |
| current | modern_metric | 43 | horizon_60s | 0.7084 | 2 | 3 | 39 | 30 | 4 | 26 |
| current | modern_metric | 43 | horizon_60s | 0.7084 | 2 | 6 | 30 | 23 | 3 | 27 |
| current | modern_metric | 43 | horizon_60s | 0.7084 | 3 | 0 | 52 | 40 | 5 | 25 |
| current | modern_metric | 43 | horizon_60s | 0.7084 | 3 | 3 | 30 | 23 | 5 | 25 |
| current | modern_metric | 43 | horizon_60s | 0.7084 | 3 | 6 | 25 | 19 | 4 | 26 |
| current | modern_metric | 44 | horizon_60s | 0.6323 | 1 | 0 | 330 | 244 | 25 | 5 |
| current | modern_metric | 44 | horizon_60s | 0.6323 | 1 | 3 | 116 | 79 | 25 | 5 |
| current | modern_metric | 44 | horizon_60s | 0.6323 | 1 | 6 | 83 | 55 | 19 | 11 |
| current | modern_metric | 44 | horizon_60s | 0.6323 | 2 | 0 | 142 | 108 | 13 | 17 |
| current | modern_metric | 44 | horizon_60s | 0.6323 | 2 | 3 | 68 | 50 | 13 | 17 |
| current | modern_metric | 44 | horizon_60s | 0.6323 | 2 | 6 | 50 | 35 | 11 | 19 |
| current | modern_metric | 44 | horizon_60s | 0.6323 | 3 | 0 | 90 | 70 | 10 | 20 |
| current | modern_metric | 44 | horizon_60s | 0.6323 | 3 | 3 | 54 | 42 | 10 | 20 |
| current | modern_metric | 44 | horizon_60s | 0.6323 | 3 | 6 | 42 | 31 | 9 | 21 |
| leave_one_workload_out:aggregation | efficient_hades | 42 | horizon_60s | 0.8668 | 1 | 0 | 0 | 0 | 0 | 3 |
| leave_one_workload_out:aggregation | efficient_hades | 42 | horizon_60s | 0.8668 | 1 | 3 | 0 | 0 | 0 | 3 |
| leave_one_workload_out:aggregation | efficient_hades | 42 | horizon_60s | 0.8668 | 1 | 6 | 0 | 0 | 0 | 3 |
| leave_one_workload_out:aggregation | efficient_hades | 42 | horizon_60s | 0.8668 | 2 | 0 | 0 | 0 | 0 | 3 |
| leave_one_workload_out:aggregation | efficient_hades | 42 | horizon_60s | 0.8668 | 2 | 3 | 0 | 0 | 0 | 3 |
| leave_one_workload_out:aggregation | efficient_hades | 42 | horizon_60s | 0.8668 | 2 | 6 | 0 | 0 | 0 | 3 |
| leave_one_workload_out:aggregation | efficient_hades | 42 | horizon_60s | 0.8668 | 3 | 0 | 0 | 0 | 0 | 3 |
| leave_one_workload_out:aggregation | efficient_hades | 42 | horizon_60s | 0.8668 | 3 | 3 | 0 | 0 | 0 | 3 |
| leave_one_workload_out:aggregation | efficient_hades | 42 | horizon_60s | 0.8668 | 3 | 6 | 0 | 0 | 0 | 3 |
| leave_one_workload_out:aggregation | efficient_hades | 43 | horizon_60s | 0.8574 | 1 | 0 | 22 | 16 | 1 | 2 |
| leave_one_workload_out:aggregation | efficient_hades | 43 | horizon_60s | 0.8574 | 1 | 3 | 9 | 7 | 1 | 2 |
| leave_one_workload_out:aggregation | efficient_hades | 43 | horizon_60s | 0.8574 | 1 | 6 | 7 | 6 | 0 | 3 |
| leave_one_workload_out:aggregation | efficient_hades | 43 | horizon_60s | 0.8574 | 2 | 0 | 8 | 5 | 1 | 2 |
| leave_one_workload_out:aggregation | efficient_hades | 43 | horizon_60s | 0.8574 | 2 | 3 | 5 | 4 | 1 | 2 |
| leave_one_workload_out:aggregation | efficient_hades | 43 | horizon_60s | 0.8574 | 2 | 6 | 4 | 3 | 1 | 2 |
| leave_one_workload_out:aggregation | efficient_hades | 43 | horizon_60s | 0.8574 | 3 | 0 | 6 | 4 | 1 | 2 |
| leave_one_workload_out:aggregation | efficient_hades | 43 | horizon_60s | 0.8574 | 3 | 3 | 4 | 3 | 1 | 2 |
| leave_one_workload_out:aggregation | efficient_hades | 43 | horizon_60s | 0.8574 | 3 | 6 | 4 | 3 | 1 | 2 |
| leave_one_workload_out:aggregation | efficient_hades | 44 | horizon_60s | 0.9152 | 1 | 0 | 0 | 0 | 0 | 3 |
| leave_one_workload_out:aggregation | efficient_hades | 44 | horizon_60s | 0.9152 | 1 | 3 | 0 | 0 | 0 | 3 |
| leave_one_workload_out:aggregation | efficient_hades | 44 | horizon_60s | 0.9152 | 1 | 6 | 0 | 0 | 0 | 3 |
| leave_one_workload_out:aggregation | efficient_hades | 44 | horizon_60s | 0.9152 | 2 | 0 | 0 | 0 | 0 | 3 |
| leave_one_workload_out:aggregation | efficient_hades | 44 | horizon_60s | 0.9152 | 2 | 3 | 0 | 0 | 0 | 3 |
| leave_one_workload_out:aggregation | efficient_hades | 44 | horizon_60s | 0.9152 | 2 | 6 | 0 | 0 | 0 | 3 |
| leave_one_workload_out:aggregation | efficient_hades | 44 | horizon_60s | 0.9152 | 3 | 0 | 0 | 0 | 0 | 3 |
| leave_one_workload_out:aggregation | efficient_hades | 44 | horizon_60s | 0.9152 | 3 | 3 | 0 | 0 | 0 | 3 |

## Files

- `comparison_metrics.csv`
- `test_predictions.csv.gz`
- `validation_predictions.csv.gz`
- `workload_metrics.csv`
- `alert_level_metrics.csv`
- `feature_manifest.json`
- `train_manifest.json`
- `latency_and_export_report.json`