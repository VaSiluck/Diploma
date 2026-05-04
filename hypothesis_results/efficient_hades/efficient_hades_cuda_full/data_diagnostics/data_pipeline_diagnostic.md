# Data Pipeline Diagnostic

- run_dir: `hypothesis_results\efficient_hades\efficient_hades_cuda_full`
- selected_model_for_distributions: `{'variant': 'modern_metric', 'seed': 42, 'target': 'y_anomaly'}`
- score_distribution_plot: `ok`

## Score Separation

| variant | seed | target | average_precision | roc_auc | precision | recall | f1 | false_positives | missed_anomalies | score_mean_positive | score_mean_negative | score_mean_gap | lift_at_top_5pct | lift_at_top_10pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| modern_metric | 42 | y_anomaly | 0.4376 | 0.6540 | 0.4337 | 0.9605 | 0.5975 | 286 | 9 | 0.7255 | 0.4358 | 0.2897 | 1.3855 | 0.9092 |
| modern_log | 42 | y_anomaly | 0.4208 | 0.6508 | 0.4202 | 0.9474 | 0.5822 | 298 | 12 | 0.6182 | 0.3936 | 0.2246 | 1.2123 | 0.6927 |
| modern_runtime | 42 | y_anomaly | 0.4388 | 0.6523 | 0.4140 | 0.9605 | 0.5786 | 310 | 9 | 0.5884 | 0.3860 | 0.2024 | 1.4721 | 0.8226 |
| original | 42 | y_anomaly | 0.4388 | 0.6523 | 0.4140 | 0.9605 | 0.5786 | 310 | 9 | 0.5884 | 0.3860 | 0.2024 | 1.4721 | 0.8226 |
| efficient_hades | 42 | y_anomaly | 0.4133 | 0.6491 | 0.4059 | 0.9737 | 0.5729 | 325 | 6 | 0.7718 | 0.4657 | 0.3061 | 1.2989 | 0.7360 |
| modern_log | 42 | y_risk | 0.1210 | 0.5379 | 0.1128 | 0.8923 | 0.2003 | 456 | 7 | 0.4914 | 0.4652 | 0.0261 | 1.2149 | 0.7593 |
| modern_runtime | 42 | y_risk | 0.1293 | 0.5327 | 0.1113 | 0.9538 | 0.1994 | 495 | 3 | 0.4534 | 0.4527 | 0.0006 | 1.8224 | 0.9112 |
| original | 42 | y_risk | 0.1293 | 0.5327 | 0.1113 | 0.9538 | 0.1994 | 495 | 3 | 0.4534 | 0.4527 | 0.0006 | 1.8224 | 0.9112 |
| modern_metric | 42 | y_risk | 0.1523 | 0.5704 | 0.1078 | 0.9538 | 0.1938 | 513 | 3 | 0.6695 | 0.5170 | 0.1525 | 1.2149 | 0.6075 |
| efficient_hades | 42 | y_risk | 0.1306 | 0.5730 | 0.1065 | 0.9846 | 0.1922 | 537 | 1 | 0.7199 | 0.5508 | 0.1691 | 0.9112 | 0.6075 |
| modern_metric | 42 | y_warning | 0.4376 | 0.6540 | 0.4337 | 0.9605 | 0.5975 | 286 | 9 | 0.7255 | 0.4358 | 0.2897 | 1.3855 | 0.9092 |
| modern_log | 42 | y_warning | 0.4208 | 0.6508 | 0.4202 | 0.9474 | 0.5822 | 298 | 12 | 0.6182 | 0.3936 | 0.2246 | 1.2123 | 0.6927 |
| modern_runtime | 42 | y_warning | 0.4388 | 0.6523 | 0.4140 | 0.9605 | 0.5786 | 310 | 9 | 0.5884 | 0.3860 | 0.2024 | 1.4721 | 0.8226 |
| original | 42 | y_warning | 0.4388 | 0.6523 | 0.4140 | 0.9605 | 0.5786 | 310 | 9 | 0.5884 | 0.3860 | 0.2024 | 1.4721 | 0.8226 |
| efficient_hades | 42 | y_warning | 0.4133 | 0.6491 | 0.4059 | 0.9737 | 0.5729 | 325 | 6 | 0.7718 | 0.4657 | 0.3061 | 1.2989 | 0.7360 |

## Interpretation

- If positive/negative score means and medians are close, threshold tuning cannot fix precision by itself.
- If `y_risk` scores separate better than `y_warning`, use separate heads or report risk-channel quality separately.
- If rolling/deviation feature ablations beat raw selected features, preprocessing should add deviation/residual signal before changing the architecture.
- If a normal workload/scenario has high median score and many false positives, treat it as split/domain shift before changing the model.

## Ambiguous Examples

| case_type | variant | seed | score | pred_label | y_anomaly | y_warning | y_risk | run_id | workload | scenario | scenario_type | failure_family | anchor_policy | window_ts | window_end | event_anchor | strict_lead_sec | macro_log_count | macro_warn_count | macro_error_count | macro_keyword_count | micro_log_count | micro_warn_count | micro_error_count | micro_keyword_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| false_positive | modern_metric | 42 | 0.9997 | 1 | 0 | 0 | 0 | svd/normal2 | svd | normal2 | normal | normal | normal_run | 2021-10-25 10:21:20+00:00 | 2021-10-25 10:21:30+00:00 | nan |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| false_positive | modern_metric | 42 | 0.9997 | 1 | 0 | 0 | 0 | svd/normal2 | svd | normal2 | normal | normal | normal_run | 2021-10-25 10:21:40+00:00 | 2021-10-25 10:21:50+00:00 | nan |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| false_positive | modern_metric | 42 | 0.9997 | 1 | 0 | 0 | 0 | svd/normal2 | svd | normal2 | normal | normal | normal_run | 2021-10-25 10:22:00+00:00 | 2021-10-25 10:22:10+00:00 | nan |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| false_positive | modern_metric | 42 | 0.9997 | 1 | 0 | 0 | 0 | svd/normal2 | svd | normal2 | normal | normal | normal_run | 2021-10-25 10:21:50+00:00 | 2021-10-25 10:22:00+00:00 | nan |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| false_positive | modern_metric | 42 | 0.9997 | 1 | 0 | 0 | 0 | svd/normal2 | svd | normal2 | normal | normal | normal_run | 2021-10-25 10:24:30+00:00 | 2021-10-25 10:24:40+00:00 | nan |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| false_positive | modern_metric | 42 | 0.9997 | 1 | 0 | 0 | 0 | svd/normal2 | svd | normal2 | normal | normal | normal_run | 2021-10-25 10:23:10+00:00 | 2021-10-25 10:23:20+00:00 | nan |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| false_positive | modern_metric | 42 | 0.9997 | 1 | 0 | 0 | 0 | svd/normal2 | svd | normal2 | normal | normal | normal_run | 2021-10-25 10:22:20+00:00 | 2021-10-25 10:22:30+00:00 | nan |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| false_positive | modern_metric | 42 | 0.9997 | 1 | 0 | 0 | 0 | svd/normal2 | svd | normal2 | normal | normal | normal_run | 2021-10-25 10:22:30+00:00 | 2021-10-25 10:22:40+00:00 | nan |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| false_positive | modern_metric | 42 | 0.9997 | 1 | 0 | 0 | 0 | svd/normal2 | svd | normal2 | normal | normal | normal_run | 2021-10-25 10:22:10+00:00 | 2021-10-25 10:22:20+00:00 | nan |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| false_positive | modern_metric | 42 | 0.9997 | 1 | 0 | 0 | 0 | svd/normal2 | svd | normal2 | normal | normal | normal_run | 2021-10-25 10:21:30+00:00 | 2021-10-25 10:21:40+00:00 | nan |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| false_positive | modern_metric | 42 | 0.9997 | 1 | 0 | 0 | 0 | svd/normal2 | svd | normal2 | normal | normal | normal_run | 2021-10-25 10:22:40+00:00 | 2021-10-25 10:22:50+00:00 | nan |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| false_positive | modern_metric | 42 | 0.9997 | 1 | 0 | 0 | 0 | svd/normal2 | svd | normal2 | normal | normal | normal_run | 2021-10-25 10:23:00+00:00 | 2021-10-25 10:23:10+00:00 | nan |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| false_positive | modern_metric | 42 | 0.9997 | 1 | 0 | 0 | 0 | svd/normal2 | svd | normal2 | normal | normal | normal_run | 2021-10-25 10:24:40+00:00 | 2021-10-25 10:24:50+00:00 | nan |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| false_positive | modern_metric | 42 | 0.9997 | 1 | 0 | 0 | 0 | svd/normal2 | svd | normal2 | normal | normal | normal_run | 2021-10-25 10:23:20+00:00 | 2021-10-25 10:23:30+00:00 | nan |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| false_positive | modern_metric | 42 | 0.9997 | 1 | 0 | 0 | 0 | svd/normal2 | svd | normal2 | normal | normal | normal_run | 2021-10-25 10:23:30+00:00 | 2021-10-25 10:23:40+00:00 | nan |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| false_positive | modern_metric | 42 | 0.9997 | 1 | 0 | 0 | 0 | svd/normal2 | svd | normal2 | normal | normal | normal_run | 2021-10-25 10:22:50+00:00 | 2021-10-25 10:23:00+00:00 | nan |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| false_positive | modern_metric | 42 | 0.9997 | 1 | 0 | 0 | 0 | svd/normal2 | svd | normal2 | normal | normal | normal_run | 2021-10-25 10:23:40+00:00 | 2021-10-25 10:23:50+00:00 | nan |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| false_positive | modern_metric | 42 | 0.9997 | 1 | 0 | 0 | 0 | svd/normal2 | svd | normal2 | normal | normal | normal_run | 2021-10-25 10:24:20+00:00 | 2021-10-25 10:24:30+00:00 | nan |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| false_positive | modern_metric | 42 | 0.9997 | 1 | 0 | 0 | 0 | svd/normal2 | svd | normal2 | normal | normal | normal_run | 2021-10-25 10:24:00+00:00 | 2021-10-25 10:24:10+00:00 | nan |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| false_positive | modern_metric | 42 | 0.9997 | 1 | 0 | 0 | 0 | svd/normal2 | svd | normal2 | normal | normal | normal_run | 2021-10-25 10:23:50+00:00 | 2021-10-25 10:24:00+00:00 | nan |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Workload/Scenario Score Shift

| workload | scenario_type | scenario | run_id | variant | seed | threshold | n | positives | prevalence | score_mean | score_median | score_min | score_max | predicted_positive_rate | false_positives | missed_anomalies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| svd | normal | normal2 | svd/normal2 | modern_metric | 42 | 0.0047 | 52 | 0 | 0.0000 | 0.7729 | 0.9859 | 0.0006 | 0.9997 | 0.9615 | 50 | 0 |
| svd | normal | normal4 | svd/normal4 | modern_metric | 42 | 0.0047 | 54 | 0 | 0.0000 | 0.6620 | 0.9991 | 0.0000 | 0.9993 | 0.7407 | 40 | 0 |
| svd | normal | normal3 | svd/normal3 | modern_metric | 42 | 0.0047 | 54 | 0 | 0.0000 | 0.6610 | 0.9991 | 0.0000 | 0.9991 | 0.7222 | 39 | 0 |
| svd | normal | normal1 | svd/normal1 | modern_metric | 42 | 0.0047 | 55 | 0 | 0.0000 | 0.6076 | 0.9976 | 0.0000 | 0.9981 | 0.6727 | 37 | 0 |
| aggregation | normal | normal4 | aggregation/normal4 | modern_metric | 42 | 0.0047 | 18 | 0 | 0.0000 | 0.2239 | 0.1130 | 0.0003 | 0.6850 | 0.8333 | 15 | 0 |
| aggregation | normal | normal5 | aggregation/normal5 | modern_metric | 42 | 0.0047 | 12 | 0 | 0.0000 | 0.2098 | 0.0224 | 0.0001 | 0.7099 | 0.7500 | 9 | 0 |
| join | normal | normal5 | join/normal5 | modern_metric | 42 | 0.0047 | 12 | 0 | 0.0000 | 0.5628 | 0.7844 | 0.0000 | 0.9418 | 0.6667 | 8 | 0 |
| aggregation | normal | normal3 | aggregation/normal3 | modern_metric | 42 | 0.0047 | 12 | 0 | 0.0000 | 0.2086 | 0.1146 | 0.0017 | 0.5991 | 0.5833 | 7 | 0 |
| aggregation | normal | normal2 | aggregation/normal2 | modern_metric | 42 | 0.0047 | 14 | 0 | 0.0000 | 0.1848 | 0.0894 | 0.0009 | 0.5567 | 0.5000 | 7 | 0 |
| join | normal | normal4 | join/normal4 | modern_metric | 42 | 0.0047 | 13 | 0 | 0.0000 | 0.3637 | 0.0007 | 0.0000 | 0.9144 | 0.4615 | 6 | 0 |
| join | normal | normal6 | join/normal6 | modern_metric | 42 | 0.0047 | 12 | 0 | 0.0000 | 0.3034 | 0.0252 | 0.0006 | 0.9980 | 0.5000 | 6 | 0 |
| rf | normal | normal4 | rf/normal4 | modern_metric | 42 | 0.0047 | 9 | 0 | 0.0000 | 0.2975 | 0.1823 | 0.0000 | 0.8750 | 0.5556 | 5 | 0 |
| rf | normal | normal1 | rf/normal1 | modern_metric | 42 | 0.0047 | 10 | 0 | 0.0000 | 0.1307 | 0.0239 | 0.0000 | 0.4017 | 0.5000 | 5 | 0 |
| aggregation | normal | normal6 | aggregation/normal6 | modern_metric | 42 | 0.0047 | 13 | 0 | 0.0000 | 0.0274 | 0.0039 | 0.0002 | 0.2875 | 0.3846 | 5 | 0 |
| rf | normal | normal6 | rf/normal6 | modern_metric | 42 | 0.0047 | 9 | 0 | 0.0000 | 0.3978 | 0.0005 | 0.0000 | 0.9954 | 0.4444 | 4 | 0 |
| rf | normal | normal7 | rf/normal7 | modern_metric | 42 | 0.0047 | 9 | 0 | 0.0000 | 0.3196 | 0.0003 | 0.0000 | 0.9582 | 0.4444 | 4 | 0 |
| rf | normal | normal5 | rf/normal5 | modern_metric | 42 | 0.0047 | 9 | 0 | 0.0000 | 0.2710 | 0.0008 | 0.0000 | 0.8713 | 0.4444 | 4 | 0 |
| join | normal | normal7 | join/normal7 | modern_metric | 42 | 0.0047 | 14 | 0 | 0.0000 | 0.2106 | 0.0003 | 0.0001 | 0.9810 | 0.2857 | 4 | 0 |
| rf | normal | normal3 | rf/normal3 | modern_metric | 42 | 0.0047 | 8 | 0 | 0.0000 | 0.1683 | 0.0268 | 0.0000 | 0.6775 | 0.5000 | 4 | 0 |
| aggregation | normal | normal7 | aggregation/normal7 | modern_metric | 42 | 0.0047 | 12 | 0 | 0.0000 | 0.1466 | 0.0003 | 0.0001 | 0.6233 | 0.3333 | 4 | 0 |
| join | normal | normal3 | join/normal3 | modern_metric | 42 | 0.0047 | 13 | 0 | 0.0000 | 0.1396 | 0.0000 | 0.0000 | 0.7004 | 0.3077 | 4 | 0 |
| aggregation | normal | normal1 | aggregation/normal1 | modern_metric | 42 | 0.0047 | 12 | 0 | 0.0000 | 0.0865 | 0.0005 | 0.0003 | 0.3846 | 0.3333 | 4 | 0 |
| svd | failure | mq_stress | svd/mq_stress | modern_metric | 42 | 0.0047 | 20 | 17 | 0.8500 | 0.9361 | 0.9775 | 0.4403 | 0.9996 | 1.0000 | 3 | 0 |
| svd | failure | resourcemanager_suspended | svd/resourcemanager_suspended | modern_metric | 42 | 0.0047 | 20 | 17 | 0.8500 | 0.9105 | 0.9452 | 0.6075 | 0.9586 | 1.0000 | 3 | 0 |
| rf | normal | normal2 | rf/normal2 | modern_metric | 42 | 0.0047 | 7 | 0 | 0.0000 | 0.1387 | 0.0025 | 0.0000 | 0.5425 | 0.4286 | 3 | 0 |
| join | normal | normal1 | join/normal1 | modern_metric | 42 | 0.0047 | 12 | 0 | 0.0000 | 0.1087 | 0.0000 | 0.0000 | 0.6071 | 0.2500 | 3 | 0 |
| join | normal | normal2 | join/normal2 | modern_metric | 42 | 0.0047 | 12 | 0 | 0.0000 | 0.1015 | 0.0000 | 0.0000 | 0.6203 | 0.2500 | 3 | 0 |
| aggregation | failure | vm_stress | aggregation/vm_stress | modern_metric | 42 | 0.0047 | 7 | 7 | 1.0000 | 0.9548 | 0.9999 | 0.7601 | 1.0000 | 1.0000 | 0 | 0 |
| join | failure | master_killed | join/master_killed | modern_metric | 42 | 0.0047 | 14 | 14 | 1.0000 | 0.9168 | 0.9997 | 0.4931 | 1.0000 | 1.0000 | 0 | 0 |
| join | failure | mq_stress | join/mq_stress | modern_metric | 42 | 0.0047 | 11 | 11 | 1.0000 | 0.9155 | 0.9860 | 0.6416 | 1.0000 | 1.0000 | 0 | 0 |

## Feature Ablation

| target | feature_set | split | features | average_precision | roc_auc | precision | recall | f1 | false_positives | missed_anomalies | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y_anomaly | contract_raw_selected | test | 80 | 0.2375 | 0.3127 | 0.3319 | 0.9912 | 0.4972 | 455 | 2 | 0.0494 |
| y_anomaly | rolling_z_delta | test | 160 | 0.2904 | 0.4454 | 0.3328 | 0.9912 | 0.4983 | 453 | 2 | 0.1456 |
| y_anomaly | raw_plus_rolling_z_delta | test | 240 | 0.2466 | 0.3362 | 0.3314 | 0.9825 | 0.4956 | 452 | 4 | 0.0124 |
| y_warning | contract_raw_selected | test | 80 | 0.2375 | 0.3127 | 0.3319 | 0.9912 | 0.4972 | 455 | 2 | 0.0494 |
| y_warning | rolling_z_delta | test | 160 | 0.2904 | 0.4454 | 0.3328 | 0.9912 | 0.4983 | 453 | 2 | 0.1456 |
| y_warning | raw_plus_rolling_z_delta | test | 240 | 0.2466 | 0.3362 | 0.3314 | 0.9825 | 0.4956 | 452 | 4 | 0.0124 |
| y_risk | contract_raw_selected | test | 80 | 0.1242 | 0.5793 | 0.0958 | 0.9846 | 0.1746 | 604 | 1 | 0.0702 |
| y_risk | rolling_z_delta | test | 160 | 0.1176 | 0.5742 | 0.0932 | 0.9692 | 0.1700 | 613 | 2 | 0.0141 |
| y_risk | raw_plus_rolling_z_delta | test | 240 | 0.1346 | 0.5949 | 0.0952 | 0.9385 | 0.1728 | 580 | 4 | 0.0232 |

## Horizon Diagnostics

| variant | seed | synthetic_target | horizon_sec | positives | average_precision | roc_auc | precision | recall | f1 | false_positives | missed_anomalies | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| modern_metric | 42 | failure_lead_30s | 30 | 65 | 0.1523 | 0.5704 | 0.1277 | 0.9231 | 0.2243 | 410 | 5 | 0.0822 |
| modern_metric | 42 | failure_lead_60s | 60 | 134 | 0.2811 | 0.5828 | 0.2617 | 0.9179 | 0.4073 | 347 | 11 | 0.0822 |
| modern_metric | 42 | failure_lead_120s | 120 | 201 | 0.3987 | 0.6381 | 0.4004 | 0.9502 | 0.5634 | 286 | 10 | 0.0714 |
| modern_metric | 42 | failure_lead_300s | 300 | 234 | 0.4438 | 0.6537 | 0.4702 | 0.9444 | 0.6278 | 249 | 13 | 0.0822 |

## Recommendation Logic

1. If `average_precision`/`roc_auc` are weak across all targets, first improve preprocessing/target alignment.
2. If `y_risk` is materially better than `y_warning`, split targets into multi-head warning/risk scoring.
3. If deviation features outperform raw contract proxies, add rolling z-score, delta, slope and residual channels to metric preprocessing.
4. If horizon targets differ strongly, rebuild chunk labels around the horizon that actually separates.