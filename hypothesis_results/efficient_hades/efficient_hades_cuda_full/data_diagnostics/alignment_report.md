# Alignment Report

- metric_segment_sec: `120`
- log_context_sec: `120`
- rows: `2883`

## Split/Target Counts

| split | y_anomaly_sum | y_anomaly_mean | y_warning_sum | y_warning_mean | y_risk_sum | y_risk_mean |
| --- | --- | --- | --- | --- | --- | --- |
| test | 228 | 0.3300 | 228 | 0.3300 | 65 | 0.0941 |
| train | 775 | 0.5222 | 775 | 0.5222 | 146 | 0.0984 |
| validation | 411 | 0.5805 | 411 | 0.5805 | 80 | 0.1130 |

## Split/Workload Target Counts

| split | workload | y_anomaly_count | y_anomaly_sum | y_anomaly_mean | y_warning_count | y_warning_sum | y_warning_mean | y_risk_count | y_risk_sum | y_risk_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | aggregation | 115 | 22 | 0.1913 | 115 | 22 | 0.1913 | 115 | 6 | 0.0522 |
| test | join | 135 | 47 | 0.3481 | 135 | 47 | 0.3481 | 135 | 8 | 0.0593 |
| test | rf | 66 | 5 | 0.0758 | 66 | 5 | 0.0758 | 66 | 2 | 0.0303 |
| test | svd | 375 | 154 | 0.4107 | 375 | 154 | 0.4107 | 375 | 49 | 0.1307 |
| train | als | 184 | 147 | 0.7989 | 184 | 147 | 0.7989 | 184 | 44 | 0.2391 |
| train | graph_pagerank | 125 | 56 | 0.4480 | 125 | 56 | 0.4480 | 125 | 18 | 0.1440 |
| train | lda | 76 | 49 | 0.6447 | 76 | 49 | 0.6447 | 76 | 13 | 0.1711 |
| train | nweight | 477 | 357 | 0.7484 | 477 | 357 | 0.7484 | 477 | 42 | 0.0881 |
| train | scan | 110 | 24 | 0.2182 | 110 | 24 | 0.2182 | 110 | 4 | 0.0364 |
| train | sleep | 110 | 0 | 0.0000 | 110 | 0 | 0.0000 | 110 | 0 | 0.0000 |
| train | sort | 60 | 9 | 0.1500 | 60 | 9 | 0.1500 | 60 | 2 | 0.0333 |
| train | web_pagerank | 255 | 105 | 0.4118 | 255 | 105 | 0.4118 | 255 | 15 | 0.0588 |
| train | wordcount | 87 | 28 | 0.3218 | 87 | 28 | 0.3218 | 87 | 8 | 0.0920 |
| validation | bayes | 235 | 189 | 0.8043 | 235 | 189 | 0.8043 | 235 | 47 | 0.2000 |
| validation | repartition | 138 | 38 | 0.2754 | 138 | 38 | 0.2754 | 138 | 8 | 0.0580 |
| validation | terasort | 335 | 184 | 0.5493 | 335 | 184 | 0.5493 | 335 | 25 | 0.0746 |

## Lead-Time Summary

| target | positive_rows | lead_min | lead_median | lead_max |
| --- | --- | --- | --- | --- |
| y_anomaly | 1414 | 10.0000 | 64.0000 | 180.0000 |
| y_warning | 1414 | 10.0000 | 64.0000 | 180.0000 |
| y_risk | 291 | 10.0000 | 20.0000 | 30.0000 |

## Leakage Checks

- metric windows crossing `event_anchor`: `0`
- rows with `window_end > event_anchor`: `0`

## Horizon Target Diagnostics

| variant | seed | synthetic_target | horizon_sec | positives | average_precision | roc_auc | precision | recall | f1 | false_positives | missed_anomalies | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| modern_metric | 42 | failure_lead_30s | 30 | 65 | 0.1523 | 0.5704 | 0.1277 | 0.9231 | 0.2243 | 410 | 5 | 0.0822 |
| modern_metric | 42 | failure_lead_60s | 60 | 134 | 0.2811 | 0.5828 | 0.2617 | 0.9179 | 0.4073 | 347 | 11 | 0.0822 |
| modern_metric | 42 | failure_lead_120s | 120 | 201 | 0.3987 | 0.6381 | 0.4004 | 0.9502 | 0.5634 | 286 | 10 | 0.0714 |
| modern_metric | 42 | failure_lead_300s | 300 | 234 | 0.4438 | 0.6537 | 0.4702 | 0.9444 | 0.6278 | 249 | 13 | 0.0822 |
