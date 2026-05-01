# Strict Forecast Experiment: `strict_forecast_01`

## Purpose

Проверяется качество до `event_anchor`: окна после anchor или ближе заданного lead gap исключаются, а targets пересчитываются заново.

## Variants

| variant | lead_gap_sec | allowed_anchor_policies | input_rows | output_rows | dropped_rows | dropped_too_close_or_after_anchor | dropped_disallowed_or_missing_failure_anchor | far_before_warning_horizon_kept_as_negative | runs | workloads | risk_positives | warning_positives | risk_prevalence | warning_prevalence | contract_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| explicit_only | 0 | ['explicit_raw_log_keyword_or_severity'] | 10237 | 3781 | 6456 | 3934 | 2522 | 993 | 274 | 16 | 573 | 1650 | 0.1515 | 0.4364 | outputs\strict_forecast\strict_forecast_01\contracts\strict_explicit_only_gap0.csv.gz |
| explicit_only | 10 | ['explicit_raw_log_keyword_or_severity'] | 10237 | 3595 | 6642 | 4120 | 2522 | 993 | 274 | 16 | 387 | 1464 | 0.1076 | 0.4072 | outputs\strict_forecast\strict_forecast_01\contracts\strict_explicit_only_gap10.csv.gz |
| explicit_only | 20 | ['explicit_raw_log_keyword_or_severity'] | 10237 | 3409 | 6828 | 4306 | 2522 | 993 | 274 | 16 | 201 | 1278 | 0.0590 | 0.3749 | outputs\strict_forecast\strict_forecast_01\contracts\strict_explicit_only_gap20.csv.gz |
| all_anchors | 0 | ['explicit_raw_log_keyword_or_severity', 'run_end_proxy', 'run_end_proxy_no_log_events'] | 10237 | 6155 | 4082 | 4082 | 0 | 1725 | 440 | 16 | 1089 | 3292 | 0.1769 | 0.5348 | outputs\strict_forecast\strict_forecast_01\contracts\strict_all_anchors_gap0.csv.gz |
| all_anchors | 10 | ['explicit_raw_log_keyword_or_severity', 'run_end_proxy', 'run_end_proxy_no_log_events'] | 10237 | 5803 | 4434 | 4434 | 0 | 1725 | 440 | 16 | 737 | 2940 | 0.1270 | 0.5066 | outputs\strict_forecast\strict_forecast_01\contracts\strict_all_anchors_gap10.csv.gz |
| all_anchors | 20 | ['explicit_raw_log_keyword_or_severity', 'run_end_proxy', 'run_end_proxy_no_log_events'] | 10237 | 5451 | 4786 | 4786 | 0 | 1725 | 440 | 16 | 385 | 2588 | 0.0706 | 0.4748 | outputs\strict_forecast\strict_forecast_01\contracts\strict_all_anchors_gap20.csv.gz |

## Dataset Summary

| split | variant | lead_gap_sec | section | rows | runs | workloads | warning_positives | risk_positives | warning_prevalence | risk_prevalence | log_available | both_modalities | anchor_policy | workload | failure_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | explicit_only | 0 | split | 841 | 71 | 4 | 367 | 140 | 0.4364 | 0.1665 | 0.5303 | 0.5303 | nan | nan | nan |
| train | explicit_only | 0 | split | 2940 | 203 | 12 | 1283 | 433 | 0.4364 | 0.1473 | 0.5878 | 0.5878 | nan | nan | nan |
| test | explicit_only | 10 | split | 795 | 71 | 4 | 321 | 94 | 0.4038 | 0.1182 | 0.5044 | 0.5044 | nan | nan | nan |
| train | explicit_only | 10 | split | 2800 | 203 | 12 | 1143 | 293 | 0.4082 | 0.1046 | 0.5771 | 0.5771 | nan | nan | nan |
| test | explicit_only | 20 | split | 749 | 71 | 4 | 275 | 48 | 0.3672 | 0.0641 | 0.4873 | 0.4873 | nan | nan | nan |
| train | explicit_only | 20 | split | 2660 | 203 | 12 | 1003 | 153 | 0.3771 | 0.0575 | 0.5684 | 0.5684 | nan | nan | nan |
| test | all_anchors | 0 | split | 1385 | 113 | 4 | 805 | 273 | 0.5812 | 0.1971 | 0.5646 | 0.5646 | nan | nan | nan |
| train | all_anchors | 0 | split | 4770 | 327 | 12 | 2487 | 816 | 0.5214 | 0.1711 | 0.5727 | 0.5725 | nan | nan | nan |
| test | all_anchors | 10 | split | 1297 | 113 | 4 | 717 | 185 | 0.5528 | 0.1426 | 0.5359 | 0.5359 | nan | nan | nan |
| train | all_anchors | 10 | split | 4506 | 327 | 12 | 2223 | 552 | 0.4933 | 0.1225 | 0.5577 | 0.5575 | nan | nan | nan |
| test | all_anchors | 20 | split | 1209 | 113 | 4 | 629 | 97 | 0.5203 | 0.0802 | 0.5103 | 0.5103 | nan | nan | nan |
| train | all_anchors | 20 | split | 4242 | 327 | 12 | 1959 | 288 | 0.4618 | 0.0679 | 0.5438 | 0.5436 | nan | nan | nan |

## Primary Metrics

| variant | lead_gap_sec | model | ap_mean | roc_mean | recall_mean | prevalence_mean |
| --- | --- | --- | --- | --- | --- | --- |
| all_anchors | 0 | log-only | 0.4805 | 0.7649 | 0.7070 | 0.1971 |
| all_anchors | 0 | log-count-only | 0.4684 | 0.7409 | 0.5641 | 0.1971 |
| all_anchors | 0 | log-only-redacted | 0.4523 | 0.7458 | 0.6850 | 0.1971 |
| all_anchors | 0 | log-count-no-severity | 0.4392 | 0.7397 | 0.6923 | 0.1971 |
| all_anchors | 0 | log-text-only | 0.4074 | 0.7307 | 0.7253 | 0.1971 |
| all_anchors | 0 | log-text-redacted | 0.4073 | 0.7308 | 0.7253 | 0.1971 |
| all_anchors | 0 | fused | 0.3182 | 0.6425 | 0.5897 | 0.1971 |
| all_anchors | 0 | availability-only | 0.2979 | 0.6925 | 0.8608 | 0.1971 |
| all_anchors | 0 | kpi-only | 0.2535 | 0.5873 | 0.6593 | 0.1971 |
| all_anchors | 10 | log-only | 0.2584 | 0.6708 | 0.6054 | 0.1426 |
| all_anchors | 10 | log-only-redacted | 0.2478 | 0.6532 | 0.5892 | 0.1426 |
| all_anchors | 10 | log-count-only | 0.2410 | 0.6431 | 0.6216 | 0.1426 |
| all_anchors | 10 | log-text-redacted | 0.2268 | 0.6352 | 0.5892 | 0.1426 |
| all_anchors | 10 | log-text-only | 0.2267 | 0.6348 | 0.5838 | 0.1426 |
| all_anchors | 10 | log-count-no-severity | 0.2114 | 0.6236 | 0.6378 | 0.1426 |
| all_anchors | 10 | availability-only | 0.2090 | 0.6655 | 0.8000 | 0.1426 |
| all_anchors | 10 | fused | 0.1643 | 0.5622 | 0.4541 | 0.1426 |
| all_anchors | 10 | kpi-only | 0.1572 | 0.5445 | 0.4973 | 0.1426 |
| all_anchors | 20 | log-count-no-severity | 0.1409 | 0.5717 | 0.5773 | 0.0802 |
| all_anchors | 20 | kpi-only | 0.1189 | 0.6130 | 0.3505 | 0.0802 |
| all_anchors | 20 | fused | 0.1148 | 0.5936 | 0.3299 | 0.0802 |
| all_anchors | 20 | log-only-redacted | 0.1130 | 0.5623 | 0.4227 | 0.0802 |
| all_anchors | 20 | log-only | 0.1091 | 0.5674 | 0.4330 | 0.0802 |
| all_anchors | 20 | log-text-redacted | 0.1087 | 0.5450 | 0.4227 | 0.0802 |
| all_anchors | 20 | log-text-only | 0.1085 | 0.5448 | 0.4227 | 0.0802 |
| all_anchors | 20 | availability-only | 0.1079 | 0.6259 | 0.7216 | 0.0802 |
| all_anchors | 20 | log-count-only | 0.1072 | 0.5803 | 0.5876 | 0.0802 |
| explicit_only | 0 | log-only | 0.2999 | 0.6964 | 0.6786 | 0.1665 |
| explicit_only | 0 | log-only-redacted | 0.2906 | 0.6766 | 0.6500 | 0.1665 |
| explicit_only | 0 | log-text-redacted | 0.2728 | 0.6596 | 0.6571 | 0.1665 |
| explicit_only | 0 | log-text-only | 0.2725 | 0.6598 | 0.6571 | 0.1665 |
| explicit_only | 0 | log-count-only | 0.2710 | 0.6665 | 0.5786 | 0.1665 |
| explicit_only | 0 | log-count-no-severity | 0.2536 | 0.6492 | 0.6214 | 0.1665 |
| explicit_only | 0 | availability-only | 0.2283 | 0.6469 | 0.7500 | 0.1665 |
| explicit_only | 0 | fused | 0.1731 | 0.5140 | 0.6714 | 0.1665 |
| explicit_only | 0 | kpi-only | 0.1615 | 0.4890 | 0.6929 | 0.1665 |
| explicit_only | 10 | log-only | 0.1790 | 0.6483 | 0.5213 | 0.1182 |
| explicit_only | 10 | log-only-redacted | 0.1760 | 0.6408 | 0.5213 | 0.1182 |
| explicit_only | 10 | log-count-only | 0.1562 | 0.6020 | 0.5532 | 0.1182 |
| explicit_only | 10 | log-text-only | 0.1551 | 0.5920 | 0.4787 | 0.1182 |
| explicit_only | 10 | log-text-redacted | 0.1551 | 0.5919 | 0.4787 | 0.1182 |
| explicit_only | 10 | log-count-no-severity | 0.1525 | 0.5795 | 0.5745 | 0.1182 |
| explicit_only | 10 | availability-only | 0.1473 | 0.5998 | 0.6489 | 0.1182 |
| explicit_only | 10 | kpi-only | 0.1209 | 0.5105 | 0.5851 | 0.1182 |
| explicit_only | 10 | fused | 0.1179 | 0.5120 | 0.6489 | 0.1182 |
| explicit_only | 20 | log-only | 0.0839 | 0.5723 | 0.2917 | 0.0641 |
| explicit_only | 20 | kpi-only | 0.0819 | 0.5680 | 0.1875 | 0.0641 |
| explicit_only | 20 | log-only-redacted | 0.0805 | 0.5659 | 0.2708 | 0.0641 |
| explicit_only | 20 | log-count-no-severity | 0.0796 | 0.5632 | 0.4792 | 0.0641 |
| explicit_only | 20 | fused | 0.0787 | 0.5527 | 0.3125 | 0.0641 |
| explicit_only | 20 | log-count-only | 0.0771 | 0.5668 | 0.4167 | 0.0641 |
| explicit_only | 20 | availability-only | 0.0708 | 0.5483 | 0.5417 | 0.0641 |
| explicit_only | 20 | log-text-only | 0.0676 | 0.5217 | 0.2708 | 0.0641 |
| explicit_only | 20 | log-text-redacted | 0.0676 | 0.5216 | 0.2708 | 0.0641 |

## Warning Metrics

| variant | lead_gap_sec | model | ap_mean | roc_mean | recall_mean | prevalence_mean |
| --- | --- | --- | --- | --- | --- | --- |
| all_anchors | 0 | fused | 0.6889 | 0.6407 | 0.4460 | 0.5812 |
| all_anchors | 0 | kpi-only | 0.6848 | 0.6334 | 0.4323 | 0.5812 |
| all_anchors | 0 | log-count-only | 0.6723 | 0.6587 | 0.5565 | 0.5812 |
| all_anchors | 0 | log-only | 0.6599 | 0.6491 | 0.5975 | 0.5812 |
| all_anchors | 0 | availability-only | 0.6562 | 0.6287 | 0.6646 | 0.5812 |
| all_anchors | 0 | log-count-no-severity | 0.6491 | 0.6242 | 0.5491 | 0.5812 |
| all_anchors | 0 | log-only-redacted | 0.6425 | 0.6200 | 0.6000 | 0.5812 |
| all_anchors | 0 | log-text-only | 0.6292 | 0.6100 | 0.5963 | 0.5812 |
| all_anchors | 0 | log-text-redacted | 0.6288 | 0.6097 | 0.5975 | 0.5812 |
| all_anchors | 10 | kpi-only | 0.6770 | 0.6521 | 0.4324 | 0.5528 |
| all_anchors | 10 | fused | 0.6692 | 0.6453 | 0.4240 | 0.5528 |
| all_anchors | 10 | log-count-only | 0.6528 | 0.6517 | 0.5160 | 0.5528 |
| all_anchors | 10 | log-only | 0.6356 | 0.6435 | 0.5132 | 0.5528 |
| all_anchors | 10 | log-count-no-severity | 0.6299 | 0.6195 | 0.5160 | 0.5528 |
| all_anchors | 10 | availability-only | 0.6159 | 0.6083 | 0.6248 | 0.5528 |
| all_anchors | 10 | log-only-redacted | 0.6151 | 0.6118 | 0.5146 | 0.5528 |
| all_anchors | 10 | log-text-redacted | 0.6133 | 0.6048 | 0.5160 | 0.5528 |
| all_anchors | 10 | log-text-only | 0.6133 | 0.6048 | 0.5160 | 0.5528 |
| all_anchors | 20 | kpi-only | 0.6632 | 0.6679 | 0.4404 | 0.5203 |
| all_anchors | 20 | fused | 0.6570 | 0.6622 | 0.4308 | 0.5203 |
| all_anchors | 20 | log-count-only | 0.6299 | 0.6558 | 0.5087 | 0.5203 |
| all_anchors | 20 | log-only | 0.6144 | 0.6467 | 0.4833 | 0.5203 |
| all_anchors | 20 | log-count-no-severity | 0.6070 | 0.6227 | 0.5087 | 0.5203 |
| all_anchors | 20 | log-only-redacted | 0.5984 | 0.6181 | 0.4833 | 0.5203 |
| all_anchors | 20 | log-text-only | 0.5929 | 0.6072 | 0.4881 | 0.5203 |
| all_anchors | 20 | log-text-redacted | 0.5928 | 0.6072 | 0.4881 | 0.5203 |
| all_anchors | 20 | availability-only | 0.5717 | 0.5897 | 0.5882 | 0.5203 |
| explicit_only | 0 | fused | 0.5326 | 0.6214 | 0.3161 | 0.4364 |
| explicit_only | 0 | log-only | 0.5307 | 0.6261 | 0.4714 | 0.4364 |
| explicit_only | 0 | kpi-only | 0.5244 | 0.6087 | 0.3297 | 0.4364 |
| explicit_only | 0 | log-only-redacted | 0.5166 | 0.5970 | 0.4768 | 0.4364 |
| explicit_only | 0 | log-text-redacted | 0.4970 | 0.5994 | 0.4768 | 0.4364 |
| explicit_only | 0 | log-text-only | 0.4969 | 0.6002 | 0.4768 | 0.4364 |
| explicit_only | 0 | log-count-only | 0.4868 | 0.5929 | 0.4905 | 0.4364 |
| explicit_only | 0 | availability-only | 0.4695 | 0.5582 | 0.5695 | 0.4364 |
| explicit_only | 0 | log-count-no-severity | 0.4634 | 0.5498 | 0.4877 | 0.4364 |
| explicit_only | 10 | kpi-only | 0.5125 | 0.6416 | 0.2773 | 0.4038 |
| explicit_only | 10 | log-only | 0.5007 | 0.6407 | 0.3427 | 0.4038 |
| explicit_only | 10 | log-only-redacted | 0.4855 | 0.6171 | 0.3458 | 0.4038 |
| explicit_only | 10 | fused | 0.4776 | 0.6174 | 0.2835 | 0.4038 |
| explicit_only | 10 | log-text-redacted | 0.4701 | 0.5862 | 0.3458 | 0.4038 |
| explicit_only | 10 | log-text-only | 0.4696 | 0.5851 | 0.3458 | 0.4038 |
| explicit_only | 10 | log-count-only | 0.4633 | 0.6023 | 0.9346 | 0.4038 |
| explicit_only | 10 | log-count-no-severity | 0.4402 | 0.5591 | 0.9346 | 0.4038 |
| explicit_only | 10 | availability-only | 0.4207 | 0.5316 | 0.5109 | 0.4038 |
| explicit_only | 20 | kpi-only | 0.4928 | 0.6588 | 0.2327 | 0.3672 |
| explicit_only | 20 | fused | 0.4826 | 0.6462 | 0.2364 | 0.3672 |
| explicit_only | 20 | log-only | 0.4584 | 0.6280 | 0.3018 | 0.3672 |
| explicit_only | 20 | log-only-redacted | 0.4503 | 0.6152 | 0.3018 | 0.3672 |
| explicit_only | 20 | log-text-only | 0.4458 | 0.5919 | 0.3055 | 0.3672 |
| explicit_only | 20 | log-text-redacted | 0.4438 | 0.5859 | 0.3055 | 0.3672 |
| explicit_only | 20 | log-count-only | 0.4194 | 0.5893 | 0.9382 | 0.3672 |
| explicit_only | 20 | log-count-no-severity | 0.4025 | 0.5570 | 0.9309 | 0.3672 |
| explicit_only | 20 | availability-only | 0.3725 | 0.5101 | 0.4691 | 0.3672 |

## Files

- `strict_dataset_summary.csv`
- `strict_model_metrics.csv`
- `strict_model_summary.csv`
- `strict_variant_summary.csv`
- `strict_config.json`
- `strict_contract_paths.json`
