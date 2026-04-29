# Model card: central fusion model

Intended use: research prototype for AIOps warning/risk scoring on the raw multi-rate contract.

Data: 10237 windows, 440 runs, 16 workloads. Targets are `y_warning` and `y_risk`; proxy anchors remain a limitation and must not be read as exact fault onset.

Architecture: log macro branch, log micro branch, KPI 1D-CNN branch, missing-aware gated fusion, dual heads.

Training protocol: group-aware inner validation, early stopping by validation PR-AUC for `y_warning`, modality dropout=0.2, loss weights risk=0.35, aux=0.12.

Primary metric: PR-AUC for `y_warning`; secondary metric: PR-AUC for `y_risk`. See `central_model_metrics.json`, `coverage_metrics.csv` and `baseline_comparison.csv`.

Limitations: proxy anchors, coverage shortcut, high-dimensional KPI tensor, small workload-holdout test, no heavy attribution beyond descriptive KPI drivers.
