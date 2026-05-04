# Efficient HADES Hypothesis: `efficient_hades_cuda_full`

This experiment preserves the HADES-like architecture: log sequence modeling, metric segment modeling, bidirectional cross-modal attention, pseudo-labeling, and binary normal/abnormal detection.

It uses raw KPI files from `raw_data`, not only pre-aggregated KPI columns from the contract.

## Test Metrics

| variant | runtime | seed | threshold_policy | average_precision | roc_auc | precision | recall | f1 | false_positives | missed_anomalies | torch_latency_ms_per_sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | torch_fp32 | 42 | validation_best_f1 | 0.4388 | 0.6523 | 0.4140 | 0.9605 | 0.5786 | 310 | 9 | 0.0598 |
| original | torch_fp32 | 42 | validation_fp_budget_0.1 | 0.4388 | 0.6523 | 0.3034 | 0.2368 | 0.2660 | 124 | 174 | 0.0598 |
| original | torch_fp32 | 43 | validation_best_f1 | 0.3648 | 0.6142 | 0.4137 | 0.9781 | 0.5815 | 316 | 5 | 0.0827 |
| original | torch_fp32 | 43 | validation_fp_budget_0.1 | 0.3648 | 0.6142 | 0.1720 | 0.1184 | 0.1403 | 130 | 201 | 0.0827 |
| original | torch_fp32 | 44 | validation_best_f1 | 0.3684 | 0.5918 | 0.3508 | 1.0000 | 0.5194 | 422 | 0 | 0.0868 |
| original | torch_fp32 | 44 | validation_fp_budget_0.1 | 0.3684 | 0.5918 | 0.1646 | 0.1184 | 0.1378 | 137 | 201 | 0.0868 |
| modern_log | torch_fp32 | 42 | validation_best_f1 | 0.4208 | 0.6508 | 0.4202 | 0.9474 | 0.5822 | 298 | 12 | 0.0910 |
| modern_log | torch_fp32 | 42 | validation_fp_budget_0.1 | 0.4208 | 0.6508 | 0.2390 | 0.1667 | 0.1964 | 121 | 190 | 0.0910 |
| modern_log | torch_fp32 | 43 | validation_best_f1 | 0.3593 | 0.6188 | 0.4060 | 0.9474 | 0.5684 | 316 | 12 | 0.0815 |
| modern_log | torch_fp32 | 43 | validation_fp_budget_0.1 | 0.3593 | 0.6188 | 0.2898 | 0.2237 | 0.2525 | 125 | 177 | 0.0815 |
| modern_log | torch_fp32 | 44 | validation_best_f1 | 0.4296 | 0.6503 | 0.3467 | 0.9868 | 0.5131 | 424 | 3 | 0.0867 |
| modern_log | torch_fp32 | 44 | validation_fp_budget_0.1 | 0.4296 | 0.6503 | 0.1429 | 0.0877 | 0.1087 | 120 | 208 | 0.0867 |
| modern_metric | torch_fp32 | 42 | validation_best_f1 | 0.4376 | 0.6540 | 0.4337 | 0.9605 | 0.5975 | 286 | 9 | 0.0968 |
| modern_metric | torch_fp32 | 42 | validation_fp_budget_0.1 | 0.4376 | 0.6540 | 0.3231 | 0.2763 | 0.2979 | 132 | 165 | 0.0968 |
| modern_metric | torch_fp32 | 43 | validation_best_f1 | 0.3795 | 0.6230 | 0.3779 | 0.9912 | 0.5472 | 372 | 2 | 0.0706 |
| modern_metric | torch_fp32 | 43 | validation_fp_budget_0.1 | 0.3795 | 0.6230 | 0.4221 | 0.5351 | 0.4720 | 167 | 106 | 0.0706 |
| modern_metric | torch_fp32 | 44 | validation_best_f1 | 0.3877 | 0.6281 | 0.4258 | 0.9561 | 0.5892 | 294 | 10 | 0.0854 |
| modern_metric | torch_fp32 | 44 | validation_fp_budget_0.1 | 0.3877 | 0.6281 | 0.4221 | 0.5702 | 0.4851 | 178 | 98 | 0.0854 |
| modern_runtime | torch_fp32 | 42 | validation_best_f1 | 0.4388 | 0.6523 | 0.4140 | 0.9605 | 0.5786 | 310 | 9 | 0.0777 |
| modern_runtime | torch_fp32 | 42 | validation_fp_budget_0.1 | 0.4388 | 0.6523 | 0.3034 | 0.2368 | 0.2660 | 124 | 174 | 0.0777 |
| modern_runtime | torch_fp32 | 43 | validation_best_f1 | 0.3648 | 0.6142 | 0.4137 | 0.9781 | 0.5815 | 316 | 5 | 0.0973 |
| modern_runtime | torch_fp32 | 43 | validation_fp_budget_0.1 | 0.3648 | 0.6142 | 0.1720 | 0.1184 | 0.1403 | 130 | 201 | 0.0973 |
| modern_runtime | torch_fp32 | 44 | validation_best_f1 | 0.3684 | 0.5918 | 0.3508 | 1.0000 | 0.5194 | 422 | 0 | 0.0284 |
| modern_runtime | torch_fp32 | 44 | validation_fp_budget_0.1 | 0.3684 | 0.5918 | 0.1646 | 0.1184 | 0.1378 | 137 | 201 | 0.0284 |
| efficient_hades | torch_fp32 | 42 | validation_best_f1 | 0.4133 | 0.6491 | 0.4059 | 0.9737 | 0.5729 | 325 | 6 | 0.0740 |
| efficient_hades | torch_fp32 | 42 | validation_fp_budget_0.1 | 0.4133 | 0.6491 | 0.4100 | 0.4693 | 0.4376 | 154 | 121 | 0.0740 |
| efficient_hades | torch_fp32 | 43 | validation_best_f1 | 0.4100 | 0.6063 | 0.3559 | 0.9912 | 0.5238 | 409 | 2 | 0.0781 |
| efficient_hades | torch_fp32 | 43 | validation_fp_budget_0.1 | 0.4100 | 0.6063 | 0.3774 | 0.4254 | 0.4000 | 160 | 131 | 0.0781 |
| efficient_hades | torch_fp32 | 44 | validation_best_f1 | 0.3359 | 0.5679 | 0.3905 | 0.9781 | 0.5582 | 348 | 5 | 0.0762 |
| efficient_hades | torch_fp32 | 44 | validation_fp_budget_0.1 | 0.3359 | 0.5679 | 0.3321 | 0.3860 | 0.3570 | 177 | 140 | 0.0762 |
| modern_runtime | onnx_int8 | 42 | validation_best_f1 | 0.6208 | 0.7492 | 0.3731 | 0.9605 | 0.5374 | 368 | 9 |  |
| modern_runtime | onnx_int8 | 42 | validation_fp_budget_0.1 | 0.6208 | 0.7492 | 0.7273 | 0.2456 | 0.3672 | 21 | 172 |  |
| modern_runtime | onnx_int8 | 43 | validation_best_f1 | 0.5463 | 0.7122 | 0.3858 | 0.9781 | 0.5533 | 355 | 5 |  |
| modern_runtime | onnx_int8 | 43 | validation_fp_budget_0.1 | 0.5463 | 0.7122 | 0.6000 | 0.1711 | 0.2662 | 26 | 189 |  |
| modern_runtime | onnx_int8 | 44 | validation_best_f1 | 0.5599 | 0.7308 | 0.3408 | 1.0000 | 0.5084 | 441 | 0 |  |
| modern_runtime | onnx_int8 | 44 | validation_fp_budget_0.1 | 0.5599 | 0.7308 | 0.5918 | 0.1272 | 0.2094 | 20 | 199 |  |
| efficient_hades | onnx_int8 | 42 | validation_best_f1 | 0.5371 | 0.6996 | 0.3752 | 0.9693 | 0.5410 | 368 | 7 |  |
| efficient_hades | onnx_int8 | 42 | validation_fp_budget_0.1 | 0.5371 | 0.6996 | 0.5196 | 0.4079 | 0.4570 | 86 | 135 |  |
| efficient_hades | onnx_int8 | 43 | validation_best_f1 | 0.6684 | 0.7954 | 0.3663 | 0.9737 | 0.5324 | 384 | 6 |  |
| efficient_hades | onnx_int8 | 43 | validation_fp_budget_0.1 | 0.6684 | 0.7954 | 0.7714 | 0.3553 | 0.4865 | 24 | 147 |  |
| efficient_hades | onnx_int8 | 44 | validation_best_f1 | 0.3486 | 0.5617 | 0.3780 | 0.9781 | 0.5452 | 367 | 5 |  |
| efficient_hades | onnx_int8 | 44 | validation_fp_budget_0.1 | 0.3486 | 0.5617 | 0.3003 | 0.4386 | 0.3565 | 233 | 128 |  |

## Variants

- `original`: FastText-like cached subword embeddings + Transformer logs + causal conv metrics + cross-modal attention.
- `modern_log`: StaticEmbedding/EmbeddingGemma cached templates + original metric encoder + cross-modal attention.
- `modern_metric`: FastText-like logs + TinyTimeMixer-lite metric segment encoder + cross-modal attention.
- `modern_runtime`: Original-style model exported to ONNX INT8 for runtime comparison.
- `efficient_hades`: Cached modern embeddings + TinyTimeMixer-lite metrics + cross-modal attention + ONNX INT8.

## Files

- `comparison_metrics.csv`: FP32 metrics by variant, seed and threshold policy.
- `runtime_metrics.csv`: ONNX INT8 metrics when export and runtime are available.
- `latency_and_export_report.json`: ONNX/INT8 export status, size and latency details.
- `feature_manifest.json`: log embedding and raw metric segment manifest.
- `train_manifest.json`: pseudo-labeling and validation monitor details.