# AGENTS.md

## Research Overview

This repository contains a **ВКР (thesis) research pipeline for AIOps risk assessment** on multimodal telemetry.

The core experimental chain is:

1. `01_eda_raw_data.ipynb` — reproducible EDA of `raw_data/` (dataset anatomy, modality coverage, temporal resolution, provenance rules).
2. `02_multirate_baseline_raw_data.ipynb` — construction of a strict multi-rate data contract (`run_id = workload/scenario`), anti-leakage split, and smoke baseline models.
3. `03_central_fusion_model_raw_data.ipynb` — central fusion multimodal model (logs + KPI + modality masks) using the contract from notebook 02.

Scientific emphasis is methodological: **transparent assumptions, local conclusions, and reproducibility**, not engineering scale.

---

## Data Pipeline

### 1) Raw data intake
- Source: `raw_data/<workload>/...`
- Modalities:
  - logs (`mapred_logs/<scenario>.log`, `parsed_log/*`)
  - KPI (`kpis/<scenario>/instruments/<host>/<metric>`)

### 2) EDA and policy definition
- Notebook: `01_eda_raw_data.ipynb`
- Role: defines inclusion/provenance policy and windowing rationale for downstream steps.
- Path configured for artifacts: `outputs/raw_eda/` (currently not present in repository snapshot).

### 3) Baseline contract assembly (main handoff)
- Notebook: `02_multirate_baseline_raw_data.ipynb`
- Inputs: `raw_data/`
- Outputs (actual, present):
  - `outputs/raw_multirate/aligned_multirate_windows.csv`
  - `outputs/raw_multirate/baseline_summary_metrics.json`
- Contract summary (from JSON): `rows=10237`, `runs=440`, `workloads=16`, micro-window `10s`, macro-window `60s`.

### 4) Central fusion modeling
- Notebook: `03_central_fusion_model_raw_data.ipynb`
- Inputs:
  - `outputs/raw_multirate/aligned_multirate_windows.csv`
  - `outputs/raw_multirate/baseline_summary_metrics.json`
- Current outputs (debug/dry-run, present):
  - `outputs/raw_central_model/feature_manifest.json`
  - `outputs/raw_central_model/preprocessing_manifest.json`
  - `outputs/raw_central_model/model_config.json`
  - `outputs/raw_central_model/dry_run_report.json`
  - `outputs/raw_central_model/cache/log_embeddings_google__embeddinggemma-300m.npz`
- Full training artifacts are generated only in `RUN_MODE="train"` and are currently absent.

### 5) Dashboard layer
- App: `dashboard/mm_dashboard.py`
- Current expected inputs:
  - primarily `outputs/lcman/*` and `outputs/final_demo/*` (legacy naming)
  - optional raw KPI overlays from `kpi_csv/<app>/<scenario>/*-intersect.csv`
- Important: this dashboard contract is **not yet aligned** with current `raw_multirate/raw_central_model` outputs.

---

## Tech Stack

### Data science / modeling
- `numpy`, `pandas`
- `matplotlib`, `seaborn`
- `scikit-learn` (GroupShuffleSplit, LogisticRegression, preprocessing/pipelines, metrics)
- `torch` (central fusion model)
- optional text embedding stack: `huggingface_hub`, `sentence_transformers`
- utilities: `joblib`, `tqdm`

### Dashboard
- `streamlit`
- `plotly` (`plotly.graph_objects`)

---

## Academic Coding Standards

### 1) Notebook cell organization (mandatory)
Each major section must follow the pattern:
1. **Markdown goal cell**: research question and assumptions.
2. **Code cell(s)**: deterministic transformation/experiment.
3. **Markdown interpretation cell**: local conclusion (`Локальный вывод`) and implication for next step.

Required conventions:
- No hidden state between sections.
- One initialization cell with constants/paths/seeds.
- Artifact-writing cells must explicitly list written files.
- Any fallback logic must be documented in markdown and manifest JSON.

### 2) Russian labels for figures/tables (mandatory)
- Titles, axis labels, legends, captions: in Russian.
- Use consistent terminology:
  - `normal` → `Норма`
  - `warning` → `Предупреждение`
  - `risk` → `Риск`
- Every plot must include units where applicable (`сек`, `%`, `кол-во`, etc.).
- Every plot must be followed by short interpretation text:
  - `Что показано`
  - `Что наблюдается`
  - `Локальный вывод`

### 3) Random state and reproducibility policy (mandatory)
Use one global seed and propagate it everywhere:
- `SEED = 42`
- `random.seed(SEED)`
- `np.random.seed(SEED)`
- `torch.manual_seed(SEED)` (+ CUDA seeds if used)
- all sklearn components with stochasticity must set `random_state=SEED` (or deterministic derivative `SEED + k` with documented rule)

Additionally:
- Save `run_mode`, seed, and split strategy in manifest/JSON.
- Prefer group-aware split for leakage control.
- Execute notebooks top-to-bottom in clean kernel before reporting metrics.

---

## Dashboard Specs

### Purpose
The dashboard is a **research demonstration interface** for replaying risk dynamics and explaining multimodal fusion decisions (logs + KPI), not an online production service.

### Main components
1. **Scenario replay controls** (case selector, play/pause, step slider).
2. **Risk-over-time panel** with state/risk trajectory.
3. **Log evidence panel** (patterns, excerpts, heuristic drivers).
4. **KPI evidence panel** (CPU/Memory/Disk/Network compact cards + detailed telemetry views).
5. **Fusion summary panel** (log-vs-KPI contribution, fallback visibility).
6. **Baseline comparison panel** (LCMAN-lite vs baseline score behavior).
7. **About/model section** (data contract table, runtime metadata, metric summaries, model card).

### Contract behavior
- Enforces artifact checks (`DataContractError`) for required inputs.
- Builds/uses timeline contract with `run_id` + `window_ts`.
- Supports degraded mode when KPI is missing (log-driven fallback).

---

## Current Progress

### Finished
- `01_eda_raw_data.ipynb`: completed EDA structure and transfer policy to notebook 02.
- `02_multirate_baseline_raw_data.ipynb`: strict baseline contract and summary artifacts generated.
- `03_central_fusion_model_raw_data.ipynb`: preprocessing + one-batch dry-run validation completed; manifests saved.
- `dashboard/mm_dashboard.py`: implemented replay UI and evidence/fusion panels with contract validation.

### Missing / pending
1. **Full training run for notebook 03** (`RUN_MODE="train"`) and associated final artifacts (`central_model.pt`, training history, metrics tables, inference store, etc.).
2. **Dashboard data alignment**: current dashboard expects `outputs/lcman`/`outputs/final_demo`, while active pipeline writes to `outputs/raw_multirate` and `outputs/raw_central_model`.
3. **KPI dashboard source alignment**: `kpi_csv/` paths expected by dashboard are not present in current repository snapshot.
4. **Environment lock/repro bundle**: no `requirements.txt`/`pyproject.toml`/env lock file currently present.
5. **Clean-run publication protocol**: final thesis-ready run should include deterministic rerun evidence and synchronized notebook outputs.

---

This project should remain notebook-first: concise code, explicit reasoning, reproducible artifacts, and interpretable local conclusions suitable for academic review without reading implementation details line-by-line.
