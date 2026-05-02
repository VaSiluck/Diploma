from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_ROOT = APP_ROOT / "raw_data"
RAW_MULTIRATE_DIR = APP_ROOT / "outputs" / "raw_multirate"
RAW_CENTRAL_MODEL_DIR = APP_ROOT / "outputs" / "raw_central_model"
FINAL_ALERT_MODEL_DIR = APP_ROOT / "outputs" / "final_alert_system"
DASHBOARD_MODEL_DIR_ENV = "DASHBOARD_MODEL_DIR"


def resolve_dashboard_model_dir(value: str | Path | None = None) -> Path:
    raw_value = value if value is not None else os.environ.get(DASHBOARD_MODEL_DIR_ENV)
    if raw_value is None or str(raw_value).strip() == "":
        if (FINAL_ALERT_MODEL_DIR / "replay_bundle.csv.gz").exists() or (FINAL_ALERT_MODEL_DIR / "dashboard_replay_store.csv.gz").exists():
            return FINAL_ALERT_MODEL_DIR.resolve()
        return RAW_CENTRAL_MODEL_DIR.resolve()
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = APP_ROOT / path
    return path.resolve()

def _first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


ALIGNED_CONTRACT_PATH = _first_existing_path(
    RAW_MULTIRATE_DIR / "aligned_multirate_windows.csv.gz",
    RAW_MULTIRATE_DIR / "aligned_multirate_windows.csv",
)
BASELINE_SUMMARY_PATH = RAW_MULTIRATE_DIR / "baseline_summary_metrics.json"

CONTRACT_ID_COLUMNS = [
    "workload",
    "scenario",
    "run_id",
    "group_id",
    "failure_family",
    "scenario_type",
    "window_ts",
    "window_end",
    "run_start",
    "run_end",
    "fail_start",
    "event_anchor",
    "anchor_policy",
    "minutes_before_fail",
    "macro_window_sec",
    "micro_window_sec",
    "window_index",
    "split",
]

CONTRACT_SIGNAL_COLUMNS = [
    "state",
    "y_warning",
    "y_risk",
    "log_text",
    "log_available",
    "macro_log_count",
    "macro_unique_components",
    "macro_warn_count",
    "macro_error_count",
    "macro_keyword_count",
    "macro_log_rate",
    "micro_log_count",
    "micro_unique_components",
    "micro_warn_count",
    "micro_error_count",
    "micro_keyword_count",
    "micro_log_rate",
    "micro_error_rate",
    "kpi_available",
    "kpi_series_available",
    "kpi_feature_count",
    "kpi_avail_ratio",
    "kpi_recency_sec",
    "has_both_modalities",
]

REQUIRED_CONTRACT_COLUMNS = [
    "run_id",
    "window_ts",
    "window_end",
    "state",
    "y_warning",
    "y_risk",
    "log_available",
    "kpi_available",
    "has_both_modalities",
]

KPI_FAMILY_ORDER = {
    "cpustat": 0,
    "vmstat": 1,
    "iostat": 2,
    "memstat": 3,
    "netstat": 4,
}

KPI_METRIC_ORDER = {
    "idle": 0,
    "user": 1,
    "system": 2,
    "iowait": 3,
    "wa": 4,
    "us": 5,
    "sy": 6,
    "id": 7,
    "memused": 0,
    "kbmemused": 1,
    "kbavail": 2,
    "commit": 3,
    "swpused": 4,
    "free": 5,
    "util": 0,
    "rkb_s": 1,
    "wkb_s": 2,
    "r_s": 3,
    "w_s": 4,
    "r_await": 5,
    "w_await": 6,
    "aqu_sz": 7,
    "rxkb_s": 0,
    "txkb_s": 1,
    "ifutil": 2,
    "rxpck_s": 3,
    "txpck_s": 4,
    "r": 8,
    "b": 9,
    "swpd": 10,
    "si": 11,
    "so": 12,
    "bi": 13,
    "bo": 14,
}

KPI_STAT_ORDER = {
    "last": 0,
    "mean": 1,
    "delta": 2,
    "slope": 3,
    "std": 4,
}


class DataContractError(RuntimeError):
    """Raised when the dashboard input contract is missing or malformed."""


@dataclass(frozen=True)
class ArtifactCheck:
    name: str
    path: Path
    required: bool
    found: bool
    note: str = ""


@dataclass
class ContractBundle:
    timeline: pd.DataFrame
    baseline_summary: dict[str, Any]
    artifact_checks: list[ArtifactCheck]
    kpi_columns: list[str]
    contract_path: Path
    summary_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise DataContractError(f"JSON artifact has unexpected shape: {path}")
    return payload


def _metric_token_score(column: str) -> int:
    lower = column.lower()
    for token, rank in KPI_METRIC_ORDER.items():
        if f"__{token}__" in lower or lower.endswith(f"__{token}"):
            return rank
    return 99


def _family_score(column: str) -> int:
    lower = column.lower()
    for family, rank in KPI_FAMILY_ORDER.items():
        if f"__{family}__" in lower:
            return rank
    return 99


def _stat_score(column: str) -> int:
    lower = column.lower()
    for stat, rank in KPI_STAT_ORDER.items():
        if lower.endswith(f"__{stat}"):
            return rank
    return 99


def _is_kpi_evidence_column(column: str) -> bool:
    lower = column.lower()
    if not lower.startswith("kpi_"):
        return False
    if not any(f"__{family}__" in lower for family in KPI_FAMILY_ORDER):
        return False
    if not any(lower.endswith(f"__{stat}") for stat in KPI_STAT_ORDER):
        return False
    return _metric_token_score(lower) < 99


def select_dashboard_columns(contract_path: Path, max_kpi_columns: int = 720) -> tuple[list[str], list[str]]:
    """Read only the columns needed for the dashboard preview and evidence panels."""
    header = pd.read_csv(contract_path, nrows=0).columns.tolist()
    header_set = set(header)

    keep = [col for col in CONTRACT_ID_COLUMNS + CONTRACT_SIGNAL_COLUMNS if col in header_set]
    kpi_candidates = [
        col
        for col in header
        if col not in keep and _is_kpi_evidence_column(col)
    ]
    kpi_candidates = sorted(
        kpi_candidates,
        key=lambda col: (
            _family_score(col),
            _metric_token_score(col),
            _stat_score(col),
            0 if "__max_dim__" in col else 1,
            col,
        ),
    )

    # Keep a balanced dashboard slice. Taking the first N sorted KPI columns
    # over-represents CPU because cpustat sorts before the other families.
    by_family: dict[int, list[str]] = {}
    for col in kpi_candidates:
        by_family.setdefault(_family_score(col), []).append(col)

    selected_kpi: list[str] = []
    family_keys = sorted(by_family)
    per_family = max(1, max_kpi_columns // max(len(family_keys), 1))
    for family_key in family_keys:
        selected_kpi.extend(by_family[family_key][:per_family])

    cursor = {family_key: min(per_family, len(cols)) for family_key, cols in by_family.items()}
    while len(selected_kpi) < max_kpi_columns:
        added = False
        for family_key in family_keys:
            pos = cursor[family_key]
            cols = by_family[family_key]
            if pos < len(cols):
                selected_kpi.append(cols[pos])
                cursor[family_key] = pos + 1
                added = True
                if len(selected_kpi) >= max_kpi_columns:
                    break
        if not added:
            break

    return keep + selected_kpi, selected_kpi


def build_artifact_checks(
    contract_path: Path = ALIGNED_CONTRACT_PATH,
    summary_path: Path = BASELINE_SUMMARY_PATH,
) -> list[ArtifactCheck]:
    return [
        ArtifactCheck(
            name="raw_data",
            path=RAW_DATA_ROOT,
            required=True,
            found=RAW_DATA_ROOT.exists(),
            note="Источник исходных логов и KPI.",
        ),
        ArtifactCheck(
            name="aligned_multirate_windows",
            path=contract_path,
            required=True,
            found=contract_path.exists(),
            note="Window-level contract из notebook 02.",
        ),
        ArtifactCheck(
            name="baseline_summary_metrics",
            path=summary_path,
            required=True,
            found=summary_path.exists(),
            note="Сводка split, coverage и smoke-baselines.",
        ),
    ]


def _normalize_state(value: Any) -> str:
    state = str(value or "").strip().lower()
    if state in {"risk", "anomaly", "critical"}:
        return "risk"
    if state == "warning":
        return "warning"
    return "normal"


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _prepare_contract_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_CONTRACT_COLUMNS if col not in df.columns]
    if missing:
        raise DataContractError(f"Contract misses required columns: {missing}")

    out = df.copy()
    out["run_id"] = out["run_id"].astype(str)
    for col in ["workload", "scenario", "failure_family", "scenario_type", "anchor_policy", "split"]:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)

    for col in ["window_ts", "window_end", "run_start", "run_end", "fail_start", "event_anchor"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")

    out = out.dropna(subset=["run_id", "window_ts"]).copy()

    numeric_cols = [
        "y_warning",
        "y_risk",
        "log_available",
        "kpi_available",
        "has_both_modalities",
        "macro_log_count",
        "macro_unique_components",
        "macro_warn_count",
        "macro_error_count",
        "macro_keyword_count",
        "macro_log_rate",
        "micro_log_count",
        "micro_unique_components",
        "micro_warn_count",
        "micro_error_count",
        "micro_keyword_count",
        "micro_log_rate",
        "micro_error_rate",
        "kpi_series_available",
        "kpi_feature_count",
        "kpi_avail_ratio",
        "kpi_recency_sec",
        "minutes_before_fail",
        "window_index",
    ]
    numeric_cols.extend([col for col in out.columns if col.startswith("kpi_")])
    _coerce_numeric(out, sorted(set(numeric_cols)))

    for col in ["y_warning", "y_risk", "log_available", "kpi_available", "has_both_modalities"]:
        out[col] = out[col].fillna(0).astype(int)

    # Defragment after selecting many KPI evidence columns; Streamlit reruns this
    # often during development, so avoiding pandas fragmentation warnings keeps
    # the console readable.
    out = out.copy()

    if "state" in out.columns:
        state_from_label = out["state"].map(_normalize_state)
    else:
        state_from_label = pd.Series("normal", index=out.index)
    state_from_targets = np.select(
        [out["y_risk"].eq(1), out["y_warning"].eq(1)],
        ["risk", "warning"],
        default="normal",
    )
    out["target_state"] = state_from_targets
    out["state_display"] = np.where(
        out["target_state"].eq("normal"),
        state_from_label,
        out["target_state"],
    )
    out["state_display"] = pd.Series(out["state_display"], index=out.index).map(_normalize_state)

    if "log_text" not in out.columns:
        out["log_text"] = ""
    out["log_text"] = out["log_text"].fillna("").astype(str)

    out = (
        out.sort_values(["run_id", "window_ts"])
        .drop_duplicates(["run_id", "window_ts"], keep="last")
        .reset_index(drop=True)
    )
    return out


def load_contract_bundle(
    contract_path: Path = ALIGNED_CONTRACT_PATH,
    summary_path: Path = BASELINE_SUMMARY_PATH,
) -> ContractBundle:
    checks = build_artifact_checks(contract_path=contract_path, summary_path=summary_path)
    missing = [check.name for check in checks if check.required and not check.found]
    if missing:
        raise DataContractError(f"Missing required artifacts: {', '.join(missing)}")

    selected_columns, kpi_columns = select_dashboard_columns(contract_path)
    timeline = pd.read_csv(
        contract_path,
        usecols=lambda col: col in set(selected_columns),
        low_memory=False,
    )
    if timeline.empty:
        raise DataContractError("Aligned multi-rate contract is empty.")

    timeline = _prepare_contract_frame(timeline)
    baseline_summary = _read_json(summary_path)

    return ContractBundle(
        timeline=timeline,
        baseline_summary=baseline_summary,
        artifact_checks=checks,
        kpi_columns=[col for col in kpi_columns if col in timeline.columns],
        contract_path=contract_path,
        summary_path=summary_path,
    )


def load_compact_replay_bundle(
    replay_store: pd.DataFrame,
    model_dir: Path | None = None,
) -> ContractBundle:
    model_dir = resolve_dashboard_model_dir(model_dir)
    if replay_store.empty:
        raise DataContractError("compact replay bundle is empty.")

    timeline = replay_store.copy()
    if "state" not in timeline.columns:
        risk = pd.to_numeric(timeline.get("y_risk", 0), errors="coerce").fillna(0).astype(int)
        warning = pd.to_numeric(timeline.get("y_warning", 0), errors="coerce").fillna(0).astype(int)
        timeline["state"] = np.select([risk.eq(1), warning.eq(1)], ["risk", "warning"], default="normal")
    if "micro_window_sec" not in timeline.columns:
        timeline["micro_window_sec"] = 10
    if "macro_window_sec" not in timeline.columns:
        timeline["macro_window_sec"] = 60

    timeline = _prepare_contract_frame(timeline)
    kpi_columns = [col for col in timeline.columns if col.startswith("kpi_dashboard_")]
    checks = [
        ArtifactCheck(
            name="replay_bundle",
            path=_first_existing_path(model_dir / "replay_bundle.csv.gz", model_dir / "replay_bundle.csv", model_dir / "dashboard_replay_store.csv.gz", model_dir / "dashboard_replay_store.csv"),
            required=True,
            found=True,
            note="Compact test/model-store окна из notebook 03.",
        ),
        ArtifactCheck(
            name="aligned_multirate_windows",
            path=ALIGNED_CONTRACT_PATH,
            required=False,
            found=ALIGNED_CONTRACT_PATH.exists(),
            note="Fallback/debug contract; в model_scoring не читается первым.",
        ),
        ArtifactCheck(
            name="baseline_summary_metrics",
            path=BASELINE_SUMMARY_PATH,
            required=False,
            found=BASELINE_SUMMARY_PATH.exists(),
            note="Fallback baseline summary из notebook 02.",
        ),
    ]
    baseline_summary = {
        "source": "replay_bundle",
        "windows": {"micro_window_sec": 10, "macro_window_sec": 60},
    }
    return ContractBundle(
        timeline=timeline,
        baseline_summary=baseline_summary,
        artifact_checks=checks,
        kpi_columns=kpi_columns,
        contract_path=_first_existing_path(model_dir / "replay_bundle.csv.gz", model_dir / "replay_bundle.csv", model_dir / "dashboard_replay_store.csv.gz", model_dir / "dashboard_replay_store.csv"),
        summary_path=_first_existing_path(model_dir / "model_report.json", model_dir / "dashboard_model_report.json"),
    )


def contract_stats(bundle: ContractBundle) -> dict[str, Any]:
    df = bundle.timeline
    summary = bundle.baseline_summary
    return {
        "rows": int(len(df)),
        "runs": int(df["run_id"].nunique()),
        "workloads": int(df["workload"].nunique()) if "workload" in df.columns else int(summary.get("workloads", 0)),
        "micro_window_sec": int(summary.get("windows", {}).get("micro_window_sec", 10)),
        "macro_window_sec": int(summary.get("windows", {}).get("macro_window_sec", 60)),
        "warning_share": float(pd.to_numeric(df["y_warning"], errors="coerce").mean()),
        "risk_share": float(pd.to_numeric(df["y_risk"], errors="coerce").mean()),
        "log_available_share": float(pd.to_numeric(df["log_available"], errors="coerce").mean()),
        "kpi_available_share": float(pd.to_numeric(df["kpi_available"], errors="coerce").mean()),
        "both_modalities_share": float(pd.to_numeric(df["has_both_modalities"], errors="coerce").mean()),
    }


def checks_to_frame(checks: list[ArtifactCheck]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Артефакт": check.name,
                "Путь": str(check.path.relative_to(APP_ROOT) if check.path.is_relative_to(APP_ROOT) else check.path),
                "Обязателен": "да" if check.required else "нет",
                "Найден": "да" if check.found else "нет",
                "Комментарий": check.note,
            }
            for check in checks
        ]
    )
