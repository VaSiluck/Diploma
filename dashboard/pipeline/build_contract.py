from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "dashboard" / "artifacts"
SEED = 42
MICRO_WINDOW_SEC = 10
MACRO_WINDOW_SEC = 60
STRICT_GAP_SEC = 10
WARNING_HORIZON_SEC = 180
RISK_HORIZON_SEC = 30
PRE_ANCHOR_CONTEXT_SEC = 240

RAW_LOG_RE = re.compile(
    r"^(?P<date>\d{2}/\d{2}/\d{2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<level>[A-Z]+)\s+"
    r"(?P<component>[^:]+):\s*"
    r"(?P<message>.*)$"
)

KEYWORDS = [
    "error",
    "exception",
    "failed",
    "failure",
    "killed",
    "lost",
    "timeout",
    "suspended",
    "blacklist",
    "retry",
    "critical",
    "fatal",
    "unhealthy",
    "refused",
    "denied",
    "warn",
]

STRONG_ANCHOR_WORDS = [
    "error",
    "exception",
    "failed",
    "failure",
    "killed",
    "lost",
    "timeout",
    "fatal",
    "critical",
    "unhealthy",
    "refused",
]

SEVERITY_WEIGHT = {
    "FATAL": 5.0,
    "ERROR": 4.0,
    "WARN": 2.0,
    "WARNING": 2.0,
    "INFO": 0.2,
    "DEBUG": 0.0,
    "TRACE": 0.0,
}

KPI_SELECTED_COLUMNS = {
    "cpustat": ["%user", "%system", "%iowait", "%idle"],
    "iostat": ["r/s", "w/s", "rkB/s", "wkB/s", "r_await", "w_await", "aqu-sz", "%util"],
    "memstat": ["kbavail", "kbmemused", "%memused", "kbcommit", "%commit", "kbswpused", "%swpused"],
    "netstat": ["rxpck/s", "txpck/s", "rxkB/s", "txkB/s", "%ifutil"],
    "vmstat": ["r", "b", "swpd", "free", "si", "so", "bi", "bo", "us", "sy", "id", "wa"],
}

LOG_NUMERIC_COLUMNS = [
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
]

MASK_COLUMNS = ["log_available", "kpi_available", "has_both_modalities", "kpi_avail_ratio"]


@dataclass(frozen=True)
class ArtifactPaths:
    artifact_dir: Path
    contract_dir: Path
    model_dir: Path
    bundle_dir: Path
    reports_dir: Path
    contract_path: Path
    contract_summary_path: Path
    baseline_summary_path: Path
    baseline_comparison_path: Path
    baseline_predictions_path: Path
    pipeline_report_path: Path


def artifact_paths(artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> ArtifactPaths:
    root = Path(artifact_dir)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    contract_dir = root / "contract"
    model_dir = root / "model"
    bundle_dir = root / "bundle"
    reports_dir = root / "reports"
    return ArtifactPaths(
        artifact_dir=root,
        contract_dir=contract_dir,
        model_dir=model_dir,
        bundle_dir=bundle_dir,
        reports_dir=reports_dir,
        contract_path=contract_dir / "strict_intensity_contract.csv.gz",
        contract_summary_path=contract_dir / "strict_contract_summary.csv",
        baseline_summary_path=reports_dir / "baseline_summary_metrics.json",
        baseline_comparison_path=reports_dir / "baseline_comparison_compact.csv",
        baseline_predictions_path=reports_dir / "baseline_predictions.csv.gz",
        pipeline_report_path=reports_dir / "final_data_pipeline_report.json",
    )


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat() if pd.notna(obj) else None
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    try:
        if pd.isna(obj) and not isinstance(obj, str):
            return None
    except Exception:
        pass
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_log_message(message: str) -> str:
    text = str(message).lower()
    text = re.sub(r"application_\d+_\d+", "<application_id>", text)
    text = re.sub(r"container_\w+", "<container_id>", text)
    text = re.sub(r"[0-9a-f]{8}-[0-9a-f-]{27,}", "<uuid>", text)
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<ip>", text)
    text = re.sub(r"hdfs://\S+|file:/\S+|/\S+", "<path>", text)
    text = re.sub(r"\b\d+\b", "<num>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_raw_log_file(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "level", "component", "message_norm", "event_score", "line_id"])
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line_id, line in enumerate(file, start=1):
            match = RAW_LOG_RE.match(line.strip())
            if not match:
                continue
            ts = pd.to_datetime(
                f"{match.group('date')} {match.group('time')}",
                format="%y/%m/%d %H:%M:%S",
                utc=True,
                errors="coerce",
            )
            if pd.isna(ts):
                continue
            level = match.group("level")
            message_norm = normalize_log_message(match.group("message"))
            lower = message_norm.lower()
            keyword_score = sum(
                (2.0 if kw in {"error", "exception", "failed", "failure", "fatal", "critical"} else 1.0)
                for kw in KEYWORDS
                if kw in lower
            )
            rows.append(
                {
                    "timestamp": ts,
                    "level": level,
                    "component": match.group("component"),
                    "message_norm": message_norm,
                    "event_score": float(SEVERITY_WEIGHT.get(level, 0.0) + keyword_score),
                    "line_id": int(line_id),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["timestamp", "level", "component", "message_norm", "event_score", "line_id"])
    return pd.DataFrame(rows).sort_values(["timestamp", "line_id"]).reset_index(drop=True)


def _scenario_type(scenario: str) -> str:
    return "normal" if str(scenario).lower().startswith("normal") else "failure"


def _failure_family(scenario: str) -> str:
    value = str(scenario).lower()
    return "normal" if value.startswith("normal") else re.sub(r"\d+$", "", value)


def _safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def scan_raw_data(raw_data: str | Path = PROJECT_ROOT / "raw_data") -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_root = Path(raw_data)
    if not raw_root.is_absolute():
        raw_root = PROJECT_ROOT / raw_root
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for workload_dir in sorted([p for p in raw_root.iterdir() if p.is_dir() and not p.name.startswith(".")]):
        workload = workload_dir.name
        log_dir = workload_dir / "mapred_logs"
        kpi_dir = workload_dir / "kpis"
        log_scenarios = {p.stem for p in log_dir.glob("*.log")} if log_dir.exists() else set()
        kpi_scenarios = {p.name for p in kpi_dir.iterdir() if p.is_dir()} if kpi_dir.exists() else set()
        for scenario in sorted(log_scenarios | kpi_scenarios):
            log_path = log_dir / f"{scenario}.log"
            scenario_kpi_dir = kpi_dir / scenario
            has_log = log_path.exists()
            has_kpi = (scenario_kpi_dir / "instruments").exists()
            rows.append(
                {
                    "workload": workload,
                    "scenario": scenario,
                    "run_id": f"{workload}/{scenario}",
                    "scenario_type": _scenario_type(scenario),
                    "failure_family": _failure_family(scenario),
                    "has_raw_log": bool(has_log),
                    "has_kpi": bool(has_kpi),
                    "pair_status": "STRICT" if has_log and has_kpi else ("LOG_ONLY" if has_log else ("KPI_ONLY" if has_kpi else "MISSING")),
                    "log_path": _safe_rel(log_path),
                    "kpi_path": _safe_rel(scenario_kpi_dir),
                }
            )
            if has_log and not has_kpi:
                issues.append({"workload": workload, "scenario": scenario, "issue": "mapred log without KPI measurements"})
            if has_kpi and not has_log:
                issues.append({"workload": workload, "scenario": scenario, "issue": "KPI measurements without mapred log"})
    return pd.DataFrame(rows), pd.DataFrame(issues)


def _floor_time(ts: pd.Timestamp, seconds: int) -> pd.Timestamp:
    return pd.Timestamp(ts).floor(f"{int(seconds)}s")


def _ceil_time(ts: pd.Timestamp, seconds: int) -> pd.Timestamp:
    return pd.Timestamp(ts).ceil(f"{int(seconds)}s")


def _extract_anchor(log_events: pd.DataFrame, scenario_type: str) -> tuple[pd.Timestamp | pd.NaT, str]:
    if scenario_type != "failure" or log_events.empty:
        return pd.NaT, "normal_run"
    level = log_events["level"].astype(str).isin(["WARN", "WARNING", "ERROR", "FATAL"])
    text = log_events["message_norm"].astype(str).str.contains("|".join(map(re.escape, STRONG_ANCHOR_WORDS)), regex=True, na=False)
    candidates = log_events.loc[level | text].copy()
    if candidates.empty:
        return pd.NaT, "no_explicit_anchor"
    candidates = candidates.sort_values(["timestamp", "line_id"])
    return pd.Timestamp(candidates.iloc[0]["timestamp"]), "explicit_log_signal"


def _keyword_count(messages: pd.Series) -> int:
    if messages.empty:
        return 0
    text = " ".join(messages.dropna().astype(str).tolist()).lower()
    return int(sum(text.count(keyword) for keyword in KEYWORDS))


def _compact_log_text(events: pd.DataFrame, limit: int = 16) -> str:
    if events.empty:
        return ""
    selected = events.sort_values(["event_score", "timestamp"], ascending=[False, False]).head(limit)
    return " ".join(selected["message_norm"].fillna("").astype(str).tolist())[:3000]


def _log_features_for_window(events: pd.DataFrame, window_ts: pd.Timestamp, window_end: pd.Timestamp) -> dict[str, Any]:
    macro_start = window_end - pd.Timedelta(seconds=MACRO_WINDOW_SEC)
    macro = events.loc[events["timestamp"].ge(macro_start) & events["timestamp"].lt(window_end)]
    micro = events.loc[events["timestamp"].ge(window_ts) & events["timestamp"].lt(window_end)]

    def features(prefix: str, part: pd.DataFrame) -> dict[str, Any]:
        count = int(len(part))
        levels = part["level"].astype(str) if count else pd.Series([], dtype=str)
        return {
            f"{prefix}_log_count": count,
            f"{prefix}_unique_components": int(part["component"].nunique()) if count else 0,
            f"{prefix}_warn_count": int(levels.isin(["WARN", "WARNING"]).sum()) if count else 0,
            f"{prefix}_error_count": int(levels.isin(["ERROR", "FATAL"]).sum()) if count else 0,
            f"{prefix}_keyword_count": _keyword_count(part["message_norm"]) if count else 0,
        }

    out = {}
    out.update(features("macro", macro))
    out.update(features("micro", micro))
    out["macro_log_rate"] = out["macro_log_count"] / float(MACRO_WINDOW_SEC)
    out["micro_log_rate"] = out["micro_log_count"] / float(MICRO_WINDOW_SEC)
    out["micro_error_rate"] = out["micro_error_count"] / float(MICRO_WINDOW_SEC)
    out["log_available"] = int(out["macro_log_count"] > 0)
    out["log_text"] = _compact_log_text(macro)
    return out


def _clean_token(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace("%", "")
    value = value.replace("/", "_s")
    value = value.replace("-", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "value"


def _epoch_seconds_to_utc(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    valid = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)) & numeric.between(946684800, 4102444800).to_numpy()
    if valid.any():
        safe_values = numeric.loc[valid].astype("float64")
        out.loc[valid] = pd.to_datetime(safe_values, unit="s", utc=True, errors="coerce")
    return out


def _safe_epoch_second(value: str) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out) or out < 946684800 or out > 4102444800:
        return None
    return out


def _read_kpi_timestamp_edges(path: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    first: float | None = None
    last: float | None = None
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            header = fh.readline().strip().split()
            if "TimeStamp" not in header:
                return None, None
            ts_idx = header.index("TimeStamp")
            for line in fh:
                parts = line.strip().split()
                if len(parts) <= ts_idx:
                    continue
                value = _safe_epoch_second(parts[ts_idx])
                if value is None:
                    continue
                if first is None:
                    first = value
                last = value
    except Exception:
        return None, None
    if first is None or last is None:
        return None, None
    start = pd.to_datetime(first, unit="s", utc=True, errors="coerce")
    end = pd.to_datetime(last, unit="s", utc=True, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return None, None
    return pd.Timestamp(start), pd.Timestamp(end)


def _read_kpi_file(path: Path, metric: str) -> pd.DataFrame:
    selected = KPI_SELECTED_COLUMNS.get(metric, [])
    if not selected or not path.exists() or path.name.startswith("."):
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, sep=r"\s+", engine="python", dtype=str)
    except Exception:
        return pd.DataFrame()
    if "TimeStamp" not in frame.columns:
        return pd.DataFrame()
    ts_num = pd.to_numeric(frame["TimeStamp"], errors="coerce")
    frame = frame.loc[ts_num.notna()].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["timestamp"] = _epoch_seconds_to_utc(frame["TimeStamp"])
    frame = frame.dropna(subset=["timestamp"]).copy()
    keep = ["timestamp"] + [col for col in selected if col in frame.columns]
    if len(keep) == 1:
        return pd.DataFrame()
    out = frame[keep].copy()
    for col in keep[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _kpi_timestamp_bounds(kpi_root: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    instruments = kpi_root / "instruments"
    if not instruments.exists():
        return None, None
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for host_dir in sorted([p for p in instruments.iterdir() if p.is_dir() and not p.name.startswith(".")]):
        for metric in KPI_SELECTED_COLUMNS:
            path = host_dir / metric
            if not path.exists() or path.name.startswith("."):
                continue
            start, end = _read_kpi_timestamp_edges(path)
            if start is None or end is None:
                continue
            starts.append(start)
            ends.append(end)
    if not starts:
        return None, None
    return min(starts), max(ends)


def _kpi_stats_for_file(path: Path, host: str, metric: str, run_start: pd.Timestamp, n_windows: int) -> pd.DataFrame:
    raw = _read_kpi_file(path, metric)
    if raw.empty:
        return pd.DataFrame(index=range(n_windows))
    offset = (raw["timestamp"] - run_start).dt.total_seconds()
    raw = raw.loc[offset.ge(0) & offset.lt(n_windows * MICRO_WINDOW_SEC)].copy()
    if raw.empty:
        return pd.DataFrame(index=range(n_windows))
    raw["window_index"] = np.floor((raw["timestamp"] - run_start).dt.total_seconds() / MICRO_WINDOW_SEC).astype(int)
    feature_frames: list[pd.DataFrame] = []
    numeric_cols = [col for col in raw.columns if col not in {"timestamp", "window_index"}]
    for source_col in numeric_cols:
        series = raw[["window_index", "timestamp", source_col]].dropna(subset=[source_col]).sort_values("timestamp")
        if series.empty:
            continue
        grouped = series.groupby("window_index", sort=True)
        mean = grouped[source_col].mean()
        std = grouped[source_col].std()
        first = grouped[source_col].first()
        last = grouped[source_col].last()
        first_ts = grouped["timestamp"].first()
        last_ts = grouped["timestamp"].last()
        duration = (last_ts - first_ts).dt.total_seconds().replace(0, np.nan)
        delta = last - first
        slope = delta / duration
        base = f"kpi_{_clean_token(host)}__{_clean_token(metric)}__{_clean_token(source_col)}__mean_dim"
        stats = pd.DataFrame(
            {
                f"{base}__mean": mean,
                f"{base}__last": last,
                f"{base}__std": std,
                f"{base}__delta": delta,
                f"{base}__slope": slope,
            }
        )
        feature_frames.append(stats)
    if not feature_frames:
        return pd.DataFrame(index=range(n_windows))
    return pd.concat(feature_frames, axis=1).reindex(range(n_windows))


def _build_kpi_features_for_run(kpi_root: Path, run_start: pd.Timestamp, n_windows: int) -> pd.DataFrame:
    instruments = kpi_root / "instruments"
    if not instruments.exists():
        return pd.DataFrame(index=range(n_windows))
    frames: list[pd.DataFrame] = []
    for host_dir in sorted([p for p in instruments.iterdir() if p.is_dir() and not p.name.startswith(".")]):
        for metric, _ in KPI_SELECTED_COLUMNS.items():
            path = host_dir / metric
            if path.exists():
                part = _kpi_stats_for_file(path, host_dir.name, metric, run_start, n_windows)
                if not part.empty:
                    frames.append(part)
    if not frames:
        return pd.DataFrame(index=range(n_windows))
    # Duplicate column names can happen when the raw file contains aliases after
    # sanitizing. Averaging them keeps the contract deterministic and compact.
    out = pd.concat(frames, axis=1)
    if out.columns.duplicated().any():
        out = out.T.groupby(level=0).mean().T
    return out.reindex(range(n_windows))


def _split_for_workload(workload: str, workloads: list[str]) -> str:
    validation = {"bayes", "repartition", "terasort"}
    test = {"aggregation", "join", "rf", "svd"}
    if workload in test:
        return "test"
    if workload in validation:
        return "validation"
    if not (validation & set(workloads)) or not (test & set(workloads)):
        ordered = sorted(workloads)
        pos = ordered.index(workload)
        if pos % 7 == 0:
            return "test"
        if pos % 5 == 0:
            return "validation"
    return "train"


def build_contract(
    raw_data: str | Path = PROJECT_ROOT / "raw_data",
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    *,
    max_runs: int | None = None,
    micro_window_sec: int = MICRO_WINDOW_SEC,
    macro_window_sec: int = MACRO_WINDOW_SEC,
    strict_gap_sec: int = STRICT_GAP_SEC,
    warning_horizon_sec: int = WARNING_HORIZON_SEC,
    risk_horizon_sec: int = RISK_HORIZON_SEC,
    pre_anchor_context_sec: int = PRE_ANCHOR_CONTEXT_SEC,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    global MICRO_WINDOW_SEC, MACRO_WINDOW_SEC, STRICT_GAP_SEC, WARNING_HORIZON_SEC, RISK_HORIZON_SEC, PRE_ANCHOR_CONTEXT_SEC
    MICRO_WINDOW_SEC = int(micro_window_sec)
    MACRO_WINDOW_SEC = int(macro_window_sec)
    STRICT_GAP_SEC = int(strict_gap_sec)
    WARNING_HORIZON_SEC = int(warning_horizon_sec)
    RISK_HORIZON_SEC = int(risk_horizon_sec)
    PRE_ANCHOR_CONTEXT_SEC = int(pre_anchor_context_sec)
    set_seed(SEED)
    raw_root = Path(raw_data)
    if not raw_root.is_absolute():
        raw_root = PROJECT_ROOT / raw_root
    if not raw_root.exists():
        raise FileNotFoundError(f"raw_data directory not found: {raw_root}")

    paths = artifact_paths(artifact_dir)
    for directory in [paths.contract_dir, paths.model_dir, paths.bundle_dir, paths.reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    inventory, issues = scan_raw_data(raw_root)
    strict_runs = inventory.loc[inventory["pair_status"].eq("STRICT")].copy()
    strict_runs = strict_runs.sort_values(["workload", "scenario"]).reset_index(drop=True)
    if max_runs:
        groups = [part for _, part in strict_runs.groupby("workload", sort=True)]
        picked: list[pd.DataFrame] = []
        cursor = 0
        while sum(len(part) for part in picked) < int(max_runs):
            added = False
            for part in groups:
                if cursor < len(part):
                    picked.append(part.iloc[[cursor]])
                    added = True
                    if sum(len(item) for item in picked) >= int(max_runs):
                        break
            if not added:
                break
            cursor += 1
        strict_runs = pd.concat(picked, ignore_index=True) if picked else strict_runs.head(int(max_runs)).copy()
    workloads = sorted(strict_runs["workload"].unique().tolist())

    rows: list[pd.DataFrame] = []
    excluded: list[dict[str, Any]] = []
    for _, run in strict_runs.iterrows():
        workload = str(run["workload"])
        scenario = str(run["scenario"])
        scenario_type = str(run["scenario_type"])
        run_id = f"{workload}/{scenario}"
        log_path = raw_root / workload / "mapred_logs" / f"{scenario}.log"
        kpi_root = raw_root / workload / "kpis" / scenario

        log_events = parse_raw_log_file(log_path)
        if log_events.empty:
            excluded.append({"run_id": run_id, "reason": "no timestamped log events"})
            continue
        anchor, anchor_policy = _extract_anchor(log_events, scenario_type)
        if scenario_type == "failure" and pd.isna(anchor):
            excluded.append({"run_id": run_id, "reason": "failure run without explicit log anchor"})
            continue

        log_start = pd.Timestamp(log_events["timestamp"].min())
        log_end = pd.Timestamp(log_events["timestamp"].max())
        kpi_start, kpi_end = _kpi_timestamp_bounds(kpi_root)
        observed_start_candidates = [log_start] + ([kpi_start] if kpi_start is not None else [])
        observed_end_candidates = [log_end] + ([kpi_end] if kpi_end is not None else [])
        if scenario_type == "failure" and pd.notna(anchor):
            context_sec = max(PRE_ANCHOR_CONTEXT_SEC, WARNING_HORIZON_SEC + STRICT_GAP_SEC)
            context_start = pd.Timestamp(anchor) - pd.Timedelta(seconds=context_sec)
            observed_start = min(observed_start_candidates)
            run_start = _floor_time(max(observed_start, context_start), MICRO_WINDOW_SEC)
            run_end = _ceil_time(anchor, MICRO_WINDOW_SEC)
        else:
            run_start = _floor_time(min(observed_start_candidates), MICRO_WINDOW_SEC)
            run_end = _ceil_time(max(observed_end_candidates) + pd.Timedelta(seconds=MICRO_WINDOW_SEC), MICRO_WINDOW_SEC)
        n_windows = int(max(1, math.ceil((run_end - run_start).total_seconds() / MICRO_WINDOW_SEC)))

        kpi = _build_kpi_features_for_run(kpi_root, run_start, n_windows)
        window_rows: list[dict[str, Any]] = []
        for window_index in range(n_windows):
            window_ts = run_start + pd.Timedelta(seconds=window_index * MICRO_WINDOW_SEC)
            window_end = window_ts + pd.Timedelta(seconds=MICRO_WINDOW_SEC)
            lead_sec = np.nan
            minutes_before_fail = np.nan
            if scenario_type == "failure" and pd.notna(anchor):
                lead_sec = float((anchor - window_end).total_seconds())
                minutes_before_fail = lead_sec / 60.0
                if lead_sec < STRICT_GAP_SEC:
                    continue
            warning_intensity = 0.0
            risk_intensity = 0.0
            y_warning = 0
            y_risk = 0
            if np.isfinite(lead_sec):
                y_warning = int(STRICT_GAP_SEC <= lead_sec <= WARNING_HORIZON_SEC)
                y_risk = int(STRICT_GAP_SEC <= lead_sec <= RISK_HORIZON_SEC)
                if y_warning:
                    warning_intensity = float((WARNING_HORIZON_SEC - lead_sec) / max(WARNING_HORIZON_SEC - STRICT_GAP_SEC, 1))
                if y_risk:
                    risk_intensity = float((RISK_HORIZON_SEC - lead_sec) / max(RISK_HORIZON_SEC - STRICT_GAP_SEC, 1))

            log_features = _log_features_for_window(log_events, window_ts, window_end)
            window_rows.append(
                {
                    "workload": workload,
                    "scenario": scenario,
                    "run_id": run_id,
                    "group_id": workload,
                    "failure_family": str(run["failure_family"]),
                    "scenario_type": scenario_type,
                    "window_ts": window_ts,
                    "window_end": window_end,
                    "run_start": run_start,
                    "run_end": run_end,
                    "fail_start": anchor if scenario_type == "failure" else pd.NaT,
                    "event_anchor": anchor if scenario_type == "failure" else pd.NaT,
                    "anchor_policy": anchor_policy,
                    "minutes_before_fail": minutes_before_fail,
                    "macro_window_sec": MACRO_WINDOW_SEC,
                    "micro_window_sec": MICRO_WINDOW_SEC,
                    "window_index": window_index,
                    "split": _split_for_workload(workload, workloads),
                    "state": "risk" if y_risk else ("warning" if y_warning else "normal"),
                    "y_warning": y_warning,
                    "y_risk": y_risk,
                    "y_anomaly": int(y_warning or y_risk),
                    "target_available": 1,
                    "target_policy": "strict_explicit_gap10",
                    "strict_variant": "strict_intensity_gap10",
                    "strict_lead_sec": lead_sec,
                    "strict_lead_gap_sec": STRICT_GAP_SEC,
                    "strict_warning_horizon_sec": WARNING_HORIZON_SEC,
                    "strict_risk_horizon_sec": RISK_HORIZON_SEC,
                    "warning_intensity": float(np.clip(warning_intensity, 0.0, 1.0)),
                    "risk_intensity": float(np.clip(risk_intensity, 0.0, 1.0)),
                    **log_features,
                }
            )
        if not window_rows:
            excluded.append({"run_id": run_id, "reason": "no windows after gap policy"})
            continue
        frame = pd.DataFrame(window_rows).reset_index(drop=True)
        if not kpi.empty:
            kpi_part = kpi.reindex(frame["window_index"].astype(int).tolist()).reset_index(drop=True)
            frame = pd.concat([frame.reset_index(drop=True), kpi_part.reset_index(drop=True)], axis=1)
        rows.append(frame)

    if not rows:
        raise RuntimeError("No strict contract rows were built from raw_data.")
    contract = pd.concat(rows, ignore_index=True)
    contract = contract.copy()
    kpi_cols = [col for col in contract.columns if col.startswith("kpi_") and "__" in col]
    if kpi_cols:
        kpi_values = contract[kpi_cols].apply(pd.to_numeric, errors="coerce")
        contract["kpi_feature_count"] = kpi_values.notna().sum(axis=1).astype(int)
        contract["kpi_series_available"] = contract["kpi_feature_count"]
        contract["kpi_avail_ratio"] = contract["kpi_feature_count"] / float(len(kpi_cols))
        contract["kpi_available"] = contract["kpi_feature_count"].gt(0).astype(int)
        contract["kpi_recency_sec"] = np.where(contract["kpi_available"].eq(1), 0.0, np.nan)
    else:
        contract["kpi_feature_count"] = 0
        contract["kpi_series_available"] = 0
        contract["kpi_avail_ratio"] = 0.0
        contract["kpi_available"] = 0
        contract["kpi_recency_sec"] = np.nan
    contract["has_both_modalities"] = (contract["log_available"].astype(int).eq(1) & contract["kpi_available"].astype(int).eq(1)).astype(int)

    contract = contract.sort_values(["run_id", "window_ts"]).reset_index(drop=True)
    contract.to_csv(paths.contract_path, index=False, compression="gzip")

    summary = {
        "source": "raw_data",
        "contract_path": _safe_rel(paths.contract_path),
        "rows": int(len(contract)),
        "runs": int(contract["run_id"].nunique()),
        "workloads": int(contract["workload"].nunique()),
        "windows": {"micro_window_sec": MICRO_WINDOW_SEC, "macro_window_sec": MACRO_WINDOW_SEC},
        "target_policy": {
            "variant": "strict_intensity_gap10",
            "gap_sec": STRICT_GAP_SEC,
            "warning_horizon_sec": WARNING_HORIZON_SEC,
            "risk_horizon_sec": RISK_HORIZON_SEC,
            "pre_anchor_context_sec": PRE_ANCHOR_CONTEXT_SEC,
        },
        "coverage": {
            "log_available_share": float(contract["log_available"].mean()),
            "kpi_available_share": float(contract["kpi_available"].mean()),
            "both_modalities_share": float(contract["has_both_modalities"].mean()),
        },
        "targets": {
            "warning_share": float(contract["y_warning"].mean()),
            "risk_share": float(contract["y_risk"].mean()),
        },
        "split_rows": contract.groupby("split").size().to_dict(),
        "split_runs": contract.groupby("split")["run_id"].nunique().to_dict(),
        "inventory": {
            "rows": int(len(inventory)),
            "strict_runs": int(strict_runs.shape[0]),
            "issues": int(len(issues)),
            "excluded_runs": excluded,
        },
    }
    pd.DataFrame(
        [
            {"metric": "rows", "value": summary["rows"]},
            {"metric": "runs", "value": summary["runs"]},
            {"metric": "workloads", "value": summary["workloads"]},
            {"metric": "warning_share", "value": summary["targets"]["warning_share"]},
            {"metric": "risk_share", "value": summary["targets"]["risk_share"]},
            {"metric": "log_available_share", "value": summary["coverage"]["log_available_share"]},
            {"metric": "kpi_available_share", "value": summary["coverage"]["kpi_available_share"]},
            {"metric": "both_modalities_share", "value": summary["coverage"]["both_modalities_share"]},
        ]
    ).to_csv(paths.contract_summary_path, index=False)
    write_json(paths.pipeline_report_path, summary)
    return contract, summary


def _safe_ap(y: pd.Series, score: pd.Series) -> float:
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    score = pd.to_numeric(score, errors="coerce").fillna(0.0)
    if y.nunique() < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def _safe_roc(y: pd.Series, score: pd.Series) -> float:
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    score = pd.to_numeric(score, errors="coerce").fillna(0.0)
    if y.nunique() < 2 or score.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def _fit_predict_baseline(train: pd.DataFrame, test: pd.DataFrame, target: str, baseline: str, numeric_cols: list[str]) -> np.ndarray:
    features = ["log_text"] + numeric_cols
    transformer = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=1600, min_df=2, ngram_range=(1, 2)), "log_text"),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))]), numeric_cols),
        ],
        remainder="drop",
    )
    if baseline in {"kpi-only", "availability-only"}:
        features = numeric_cols
        transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))])
    model = Pipeline(
        [
            ("features", transformer),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear", random_state=SEED)),
        ]
    )
    y_train = train[target].astype(int)
    if y_train.nunique() < 2:
        return np.full(len(test), float(y_train.mean()) if len(y_train) else 0.0)
    model.fit(train[features], y_train)
    return model.predict_proba(test[features])[:, 1]


def build_baselines(contract: pd.DataFrame, artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    paths = artifact_paths(artifact_dir)
    train = contract.loc[contract["split"].eq("train")].copy()
    test = contract.loc[contract["split"].eq("test")].copy()
    if train.empty or test.empty:
        raise RuntimeError("Contract must contain non-empty train and test splits for baselines.")

    kpi_cols = [col for col in contract.columns if col.startswith("kpi_") and "__" in col]
    log_numeric = [col for col in LOG_NUMERIC_COLUMNS if col in contract.columns]
    mask_cols = [col for col in MASK_COLUMNS if col in contract.columns]
    kpi_numeric = kpi_cols + mask_cols
    feature_sets = {
        "log-only": log_numeric + ["log_available"],
        "kpi-only": kpi_numeric,
        "availability-only": mask_cols,
        "fused": log_numeric + kpi_numeric,
    }

    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for target in ["y_warning", "y_risk"]:
        for baseline, numeric_cols in feature_sets.items():
            if baseline != "log-only" and len(numeric_cols) == 0:
                score = np.full(len(test), train[target].mean())
            else:
                score = _fit_predict_baseline(train, test, target, baseline, numeric_cols)
            pred = test[["run_id", "window_ts", "split", target]].copy()
            pred = pred.rename(columns={target: "label"})
            pred["target"] = target
            pred["baseline"] = baseline
            pred["score"] = score
            predictions.append(pred)
            y = test[target].astype(int)
            labels = (pd.Series(score, index=test.index) >= 0.5).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(y, labels, average="binary", zero_division=0)
            metrics.append(
                {
                    "target": target,
                    "baseline": baseline,
                    "average_precision": _safe_ap(y, pd.Series(score, index=test.index)),
                    "roc_auc": _safe_roc(y, pd.Series(score, index=test.index)),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "n": int(len(test)),
                    "positives": int(y.sum()),
                    "prevalence": float(y.mean()),
                }
            )

    pred_df = pd.concat(predictions, ignore_index=True)
    metrics_df = pd.DataFrame(metrics).sort_values(["target", "average_precision"], ascending=[True, False]).reset_index(drop=True)
    pred_df.to_csv(paths.baseline_predictions_path, index=False, compression="gzip")
    metrics_df.to_csv(paths.baseline_comparison_path, index=False)

    payload = {
        "source": "dashboard.pipeline.build_contract",
        "contract_path": _safe_rel(paths.contract_path),
        "baseline_predictions_path": _safe_rel(paths.baseline_predictions_path),
        "baseline_comparison_path": _safe_rel(paths.baseline_comparison_path),
        "baseline_metrics": metrics_df.to_dict("records"),
        "windows": {"micro_window_sec": MICRO_WINDOW_SEC, "macro_window_sec": MACRO_WINDOW_SEC},
    }
    write_json(paths.baseline_summary_path, payload)
    return metrics_df, pred_df, payload


def run_pipeline(
    raw_data: str | Path = PROJECT_ROOT / "raw_data",
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    *,
    max_runs: int | None = None,
    micro_window_sec: int = MICRO_WINDOW_SEC,
    macro_window_sec: int = MACRO_WINDOW_SEC,
    strict_gap_sec: int = STRICT_GAP_SEC,
    warning_horizon_sec: int = WARNING_HORIZON_SEC,
    risk_horizon_sec: int = RISK_HORIZON_SEC,
    pre_anchor_context_sec: int = PRE_ANCHOR_CONTEXT_SEC,
) -> dict[str, Any]:
    contract, summary = build_contract(
        raw_data=raw_data,
        artifact_dir=artifact_dir,
        max_runs=max_runs,
        micro_window_sec=micro_window_sec,
        macro_window_sec=macro_window_sec,
        strict_gap_sec=strict_gap_sec,
        warning_horizon_sec=warning_horizon_sec,
        risk_horizon_sec=risk_horizon_sec,
        pre_anchor_context_sec=pre_anchor_context_sec,
    )
    baseline_metrics, _, baseline_payload = build_baselines(contract, artifact_dir=artifact_dir)
    paths = artifact_paths(artifact_dir)
    report = {
        "status": "ok",
        "artifact_dir": _safe_rel(paths.artifact_dir),
        "contract": summary,
        "baseline_rows": int(len(baseline_metrics)),
        "written_files": {
            "contract": _safe_rel(paths.contract_path),
            "contract_summary": _safe_rel(paths.contract_summary_path),
            "baseline_summary": _safe_rel(paths.baseline_summary_path),
            "baseline_comparison": _safe_rel(paths.baseline_comparison_path),
            "baseline_predictions": _safe_rel(paths.baseline_predictions_path),
            "pipeline_report": _safe_rel(paths.pipeline_report_path),
        },
        "baseline": baseline_payload,
    }
    write_json(paths.pipeline_report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final strict/intensity contract and baselines from raw_data.")
    parser.add_argument("--raw-data", default="raw_data")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR.relative_to(PROJECT_ROOT)))
    parser.add_argument("--max-runs", type=int, default=None, help="Optional smoke-run limiter.")
    parser.add_argument("--micro-window-sec", type=int, default=MICRO_WINDOW_SEC)
    parser.add_argument("--macro-window-sec", type=int, default=MACRO_WINDOW_SEC)
    parser.add_argument("--strict-gap-sec", type=int, default=STRICT_GAP_SEC)
    parser.add_argument("--warning-horizon-sec", type=int, default=WARNING_HORIZON_SEC)
    parser.add_argument("--risk-horizon-sec", type=int, default=RISK_HORIZON_SEC)
    parser.add_argument("--pre-anchor-context-sec", type=int, default=PRE_ANCHOR_CONTEXT_SEC)
    args = parser.parse_args()
    report = run_pipeline(
        raw_data=args.raw_data,
        artifact_dir=args.artifact_dir,
        max_runs=args.max_runs,
        micro_window_sec=args.micro_window_sec,
        macro_window_sec=args.macro_window_sec,
        strict_gap_sec=args.strict_gap_sec,
        warning_horizon_sec=args.warning_horizon_sec,
        risk_horizon_sec=args.risk_horizon_sec,
        pre_anchor_context_sec=args.pre_anchor_context_sec,
    )
    print(json.dumps(_json_safe(report["written_files"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
