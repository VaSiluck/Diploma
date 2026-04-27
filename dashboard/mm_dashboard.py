from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - runtime dependency guard
    go = None


APP_ROOT = Path(__file__).resolve().parents[1]
LCMAN_DIR = APP_ROOT / "outputs" / "lcman"
FINAL_DEMO_DIR = APP_ROOT / "outputs" / "final_demo"
DASHBOARD_TIMELINE_PATH = LCMAN_DIR / "lcman_dashboard_timeline.csv"
RUNTIME_META_JSON_PATH = LCMAN_DIR / "lcman_runtime_metadata.json"
RUNTIME_META_CSV_PATH = LCMAN_DIR / "lcman_runtime_metadata.csv"
KPI_RAW_ROOT = APP_ROOT / "kpi_csv"


RAW_KPI_TIME_COLUMNS = ("timestamp", "Timestamp", "time", "Time")
RAW_KPI_FAMILIES = ("cpu", "memstat", "iostat", "netstat")
RAW_KPI_CURRENT_TOLERANCE = pd.Timedelta(minutes=2, seconds=30)
RAW_KPI_PANEL_META: dict[str, dict[str, Any]] = {
    "cpu": {
        "title": "CPU",
        "color": "#2563eb",
        "priority": ["cpu__user", "cpu__system", "cpu__iowait", "cpu__idle"],
    },
    "memstat": {
        "title": "Memory",
        "color": "#059669",
        "priority": ["memstat__memused", "memstat__commit"],
    },
    "iostat": {
        "title": "Disk",
        "color": "#d97706",
        "priority": [
            "iostat__util",
            "iostat__total_kB_s",
            "iostat__wkB_s",
            "iostat__rkB_s",
            "iostat__w_await",
            "iostat__r_await",
        ],
    },
    "netstat": {
        "title": "Network",
        "color": "#7c3aed",
        "priority": ["netstat__total_kB_s", "netstat__rxkB_s", "netstat__txkB_s"],
    },
}


STATE_LABELS = {
    "normal": "Норма",
    "warning": "Предупреждение",
    "risk": "Риск",
}
STATE_COLORS = {
    "normal": "#16a34a",
    "warning": "#f59e0b",
    "risk": "#ef4444",
}

CASE_UI_LABELS = {
    "multimodal_risk": "мультириск",
    "warning_only": "предупреждение",
    "kpi_missing": "KPI отсутствуют",
    "normal": "норма",
}
CASE_UI_ORDER = ["multimodal_risk", "warning_only", "kpi_missing", "normal"]

RISK_NORMAL_MAX = 0.30
RISK_WARNING_MAX = 0.70

LOG_PATTERN_LIBRARY = [
    ("blk_update_request", "blk_update_request: I/O error"),
    ("task blocked", "task blocked > 120s"),
    ("connection pool timeout", "connection pool timeout"),
    ("write latency", "write latency exceeded"),
    ("buffer i/o", "kernel buffer I/O error"),
    ("registering the applicationmaster", "Registering the ApplicationMaster"),
    ("connecting to resourcemanager", "Connecting to ResourceManager"),
    ("launching container", "Launching container"),
    ("received new token", "Received new token"),
    ("started remote fetches", "Started remote fetches"),
    ("memorystore", "MemoryStore operations"),
    ("__no_log__", "Окно без логов"),
]

SERIES_LABELS = {
    "cpu__idle": "CPU idle",
    "cpu__iowait": "CPU iowait",
    "cpu__system": "CPU system",
    "cpu__user": "CPU user",
    "iostat__r_await": "Disk read await",
    "iostat__rkB_s": "Disk read KB/s",
    "iostat__util": "Disk util",
    "iostat__w_await": "Disk write await",
    "iostat__wkB_s": "Disk write KB/s",
    "memstat__commit": "Memory commit",
    "memstat__memused": "Memory used",
    "netstat__rxkB_s": "Network RX KB/s",
    "netstat__txkB_s": "Network TX KB/s",
}

KPI_DENSE_PRIORITY = [
    "kpi_last",
    "kpi_mean",
    "kpi_roll_mean_60",
    "kpi_roll_mean_120",
    "kpi_delta_60",
    "kpi_delta_120",
    "kpi_slope_60",
    "kpi_slope_120",
    "cpu__last__family_pool",
    "cpu__mean__family_pool",
    "cpu__delta__family_pool",
    "cpu__slope__family_pool",
    "iostat__last__family_pool",
    "iostat__mean__family_pool",
    "iostat__delta__family_pool",
    "iostat__slope__family_pool",
    "memstat__last__family_pool",
    "memstat__mean__family_pool",
    "memstat__delta__family_pool",
    "memstat__slope__family_pool",
    "netstat__last__family_pool",
    "netstat__mean__family_pool",
    "netstat__delta__family_pool",
    "netstat__slope__family_pool",
]

KPI_INTERP_SUFFIXES = ("__last", "__mean", "__delta", "__slope")


class DataContractError(RuntimeError):
    pass


def normalize_state(value: Any) -> str:
    if value is None:
        return "normal"
    state = str(value).strip().lower()
    if state in {"anomaly", "critical"}:
        state = "risk"
    if state in STATE_LABELS:
        return state
    return "normal"


def risk_to_state(risk: float) -> str:
    r = float(np.clip(risk, 0.0, 1.0))
    if r < RISK_NORMAL_MAX:
        return "normal"
    if r < RISK_WARNING_MAX:
        return "warning"
    return "risk"


def score_to_state(score: float) -> str:
    return risk_to_state(score)


def confidence_from_risk(risk: float, state: str) -> float:
    r = float(np.clip(risk, 0.0, 1.0))
    # calibration for demo UX: confidence reflects distance from indecision 0.5
    conf = 0.58 + 0.85 * abs(r - 0.5)
    return float(np.clip(conf, 0.05, 0.99))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if np.isnan(out) or np.isinf(out):
        return default
    return out


def find_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _format_run_title(run_id: str) -> str:
    if "/" in run_id:
        app, scenario = run_id.split("/", 1)
        return f"{app.title()} / {scenario}"
    return run_id


def _make_default_explanation(row: pd.Series) -> str:
    kpi_available = int(safe_float(row.get("kpi_available", 0), 0))
    delta = safe_float(
        row.get("delta_risk_vs_fused", row.get("delta_vs_log", 0.0)),
        0.0,
    )
    gate = safe_float(row.get("lcman_gate", np.nan), np.nan)

    if kpi_available == 0:
        return (
            "Риск оценён в деградированном мультимодальном режиме: KPI в окне отсутствуют, "
            "поэтому решение опирается на логовую ветку с missing-aware fallback."
        )
    if delta > 0.08:
        return (
            "Риск вырос за счёт мультимодального объединения: KPI подтверждают логовый сигнал и усиливают итоговую оценку."
        )
    if delta < -0.08:
        return (
            "KPI-сигналы в текущем окне смягчают риск относительно fused baseline, "
            "что указывает на противоречивость модальностей."
        )

    if np.isnan(gate):
        return "Обе модальности имеют сопоставимый вклад, итоговая оценка остаётся умеренной."

    if gate >= 0.75:
        return "Fusion лог-доминантна: решение в основном определяется семантикой логов."
    if gate <= 0.35:
        return "Fusion KPI-доминантна: временные ряды дают основной вклад в оценку риска."
    return "Обе модальности участвуют сбалансированно, без выраженного доминирования одной ветки."


def _infer_case_kind(row: pd.Series) -> str:
    kpi = int(safe_float(row.get("kpi_available", 0), 0))
    risk = safe_float(row.get("risk_score", 0.0), 0.0)
    state = normalize_state(row.get("pred_state", risk_to_state(risk)))
    delta = safe_float(
        row.get("delta_risk_vs_fused", row.get("delta_vs_log", 0.0)),
        0.0,
    )

    if kpi == 1 and state == "risk" and delta > 0.08:
        return "multimodal_risk"
    if state == "warning" and kpi == 0:
        return "warning_only"
    if kpi == 0:
        return "kpi_missing"
    return "normal"


def _make_case_description(kind: str, row: pd.Series) -> str:
    if kind == "multimodal_risk":
        return "Кейс near-failure: мультимодальное объединение усиливает риск относительно fused baseline."
    if kind == "warning_only":
        return "Warning-кейс: KPI в окне отсутствуют, решение остаётся log-driven и демонстрирует честный fallback."
    if kind == "kpi_missing":
        return "Сценарий с пропуском KPI: система работает в деградированном мультимодальном режиме без потери устойчивости."
    return "Стабильный нормальный фрагмент: низкий риск и отсутствие признаков деградации."


def _pick_best_row(df: pd.DataFrame, condition: pd.Series, sort_cols: list[tuple[str, bool]]) -> pd.Series | None:
    sub = df.loc[condition].copy()
    if sub.empty:
        return None
    by = [name for name, _ in sort_cols]
    asc = [flag for _, flag in sort_cols]
    sub = sub.sort_values(by=by, ascending=asc)
    return sub.iloc[0]


def pick_first_series(frame: pd.DataFrame, candidates: list[str], default: float | str = 0.0) -> pd.Series:
    for col in candidates:
        if col in frame.columns:
            return frame[col]
    return pd.Series(default, index=frame.index)


def _aggregate_series_stat(df: pd.DataFrame, stat: str) -> pd.Series:
    series_cols = [
        c
        for c in df.columns
        if c.endswith(f"__{stat}")
        and "__family_pool" not in c
        and c.startswith(("cpu__", "iostat__", "memstat__", "netstat__"))
    ]
    if series_cols:
        return df[series_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)

    family_cols = [
        c
        for c in df.columns
        if c.endswith(f"__{stat}__family_pool")
        and c.startswith(("cpu__", "iostat__", "memstat__", "netstat__"))
    ]
    if family_cols:
        return df[family_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)

    return pd.Series(np.nan, index=df.index, dtype=float)


def _ensure_dense_kpi_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    defaults = {
        "kpi_last": "last",
        "kpi_mean": "mean",
        "kpi_delta_60": "delta",
        "kpi_delta_120": "delta",
        "kpi_slope_60": "slope",
        "kpi_slope_120": "slope",
    }
    for col, stat in defaults.items():
        if col not in out.columns:
            out[col] = _aggregate_series_stat(out, stat)

    for col in defaults:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "kpi_roll_mean_60" not in out.columns:
        out["kpi_roll_mean_60"] = out["kpi_last"] - out["kpi_delta_60"]
    if "kpi_roll_mean_120" not in out.columns:
        out["kpi_roll_mean_120"] = out["kpi_last"] - out["kpi_delta_120"]

    out["kpi_roll_mean_60"] = pd.to_numeric(out["kpi_roll_mean_60"], errors="coerce")
    out["kpi_roll_mean_120"] = pd.to_numeric(out["kpi_roll_mean_120"], errors="coerce")

    if "kpi_slope_60" in out.columns and "kpi_delta_60" in out.columns:
        out["kpi_slope_60"] = pd.to_numeric(out["kpi_slope_60"], errors="coerce").fillna(
            pd.to_numeric(out["kpi_delta_60"], errors="coerce") / 60.0
        )
    if "kpi_slope_120" in out.columns and "kpi_delta_120" in out.columns:
        out["kpi_slope_120"] = pd.to_numeric(out["kpi_slope_120"], errors="coerce").fillna(
            pd.to_numeric(out["kpi_delta_120"], errors="coerce") / 120.0
        )

    return out


def _extract_kpi_dense_columns(df: pd.DataFrame) -> list[str]:
    explicit = [col for col in KPI_DENSE_PRIORITY if col in df.columns]
    if explicit:
        return explicit

    dynamic = [
        col
        for col in df.columns
        if col.startswith(("kpi_", "cpu__", "iostat__", "memstat__", "netstat__"))
        and (
            col.endswith(("_last", "_mean", "_delta", "_slope", "_60", "_120"))
            or col.endswith(("__last", "__mean", "__delta", "__slope"))
            or col.endswith(("__last__family_pool", "__mean__family_pool", "__delta__family_pool", "__slope__family_pool"))
        )
        and "__avail" not in col
        and "__count" not in col
        and "coverage" not in col
    ]
    return sorted(dynamic)


def _extract_task_manager_kpi_columns(df: pd.DataFrame) -> list[str]:
    primary = [
        col
        for col in df.columns
        if col.startswith(("cpu__", "iostat__", "memstat__", "netstat__"))
        and col.endswith("__last")
        and "__family_pool" not in col
    ]
    if primary:
        return sorted(primary)

    secondary = [
        col
        for col in df.columns
        if col.startswith(("cpu__", "iostat__", "memstat__", "netstat__"))
        and col.endswith("__mean")
        and "__family_pool" not in col
    ]
    if secondary:
        return sorted(secondary)

    return _extract_kpi_dense_columns(df)


def _kpi_interp_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []

    for col in _extract_kpi_dense_columns(df):
        if col in df.columns:
            cols.append(col)

    dense_stats = [
        col
        for col in df.columns
        if col.startswith(("cpu__", "iostat__", "memstat__", "netstat__"))
        and col.endswith(KPI_INTERP_SUFFIXES)
        and "__family_pool" not in col
    ]
    cols.extend(dense_stats)
    return sorted(set(cols))


def _safe_rel_minutes(df: pd.DataFrame) -> pd.Series:
    if "relative_min" in df.columns:
        out = pd.to_numeric(df["relative_min"], errors="coerce")
        if out.notna().any():
            return out
    if "fail_start" in df.columns:
        fail_ts = pd.to_datetime(df["fail_start"], utc=True, errors="coerce")
        return (pd.to_datetime(df["window_ts"], utc=True, errors="coerce") - fail_ts) / pd.Timedelta(minutes=1)
    return pd.Series(np.nan, index=df.index)


def build_replay_frame(run_df: pd.DataFrame, steps: int = 36) -> pd.DataFrame:
    """Build replay-friendly frame by interpolating numeric timeline to fixed steps.

    Source observations remain untouched; each replay point keeps pointer `_src_idx`
    to the nearest real aligned window for evidences and categorical fields.
    """
    if run_df.empty:
        return run_df.copy()

    base = run_df.sort_values("window_ts").reset_index(drop=True).copy()
    if len(base) == 1:
        base["_src_idx"] = 0
        return base

    steps = max(2, int(steps))

    ts = pd.to_datetime(base["window_ts"], utc=True, errors="coerce").astype("int64")
    x = ts.to_numpy(dtype=np.float64)
    new_x = np.linspace(float(x.min()), float(x.max()), steps)

    out = pd.DataFrame()
    out["window_ts"] = pd.to_datetime(new_x.astype(np.int64), utc=True)

    # nearest source index for categorical and evidence fields
    src_idx = np.abs(x.reshape(-1, 1) - new_x.reshape(1, -1)).argmin(axis=0)
    out["_src_idx"] = src_idx.astype(int)

    numeric_interp_cols = [
        "warning_score",
        "risk_score",
        "fused_warn_score",
        "fused_risk_score",
        "baseline_score",
        "delta_warn_vs_fused",
        "delta_risk_vs_fused",
        "delta_vs_log",
        "lcman_gate",
        "confidence",
        "relative_min",
    ]
    numeric_interp_cols.extend(_kpi_interp_columns(base))

    interp_payload: dict[str, np.ndarray] = {}
    for col in numeric_interp_cols:
        if col in base.columns:
            y = pd.to_numeric(base[col], errors="coerce").to_numpy(dtype=np.float64)
            valid = np.isfinite(y)
            if valid.sum() >= 2:
                interp_payload[col] = np.interp(new_x, x[valid], y[valid])
            else:
                interp_payload[col] = y[src_idx]

    if interp_payload:
        out = pd.concat([out, pd.DataFrame(interp_payload)], axis=1)

    # carry over remaining columns from nearest source
    carry_payload: dict[str, np.ndarray] = {}
    for col in base.columns:
        if col in out.columns:
            continue
        carry_payload[col] = base.iloc[src_idx][col].to_numpy()

    if carry_payload:
        out = pd.concat([out, pd.DataFrame(carry_payload)], axis=1)

    # recalc pred_state from source when available, fallback by risk
    if "pred_state" in out.columns:
        out["pred_state"] = out["pred_state"].map(normalize_state)
    else:
        out["pred_state"] = out["risk_score"].map(risk_to_state)

    if "warning_score" not in out.columns:
        out["warning_score"] = out["risk_score"]

    if "true_state" in out.columns:
        out["true_state"] = out["true_state"].map(normalize_state)

    if "confidence" not in out.columns:
        out["confidence"] = [
            confidence_from_risk(safe_float(r, 0.0), str(s))
            for r, s in zip(out["risk_score"], out["pred_state"])  # noqa: B905
        ]

    return out.reset_index(drop=True)


def _build_contract_rows(contract: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for logical_name, item in contract.items():
        rows.append(
            {
                "artifact": logical_name,
                "path": str(item["path"]),
                "required": "yes" if item.get("required", False) else "no",
                "found": "yes" if item.get("found", False) else "no",
                "required_fields": ", ".join(item.get("required_fields", [])),
                "optional_fields": ", ".join(item.get("optional_fields", [])),
            }
        )
    return pd.DataFrame(rows)


def _parse_model_card_runtime(model_card_text: str) -> dict[str, Any]:
    text = str(model_card_text or "")
    out: dict[str, Any] = {}
    patterns = {
        "data_source": r"-\s*источник:\s*(.+)",
        "data_mode": r"-\s*режим контракта:\s*(.+)",
        "encoder_name": r"-\s*encoder:\s*(.+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            out[key] = match.group(1).strip()
    return out


def _infer_embedding_dim_from_cache() -> int | None:
    cache_dir = LCMAN_DIR / "cache"
    if not cache_dir.exists():
        return None
    for path in sorted(cache_dir.glob("log_embeddings_*.npz")):
        try:
            with np.load(path, allow_pickle=False) as payload:
                for key in ["embeddings", *payload.files]:
                    if key not in payload.files:
                        continue
                    arr = payload[key]
                    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] > 0:
                        return int(arr.shape[1])
        except Exception:
            continue
    return None


def _read_runtime_metadata(model_card_text: str) -> dict[str, Any]:
    runtime: dict[str, Any] = {}

    if RUNTIME_META_JSON_PATH.exists():
        try:
            payload = json.loads(RUNTIME_META_JSON_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                runtime.update(payload)
        except Exception:
            pass
    elif RUNTIME_META_CSV_PATH.exists():
        try:
            meta_csv = pd.read_csv(RUNTIME_META_CSV_PATH)
            if not meta_csv.empty:
                runtime.update(meta_csv.iloc[0].to_dict())
        except Exception:
            pass

    runtime_from_card = _parse_model_card_runtime(model_card_text)
    for key, value in runtime_from_card.items():
        runtime.setdefault(key, value)

    if runtime.get("embedding_dim") in {None, "", "nan", "NaN"}:
        emb_dim = _infer_embedding_dim_from_cache()
        if emb_dim is not None:
            runtime["embedding_dim"] = emb_dim

    cleaned: dict[str, Any] = {}
    for key in ["encoder_name", "embedding_dim", "data_mode", "data_source"]:
        if key not in runtime:
            continue
        value = runtime[key]
        if isinstance(value, str):
            value = value.strip()
            if not value or value.lower() == "nan":
                continue
        if key == "embedding_dim":
            try:
                value = int(float(value))
            except Exception:
                continue
        cleaned[key] = value

    if cleaned and not RUNTIME_META_JSON_PATH.exists():
        try:
            RUNTIME_META_JSON_PATH.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    return cleaned


def _harmonize_timeline_contract(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    if "run_id" not in df.columns or "window_ts" not in df.columns:
        raise DataContractError("Timeline base misses required fields: ['run_id', 'window_ts']")

    df["run_id"] = df["run_id"].astype(str)
    df["window_ts"] = pd.to_datetime(df["window_ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["run_id", "window_ts"]).copy()

    if "fail_start" in df.columns:
        df["fail_start"] = pd.to_datetime(df["fail_start"], utc=True, errors="coerce")
    else:
        df["fail_start"] = pd.NaT

    if df["fail_start"].isna().all():
        df["fail_start"] = df["window_ts"]

    df["relative_min"] = _safe_rel_minutes(df)
    if df["relative_min"].isna().all() and "minutes_before_fail" in df.columns:
        df["relative_min"] = -pd.to_numeric(df["minutes_before_fail"], errors="coerce")

    if "state" in df.columns:
        state_series = df["state"]
    elif "true_state" in df.columns:
        state_series = df["true_state"]
    elif "pred_state" in df.columns:
        state_series = df["pred_state"]
    else:
        state_series = pd.Series("normal", index=df.index)
    state_series = (
        state_series.fillna("normal").astype(str).str.strip().str.lower().replace({"anomaly": "risk", "critical": "risk"})
    )

    if "y_warning" in df.columns:
        y_warning = pd.to_numeric(df["y_warning"], errors="coerce").fillna(0).astype(int)
    else:
        y_warning = state_series.eq("warning").astype(int)

    if "y_risk" in df.columns:
        y_risk = pd.to_numeric(df["y_risk"], errors="coerce").fillna(0).astype(int)
    else:
        y_risk = state_series.eq("risk").astype(int)

    df["y_warning"] = y_warning
    df["y_risk"] = y_risk
    df["state"] = np.select([df["y_risk"].eq(1), df["y_warning"].eq(1)], ["risk", "warning"], default="normal")

    if "true_state" in df.columns:
        df["true_state"] = df["true_state"].map(normalize_state)
    else:
        df["true_state"] = df["state"].map(normalize_state)

    if "pred_state" in df.columns:
        df["pred_state"] = df["pred_state"].map(normalize_state)
    else:
        df["pred_state"] = ""

    df["log_count"] = pd.to_numeric(pick_first_series(df, ["log_count", "macro_log_count", "micro_log_count"], 0.0), errors="coerce").fillna(0.0)
    df["unique_templates"] = pd.to_numeric(
        pick_first_series(df, ["unique_templates", "macro_unique_templates", "micro_unique_templates"], 0.0),
        errors="coerce",
    ).fillna(0.0)
    df["error_count"] = pd.to_numeric(pick_first_series(df, ["error_count", "macro_error_count", "micro_error_count"], 0.0), errors="coerce").fillna(0.0)

    if "log_available" in df.columns:
        df["log_available"] = pd.to_numeric(df["log_available"], errors="coerce").fillna(0)
    else:
        df["log_available"] = df["log_count"].gt(0).astype(int)
    df["log_available"] = df["log_available"].gt(0).astype(int)

    series_avail_cols = [
        c
        for c in df.columns
        if c.endswith("__avail")
        and "__family_pool" not in c
        and c.startswith(("cpu__", "iostat__", "memstat__", "netstat__"))
    ]

    if "kpi_series_available" in df.columns:
        df["kpi_series_available"] = pd.to_numeric(df["kpi_series_available"], errors="coerce").fillna(0)
    elif series_avail_cols:
        avail_block = df[series_avail_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        df["kpi_series_available"] = avail_block.gt(0).sum(axis=1)
    else:
        df["kpi_series_available"] = 0
    df["kpi_series_available"] = pd.to_numeric(df["kpi_series_available"], errors="coerce").fillna(0).astype(int)

    if "kpi_available" in df.columns:
        df["kpi_available"] = pd.to_numeric(df["kpi_available"], errors="coerce").fillna(0)
    else:
        df["kpi_available"] = df["kpi_series_available"].gt(0).astype(int)
    df["kpi_available"] = df["kpi_available"].gt(0).astype(int)

    if "has_both_modalities" in df.columns:
        has_both = pd.to_numeric(df["has_both_modalities"], errors="coerce").fillna(0)
        df["has_both_modalities"] = has_both.gt(0).astype(int)
    else:
        df["has_both_modalities"] = (df["log_available"].eq(1) & df["kpi_available"].eq(1)).astype(int)

    if "kpi_feature_count" not in df.columns:
        df["kpi_feature_count"] = pd.to_numeric(df["kpi_series_available"], errors="coerce").fillna(0.0)

    if "kpi_avail_ratio" not in df.columns:
        series_denom = max(len({c[: -len("__avail")] for c in series_avail_cols}), 1)
        df["kpi_avail_ratio"] = pd.to_numeric(df["kpi_series_available"], errors="coerce").fillna(0.0) / series_denom

    df["log_text"] = pick_first_series(df, ["log_text"], default="__NO_LOG__").fillna("__NO_LOG__").astype(str)
    if "explanation" not in df.columns:
        df["explanation"] = ""
    df["explanation"] = df["explanation"].fillna("").astype(str)

    df = _ensure_dense_kpi_features(df)

    return (
        df.sort_values(["run_id", "window_ts"])
        .drop_duplicates(["run_id", "window_ts"], keep="last")
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict[str, Any]:
    contract: dict[str, dict[str, Any]] = {}

    scored_candidates = [
        DASHBOARD_TIMELINE_PATH,
        LCMAN_DIR / "scored_windows_reference.csv",
        FINAL_DEMO_DIR / "scored_windows_reference.csv",
    ]
    scored_path = find_first_existing(scored_candidates)

    contract["timeline_base"] = {
        "path": scored_path if scored_path is not None else scored_candidates[0],
        "required": True,
        "found": scored_path is not None,
        "required_fields": ["run_id", "window_ts"],
        "optional_fields": [
            "state",
            "y_warning",
            "y_risk",
            "log_available",
            "kpi_available",
            "kpi_series_available",
            "has_both_modalities",
            "kpi_last",
            "kpi_mean",
            "kpi_roll_mean_60",
            "kpi_delta_60",
            "kpi_slope_60",
        ],
    }

    inference_path = LCMAN_DIR / "lcman_inference_store.csv"
    demo_cases_path = LCMAN_DIR / "lcman_demo_cases.csv"
    metrics_summary_path = LCMAN_DIR / "lcman_metrics_summary.csv"
    vs_baselines_path = LCMAN_DIR / "lcman_vs_baselines.csv"
    modality_path = LCMAN_DIR / "lcman_modality_analysis.csv"
    model_card_path = LCMAN_DIR / "model_card_lcman.md"

    contract["lcman_inference_store"] = {
        "path": inference_path,
        "required": False,
        "found": inference_path.exists(),
        "required_fields": ["run_id", "window_ts", "warning_score", "risk_score"],
        "optional_fields": [
            "pred_state",
            "lcman_gate",
            "kpi_available",
            "kpi_series_available",
            "delta_warn_vs_fused",
            "delta_risk_vs_fused",
        ],
    }
    contract["lcman_demo_cases"] = {
        "path": demo_cases_path,
        "required": False,
        "found": demo_cases_path.exists(),
        "required_fields": ["run_id", "window_ts"],
        "optional_fields": [
            "pred_state",
            "warning_score",
            "risk_score",
            "delta_warn_vs_fused",
            "comment",
            "kpi_drivers",
            "log_excerpt",
        ],
    }
    contract["lcman_vs_baselines"] = {
        "path": vs_baselines_path,
        "required": True,
        "found": vs_baselines_path.exists(),
        "required_fields": ["Система"],
        "optional_fields": [
            "PR-AUC",
            "F1",
            "ROC-AUC",
            "PR-AUC (y_warning, all)",
            "F1 (y_warning, all)",
            "ROC-AUC (y_warning, all)",
            "Комментарий",
        ],
    }
    contract["lcman_metrics_summary"] = {
        "path": metrics_summary_path,
        "required": False,
        "found": metrics_summary_path.exists(),
        "required_fields": ["model", "target", "pr_auc_mean", "f1_mean"],
        "optional_fields": ["slice", "roc_auc_mean", "pr_auc_std", "f1_std", "roc_auc_std"],
    }
    contract["lcman_modality_analysis"] = {
        "path": modality_path,
        "required": False,
        "found": modality_path.exists(),
        "required_fields": ["slice", "model", "rows", "f1"],
        "optional_fields": ["pr_auc", "mean_risk"],
    }
    contract["model_card"] = {
        "path": model_card_path,
        "required": False,
        "found": model_card_path.exists(),
        "required_fields": [],
        "optional_fields": [],
    }
    contract["runtime_metadata"] = {
        "path": RUNTIME_META_JSON_PATH if RUNTIME_META_JSON_PATH.exists() else RUNTIME_META_CSV_PATH,
        "required": False,
        "found": RUNTIME_META_JSON_PATH.exists() or RUNTIME_META_CSV_PATH.exists(),
        "required_fields": ["encoder_name", "embedding_dim", "data_mode", "data_source"],
        "optional_fields": [],
    }

    missing_required = [k for k, v in contract.items() if v.get("required") and not v.get("found")]
    if missing_required:
        missing_txt = ", ".join(missing_required)
        raise DataContractError(f"Missing required artifacts: {missing_txt}")

    if scored_path is None:
        raise DataContractError("Timeline base artifact was not found")

    timeline = pd.read_csv(scored_path, low_memory=False)
    if timeline.empty:
        raise DataContractError("Timeline base artifact is empty")
    timeline = _harmonize_timeline_contract(timeline)
    inf_cols_existing = [col for col in timeline.columns if col.startswith("inf_")]
    if inf_cols_existing:
        timeline = timeline.drop(columns=inf_cols_existing, errors="ignore")

    inference = read_optional_csv(inference_path)
    if not inference.empty and {"run_id", "window_ts"}.issubset(inference.columns):
        inference = inference.copy()
        inference["run_id"] = inference["run_id"].astype(str)
        inference["window_ts"] = pd.to_datetime(inference["window_ts"], utc=True, errors="coerce")
        inference = inference.dropna(subset=["run_id", "window_ts"]).drop_duplicates(["run_id", "window_ts"], keep="last")
        inference = inference.rename(
            columns={
                col: f"inf_{col}"
                for col in inference.columns
                if col not in {"run_id", "window_ts"}
            }
        )
        timeline = timeline.merge(inference, on=["run_id", "window_ts"], how="left")

    warning_series = pd.Series(np.nan, index=timeline.index, dtype=float)
    for col in ["inf_warning_score", "warning_score", "lcman_warning", "y_warning"]:
        if col in timeline.columns:
            warning_series = warning_series.fillna(pd.to_numeric(timeline[col], errors="coerce"))
    timeline["warning_score"] = warning_series.fillna(0.0).clip(0.0, 1.0)

    risk_series = pd.Series(np.nan, index=timeline.index, dtype=float)
    for col in ["inf_risk_score", "risk_score", "lcman_risk", "missing_aware_fusion", "y_risk"]:
        if col in timeline.columns:
            risk_series = risk_series.fillna(pd.to_numeric(timeline[col], errors="coerce"))
    timeline["risk_score"] = risk_series.fillna(0.0).clip(0.0, 1.0)

    delta_warn_inf = pd.Series(np.nan, index=timeline.index, dtype=float)
    for col in ["inf_delta_warn_vs_fused", "delta_warn_vs_fused"]:
        if col in timeline.columns:
            delta_warn_inf = delta_warn_inf.fillna(pd.to_numeric(timeline[col], errors="coerce"))

    delta_risk_inf = pd.Series(np.nan, index=timeline.index, dtype=float)
    for col in ["inf_delta_risk_vs_fused", "delta_risk_vs_fused", "inf_delta_vs_log"]:
        if col in timeline.columns:
            delta_risk_inf = delta_risk_inf.fillna(pd.to_numeric(timeline[col], errors="coerce"))

    fused_warn = pd.Series(np.nan, index=timeline.index, dtype=float)
    for col in ["inf_fused_warn", "fused_warn", "warning_baseline_score"]:
        if col in timeline.columns:
            fused_warn = fused_warn.fillna(pd.to_numeric(timeline[col], errors="coerce"))
    fused_warn = fused_warn.fillna(timeline["warning_score"] - delta_warn_inf)
    fused_warn = fused_warn.fillna(timeline["warning_score"]).clip(0.0, 1.0)
    timeline["fused_warn_score"] = fused_warn

    fused_risk = pd.Series(np.nan, index=timeline.index, dtype=float)
    for col in ["inf_fused_risk", "fused_risk", "baseline_score", "log_only_ref", "log_only"]:
        if col in timeline.columns:
            fused_risk = fused_risk.fillna(pd.to_numeric(timeline[col], errors="coerce"))
    fused_risk = fused_risk.fillna(timeline["risk_score"] - delta_risk_inf)
    fused_risk = fused_risk.fillna(timeline["risk_score"]).clip(0.0, 1.0)
    timeline["fused_risk_score"] = fused_risk

    timeline["delta_warn_vs_fused"] = timeline["warning_score"] - timeline["fused_warn_score"]
    if delta_warn_inf.notna().any():
        timeline["delta_warn_vs_fused"] = delta_warn_inf.fillna(timeline["delta_warn_vs_fused"])

    timeline["delta_risk_vs_fused"] = timeline["risk_score"] - timeline["fused_risk_score"]
    if delta_risk_inf.notna().any():
        timeline["delta_risk_vs_fused"] = delta_risk_inf.fillna(timeline["delta_risk_vs_fused"])

    timeline["baseline_score"] = timeline["fused_risk_score"]
    timeline["delta_vs_log"] = timeline["delta_risk_vs_fused"]

    if "inf_kpi_available" in timeline.columns:
        timeline["kpi_available"] = pd.to_numeric(timeline["inf_kpi_available"], errors="coerce").fillna(timeline["kpi_available"])
    timeline["kpi_available"] = pd.to_numeric(timeline["kpi_available"], errors="coerce").fillna(0).gt(0).astype(int)

    if "inf_kpi_series_available" in timeline.columns:
        timeline["kpi_series_available"] = pd.to_numeric(timeline["inf_kpi_series_available"], errors="coerce").fillna(timeline["kpi_series_available"])
    timeline["kpi_series_available"] = pd.to_numeric(timeline["kpi_series_available"], errors="coerce").fillna(0).astype(int)

    gate_series = pd.Series(np.nan, index=timeline.index, dtype=float)
    for col in ["inf_lcman_gate", "lcman_gate"]:
        if col in timeline.columns:
            gate_series = gate_series.fillna(pd.to_numeric(timeline[col], errors="coerce"))
    timeline["lcman_gate"] = gate_series
    timeline.loc[timeline["kpi_available"].eq(0), "lcman_gate"] = 1.0
    timeline["lcman_gate"] = timeline["lcman_gate"].fillna(0.5).clip(0.0, 1.0)

    pred_series = pd.Series("", index=timeline.index, dtype="object")
    for col in ["inf_pred_state", "pred_state", "state"]:
        if col in timeline.columns:
            vals = timeline[col].where(pd.notna(timeline[col]), "").astype(str).str.strip()
            fill_mask = pred_series.eq("") & vals.ne("")
            pred_series.loc[fill_mask] = vals.loc[fill_mask]
    pred_series = pred_series.map(normalize_state)
    pred_from_risk = timeline["risk_score"].map(risk_to_state)
    timeline["pred_state"] = np.where(pred_series.astype(str).eq(""), pred_from_risk, pred_series)

    if "true_state" in timeline.columns:
        timeline["true_state"] = timeline["true_state"].map(normalize_state)
    else:
        timeline["true_state"] = np.select(
            [timeline["y_risk"].eq(1), timeline["y_warning"].eq(1)],
            ["risk", "warning"],
            default="normal",
        )

    if "inf_has_both_modalities" in timeline.columns:
        inferred_both = pd.to_numeric(timeline["inf_has_both_modalities"], errors="coerce")
        timeline["has_both_modalities"] = inferred_both.fillna(timeline["has_both_modalities"])
    timeline["has_both_modalities"] = pd.to_numeric(timeline["has_both_modalities"], errors="coerce").fillna(0).astype(int)
    timeline["has_both_modalities"] = np.where(
        timeline["has_both_modalities"].gt(0),
        1,
        (timeline["log_available"].eq(1) & timeline["kpi_available"].eq(1)).astype(int),
    )

    timeline["confidence"] = [
        confidence_from_risk(risk=safe_float(r, 0.0), state=str(s))
        for r, s in zip(timeline["risk_score"], timeline["pred_state"])  # noqa: B905
    ]

    missing_expl = timeline["explanation"].astype(str).str.strip().eq("")
    if missing_expl.any():
        timeline.loc[missing_expl, "explanation"] = timeline.loc[missing_expl].apply(_make_default_explanation, axis=1)

    timeline = _ensure_dense_kpi_features(timeline)
    timeline = timeline.sort_values(["run_id", "window_ts"]).drop_duplicates(["run_id", "window_ts"], keep="last").reset_index(drop=True)

    if (not DASHBOARD_TIMELINE_PATH.exists()) or (scored_path != DASHBOARD_TIMELINE_PATH):
        try:
            timeline_store = timeline.drop(columns=[c for c in timeline.columns if c.startswith("inf_")], errors="ignore")
            timeline_store.to_csv(DASHBOARD_TIMELINE_PATH, index=False)
        except Exception:
            pass

    demo_cases = read_optional_csv(demo_cases_path)
    if not demo_cases.empty and {"run_id", "window_ts"}.issubset(demo_cases.columns):
        demo_cases = demo_cases.copy()
        demo_cases["run_id"] = demo_cases["run_id"].astype(str)
        demo_cases["window_ts"] = pd.to_datetime(demo_cases["window_ts"], utc=True, errors="coerce")
        demo_cases = demo_cases.dropna(subset=["run_id", "window_ts"])

    case_rows: list[dict[str, Any]] = []
    used_focuses: set[tuple[str, pd.Timestamp]] = set()

    def add_case(row: pd.Series, kind: str, source: str, note: str = "") -> None:
        key = (str(row["run_id"]), pd.Timestamp(row["window_ts"]))
        if key in used_focuses:
            return
        used_focuses.add(key)
        case_id = f"{kind}::{key[0]}::{pd.Timestamp(key[1]).strftime('%Y%m%d%H%M')}"
        case_rows.append(
            {
                "case_id": case_id,
                "kind": kind,
                "label": CASE_UI_LABELS[kind],
                "run_id": str(row["run_id"]),
                "focus_window_ts": pd.Timestamp(row["window_ts"]),
                "focus_relative_min": safe_float(row.get("relative_min", np.nan), np.nan),
                "focus_risk": safe_float(row.get("risk_score", row.get("missing_aware_fusion", 0.0)), 0.0),
                "source": source,
                "note": note.strip() or _make_case_description(kind, row),
            }
        )

    row_multimodal = _pick_best_row(
        timeline,
        timeline["kpi_available"].eq(1) & timeline["risk_score"].ge(RISK_WARNING_MAX),
        sort_cols=[("delta_risk_vs_fused", False), ("risk_score", False)],
    )
    if row_multimodal is not None:
        add_case(row_multimodal, "multimodal_risk", "auto")

    row_warning = _pick_best_row(
        timeline,
        timeline["kpi_available"].eq(0) & timeline["pred_state"].eq("warning"),
        sort_cols=[("warning_score", False), ("relative_min", True)],
    )
    if row_warning is not None:
        add_case(row_warning, "warning_only", "auto")

    row_missing = _pick_best_row(
        timeline,
        timeline["kpi_available"].eq(0),
        sort_cols=[("delta_risk_vs_fused", True), ("risk_score", False)],
    )
    if row_missing is not None:
        add_case(row_missing, "kpi_missing", "auto")

    row_normal = _pick_best_row(
        timeline,
        timeline["pred_state"].eq("normal"),
        sort_cols=[("risk_score", True), ("relative_min", True)],
    )
    if row_normal is not None:
        add_case(row_normal, "normal", "auto")

    if not demo_cases.empty:
        merged_demo = demo_cases.merge(timeline, on=["run_id", "window_ts"], how="left", suffixes=("_demo", ""))
        merged_demo = merged_demo.dropna(subset=["run_id", "window_ts"]) if not merged_demo.empty else merged_demo
        present = {item["kind"] for item in case_rows}
        for _, row in merged_demo.iterrows():
            kind = _infer_case_kind(row)
            if kind in present:
                continue
            note = str(row.get("comment", "")).strip()
            add_case(row, kind=kind, source="lcman_demo_cases", note=note)
            present.add(kind)

    cases = pd.DataFrame(case_rows)
    if cases.empty:
        first = timeline.iloc[0]
        add_case(first, kind="normal", source="fallback")
        cases = pd.DataFrame(case_rows)

    picked: list[dict[str, Any]] = []
    for kind in CASE_UI_ORDER:
        sub = cases[cases["kind"].eq(kind)]
        if not sub.empty:
            if kind == "normal":
                sub = sub.sort_values(["focus_risk", "focus_relative_min"], ascending=[True, True])
            else:
                sub = sub.sort_values(["focus_risk", "focus_relative_min"], ascending=[False, True])
            picked.append(sub.iloc[0].to_dict())

    if len(picked) < 4:
        leftovers = cases[~cases["case_id"].isin([r["case_id"] for r in picked])]
        for _, row in leftovers.iterrows():
            picked.append(row.to_dict())
            if len(picked) >= 4:
                break

    cases = pd.DataFrame(picked).reset_index(drop=True)

    run_ids_sorted = sorted(timeline["run_id"].unique().tolist())
    run_number_map = {rid: idx + 1 for idx, rid in enumerate(run_ids_sorted)}
    cases["run_number"] = cases["run_id"].map(run_number_map).fillna(0).astype(int)

    metrics_summary = read_optional_csv(metrics_summary_path)
    vs_baselines = read_optional_csv(vs_baselines_path)
    modality = read_optional_csv(modality_path)
    model_card = model_card_path.read_text(encoding="utf-8") if model_card_path.exists() else ""
    runtime_metadata = _read_runtime_metadata(model_card)

    contract_df = _build_contract_rows(contract)

    return {
        "timeline": timeline,
        "cases": cases,
        "metrics_summary": metrics_summary,
        "vs_baselines": vs_baselines,
        "modality": modality,
        "model_card": model_card,
        "runtime_metadata": runtime_metadata,
        "contract_table": contract_df,
    }


def _state_pill_html(state: str) -> str:
    s = normalize_state(state)
    color = STATE_COLORS.get(s, "#334155")
    label = STATE_LABELS.get(s, s)
    return (
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        f"background:{color}22;color:{color};font-weight:600;font-size:13px'>{html.escape(label)}</span>"
    )


def _metric_card(title: str, value: str, subtitle: str = "", tone: str = "default", icon: str = "") -> str:
    tone_map = {
        "default": "#ffffff",
        "accent": "#eff6ff",
        "warning": "#fff7ed",
        "danger": "#fef2f2",
        "success": "#f0fdf4",
    }
    border_map = {
        "default": "#d1d5db",
        "accent": "#93c5fd",
        "warning": "#fdba74",
        "danger": "#fca5a5",
        "success": "#86efac",
    }
    bg = tone_map.get(tone, tone_map["default"])
    border = border_map.get(tone, border_map["default"])
    value_color = {
        "default": "#0f172a",
        "accent": "#1d4ed8",
        "warning": "#c2410c",
        "danger": "#dc2626",
        "success": "#15803d",
    }.get(tone, "#0f172a")
    title_bits = f"<span class='metric-icon'>{html.escape(icon)}</span>" if icon else ""
    return (
        f"<div class='metric-shell metric-shell-{html.escape(tone)}' style='background:{bg};border:1px solid {border}'>"
        f"<div class='metric-title-row'>{title_bits}<span>{html.escape(title)}</span></div>"
        f"<div class='metric-value' style='color:{value_color}'>{html.escape(value)}</div>"
        f"<div class='metric-subtitle'>{html.escape(subtitle)}</div>"
        "</div>"
    )


def _nearest_time_index(df: pd.DataFrame, target_ts: pd.Timestamp, ts_col: str = "window_ts") -> int:
    if df.empty or ts_col not in df.columns:
        return 0
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    target = pd.to_datetime(target_ts, utc=True, errors="coerce")
    if pd.isna(target) or ts.isna().all():
        return 0
    return int((ts - pd.Timestamp(target)).abs().idxmin())


def _replay_scale_html(total_steps: int, current_step: int) -> str:
    dots = []
    for idx in range(max(int(total_steps), 1)):
        cls = "future"
        if idx < current_step:
            cls = "done"
        elif idx == current_step:
            cls = "current"
        dots.append(f"<span class='replay-dot replay-dot-{cls}' title='Шаг {idx + 1}'></span>")

    return (
        "<div class='replay-scale-shell'>"
        f"<div class='replay-dots'>{''.join(dots)}</div>"
        "<div class='replay-scale-labels'>"
        "<span>1</span>"
        f"<span class='replay-scale-current'>{current_step + 1}</span>"
        f"<span>{total_steps}</span>"
        "</div>"
        "</div>"
    )


def _meta_chips_html(items: list[tuple[str, str]]) -> str:
    chips = []
    for label, value in items:
        chips.append(
            "<span class='meta-chip'>"
            f"<span class='meta-chip-label'>{html.escape(label)}:</span> "
            f"<span class='meta-chip-value'>{html.escape(value)}</span>"
            "</span>"
        )
    return f"<div class='meta-chip-row'>{''.join(chips)}</div>"


def _render_log_evidence_table(log_df: pd.DataFrame) -> None:
    rows_html: list[str] = []
    for _, row in log_df.iterrows():
        rank = int(safe_float(row.get("Ранг", 0), 0))
        label = str(row.get("Шаблон (лог-паттерн)", "—"))
        freq = int(safe_float(row.get("Частота (replay-контекст)", 0), 0))
        delta_txt = str(row.get("Δ к baseline (%)", "n/a"))
        contrib = float(np.clip(safe_float(row.get("Эвристический вклад", 0.0), 0.0), 0.0, 1.0))
        direction = "↑" if str(delta_txt).strip().startswith("+") else ("↓" if str(delta_txt).strip().startswith("-") else "•")
        delta_class = "up" if str(delta_txt).strip().startswith("+") else ("down" if str(delta_txt).strip().startswith("-") else "flat")
        rows_html.append(
            "<div class='log-ev-row'>"
            f"<div class='log-rank'>{rank}</div>"
            f"<div class='log-pattern'>{html.escape(label)}</div>"
            f"<div class='log-freq'>{freq:,}</div>"
            f"<div class='log-delta {delta_class}'>{direction} {html.escape(delta_txt)}</div>"
            "<div class='log-contrib-cell'>"
            "<div class='log-contrib-track'>"
            f"<div class='log-contrib-fill' style='width:{contrib * 100:.1f}%'></div>"
            "</div>"
            f"<div class='log-contrib-text'>{contrib * 100:.1f}%</div>"
            "</div>"
            "</div>"
        )

    header = (
        "<div class='log-ev-head'>"
        "<div></div><div>Шаблон</div><div>Частота</div><div>Δ к baseline</div><div>Вклад</div>"
        "</div>"
    )
    st.markdown(f"<div class='log-ev-table'>{header}{''.join(rows_html)}</div>", unsafe_allow_html=True)


def _render_kpi_signal_cards(
    raw_kpi: dict[str, Any],
    replay_ts: pd.Timestamp,
    current_has_kpi: bool,
) -> None:
    builders: list[tuple[str, Any]] = [
        ("cpu", build_cpu_family_chart),
        ("memstat", build_memory_family_chart),
        ("iostat", build_disk_family_chart),
        ("netstat", build_network_family_chart),
    ]

    ts_now = pd.to_datetime(replay_ts, utc=True, errors="coerce")
    x_min = pd.to_datetime(raw_kpi.get("time_min"), utc=True, errors="coerce")
    x_max = pd.to_datetime(raw_kpi.get("time_max"), utc=True, errors="coerce")
    x_domain: tuple[pd.Timestamp, pd.Timestamp] | None = None
    if pd.notna(x_min) and pd.notna(x_max):
        x_left = pd.Timestamp(x_min)
        x_right = pd.Timestamp(x_max)
        if x_left == x_right:
            x_right = x_right + pd.Timedelta(seconds=1)
        x_domain = (x_left, x_right)

    cols = st.columns(2, gap="small")
    for idx, (family, builder) in enumerate(builders):
        fig, status = builder(raw_kpi, current_ts=pd.Timestamp(ts_now) if pd.notna(ts_now) else pd.NaT, x_domain=x_domain)
        fig.update_layout(height=126, margin={"l": 4, "r": 4, "t": 4, "b": 4}, paper_bgcolor="#ffffff", plot_bgcolor="#fbfdff")
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        panel_title = str(RAW_KPI_PANEL_META.get(family, {}).get("title", family.upper()))
        metric_label = str(status.get("primary_label", "")).strip() or panel_title
        value = status.get("current")
        if not bool(status.get("current_in_tolerance", False)):
            value_txt = "Отсутствует"
            delta_txt = "Нет данных в окне"
            signal_txt = "Отсутствует"
            signal_color = "#94a3b8"
        else:
            value_txt = _format_status_value(value)
            delta_pct = status.get("delta_pct")
            delta_txt = (
                f"{float(delta_pct):+.1f}% к baseline"
                if pd.notna(delta_pct)
                else "baseline недоступен"
            )
            signal_txt, signal_color = _signal_level(abs(safe_float(delta_pct, 0.0)) / 40.0)

        with cols[idx % 2]:
            with st.container(border=True):
                st.markdown(
                    (
                        "<div class='kpi-signal-card'>"
                        f"<div class='kpi-signal-title'>{html.escape(metric_label)}</div>"
                        f"<div class='kpi-signal-value'>{html.escape(value_txt)}</div>"
                        f"<div class='kpi-signal-meta'>{html.escape(delta_txt)}</div>"
                        "<div class='kpi-signal-status-row'>"
                        f"<span class='kpi-family-chip'>{html.escape(panel_title)}</span>"
                        f"<span class='kpi-signal-pill' style='color:{signal_color};border-color:{signal_color}33;background:{signal_color}14'>{html.escape(signal_txt)}</span>"
                        "</div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
                chart_key = f"kpi_signal::{family}::{pd.Timestamp(ts_now).value if pd.notna(ts_now) else 'na'}::{idx}"
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=chart_key)

    note = (
        "Панели отражают фактическое raw KPI-покрытие run и ближайшее реальное окно evidence."
        if current_has_kpi
        else "Серые и пустые состояния означают: в текущем 5-мин окне KPI не подтверждают решение; ниже показан только run-контекст."
    )
    st.caption(note)


def build_timeline_figure(run_df: pd.DataFrame, current_idx: int, target_view: str = "both") -> go.Figure:
    run_df = run_df.sort_values("window_ts").reset_index(drop=True)
    cur = run_df.iloc[current_idx]

    fig = go.Figure()

    x_series = run_df["window_ts"]
    y_risk = run_df["risk_score"]
    y_warning = pd.to_numeric(run_df.get("warning_score", y_risk), errors="coerce").fillna(y_risk)

    show_risk = target_view in {"risk", "both"}
    show_warning = target_view in {"warning", "both"}

    # risk bands
    fig.add_hrect(y0=0.0, y1=RISK_NORMAL_MAX, fillcolor="#dcfce7", opacity=0.18, line_width=0, layer="below")
    fig.add_hrect(y0=RISK_NORMAL_MAX, y1=RISK_WARNING_MAX, fillcolor="#ffedd5", opacity=0.14, line_width=0, layer="below")
    fig.add_hrect(y0=RISK_WARNING_MAX, y1=1.05, fillcolor="#fee2e2", opacity=0.12, line_width=0, layer="below")

    # KPI missing zones
    missing = run_df["kpi_available"].eq(0).to_numpy()
    if missing.any():
        start = None
        for i, flag in enumerate(missing):
            if flag and start is None:
                start = i
            if (not flag or i == len(missing) - 1) and start is not None:
                end_idx = i if not flag else i
                x0 = run_df.iloc[start]["window_ts"]
                x1 = run_df.iloc[end_idx]["window_ts"]
                fig.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor="#e2e8f0",
                    opacity=0.16,
                    layer="below",
                    line_width=0,
                )
                start = None

    # state thresholds
    fig.add_trace(
        go.Scatter(
            x=x_series,
            y=[RISK_NORMAL_MAX] * len(x_series),
            mode="lines",
            name="Норма (<0.3)",
            line={"color": "#22c55e", "width": 1.2, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_series,
            y=[(RISK_NORMAL_MAX + RISK_WARNING_MAX) / 2.0] * len(x_series),
            mode="lines",
            name="Предупреждение (0.3–0.7)",
            line={"color": "#f59e0b", "width": 1.0, "dash": "dot"},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_series,
            y=[RISK_WARNING_MAX] * len(x_series),
            mode="lines",
            name="Аномалия (>0.7)",
            line={"color": "#ef4444", "width": 1.2, "dash": "dash"},
            hoverinfo="skip",
        )
    )

    if show_warning:
        fig.add_trace(
            go.Scatter(
                x=x_series,
                y=y_warning,
                mode="lines",
                name="Warning score",
                line={"color": "#7c3aed", "width": 2.1, "dash": "dot" if show_risk else "solid"},
            )
        )

    if show_risk:
        fig.add_trace(
            go.Scatter(
                x=x_series,
                y=y_risk,
                mode="lines",
                name="Оценка риска",
                line={"color": "#2563eb", "width": 2.8},
            )
        )

    real_points = run_df.drop_duplicates("_src_idx") if "_src_idx" in run_df.columns else run_df
    if show_risk and not real_points.empty:
        fig.add_trace(
            go.Scatter(
                x=real_points["window_ts"],
                y=pd.to_numeric(real_points["risk_score"], errors="coerce"),
                mode="markers",
                name="Реальные окна",
                marker={"size": 5.5, "color": "#ffffff", "line": {"color": "#2563eb", "width": 1.3}},
            )
        )

    if show_warning:
        fig.add_trace(
            go.Scatter(
                x=[cur["window_ts"]],
                y=[safe_float(cur.get("warning_score", cur["risk_score"]), 0.0)],
                mode="markers",
                name="Текущий warning",
                marker={"size": 7, "color": "#ffffff", "line": {"color": "#6d28d9", "width": 1.8}},
            )
        )

    if show_risk:
        fig.add_trace(
            go.Scatter(
                x=[cur["window_ts"]],
                y=[cur["risk_score"]],
                mode="markers",
                name="Текущий шаг",
                marker={"size": 9, "color": "#ffffff", "line": {"color": "#1e3a8a", "width": 1.8}},
            )
        )

    # selected window shading
    window_center = pd.Timestamp(cur["window_ts"])
    window_start = window_center - pd.Timedelta(minutes=2, seconds=30)
    window_end = window_center + pd.Timedelta(minutes=2, seconds=30)
    fig.add_vrect(
        x0=window_start,
        x1=window_end,
        fillcolor="#dbeafe",
        opacity=0.22,
        layer="below",
        line_width=0,
    )

    fig.add_vline(x=cur["window_ts"], line_width=1.2, line_color="#1d4ed8")

    # fail marker
    fail_ts = pd.to_datetime(cur.get("fail_start", pd.NaT), utc=True, errors="coerce")
    if pd.notna(fail_ts):
        fig.add_vline(x=fail_ts, line_width=1.3, line_dash="dash", line_color="#7c3aed")
        fig.add_annotation(
            x=fail_ts,
            y=1.03,
            yref="y",
            text="Начало сбоя",
            showarrow=False,
            font={"size": 11, "color": "#7c3aed"},
        )

    fig.add_annotation(
        x=cur["window_ts"],
        y=min(
            1.0,
            float(
                safe_float(
                    cur.get("risk_score", 0.0) if show_risk else cur.get("warning_score", 0.0),
                    0.0,
                )
            )
            + 0.12,
        ),
        text=f"Replay-шаг<br>{pd.Timestamp(cur['window_ts']).strftime('%H:%M')}",
        showarrow=True,
        arrowhead=2,
        bgcolor="#1d4ed8",
        bordercolor="#1d4ed8",
        font={"color": "white", "size": 11},
        ax=0,
        ay=-38,
    )

    # zone labels on the left
    x_left = pd.Timestamp(run_df["window_ts"].min()) + pd.Timedelta(minutes=1)
    fig.add_annotation(
        x=x_left,
        y=0.90,
        text="Аномалия",
        showarrow=False,
        font={"color": "#ffffff", "size": 12},
        bgcolor="#ef4444",
        bordercolor="#ef4444",
        borderpad=4,
    )
    fig.add_annotation(
        x=x_left,
        y=0.57,
        text="Предупреждение",
        showarrow=False,
        font={"color": "#ea580c", "size": 12},
        bgcolor="#fff7ed",
        bordercolor="#fdba74",
        borderpad=4,
    )
    fig.add_annotation(
        x=x_left,
        y=0.16,
        text="Норма",
        showarrow=False,
        font={"color": "#166534", "size": 12},
        bgcolor="#f0fdf4",
        bordercolor="#86efac",
        borderpad=4,
    )

    fig.update_layout(
        height=286,
        margin={"l": 18, "r": 10, "t": 16, "b": 12},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "font": {"size": 10}},
        font={"color": "#334155"},
    )
    fig.update_yaxes(
        title_text="Оценка риска",
        range=[0.0, 1.08],
        gridcolor="#e2e8f0",
        tickfont={"color": "#64748b"},
        title_font={"color": "#64748b"},
    )
    fig.update_xaxes(
        title_text="",
        tickformat="%H:%M",
        gridcolor="#e2e8f0",
        tickfont={"color": "#64748b"},
    )

    return fig


def build_fusion_pie(cur: pd.Series) -> go.Figure:
    log_share = float(np.clip(safe_float(cur.get("lcman_gate", 0.5), 0.5), 0.0, 1.0))
    kpi_share = 0.0 if int(safe_float(cur.get("kpi_available", 0), 0)) == 0 else 1.0 - log_share
    if kpi_share <= 0.0:
        log_share = 1.0

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Логи (semantic)", "KPI (time-series)"],
                values=[log_share, max(kpi_share, 1e-6)],
                hole=0.78,
                marker={"colors": ["#2563eb", "#7c3aed"]},
                textinfo="none",
                sort=False,
            )
        ]
    )
    fig.update_layout(
        height=108,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
        paper_bgcolor="#ffffff",
        font={"color": "#334155"},
        annotations=[
            {
                "text": "Вклад\nfusion",
                "font": {"size": 13, "color": "#334155"},
                "showarrow": False,
            }
        ],
    )
    return fig


def _contains_pattern(text: str, token: str) -> bool:
    return token.lower() in text.lower()


def build_log_evidence(run_df: pd.DataFrame, current_idx: int, top_n: int = 5) -> pd.DataFrame:
    scope = run_df.iloc[: current_idx + 1].copy()
    baseline_scope = run_df[run_df["relative_min"].le(-20)].copy()
    if baseline_scope.empty:
        baseline_scope = run_df.head(max(3, len(run_df) // 4)).copy()

    rows = []
    for token, label in LOG_PATTERN_LIBRARY:
        active_count = 0
        baseline_count = 0

        for _, row in scope.iterrows():
            if _contains_pattern(str(row.get("log_text", "")), token):
                active_count += int(max(1, safe_float(row.get("log_count", 0.0), 0.0)))

        for _, row in baseline_scope.iterrows():
            if _contains_pattern(str(row.get("log_text", "")), token):
                baseline_count += int(max(1, safe_float(row.get("log_count", 0.0), 0.0)))

        if active_count <= 0:
            continue

        baseline_norm = baseline_count / max(len(baseline_scope), 1)
        active_norm = active_count / max(len(scope), 1)
        delta_pct = ((active_norm + 1e-6) / (baseline_norm + 1e-6) - 1.0) * 100.0

        rows.append(
            {
                "Шаблон (лог-паттерн)": label,
                "Частота (replay-контекст)": int(active_count),
                "Δ к baseline (%)": float(delta_pct),
            }
        )

    if not rows:
        rows = [
            {
                "Шаблон (лог-паттерн)": "нет выраженных логовых паттернов",
                "Частота (replay-контекст)": 0,
                "Δ к baseline (%)": 0.0,
            }
        ]

    out = pd.DataFrame(rows)
    out = out.sort_values(["Частота (replay-контекст)", "Δ к baseline (%)"], ascending=[False, False]).head(top_n).reset_index(drop=True)

    # relative contribution approximation
    raw_contrib = np.maximum(out["Частота (replay-контекст)"].astype(float), 0.0) * (1.0 + np.maximum(out["Δ к baseline (%)"], -95.0) / 100.0)
    norm = float(raw_contrib.sum()) if float(raw_contrib.sum()) > 0 else 1.0
    out["Эвристический вклад"] = (raw_contrib / norm).round(3)
    out.insert(0, "Ранг", np.arange(1, len(out) + 1))

    out["Δ к baseline (%)"] = out["Δ к baseline (%)"].map(lambda x: f"{x:+.0f}%")
    return out


def _signal_level(z_abs: float) -> tuple[str, str]:
    if z_abs >= 2.0:
        return "Высокий", "#dc2626"
    if z_abs >= 1.0:
        return "Средний", "#f59e0b"
    return "Низкий", "#16a34a"


def _series_label_from_col(col: str) -> str:
    named = {
        "kpi_last": "KPI aggregate · last",
        "kpi_mean": "KPI aggregate · mean",
        "kpi_roll_mean_60": "KPI roll mean · 60s",
        "kpi_roll_mean_120": "KPI roll mean · 120s",
        "kpi_delta_60": "KPI delta · 60s",
        "kpi_delta_120": "KPI delta · 120s",
        "kpi_slope_60": "KPI slope · 60s",
        "kpi_slope_120": "KPI slope · 120s",
    }
    if col in named:
        return named[col]

    if col.endswith("__family_pool"):
        parts = col.split("__")
        if len(parts) >= 3:
            fam = parts[0]
            stat = parts[1]
            return f"{fam} family · {stat}"

    for stat in ["__mean", "__last", "__delta", "__slope"]:
        if col.endswith(stat):
            key = col[: -len(stat)]
            label = SERIES_LABELS.get(key, key)
            return f"{label} · {stat.replace('__', '')}"

    return col


def build_kpi_evidence(run_df: pd.DataFrame, current_idx: int, top_n: int = 6) -> list[dict[str, Any]]:
    cur = run_df.iloc[current_idx]
    feature_cols = _extract_task_manager_kpi_columns(run_df)

    if not feature_cols:
        return []

    if "relative_min" in run_df.columns:
        baseline_mask = pd.to_numeric(run_df["relative_min"], errors="coerce").le(-20)
    else:
        baseline_mask = pd.Series(False, index=run_df.index)
    if not baseline_mask.any():
        baseline_mask = run_df.index < max(3, len(run_df) // 4)

    evidence: list[dict[str, Any]] = []
    for col in feature_cols:
        series_values = pd.to_numeric(run_df[col], errors="coerce")
        cur_val = pd.to_numeric(pd.Series([cur.get(col)]), errors="coerce").iloc[0]
        valid_points = int(series_values.notna().sum())
        if valid_points <= 0:
            continue

        if pd.isna(cur_val):
            z_abs = 0.0
            delta_pct = np.nan
            baseline_median = np.nan
            trend = 0.0
            available = False
        else:
            baseline_values = series_values[baseline_mask].dropna()
            if baseline_values.empty:
                baseline_values = series_values.dropna()
            baseline_median = float(baseline_values.median()) if not baseline_values.empty else float(cur_val)
            baseline_std = float(baseline_values.std(ddof=0)) if not baseline_values.empty else 0.0
            z_abs = abs(float(cur_val) - baseline_median) / max(baseline_std, 1e-6)
            delta_pct = ((float(cur_val) - baseline_median) / (abs(baseline_median) + 1e-6)) * 100.0

            local = series_values.iloc[max(0, current_idx - 4) : current_idx + 1].dropna().to_numpy(dtype=float)
            if local.size >= 2:
                trend = float(np.polyfit(np.arange(local.size, dtype=float), local, 1)[0])
            else:
                trend = 0.0
            available = True

        signal_name, signal_color = _signal_level(z_abs)

        evidence.append(
            {
                "col": col,
                "label": _series_label_from_col(col),
                "value": float(cur_val) if not pd.isna(cur_val) else np.nan,
                "baseline": float(baseline_median) if not pd.isna(baseline_median) else np.nan,
                "delta_pct": float(delta_pct) if not pd.isna(delta_pct) else np.nan,
                "trend": float(trend),
                "signal": signal_name,
                "signal_color": signal_color,
                "score": float(z_abs),
                "available": available,
                "valid_points": valid_points,
                "x": run_df["window_ts"],
                "y": series_values,
            }
        )

    evidence.sort(key=lambda item: (item["available"], item["score"], item.get("valid_points", 0)), reverse=True)
    return evidence[:top_n]


def build_kpi_sparkline(item: dict[str, Any], current_ts: pd.Timestamp) -> go.Figure:
    y = pd.to_numeric(pd.Series(item["y"]), errors="coerce")
    x = pd.to_datetime(pd.Series(item["x"]), utc=True, errors="coerce")
    valid = x.notna() & y.notna()

    fig = go.Figure()

    if int(valid.sum()) >= 2:
        xv = x[valid]
        yv = y[valid]
        fig.add_trace(
            go.Scatter(
                x=xv,
                y=yv,
                mode="lines",
                line={"color": "#2563eb", "width": 1.9, "shape": "spline", "smoothing": 0.45},
                fill="tozeroy",
                fillcolor="rgba(37, 99, 235, 0.16)",
                connectgaps=True,
                hovertemplate="%{x|%H:%M}<br>%{y:.3g}<extra></extra>",
                showlegend=False,
            )
        )

        y_min = float(yv.min())
        y_max = float(yv.max())
        span = max(y_max - y_min, 1e-6)
        pad = max(0.05 * span, 1e-3)
        fig.update_yaxes(range=[y_min - pad, y_max + pad])

        # current marker as a lightweight "task-manager" cursor hint
        if pd.notna(current_ts):
            nearest_idx = (xv - pd.Timestamp(current_ts)).abs().idxmin()
            fig.add_trace(
                go.Scatter(
                    x=[xv.loc[nearest_idx]],
                    y=[yv.loc[nearest_idx]],
                    mode="markers",
                    marker={"size": 5, "color": "#1d4ed8", "line": {"color": "#ffffff", "width": 1}},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    elif int(valid.sum()) == 1:
        xv = pd.to_datetime(x[valid].iloc[0], utc=True)
        yv = float(y[valid].iloc[0])
        x_min = x.min()
        x_max = x.max()
        if pd.notna(x_min) and pd.notna(x_max) and pd.Timestamp(x_min) != pd.Timestamp(x_max):
            fig.add_trace(
                go.Scatter(
                    x=[x_min, x_max],
                    y=[yv, yv],
                    mode="lines",
                    line={"color": "#93c5fd", "width": 1.3, "dash": "dot"},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[xv],
                y=[yv],
                mode="markers",
                marker={"size": 8, "color": "#2563eb", "line": {"color": "#1d4ed8", "width": 1}},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    else:
        fig.add_annotation(
            text="Нет KPI данных",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 11, "color": "#94a3b8"},
        )

    if pd.notna(current_ts):
        fig.add_vline(x=current_ts, line_width=1.0, line_dash="dot", line_color="#64748b")

    fig.update_layout(
        height=118,
        margin={"l": 4, "r": 4, "t": 4, "b": 4},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fbff",
        font={"color": "#334155"},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#dbe7ff", zeroline=False, showticklabels=False, fixedrange=True)
    fig.update_yaxes(showgrid=True, gridcolor="#dbe7ff", zeroline=False, showticklabels=False, fixedrange=True)
    return fig


def _split_run_id(run_id: str) -> tuple[str, str] | None:
    token = str(run_id or "").strip()
    if "/" not in token:
        return None
    app, scenario = token.split("/", 1)
    app = app.strip()
    scenario = scenario.strip()
    if not app or not scenario:
        return None
    return app, scenario


def _build_raw_kpi_sources(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in RAW_KPI_FAMILIES:
        path = run_dir / f"{family}-intersect.csv"
        rows.append(
            {
                "family": family,
                "path": str(path),
                "found": bool(path.exists()),
            }
        )
    return rows


def _empty_raw_kpi_payload(
    run_id: str,
    run_dir: Path | None = None,
    reason: str = "",
    source_files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    families = {family: pd.DataFrame(columns=["timestamp"]) for family in RAW_KPI_FAMILIES}
    coverage = {
        family: {
            "series_count": 0,
            "points": 0,
            "columns": [],
            "time_min": pd.NaT,
            "time_max": pd.NaT,
        }
        for family in RAW_KPI_FAMILIES
    }

    return {
        "run_id": str(run_id),
        "run_dir": str(run_dir) if run_dir is not None else "",
        "available": False,
        "reason": reason,
        "source_files": source_files or [],
        "long": pd.DataFrame(columns=["timestamp", "family", "series", "feature_series", "value", "label"]),
        "wide": pd.DataFrame(columns=["timestamp"]),
        "families": families,
        "coverage": coverage,
        "time_min": pd.NaT,
        "time_max": pd.NaT,
    }


def _prepare_raw_kpi_family_frame(raw: pd.DataFrame, family: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["timestamp", "family", "series", "feature_series", "value", "label"])

    time_col = next((col for col in RAW_KPI_TIME_COLUMNS if col in raw.columns), None)
    if time_col is None:
        return pd.DataFrame(columns=["timestamp", "family", "series", "feature_series", "value", "label"])

    label_col = next((col for col in ["label", "Label"] if col in raw.columns), None)

    if {"series", "value"}.issubset(raw.columns):
        cols = [time_col, "series", "value"]
        if label_col:
            cols.append(label_col)
        long = raw[cols].copy()
    else:
        prepared = raw.copy()
        numeric_cols: list[str] = []
        for col in prepared.columns:
            if col in {time_col, label_col}:
                continue
            numeric = pd.to_numeric(prepared[col], errors="coerce")
            if numeric.notna().any():
                prepared[col] = numeric
                numeric_cols.append(col)
        if not numeric_cols:
            return pd.DataFrame(columns=["timestamp", "family", "series", "feature_series", "value", "label"])

        id_cols = [time_col] + ([label_col] if label_col else [])
        long = prepared[id_cols + numeric_cols].melt(
            id_vars=id_cols,
            value_vars=numeric_cols,
            var_name="series",
            value_name="value",
        )

    long["timestamp"] = pd.to_datetime(long[time_col], utc=True, errors="coerce")
    long["series"] = long["series"].astype(str).str.strip()
    long.loc[long["series"].eq(""), "series"] = np.nan
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["timestamp", "series", "value"]).copy()
    if long.empty:
        return pd.DataFrame(columns=["timestamp", "family", "series", "feature_series", "value", "label"])

    if label_col and label_col in long.columns:
        long["label"] = pd.to_numeric(long[label_col], errors="coerce").fillna(0).astype(int)
    else:
        long["label"] = 0

    long["family"] = family
    long["feature_series"] = family + "__" + long["series"].astype(str)
    return long[["timestamp", "family", "series", "feature_series", "value", "label"]].sort_values("timestamp").reset_index(drop=True)


def _raw_kpi_cache_fingerprint(run_id: str) -> str:
    parsed = _split_run_id(run_id)
    if parsed is None:
        return "invalid-run-id"

    app, scenario = parsed
    run_dir = KPI_RAW_ROOT / app / scenario
    if not run_dir.exists():
        return f"{run_dir}::missing"

    parts: list[str] = []
    for family in RAW_KPI_FAMILIES:
        path = run_dir / f"{family}-intersect.csv"
        if not path.exists():
            parts.append(f"{family}:missing")
            continue
        try:
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
        except Exception:
            parts.append(f"{path.name}:stat-error")
    return "|".join(parts)


@st.cache_data(show_spinner=False)
def _load_raw_kpi_timeseries_cached(run_id: str, cache_token: str) -> dict[str, Any]:
    del cache_token

    parsed = _split_run_id(run_id)
    if parsed is None:
        return _empty_raw_kpi_payload(
            run_id=run_id,
            reason="run_id не соответствует формату app/scenario.",
        )

    app, scenario = parsed
    run_dir = KPI_RAW_ROOT / app / scenario
    source_files = _build_raw_kpi_sources(run_dir)
    if not run_dir.exists():
        return _empty_raw_kpi_payload(
            run_id=run_id,
            run_dir=run_dir,
            reason=f"Каталог raw KPI не найден: {run_dir}",
            source_files=source_files,
        )

    long_parts: list[pd.DataFrame] = []
    for family in RAW_KPI_FAMILIES:
        csv_path = run_dir / f"{family}-intersect.csv"
        if not csv_path.exists():
            continue
        try:
            raw = pd.read_csv(csv_path)
        except Exception:
            continue

        prepared = _prepare_raw_kpi_family_frame(raw, family)
        if not prepared.empty:
            long_parts.append(prepared)

    if not long_parts:
        return _empty_raw_kpi_payload(
            run_id=run_id,
            run_dir=run_dir,
            reason="Файлы *-intersect.csv есть, но после парсинга не осталось валидных временных рядов.",
            source_files=source_files,
        )

    long_df = pd.concat(long_parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    wide = (
        long_df.pivot_table(
            index="timestamp",
            columns="feature_series",
            values="value",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    if {"netstat__rxkB_s", "netstat__txkB_s"}.issubset(wide.columns):
        wide["netstat__total_kB_s"] = wide["netstat__rxkB_s"] + wide["netstat__txkB_s"]
    if {"iostat__rkB_s", "iostat__wkB_s"}.issubset(wide.columns):
        wide["iostat__total_kB_s"] = wide["iostat__rkB_s"] + wide["iostat__wkB_s"]

    wide = wide.sort_index(axis=1)
    wide_df = wide.reset_index().rename(columns={"index": "timestamp"})

    families: dict[str, pd.DataFrame] = {}
    coverage: dict[str, dict[str, Any]] = {}
    for family in RAW_KPI_FAMILIES:
        family_cols = [col for col in wide.columns if col.startswith(f"{family}__")]
        family_frame = pd.DataFrame(index=wide.index)
        family_frame.index.name = "timestamp"
        for col in family_cols:
            family_frame[col] = wide[col]
        family_frame = family_frame.reset_index().rename(columns={"index": "timestamp"})

        if family_cols:
            family_frame = family_frame.loc[family_frame[family_cols].notna().any(axis=1)].reset_index(drop=True)
        else:
            family_frame = pd.DataFrame(columns=["timestamp"])

        families[family] = family_frame
        points = int(family_frame[family_cols].notna().sum().sum()) if family_cols and not family_frame.empty else 0
        coverage[family] = {
            "series_count": int(len(family_cols)),
            "points": points,
            "columns": family_cols,
            "time_min": pd.Timestamp(family_frame["timestamp"].min()) if not family_frame.empty else pd.NaT,
            "time_max": pd.Timestamp(family_frame["timestamp"].max()) if not family_frame.empty else pd.NaT,
        }

    return {
        "run_id": str(run_id),
        "run_dir": str(run_dir),
        "available": True,
        "reason": "",
        "source_files": source_files,
        "long": long_df,
        "wide": wide_df,
        "families": families,
        "coverage": coverage,
        "time_min": pd.Timestamp(long_df["timestamp"].min()),
        "time_max": pd.Timestamp(long_df["timestamp"].max()),
    }


def load_raw_kpi_timeseries(run_id: str) -> dict[str, Any]:
    cache_token = _raw_kpi_cache_fingerprint(run_id)
    return _load_raw_kpi_timeseries_cached(str(run_id), cache_token)


def _series_label_for_family_panel(col: str) -> str:
    if col in SERIES_LABELS:
        return SERIES_LABELS[col]
    if "__" in col:
        return col.replace("__", " · ")
    return col


def _pick_family_primary_column(family_df: pd.DataFrame, family: str) -> str | None:
    if family_df.empty:
        return None
    metric_cols = [col for col in family_df.columns if col != "timestamp"]
    if not metric_cols:
        return None

    priority = RAW_KPI_PANEL_META.get(family, {}).get("priority", [])
    for col in priority:
        if col in metric_cols and pd.to_numeric(family_df[col], errors="coerce").notna().any():
            return col

    for col in sorted(metric_cols):
        if pd.to_numeric(family_df[col], errors="coerce").notna().any():
            return col
    return None


def _nearest_series_value(
    timestamps: pd.Series,
    values: pd.Series,
    current_ts: pd.Timestamp,
    tolerance: pd.Timedelta,
) -> tuple[float | None, pd.Timestamp, bool]:
    valid_mask = timestamps.notna() & values.notna()
    if int(valid_mask.sum()) == 0:
        return None, pd.NaT, False

    xv = pd.to_datetime(timestamps[valid_mask], utc=True, errors="coerce")
    yv = pd.to_numeric(values[valid_mask], errors="coerce")
    valid = xv.notna() & yv.notna()
    if int(valid.sum()) == 0:
        return None, pd.NaT, False

    xv = xv[valid]
    yv = yv[valid]
    nearest_idx = (xv - pd.Timestamp(current_ts)).abs().idxmin()
    nearest_ts = pd.Timestamp(xv.loc[nearest_idx])
    delta = abs(nearest_ts - pd.Timestamp(current_ts))
    if delta > tolerance:
        return None, nearest_ts, False
    return float(yv.loc[nearest_idx]), nearest_ts, True


def _compute_family_status(family_df: pd.DataFrame, family: str, primary_col: str | None, current_ts: pd.Timestamp) -> dict[str, Any]:
    out: dict[str, Any] = {
        "family": family,
        "has_series": False,
        "primary_col": primary_col,
        "primary_label": "",
        "baseline": np.nan,
        "current": np.nan,
        "current_ts": pd.NaT,
        "current_in_tolerance": False,
        "delta": np.nan,
        "delta_pct": np.nan,
    }

    if family_df.empty or primary_col is None or primary_col not in family_df.columns:
        return out

    work = family_df[["timestamp", primary_col]].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work[primary_col] = pd.to_numeric(work[primary_col], errors="coerce")
    work = work.dropna(subset=["timestamp", primary_col]).sort_values("timestamp").reset_index(drop=True)
    if work.empty:
        return out

    out["has_series"] = True
    out["primary_label"] = _series_label_for_family_panel(primary_col)

    ts_now = pd.to_datetime(current_ts, utc=True, errors="coerce")
    if pd.notna(ts_now):
        cur_val, cur_point_ts, within = _nearest_series_value(
            work["timestamp"],
            work[primary_col],
            pd.Timestamp(ts_now),
            tolerance=RAW_KPI_CURRENT_TOLERANCE,
        )
    else:
        cur_val = float(work.iloc[-1][primary_col])
        cur_point_ts = pd.Timestamp(work.iloc[-1]["timestamp"])
        within = True

    out["current"] = np.nan if cur_val is None else float(cur_val)
    out["current_ts"] = cur_point_ts
    out["current_in_tolerance"] = bool(within)

    before = work
    if pd.notna(ts_now):
        before = work[work["timestamp"].le(pd.Timestamp(ts_now))]
        if before.empty:
            before = work

    if len(before) >= 8:
        baseline_scope = before.head(max(6, int(np.ceil(len(before) * 0.35))))
    elif len(before) >= 3:
        baseline_scope = before
    else:
        baseline_scope = work.head(max(3, int(np.ceil(len(work) * 0.35))))
        if baseline_scope.empty:
            baseline_scope = work

    baseline = pd.to_numeric(baseline_scope[primary_col], errors="coerce").median()
    out["baseline"] = float(baseline) if pd.notna(baseline) else np.nan

    if pd.notna(out["current"]) and pd.notna(out["baseline"]):
        delta = float(out["current"] - out["baseline"])
        out["delta"] = delta
        out["delta_pct"] = (delta / max(abs(float(out["baseline"])), 1e-6)) * 100.0

    return out


def _format_status_value(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    val = float(value)
    abs_val = abs(val)
    if abs_val >= 1000:
        return f"{val:,.0f}".replace(",", " ")
    if abs_val >= 100:
        return f"{val:.1f}"
    if abs_val >= 10:
        return f"{val:.2f}"
    return f"{val:.3g}"


def _status_line_for_family(status: dict[str, Any]) -> str:
    if not bool(status.get("has_series", False)):
        return "current: n/a | baseline: n/a | delta: n/a"

    current_txt = _format_status_value(status.get("current"))
    baseline_txt = _format_status_value(status.get("baseline"))

    delta = status.get("delta")
    delta_pct = status.get("delta_pct")
    if pd.isna(delta) or pd.isna(delta_pct):
        delta_txt = "n/a"
    else:
        delta_txt = f"{float(delta):+.3g} ({float(delta_pct):+.1f}%)"

    suffix = ""
    if not bool(status.get("current_in_tolerance", False)):
        suffix = " · replay-курсор вне raw-покрытия"
    return f"current: {current_txt} | baseline: {baseline_txt} | delta: {delta_txt}{suffix}"


def _hex_to_rgba(color: str, alpha: float) -> str:
    token = str(color).strip().lstrip("#")
    if len(token) != 6:
        return f"rgba(37,99,235,{alpha:.3f})"
    try:
        r = int(token[0:2], 16)
        g = int(token[2:4], 16)
        b = int(token[4:6], 16)
    except Exception:
        return f"rgba(37,99,235,{alpha:.3f})"
    return f"rgba({r},{g},{b},{alpha:.3f})"


def _resolve_cursor_for_domain(
    current_ts: pd.Timestamp,
    x_domain: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> tuple[pd.Timestamp, str]:
    ts_now = pd.to_datetime(current_ts, utc=True, errors="coerce")
    if x_domain is None or pd.isna(ts_now):
        return pd.NaT, "unknown"

    left, right = x_domain
    if pd.isna(left) or pd.isna(right):
        return pd.NaT, "unknown"

    left = pd.Timestamp(left)
    right = pd.Timestamp(right)
    ts_now = pd.Timestamp(ts_now)
    if ts_now < left:
        return left, "before"
    if ts_now > right:
        return right, "after"
    return ts_now, "inside"


def _build_family_task_manager_chart(
    family: str,
    family_df: pd.DataFrame,
    current_ts: pd.Timestamp,
    x_domain: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> tuple[go.Figure, dict[str, Any]]:
    meta = RAW_KPI_PANEL_META.get(family, {})
    color = str(meta.get("color", "#2563eb"))

    fig = go.Figure()
    status: dict[str, Any] = {
        "family": family,
        "has_series": False,
        "primary_col": None,
        "primary_label": "",
        "baseline": np.nan,
        "current": np.nan,
        "current_ts": pd.NaT,
        "current_in_tolerance": False,
        "delta": np.nan,
        "delta_pct": np.nan,
        "cursor_state": "unknown",
    }

    if not family_df.empty and "timestamp" in family_df.columns:
        work = family_df.copy()
        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
        work = work.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        primary_col = _pick_family_primary_column(work, family)
        status = _compute_family_status(work, family, primary_col, current_ts)
        status["cursor_state"] = "unknown"

        if bool(status.get("has_series", False)) and primary_col is not None:
            x = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
            y = pd.to_numeric(work[primary_col], errors="coerce")
            valid = x.notna() & y.notna()

            if int(valid.sum()) >= 2:
                xv = x[valid]
                yv = y[valid]
                fig.add_trace(
                    go.Scatter(
                        x=xv,
                        y=yv,
                        mode="lines",
                        line={"color": color, "width": 2.4},
                        fill="tozeroy",
                        fillcolor=_hex_to_rgba(color, 0.14),
                        connectgaps=False,
                        hovertemplate="%{x|%H:%M:%S}<br>%{y:.3g}<extra></extra>",
                        showlegend=False,
                    )
                )

                y_min = float(yv.min())
                y_max = float(yv.max())
                span = max(y_max - y_min, 1e-6)
                pad = max(span * 0.08, 1e-3)
                fig.update_yaxes(range=[y_min - pad, y_max + pad])
            elif int(valid.sum()) == 1:
                xv = pd.Timestamp(x[valid].iloc[0])
                yv = float(y[valid].iloc[0])
                fig.add_trace(
                    go.Scatter(
                        x=[xv],
                        y=[yv],
                        mode="markers",
                        marker={"size": 8, "color": color, "line": {"color": "#ffffff", "width": 1}},
                        hovertemplate="%{x|%H:%M:%S}<br>%{y:.3g}<extra></extra>",
                        showlegend=False,
                    )
                )
            else:
                fig.add_annotation(
                    text="Нет KPI данных",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font={"size": 11, "color": "#94a3b8"},
                )

            if pd.notna(status.get("current")) and pd.notna(status.get("current_ts")):
                fig.add_trace(
                    go.Scatter(
                        x=[status["current_ts"]],
                        y=[float(status["current"])],
                        mode="markers",
                        marker={"size": 7, "color": "#0f172a", "line": {"color": "#ffffff", "width": 1}},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    if not fig.data and not fig.layout.annotations:
        fig.add_annotation(
            text="Нет KPI данных",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 11, "color": "#94a3b8"},
        )

    cursor_ts, cursor_state = _resolve_cursor_for_domain(current_ts, x_domain)
    status["cursor_state"] = cursor_state
    if pd.notna(cursor_ts):
        fig.add_vline(x=cursor_ts, line_width=1.15, line_dash="dot", line_color="#0f172a")

    if x_domain is not None:
        left, right = x_domain
        if pd.notna(left) and pd.notna(right):
            fig.update_xaxes(range=[left, right])

    if cursor_state in {"before", "after"}:
        fig.add_annotation(
            text="Курсор replay вне диапазона raw KPI",
            x=0.99,
            y=0.02,
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            font={"size": 10, "color": "#64748b"},
            bgcolor="#ffffff",
            bordercolor="#e2e8f0",
            borderpad=3,
        )

    fig.update_layout(
        height=220,
        margin={"l": 6, "r": 6, "t": 8, "b": 8},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fbff",
        font={"color": "#334155"},
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#dbe7ff",
        zeroline=False,
        fixedrange=True,
        tickformat="%H:%M",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#dbe7ff",
        zeroline=False,
        fixedrange=True,
    )
    return fig, status


def build_cpu_family_chart(
    raw_kpi: dict[str, Any],
    current_ts: pd.Timestamp,
    x_domain: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> tuple[go.Figure, dict[str, Any]]:
    families = raw_kpi.get("families", {}) if isinstance(raw_kpi, dict) else {}
    family_df = families.get("cpu", pd.DataFrame(columns=["timestamp"]))
    return _build_family_task_manager_chart("cpu", family_df, current_ts=current_ts, x_domain=x_domain)


def build_memory_family_chart(
    raw_kpi: dict[str, Any],
    current_ts: pd.Timestamp,
    x_domain: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> tuple[go.Figure, dict[str, Any]]:
    families = raw_kpi.get("families", {}) if isinstance(raw_kpi, dict) else {}
    family_df = families.get("memstat", pd.DataFrame(columns=["timestamp"]))
    return _build_family_task_manager_chart("memstat", family_df, current_ts=current_ts, x_domain=x_domain)


def build_disk_family_chart(
    raw_kpi: dict[str, Any],
    current_ts: pd.Timestamp,
    x_domain: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> tuple[go.Figure, dict[str, Any]]:
    families = raw_kpi.get("families", {}) if isinstance(raw_kpi, dict) else {}
    family_df = families.get("iostat", pd.DataFrame(columns=["timestamp"]))
    return _build_family_task_manager_chart("iostat", family_df, current_ts=current_ts, x_domain=x_domain)


def build_network_family_chart(
    raw_kpi: dict[str, Any],
    current_ts: pd.Timestamp,
    x_domain: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> tuple[go.Figure, dict[str, Any]]:
    families = raw_kpi.get("families", {}) if isinstance(raw_kpi, dict) else {}
    family_df = families.get("netstat", pd.DataFrame(columns=["timestamp"]))
    return _build_family_task_manager_chart("netstat", family_df, current_ts=current_ts, x_domain=x_domain)


def _render_raw_kpi_missing(raw_kpi: dict[str, Any], run_id: str) -> None:
    st.warning("Сырые KPI-таймсерии недоступны. Dashboard остаётся в режиме log-driven fallback без имитации сигнала.")
    reason = str(raw_kpi.get("reason", "")).strip()
    if reason:
        st.caption(reason)
    st.caption(f"run_id: `{run_id}`")

    rows = []
    for item in raw_kpi.get("source_files", []):
        rows.append(
            {
                "Семейство": str(item.get("family", "")),
                "Файл": str(item.get("path", "")),
                "Статус": "найден" if bool(item.get("found", False)) else "нет",
            }
        )
    if rows:
        with st.expander("Проверка raw KPI источников", expanded=False):
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_legacy_kpi_evidence(run_df: pd.DataFrame, current_idx: int, current_ts: pd.Timestamp) -> None:
    evidence = build_kpi_evidence(run_df, current_idx, top_n=6)
    if not evidence:
        st.info("Нет доступных KPI-признаков в 5-мин timeline для legacy-view.")
        return

    sparse_counts = [int(item.get("valid_points", 0)) for item in evidence]
    if sparse_counts and max(sparse_counts) <= 1:
        st.caption("Legacy-view разрежен: по большинству рядов доступно не более одной точки в 5-мин timeline.")

    cols = st.columns(2, gap="small")
    for idx, item in enumerate(evidence):
        with cols[idx % 2]:
            with st.container(border=True):
                st.markdown(f"**{item['label']}**")
                if bool(item.get("available", False)):
                    value_txt = f"{float(item['value']):.3g}"
                    delta_txt = f"{float(item['delta_pct']):+.1f}%" if pd.notna(item.get("delta_pct")) else "n/a"
                else:
                    value_txt = "n/a"
                    delta_txt = "n/a"

                st.caption(f"Текущее: {value_txt} | Δ baseline: {delta_txt}")
                fig = build_kpi_sparkline(item, current_ts=current_ts)
                chart_key = f"legacy_kpi::{item['col']}::{current_idx}::{idx}"
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=chart_key)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --primary-color: #2563eb;
        }
        html, body, [data-testid="stAppViewContainer"], .stApp {
            color-scheme: light;
        }
        .stApp {
            background-color: #f8fafc;
            color: #0f172a;
        }
        header[data-testid="stHeader"] {
            display: none;
        }
        div[data-testid="stToolbar"] {
            display: none;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding-top: 0.22rem;
            padding-bottom: 0.45rem;
            max-width: 1840px;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 16px !important;
            border-color: #e5edf7 !important;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05), 0 10px 28px rgba(15, 23, 42, 0.035) !important;
            background: #ffffff;
        }
        .app-title {
            display:flex;
            align-items:center;
            gap:12px;
            margin-bottom:2px;
            margin-top:2px;
        }
        .shield {
            width:44px;
            height:44px;
            border-radius:12px;
            background:linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%);
            display:flex;
            align-items:center;
            justify-content:center;
            color:white;
            font-size:22px;
            font-weight:800;
            box-shadow: 0 6px 14px rgba(37,99,235,0.18);
        }
        .subtitle {
            color:#64748b;
            font-size:11px;
            margin-top:-2px;
            margin-bottom:2px;
        }
        .panel {
            border:1px solid #e2e8f0;
            border-radius:12px;
            padding:10px 12px;
            background:#ffffff;
        }
        .panel-title {
            font-size:19px;
            font-weight:700;
            color:#0f172a;
            margin-bottom:6px;
        }
        .tiny-muted {
            color:#64748b;
            font-size:12px;
        }
        .metric-shell {
            padding: 7px 10px;
            border-radius: 14px;
            min-height: 60px;
            display:flex;
            flex-direction:column;
            justify-content:center;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 8px 22px rgba(15, 23, 42, 0.03);
        }
        .metric-title-row {
            display:flex;
            align-items:center;
            gap:6px;
            font-size:11px;
            color:#475569;
            margin-bottom:4px;
            line-height:1.1;
        }
        .metric-icon {
            font-size:12px;
            color:#64748b;
        }
        .metric-value {
            font-size:15px;
            line-height:1.1;
            font-weight:700;
            letter-spacing:-0.01em;
        }
        .metric-subtitle {
            font-size:11px;
            color:#64748b;
            margin-top:4px;
        }

        .stButton > button {
            border-radius: 10px;
            min-height: 32px;
            font-weight: 600;
            font-size: 12px;
            border: 1px solid #d7e1ee;
            background: #fbfdff;
            color: #1f2937;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            box-shadow: 0 1px 1px rgba(15,23,42,0.03);
        }
        .stButton > button[data-testid="baseButton-primary"] {
            background: #2563eb !important;
            border-color: #2563eb !important;
            color: white !important;
        }
        .stButton > button[kind="primary"] {
            background: #2563eb !important;
            border-color: #2563eb !important;
            color: white !important;
        }
        .stButton > button:hover {
            border-color: #93c5fd;
            color: #1d4ed8;
            background: #f8fbff;
        }

        div[data-baseweb="slider"] {
            padding-top: 0.02rem;
        }
        div[data-baseweb="slider"] div[role="slider"] {
            background: #2563eb !important;
            border-color: #2563eb !important;
            box-shadow: 0 0 0 4px rgba(37,99,235,0.12);
        }
        div[data-baseweb="slider"] div[role="slider"] + div,
        div[data-baseweb="slider"] div[role="slider"] ~ div {
            border-color: transparent !important;
            background: transparent !important;
            box-shadow: none !important;
        }
        .replay-scale-shell {
            margin: 0 0 0.25rem 0;
        }
        .replay-dots {
            display:grid;
            grid-template-columns: repeat(auto-fit, minmax(0, 1fr));
            gap:5px;
            align-items:center;
        }
        .replay-dot {
            display:block;
            height:8px;
            border-radius:999px;
            background:#cbd5e1;
        }
        .replay-dot-done {
            background:#60a5fa;
        }
        .replay-dot-current {
            background:#2563eb;
            height:12px;
            box-shadow: 0 0 0 3px rgba(37,99,235,0.14);
        }
        .replay-scale-labels {
            display:flex;
            justify-content:space-between;
            align-items:center;
            margin-top:5px;
            font-size:11px;
            color:#64748b;
        }
        .replay-scale-current {
            color:#2563eb;
            font-weight:700;
        }
        .meta-chip-row {
            display:flex;
            gap:8px;
            flex-wrap:wrap;
            margin-top:2px;
        }
        .meta-chip {
            display:inline-flex;
            align-items:center;
            gap:4px;
            border:1px solid #e2e8f0;
            border-radius:999px;
            background:#f8fafc;
            padding:4px 9px;
            font-size:12px;
            color:#475569;
        }
        .meta-chip-label {
            color:#64748b;
        }
        .meta-chip-value {
            color:#0f172a;
            font-weight:600;
        }

        h3 {
            margin-top: 0.15rem;
            margin-bottom: 0.24rem;
            color: #1f2937;
            font-size: 0.96rem;
        }
        h4 {
            margin-top: 0.18rem;
            margin-bottom: 0.2rem;
            font-size: 0.94rem;
            color: #1f2937;
        }

        h1, h2, h5, h6, p, li, span, label, small {
            color: #0f172a;
        }

        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        [data-testid="stCaptionContainer"] span {
            color: #64748b !important;
        }

        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span {
            color: #0f172a;
        }

        [data-testid="stDataFrame"] * {
            color: #0f172a;
        }

        .summary-list {
            border: 1px solid #e8eef6;
            border-radius: 14px;
            background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
            overflow: hidden;
        }
        .summary-hero {
            padding: 10px 11px 8px 11px;
            border-bottom: 1px solid #edf2f7;
            background: linear-gradient(180deg, rgba(37,99,235,0.06) 0%, rgba(37,99,235,0.01) 100%);
        }
        .summary-hero-label {
            font-size: 12px;
            color:#64748b;
            margin-bottom:2px;
        }
        .summary-hero-score {
            font-size: 24px;
            line-height:1.0;
            font-weight:800;
            color:#dc2626;
            letter-spacing:-0.03em;
        }
        .summary-row {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 10px;
            padding: 6px 10px;
            border-bottom: 1px solid #eef2f7;
            font-size: 13px;
        }
        .summary-row:last-child {
            border-bottom: none;
        }
        .summary-k {
            color: #334155;
        }
        .summary-v {
            color: #0f172a;
            font-weight: 600;
        }
        .fusion-legend {
            display:flex;
            flex-direction:column;
            gap:10px;
            margin-top:10px;
        }
        .fusion-row {
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:8px;
            font-size:13px;
        }
        .fusion-left {
            display:flex;
            align-items:center;
            gap:8px;
            color:#334155;
        }
        .fusion-dot {
            width:8px;
            height:8px;
            border-radius:999px;
            display:inline-block;
        }
        .fusion-value {
            font-weight:700;
            color:#0f172a;
        }
        .explanation-card {
            border: 1px solid #e8eef6;
            border-radius: 14px;
            background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
            padding: 10px 11px;
            font-size: 13px;
            line-height: 1.5;
            color:#334155;
        }
        .explanation-note {
            margin-top:8px;
            font-size:11px;
            color:#64748b;
        }

        .baseline-table {
            border: 1px solid #e8eef6;
            border-radius: 14px;
            overflow: hidden;
            background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
        }
        .baseline-head,
        .baseline-row {
            display: grid;
            grid-template-columns: 1.35fr .72fr .92fr .88fr;
            gap: 8px;
            padding: 8px 10px;
            font-size: 13px;
            border-bottom: 1px solid #eef2f7;
        }
        .baseline-head {
            font-weight: 700;
            color: #475569;
            background: #f8fafc;
        }
        .baseline-row:last-child {
            border-bottom: none;
        }
        .baseline-up {
            color:#dc2626;
            font-weight:700;
        }
        .baseline-down {
            color:#0f766e;
            font-weight:700;
        }
        .log-ev-table {
            border:1px solid #e8eef6;
            border-radius:14px;
            overflow:hidden;
            background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
        }
        .log-ev-head,
        .log-ev-row {
            display:grid;
            grid-template-columns: 34px 1.7fr .65fr .9fr 1fr;
            gap:10px;
            align-items:center;
            padding:8px 10px;
            border-bottom:1px solid #edf2f7;
            font-size:13px;
        }
        .log-ev-head {
            background:#f8fafc;
            font-weight:700;
            color:#475569;
        }
        .log-ev-row:last-child {
            border-bottom:none;
        }
        .log-rank {
            width:24px;
            height:24px;
            border-radius:999px;
            background:#2563eb;
            color:#ffffff;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:12px;
            font-weight:700;
        }
        .log-pattern {
            color:#0f172a;
            font-weight:600;
        }
        .log-freq {
            color:#334155;
            font-variant-numeric: tabular-nums;
        }
        .log-delta.up {
            color:#dc2626;
            font-weight:700;
        }
        .log-delta.down {
            color:#0f766e;
            font-weight:700;
        }
        .log-delta.flat {
            color:#64748b;
            font-weight:600;
        }
        .log-contrib-cell {
            display:flex;
            align-items:center;
            gap:8px;
        }
        .log-contrib-track {
            flex:1;
            height:8px;
            border-radius:999px;
            background:#e2e8f0;
            overflow:hidden;
        }
        .log-contrib-fill {
            height:100%;
            border-radius:999px;
            background:linear-gradient(90deg, #60a5fa 0%, #2563eb 100%);
        }
        .log-contrib-text {
            min-width:46px;
            text-align:right;
            font-size:12px;
            color:#334155;
            font-variant-numeric: tabular-nums;
        }
        .kpi-signal-card {
            display:flex;
            flex-direction:column;
            gap:4px;
        }
        .kpi-signal-title {
            font-size:13px;
            color:#0f172a;
            font-weight:700;
        }
        .kpi-signal-value {
            font-size:20px;
            line-height:1.0;
            font-weight:800;
            color:#334155;
        }
        .kpi-signal-meta {
            font-size:12px;
            color:#64748b;
        }
        .kpi-signal-status-row {
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:8px;
            margin-top:2px;
        }
        .kpi-family-chip {
            display:inline-flex;
            align-items:center;
            border:1px solid #dbe5f1;
            border-radius:999px;
            background:#f8fafc;
            padding:3px 8px;
            font-size:11px;
            color:#475569;
        }
        .kpi-signal-pill {
            display:inline-flex;
            align-items:center;
            border:1px solid;
            border-radius:999px;
            padding:3px 8px;
            font-size:11px;
            font-weight:700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_header(case_row: pd.Series, current_row: pd.Series, step_idx: int, total_steps: int) -> None:
    state = normalize_state(current_row["pred_state"])
    risk = safe_float(current_row["risk_score"], 0.0)

    left, c1, c2, c3, c4, c5, c6 = st.columns([2.75, 1.0, 1.8, 1.18, 1.0, 1.4, 1.0], gap="small")

    with left:
        st.markdown(
            """
            <div class="app-title">
              <div class="shield">⛨</div>
              <div>
                <div style="font-size:28px;line-height:1.06;font-weight:700;color:#0f172a">LCMAN Dashboard</div>
                <div class="subtitle">Легковесная мультимодальная оценка риска для программных систем</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c1:
        st.markdown(_metric_card("Режим", "Replay", "демо-режим", tone="accent", icon="↻"), unsafe_allow_html=True)

    with c2:
        run_title = _format_run_title(str(case_row["run_id"]))
        run_title_card = run_title if len(run_title) <= 28 else (run_title[:27] + "…")
        st.markdown(
            _metric_card(
                "Сценарий",
                run_title_card,
                f"Run #{int(case_row['run_number']):03d}",
                tone="default",
                icon="▤",
            ),
            unsafe_allow_html=True,
        )

    with c3:
        window_ts = pd.Timestamp(current_row["window_ts"])
        wnd_text = f"{window_ts.strftime('%H:%M')}–{(window_ts + pd.Timedelta(minutes=5)).strftime('%H:%M')}"
        st.markdown(_metric_card("Окно", wnd_text, "ближайшее evidence-окно", tone="default", icon="◷"), unsafe_allow_html=True)

    with c4:
        st.markdown(
            _metric_card("Шаг", f"{step_idx + 1} / {total_steps}", "36 replay-шагов", tone="default", icon="☰"),
            unsafe_allow_html=True,
        )

    with c5:
        tone = "success" if state == "normal" else ("warning" if state == "warning" else "danger")
        st.markdown(
            _metric_card("Текущее состояние", STATE_LABELS[state], "по текущему окну", tone=tone, icon="⚑"),
            unsafe_allow_html=True,
        )

    with c6:
        tone = "danger" if risk >= RISK_WARNING_MAX else ("warning" if risk >= RISK_NORMAL_MAX else "success")
        st.markdown(_metric_card("Оценка риска", f"{risk:.2f}", "текущий replay-шаг", tone=tone, icon="◔"), unsafe_allow_html=True)


def render_case_and_controls(cases: pd.DataFrame, timeline: pd.DataFrame) -> tuple[str, int, pd.DataFrame]:
    state = st.session_state
    changed = False

    def default_case_id() -> str:
        warning = cases[cases["kind"].eq("warning_only")]
        if not warning.empty:
            return str(warning.iloc[0]["case_id"])
        return str(cases.iloc[0]["case_id"])

    if "selected_case_id" not in state:
        state.selected_case_id = default_case_id()
    if "replay_step" not in state:
        state.replay_step = 0
    if "autoplay" not in state:
        state.autoplay = False

    selected_case = cases[cases["case_id"].eq(state.selected_case_id)]
    if selected_case.empty:
        state.selected_case_id = default_case_id()
        selected_case = cases[cases["case_id"].eq(state.selected_case_id)]

    selected_case = selected_case.iloc[0]
    run_df_raw = timeline[timeline["run_id"].eq(str(selected_case["run_id"]))].sort_values("window_ts").reset_index(drop=True)
    run_df = build_replay_frame(run_df_raw, steps=36)
    if run_df.empty:
        run_df = build_replay_frame(timeline.sort_values(["run_id", "window_ts"]).reset_index(drop=True), steps=36)

    top = st.columns([3.2, 2.35, 2.45], gap="small")

    with top[0]:
        st.markdown("#### Демо-сценарии ⓘ")
        kind_to_case = {row["kind"]: row for _, row in cases.iterrows()}
        btn_cols = st.columns([1.32, 1.02, 1.0, 0.86], gap="small")
        for idx, kind in enumerate(CASE_UI_ORDER):
            if kind not in kind_to_case:
                with btn_cols[idx]:
                    st.button(CASE_UI_LABELS[kind], disabled=True, use_container_width=True, key=f"case_disabled_{kind}")
                continue

            row = kind_to_case[kind]
            is_active = str(row["case_id"]) == state.selected_case_id
            with btn_cols[idx]:
                if st.button(
                    CASE_UI_LABELS[kind],
                    key=f"case_btn_{kind}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    state.selected_case_id = str(row["case_id"])
                    state.autoplay = False
                    changed = True

    # refresh selected case after button press
    selected_case = cases[cases["case_id"].eq(state.selected_case_id)].iloc[0]
    run_df_raw = timeline[timeline["run_id"].eq(str(selected_case["run_id"]))].sort_values("window_ts").reset_index(drop=True)
    run_df = build_replay_frame(run_df_raw, steps=36)
    if run_df.empty:
        run_df = build_replay_frame(timeline.sort_values(["run_id", "window_ts"]).reset_index(drop=True), steps=36)

    focus_ts = pd.Timestamp(selected_case["focus_window_ts"])
    focus_idx = _nearest_time_index(run_df, focus_ts)

    if state.get("_last_case") != state.selected_case_id:
        state.replay_step = focus_idx
        state._last_case = state.selected_case_id

    total_steps = len(run_df)
    state.replay_step = int(np.clip(state.replay_step, 0, max(total_steps - 1, 0)))
    real_windows = int(run_df_raw["window_ts"].nunique()) if not run_df_raw.empty else 0

    with top[1]:
        st.markdown("#### Управление replay")
        c_prev, c_play, c_pause, c_next = st.columns(4, gap="small")
        with c_prev:
            if st.button("◀ Назад", use_container_width=True, key="replay_prev", disabled=state.replay_step <= 0):
                state.replay_step = max(0, state.replay_step - 1)
                state.autoplay = False
                changed = True
        with c_play:
            play_label = "↻ Replay" if state.replay_step >= total_steps - 1 else "▶ Play"
            if st.button(play_label, use_container_width=True, key="replay_play"):
                if state.replay_step >= total_steps - 1:
                    state.replay_step = 0
                state.autoplay = True
                changed = True
        with c_pause:
            if st.button("⏸ Pause", use_container_width=True, key="replay_pause", type="primary" if state.autoplay else "secondary"):
                state.autoplay = False
                changed = True
        with c_next:
            if st.button("Далее ▶", use_container_width=True, key="replay_next", disabled=state.replay_step >= total_steps - 1):
                state.replay_step = min(total_steps - 1, state.replay_step + 1)
                state.autoplay = False
                changed = True

    with top[2]:
        st.markdown("#### Шкала replay (окна) ⓘ")
        st.markdown(_replay_scale_html(total_steps, state.replay_step), unsafe_allow_html=True)
        slider_val = st.slider(
            "",
            min_value=1,
            max_value=max(total_steps, 1),
            value=state.replay_step + 1,
            key=f"replay_slider_{state.selected_case_id}",
            label_visibility="collapsed",
        )
        if slider_val - 1 != state.replay_step:
            state.replay_step = slider_val - 1
            state.autoplay = False
            changed = True
        st.caption(f"{state.replay_step + 1}/{total_steps} · реальные окна: {real_windows}")

    state.replay_step = int(np.clip(state.replay_step, 0, max(total_steps - 1, 0)))

    if changed:
        st.rerun()

    return str(state.selected_case_id), int(state.replay_step), run_df


def render_right_panel(cur: pd.Series, vs_baselines: pd.DataFrame) -> None:
    warning = safe_float(cur.get("warning_score", cur.get("risk_score", 0.0)), 0.0)
    risk = safe_float(cur["risk_score"], 0.0)
    state = normalize_state(cur["pred_state"])
    conf = safe_float(cur["confidence"], 0.0)
    mode_text = (
        "Деградированный мультимодальный"
        if int(safe_float(cur.get("kpi_available", 0), 0)) == 0
        else "Полный мультимодальный"
    )

    with st.container(border=True):
        st.markdown("### Сводка по риску")
        hero_color = "#dc2626" if risk >= RISK_WARNING_MAX else ("#ea580c" if risk >= RISK_NORMAL_MAX else "#16a34a")
        summary_rows = [
            ("Состояние", STATE_LABELS[state]),
            ("Warning score", f"{warning:.2f}"),
            ("Уверенность", f"{conf:.2f}"),
            ("Режим", mode_text),
        ]
        summary_html = "".join(
            f"<div class='summary-row'><div class='summary-k'>{html.escape(k)}</div><div class='summary-v'>{html.escape(v)}</div></div>"
            for k, v in summary_rows
        )
        hero = (
            "<div class='summary-hero'>"
            "<div class='summary-hero-label'>Текущая оценка риска</div>"
            f"<div class='summary-hero-score' style='color:{hero_color}'>{risk:.2f}</div>"
            "</div>"
        )
        st.markdown(f"<div class='summary-list'>{hero}{summary_html}</div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### Сводка по объединению")
        log_share = safe_float(cur.get("lcman_gate", 0.5), 0.5)
        kpi_share = 0.0 if int(safe_float(cur.get("kpi_available", 0), 0)) == 0 else 1.0 - log_share
        if int(safe_float(cur.get("kpi_available", 0), 0)) == 0:
            fusion_html = (
                "<div class='summary-list'>"
                "<div class='summary-row'><div class='summary-k'>Режим fusion</div><div class='summary-v'>log-only fallback</div></div>"
                f"<div class='summary-row'><div class='summary-k'>Логи (semantic)</div><div class='summary-v'>{log_share * 100:.0f}%</div></div>"
                "<div class='summary-row'><div class='summary-k'>KPI (time-series)</div><div class='summary-v'>нет покрытия</div></div>"
                "</div>"
            )
            st.markdown(fusion_html, unsafe_allow_html=True)
            st.caption("В текущем окне fusion деградирует до log-driven режима без KPI-подтверждения.")
        else:
            pie = build_fusion_pie(cur)
            left, right = st.columns([0.88, 1.12], gap="small")
            with left:
                st.plotly_chart(pie, use_container_width=True, config={"displayModeBar": False})
            with right:
                fusion_html = (
                    "<div class='fusion-legend'>"
                    f"<div class='fusion-row'><div class='fusion-left'><span class='fusion-dot' style='background:#2563eb'></span><span>Логи (semantic)</span></div><div class='fusion-value'>{log_share * 100:.0f}%</div></div>"
                    f"<div class='fusion-row'><div class='fusion-left'><span class='fusion-dot' style='background:#7c3aed'></span><span>KPI (time-series)</span></div><div class='fusion-value'>{max(kpi_share, 0.0) * 100:.0f}%</div></div>"
                    "</div>"
                )
                st.markdown(fusion_html, unsafe_allow_html=True)
                st.caption("Решение: weighted late fusion с fallback при отсутствии KPI в окне.")

    with st.container(border=True):
        st.markdown("### Пояснение")
        full_expl = str(cur.get("explanation", _make_default_explanation(cur))).strip()
        note = (
            "Replay densified до 36 шагов; доказательства ниже привязаны к ближайшему реальному окну."
            if int(safe_float(cur.get("kpi_available", 0), 0)) == 1
            else "В этом окне KPI не подтверждают решение; KPI-панель ниже показывает run-контекст, а не прямое подтверждение сигнала."
        )
        explanation_html = (
            f"<div class='explanation-card'>{html.escape(full_expl)}"
            f"<div class='explanation-note'>{html.escape(note)}</div></div>"
        )
        st.markdown(explanation_html, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### Сравнение с baseline")
        delta_risk = pd.to_numeric(
            pd.Series([cur.get("delta_risk_vs_fused", cur.get("delta_vs_log"))]),
            errors="coerce",
        ).iloc[0]
        fused_risk = safe_float(
            cur.get("fused_risk_score", cur.get("baseline_score", risk - (0.0 if pd.isna(delta_risk) else float(delta_risk)))),
            risk,
        )
        if pd.isna(delta_risk):
            delta_risk = risk - fused_risk

        risk_pct = (float(delta_risk) / max(abs(fused_risk), 1e-6)) * 100.0
        improve_cls = "baseline-up" if float(delta_risk) >= 0 else "baseline-down"
        improve_txt = f"{float(delta_risk):+.2f} ({risk_pct:+.1f}%)"
        baseline_html = (
            "<div class='baseline-table'>"
            "<div class='baseline-head'><div>Модель</div><div>Оценка риска</div><div>Состояние</div><div>Улучшение</div></div>"
            f"<div class='baseline-row'><div>Baseline (логовый ref)</div><div>{fused_risk:.2f}</div><div>{STATE_LABELS[score_to_state(fused_risk)]}</div><div>—</div></div>"
            f"<div class='baseline-row'><div>LCMAN-lite</div><div>{risk:.2f}</div><div>{STATE_LABELS[score_to_state(risk)]}</div><div class='{improve_cls}'>{improve_txt}</div></div>"
            "</div>"
        )
        st.markdown(baseline_html, unsafe_allow_html=True)
        st.caption("Положительное Δ означает: мультимодальное объединение раньше усиливает риск относительно логового ref baseline.")

        if not vs_baselines.empty:
            with st.expander("Метрики LCMAN vs baselines", expanded=False):
                st.dataframe(vs_baselines, use_container_width=True, hide_index=True)


def render_log_panel(run_df: pd.DataFrame, current_idx: int, cur: pd.Series) -> None:
    st.markdown("### Лог-доказательства")
    log_df = build_log_evidence(run_df, current_idx, top_n=5)
    _render_log_evidence_table(log_df)

    st.caption(
        "Таблица показывает эвристические драйверы риска по паттернам логов. "
        "Доказательства относятся к ближайшему реальному окну и его накопленному контексту replay."
    )

    excerpt = str(cur.get("log_text", "__NO_LOG__"))
    excerpt = re.sub(r"\s+", " ", excerpt).strip()
    if len(excerpt) > 520:
        excerpt = excerpt[:520] + "…"

    with st.expander("Фрагмент логов выбранного окна", expanded=False):
        st.code(excerpt if excerpt else "__NO_LOG__", language="text")


def render_kpi_panel(
    run_id: str,
    replay_ts: pd.Timestamp,
    run_df_legacy: pd.DataFrame,
    legacy_idx: int,
    cur_legacy: pd.Series,
) -> None:
    st.markdown("### KPI-доказательства")

    raw_kpi = load_raw_kpi_timeseries(run_id)
    families = raw_kpi.get("families", {}) if isinstance(raw_kpi, dict) else {}
    has_family_data = any(
        isinstance(families.get(family), pd.DataFrame) and not families.get(family).empty
        for family in RAW_KPI_FAMILIES
    )

    if not bool(raw_kpi.get("available", False)) or not has_family_data:
        _render_raw_kpi_missing(raw_kpi, run_id)
        with st.expander("Legacy KPI evidence (5-мин признаки)", expanded=False):
            _render_legacy_kpi_evidence(run_df_legacy, legacy_idx, current_ts=replay_ts)
        return

    st.caption(
        "Источник: `kpi_csv/<app>/<scenario>/*-intersect.csv` · компактные карточки показывают ближайший реальный KPI-сигнал и mini-trend."
    )

    if int(safe_float(cur_legacy.get("kpi_available", 0), 0)) == 0:
        st.caption(
            "Текущее окно отмечено как `kpi_available=0`: решение остаётся log-driven, а KPI-карточки ниже показывают только доступный run-контекст."
        )

    _render_kpi_signal_cards(
        raw_kpi=raw_kpi,
        replay_ts=replay_ts,
        current_has_kpi=int(safe_float(cur_legacy.get("kpi_available", 0), 0)) == 1,
    )

    with st.expander("Подробная raw телеметрия", expanded=False):
        st.caption("Подробный просмотр по семействам KPI для отладки и проверки покрытия сырой телеметрии.")
        ts_now = pd.to_datetime(replay_ts, utc=True, errors="coerce")
        x_min = pd.to_datetime(raw_kpi.get("time_min"), utc=True, errors="coerce")
        x_max = pd.to_datetime(raw_kpi.get("time_max"), utc=True, errors="coerce")
        x_domain: tuple[pd.Timestamp, pd.Timestamp] | None = None
        if pd.notna(x_min) and pd.notna(x_max):
            x_left = pd.Timestamp(x_min)
            x_right = pd.Timestamp(x_max)
            if x_left == x_right:
                x_right = x_right + pd.Timedelta(seconds=1)
            x_domain = (x_left, x_right)

        builders: list[tuple[str, Any]] = [
            ("cpu", build_cpu_family_chart),
            ("memstat", build_memory_family_chart),
            ("iostat", build_disk_family_chart),
            ("netstat", build_network_family_chart),
        ]

        cols = st.columns(2, gap="small")
        ts_token = str(pd.Timestamp(ts_now).value) if pd.notna(ts_now) else "na"
        ts_for_build = pd.Timestamp(ts_now) if pd.notna(ts_now) else pd.NaT
        for idx, (family, builder) in enumerate(builders):
            fig, status = builder(raw_kpi, current_ts=ts_for_build, x_domain=x_domain)
            panel_title = str(RAW_KPI_PANEL_META.get(family, {}).get("title", family.upper()))
            primary_label = str(status.get("primary_label", "")).strip()
            with cols[idx % 2]:
                with st.container(border=True):
                    title = f"**{panel_title}**"
                    if primary_label:
                        title += f" · {primary_label}"
                    st.markdown(title)
                    st.caption(_status_line_for_family(status))
                    chart_key = f"raw_kpi::{run_id}::{family}::{legacy_idx}::{ts_token}"
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=chart_key)

        with st.expander("Legacy KPI evidence (5-мин признаки)", expanded=False):
            st.caption("Вторичный debug-view: прежняя feature-сетка из timeline (`*__last/*__mean`).")
            _render_legacy_kpi_evidence(run_df_legacy, legacy_idx, current_ts=ts_for_build)


def render_about_section(bundle: dict[str, Any]) -> None:
    st.markdown("---")
    with st.expander("Model / About", expanded=False):
        st.markdown("#### Data contract dashboard")
        st.dataframe(bundle["contract_table"], use_container_width=True, hide_index=True)

        runtime_meta = bundle.get("runtime_metadata", {}) or {}
        if runtime_meta:
            st.markdown("#### Runtime / Encoder")
            runtime_rows = [
                {"Параметр": "Encoder", "Значение": str(runtime_meta.get("encoder_name", "n/a"))},
                {"Параметр": "Embedding dim", "Значение": str(runtime_meta.get("embedding_dim", "n/a"))},
                {"Параметр": "Data mode", "Значение": str(runtime_meta.get("data_mode", "n/a"))},
                {"Параметр": "Data source", "Значение": str(runtime_meta.get("data_source", "n/a"))},
            ]
            st.dataframe(pd.DataFrame(runtime_rows), use_container_width=True, hide_index=True)

        metrics_summary = bundle["metrics_summary"]
        if not metrics_summary.empty:
            st.markdown("#### Сводка метрик LCMAN")
            st.dataframe(metrics_summary, use_container_width=True, hide_index=True)

        modality = bundle["modality"]
        if not modality.empty:
            st.markdown("#### Анализ по модальностям")
            st.dataframe(modality, use_container_width=True, hide_index=True)

        model_card = str(bundle.get("model_card", "")).strip()
        if model_card:
            st.markdown("#### Model card")
            st.markdown(model_card)


def main() -> None:
    st.set_page_config(
        page_title="LCMAN Dashboard",
        page_icon="L",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if go is None:
        st.error("Не найден пакет `plotly`. Установите зависимость: `pip install plotly`.")
        st.stop()

    inject_css()

    try:
        bundle = load_dashboard_data()
    except DataContractError as exc:
        st.error(f"Ошибка data contract: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Не удалось загрузить dashboard artifacts: {exc}")
        st.stop()

    timeline = bundle["timeline"]
    cases = bundle["cases"]

    if timeline.empty or cases.empty:
        st.error("Недостаточно данных для replay-dashboard (пустой timeline/cases).")
        st.stop()

    header_slot = st.empty()
    selected_case_id, step_idx, run_df = render_case_and_controls(cases, timeline)

    selected_case = cases[cases["case_id"].eq(selected_case_id)].iloc[0]
    step_idx = int(np.clip(step_idx, 0, max(len(run_df) - 1, 0)))
    cur = run_df.iloc[step_idx]

    run_df_raw = timeline[timeline["run_id"].eq(str(selected_case["run_id"]))].sort_values("window_ts").reset_index(drop=True)
    src_idx = int(np.clip(safe_float(cur.get("_src_idx", step_idx), float(step_idx)), 0, max(len(run_df_raw) - 1, 0)))
    cur_source = run_df_raw.iloc[src_idx].copy()
    for col in [
        "warning_score",
        "risk_score",
        "fused_warn_score",
        "fused_risk_score",
        "baseline_score",
        "delta_warn_vs_fused",
        "delta_risk_vs_fused",
        "delta_vs_log",
        "lcman_gate",
        "confidence",
        "pred_state",
    ]:
        if col in cur.index:
            cur_source[col] = cur[col]
    with header_slot.container():
        render_top_header(selected_case, cur_source, step_idx, len(run_df))

    evidence_window_ts = pd.Timestamp(run_df_raw.iloc[src_idx]["window_ts"]) if not run_df_raw.empty else pd.Timestamp(cur["window_ts"])
    evidence_window_txt = f"{evidence_window_ts.strftime('%H:%M')}–{(evidence_window_ts + pd.Timedelta(minutes=5)).strftime('%H:%M')}"
    st.markdown(
        _meta_chips_html(
            [
                ("Кейс", str(selected_case["label"])),
                ("Replay-курсор", pd.Timestamp(cur["window_ts"]).strftime("%H:%M")),
                ("Evidence-окно", evidence_window_txt),
                ("Источник", str(selected_case["source"])),
            ]
        ),
        unsafe_allow_html=True,
    )
    st.caption(str(selected_case["note"]))

    main_cols = st.columns([3.85, 1.15], gap="large")

    with main_cols[0]:
        with st.container(border=True):
            st.markdown("### Оценка риска во времени")
            fig = build_timeline_figure(run_df, step_idx, target_view="risk")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            fail_start = pd.to_datetime(cur_source.get("fail_start"), utc=True, errors="coerce")
            st.markdown(
                _meta_chips_html(
                    [
                        ("Логи", "доступны" if int(cur_source["log_available"]) == 1 else "отсутствуют"),
                        ("KPI", "доступны" if int(cur_source["kpi_available"]) == 1 else "частично/отсутствуют"),
                        ("Начало сбоя", fail_start.strftime("%H:%M") if pd.notna(fail_start) else "n/a"),
                    ]
                ),
                unsafe_allow_html=True,
            )
            st.caption("Плавная линия показывает densified replay, пустые/серые зоны обозначают отсутствие KPI, маркеры на линии соответствуют реальным 5-мин окнам.")

        low_cols = st.columns([1.25, 1.75], gap="small")
        with low_cols[0]:
            with st.container(border=True):
                render_log_panel(run_df_raw, src_idx, cur_source)
        with low_cols[1]:
            with st.container(border=True):
                render_kpi_panel(
                    run_id=str(selected_case["run_id"]),
                    replay_ts=pd.Timestamp(cur["window_ts"]),
                    run_df_legacy=run_df_raw,
                    legacy_idx=src_idx,
                    cur_legacy=cur_source,
                )

    with main_cols[1]:
        render_right_panel(cur_source, bundle["vs_baselines"])

    render_about_section(bundle)

    # autoplay loop
    if st.session_state.get("autoplay", False):
        if st.session_state.replay_step < len(run_df) - 1:
            time.sleep(0.65)
            st.session_state.replay_step += 1
            st.rerun()
        else:
            st.session_state.autoplay = False


if __name__ == "__main__":
    main()
