from __future__ import annotations

import json
import os
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = ROOT / "dashboard" / "artifacts"
ARTIFACT_DIR = Path(os.environ.get("DASHBOARD_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR))
if not ARTIFACT_DIR.is_absolute():
    ARTIFACT_DIR = ROOT / ARTIFACT_DIR
MODEL_RUN_DIR = Path(
    os.environ.get(
        "FINAL_MULTIMODAL_RUN_DIR",
        ARTIFACT_DIR / "model" / "final_multimodal_strict_intensity_gap10",
    )
)
STRICT_CONTRACT_PATH = Path(
    os.environ.get(
        "FINAL_STRICT_CONTRACT_PATH",
        ARTIFACT_DIR / "contract" / "strict_intensity_contract.csv.gz",
    )
)
OUT_DIR = ARTIFACT_DIR / "bundle"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _kpi_info(column: str) -> dict[str, str]:
    parts = column.removeprefix("kpi_").split("__")
    return {
        "host": parts[0] if len(parts) > 0 else "",
        "family": parts[1] if len(parts) > 1 else "",
        "metric": parts[2] if len(parts) > 2 else "",
        "dim": parts[3] if len(parts) > 3 else "",
        "stat": parts[-1] if parts else "",
    }


def _metric_matches(metric: str, token: str) -> bool:
    metric = str(metric).lower()
    token = str(token).lower()
    return metric == token or metric.replace("_ss", "_s") == token or token.replace("_s", "_ss") == metric


def _score_columns(
    frame: pd.DataFrame,
    family: str | None = None,
    tokens: tuple[str, ...] = (),
    stats: tuple[str, ...] | None = None,
) -> list[str]:
    cols = [col for col in frame.columns if col.startswith("kpi_") and "__" in col]
    if family:
        cols = [col for col in cols if f"__{family}__" in col]
    if tokens:
        cols = [
            col
            for col in cols
            if any(_metric_matches(_kpi_info(col)["metric"], token) for token in tokens)
        ]
    if stats:
        stat_set = {str(stat).lower() for stat in stats}
        cols = [col for col in cols if _kpi_info(col)["stat"].lower() in stat_set]
    return cols


def _row_aggregate(frame: pd.DataFrame, cols: list[str], mode: str = "median") -> pd.Series:
    if not cols:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = frame[cols].apply(pd.to_numeric, errors="coerce")
    if mode == "max":
        return values.max(axis=1)
    if mode == "min":
        return values.min(axis=1)
    if mode == "sum":
        return values.sum(axis=1, min_count=1)
    return values.median(axis=1)


def _source_summary(cols: list[str], limit: int = 10) -> str:
    if not cols:
        return ""
    shown = cols[:limit]
    suffix = f";...(+{len(cols) - limit})" if len(cols) > limit else ""
    return ";".join(shown) + suffix


def _sum_metrics_by_host(
    frame: pd.DataFrame,
    family: str,
    metrics: tuple[str, ...],
    stats: tuple[str, ...] = ("last", "mean"),
) -> tuple[pd.Series, list[str]]:
    cols = _score_columns(frame, family, metrics, stats=stats)
    if not cols:
        return pd.Series(np.nan, index=frame.index, dtype=float), []

    grouped: dict[tuple[str, str, str], list[str]] = {}
    for col in cols:
        info = _kpi_info(col)
        grouped.setdefault((info["host"], info["dim"], info["stat"]), []).append(col)

    pieces = []
    for group_cols in grouped.values():
        pieces.append(frame[group_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1))
    if not pieces:
        return pd.Series(np.nan, index=frame.index, dtype=float), cols
    values = pd.concat(pieces, axis=1)
    return values.median(axis=1, skipna=True), cols


def _normalise_pct(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    finite = series.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    high = float(finite.quantile(0.98))
    low = float(finite.quantile(0.02))
    if high <= 1.5:
        out = series * 100.0
    elif high <= 100.0 and low >= 0.0:
        out = series
    else:
        denom = high if high > 0 else float(finite.max())
        out = series / max(denom, 1e-9) * 100.0
    return out.clip(0.0, 100.0)


def _add_dashboard_kpis(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "kpi_dashboard_cpu_pct" not in out.columns:
        idle_cols = _score_columns(out, "cpustat", ("idle",), stats=("last", "mean"))
        idle = _row_aggregate(out, idle_cols, "median")
        busy, busy_cols = _sum_metrics_by_host(out, "cpustat", ("user", "system", "iowait", "wa"), stats=("last", "mean"))
        cpu_pct = np.where(idle.notna(), 100.0 - _normalise_pct(idle), _normalise_pct(busy))
        out["kpi_dashboard_cpu_pct"] = pd.Series(cpu_pct, index=out.index).clip(0.0, 100.0)
        out["kpi_dashboard_cpu_pct_source"] = _source_summary(idle_cols or busy_cols)

    if "kpi_dashboard_memory_pct" not in out.columns:
        mem_cols = _score_columns(out, "memstat", ("memused",), stats=("last", "mean"))
        memory = _normalise_pct(_row_aggregate(out, mem_cols, "median"))
        if memory.isna().all():
            used_cols = _score_columns(out, "memstat", ("kbmemused",), stats=("last", "mean"))
            avail_cols = _score_columns(out, "memstat", ("kbavail",), stats=("last", "mean"))
            used = _row_aggregate(out, used_cols, "median")
            avail = _row_aggregate(out, avail_cols, "median")
            memory = (used / (used + avail).replace(0.0, np.nan) * 100.0).clip(0.0, 100.0)
            mem_cols = used_cols + avail_cols
        out["kpi_dashboard_memory_pct"] = memory
        out["kpi_dashboard_memory_pct_source"] = _source_summary(mem_cols)

    if "kpi_dashboard_disk_pct" not in out.columns:
        disk_cols = _score_columns(out, "iostat", ("util",), stats=("last", "mean"))
        out["kpi_dashboard_disk_pct"] = _normalise_pct(_row_aggregate(out, disk_cols, "max"))
        out["kpi_dashboard_disk_pct_source"] = _source_summary(disk_cols)

    if "kpi_dashboard_network_mbps" not in out.columns:
        net_cols = _score_columns(out, "netstat", ("rxkb_s", "txkb_s", "rxkb_ss", "txkb_ss"), stats=("last", "mean"))
        out["kpi_dashboard_network_mbps"] = (_row_aggregate(out, net_cols, "sum") / 1024.0).clip(lower=0.0)
        out["kpi_dashboard_network_mbps_source"] = _source_summary(net_cols)

    for col in [
        "kpi_dashboard_cpu_pct",
        "kpi_dashboard_memory_pct",
        "kpi_dashboard_disk_pct",
        "kpi_dashboard_network_mbps",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _load_strict_predictions(model_run_dir: Path = MODEL_RUN_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_path = model_run_dir / "ensemble_predictions.csv"
    val_path = model_run_dir / "validation_ensemble_predictions.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing strict/intensity test predictions: {test_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing strict/intensity validation predictions: {val_path}")
    test = pd.read_csv(test_path)
    val = pd.read_csv(val_path)
    for frame in (test, val):
        frame["run_id"] = frame["run_id"].astype(str)
        frame["window_ts"] = pd.to_datetime(frame["window_ts"], utc=True, errors="coerce")
        frame["split"] = frame.get("split", "test").fillna("test").astype(str)
    return test, val


def _load_strict_contract(contract_path: Path = STRICT_CONTRACT_PATH) -> pd.DataFrame:
    if not contract_path.exists():
        raise FileNotFoundError(f"Missing final strict/intensity contract: {contract_path}")
    raw = pd.read_csv(contract_path, low_memory=False)
    raw["run_id"] = raw["run_id"].astype(str)
    raw["window_ts"] = pd.to_datetime(raw["window_ts"], utc=True, errors="coerce")
    if "window_end" in raw.columns:
        raw["window_end"] = pd.to_datetime(raw["window_end"], utc=True, errors="coerce")
    return raw


def _merge_predictions_with_contract(pred: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    raw_dedup = raw.drop_duplicates(["run_id", "window_ts"], keep="last")
    merged = pred.merge(raw_dedup, on=["run_id", "window_ts"], how="left", suffixes=("", "_raw"))
    if "window_end" not in merged.columns or merged["window_end"].isna().all():
        merged["window_end"] = merged["window_ts"] + pd.to_timedelta(10, unit="s")
    else:
        merged["window_end"] = pd.to_datetime(merged["window_end"], utc=True, errors="coerce")
        merged["window_end"] = merged["window_end"].fillna(merged["window_ts"] + pd.to_timedelta(10, unit="s"))
    if "event_anchor" not in merged.columns or merged["event_anchor"].isna().all():
        lead = pd.to_numeric(merged.get("strict_lead_sec", np.nan), errors="coerce")
        merged["event_anchor"] = merged["window_ts"] + pd.to_timedelta(lead, unit="s")
    merged["event_anchor"] = pd.to_datetime(merged["event_anchor"], utc=True, errors="coerce")
    return merged


def _prepare_threshold_frame(validation: pd.DataFrame) -> pd.DataFrame:
    out = validation.copy()
    out["alert_score"] = pd.to_numeric(out["warning_intensity_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["model_score"] = out["alert_score"]
    out["run_id"] = out["run_id"].astype(str)
    out["window_ts"] = pd.to_datetime(out["window_ts"], utc=True, errors="coerce")
    if "window_end" not in out.columns:
        out["window_end"] = out["window_ts"] + pd.to_timedelta(10, unit="s")
    else:
        out["window_end"] = pd.to_datetime(out["window_end"], utc=True, errors="coerce")
        out["window_end"] = out["window_end"].fillna(out["window_ts"] + pd.to_timedelta(10, unit="s"))
    if "event_anchor" not in out.columns or out["event_anchor"].isna().all():
        lead = pd.to_numeric(out.get("strict_lead_sec", np.nan), errors="coerce")
        out["event_anchor"] = out["window_ts"] + pd.to_timedelta(lead, unit="s")
    out["event_anchor"] = pd.to_datetime(out["event_anchor"], utc=True, errors="coerce")
    if "scenario_type" not in out.columns:
        out["scenario_type"] = np.where(pd.to_numeric(out.get("y_warning", 0), errors="coerce").fillna(0).gt(0), "failure", "normal")
    return out


def _choose_thresholds(validation: pd.DataFrame) -> dict[str, Any]:
    threshold_frame = _prepare_threshold_frame(validation)
    score = pd.to_numeric(threshold_frame["alert_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if score.empty:
        return {
            "watch_threshold": 0.20,
            "alert_threshold": 0.30,
            "min_consecutive_windows": 2.0,
            "threshold_selection_warning": "validation_score_empty",
        }

    quantiles = np.linspace(0.55, 0.98, 24)
    thresholds = sorted({float(score.quantile(q)) for q in quantiles if np.isfinite(score.quantile(q))})
    thresholds = [value for value in thresholds if value > 0.0]
    if not thresholds:
        thresholds = [float(score.max()) if float(score.max()) > 0 else 0.01]

    q70 = float(score.quantile(0.70))
    candidates: list[dict[str, Any]] = []
    for alert_threshold in thresholds:
        watch_threshold = min(alert_threshold * 0.75, q70)
        watch_threshold = max(0.0, min(watch_threshold, alert_threshold - 1e-6))
        for consecutive in [1, 2, 3]:
            trial_thresholds = {
                "watch_threshold": watch_threshold,
                "alert_threshold": alert_threshold,
                "min_consecutive_windows": float(consecutive),
            }
            trial = _apply_alert_layer(threshold_frame, trial_thresholds)
            events = _run_events(trial)
            metrics = _metrics_summary(trial, events).iloc[0].to_dict()
            candidates.append(
                {
                    **trial_thresholds,
                    "run_alert_f1": _safe_float(metrics.get("run_alert_f1")),
                    "failure_run_recall": _safe_float(metrics.get("failure_run_recall")),
                    "run_alert_precision": _safe_float(metrics.get("run_alert_precision")),
                    "false_alert_run_rate": _safe_float(metrics.get("false_alert_run_rate")),
                    "alert_density": _safe_float(metrics.get("alert_density")),
                    "median_lead_time_sec": _safe_float(metrics.get("median_lead_time_sec")),
                }
            )

    feasible = [
        row
        for row in candidates
        if row["false_alert_run_rate"] <= 0.20 and row["alert_density"] <= 0.35
    ]
    if feasible:
        best = sorted(
            feasible,
            key=lambda row: (
                row["run_alert_f1"],
                row["failure_run_recall"],
                row["run_alert_precision"],
                -row["alert_density"],
            ),
            reverse=True,
        )[0]
        warning = ""
    else:
        best = sorted(
            candidates,
            key=lambda row: (
                -row["false_alert_run_rate"],
                -row["alert_density"],
                row["run_alert_f1"],
                row["failure_run_recall"],
            ),
            reverse=True,
        )[0]
        warning = "no_validation_policy_met_false_alert_or_density_caps"

    best.update(
        {
            "raw_score_q70": float(score.quantile(0.70)),
            "raw_score_q85": float(score.quantile(0.85)),
            "raw_score_q95": float(score.quantile(0.95)),
            "candidate_count": int(len(candidates)),
            "feasible_candidate_count": int(len(feasible)),
            "threshold_selection_warning": warning,
            "selection_split": "validation",
            "score_column": "warning_intensity_score",
        }
    )
    return best


def _rank_calibrate_score(values: pd.Series, validation_values: pd.Series) -> pd.Series:
    train = pd.to_numeric(validation_values, errors="coerce").dropna().to_numpy(dtype=float)
    if train.size == 0:
        return pd.to_numeric(values, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    train = np.sort(train)
    score = pd.to_numeric(values, errors="coerce").fillna(float(np.nanmedian(train))).to_numpy(dtype=float)
    ranks = np.searchsorted(train, score, side="right") / float(train.size)
    return pd.Series(ranks, index=values.index, dtype=float).clip(0.0, 1.0)


def _apply_alert_layer(frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    out = frame.sort_values(["run_id", "window_ts"]).copy()
    score_col = "alert_score" if "alert_score" in out.columns else "model_score"
    score = pd.to_numeric(out[score_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["watch_threshold"] = float(thresholds["watch_threshold"])
    out["alert_threshold"] = float(thresholds["alert_threshold"])
    high = score.ge(out["alert_threshold"])

    confirmed = pd.Series(False, index=out.index)
    confirmed_now = pd.Series(False, index=out.index)
    for _, idx in out.groupby("run_id", sort=False).groups.items():
        run_high = high.loc[idx].astype(int)
        run_confirmed = run_high.rolling(int(thresholds["min_consecutive_windows"]), min_periods=int(thresholds["min_consecutive_windows"])).sum()
        current = run_confirmed.ge(int(thresholds["min_consecutive_windows"]))
        confirmed_now.loc[idx] = current.to_numpy()
        confirmed.loc[idx] = current.cummax().to_numpy()

    out["alert_fired"] = confirmed.astype(int)
    out["alert_confirmed_now"] = confirmed_now.astype(int)
    started = pd.Series(False, index=out.index)
    for _, idx in out.groupby("run_id", sort=False).groups.items():
        run_alert = confirmed.loc[idx].astype(int)
        started.loc[idx] = run_alert.eq(1) & run_alert.shift(1, fill_value=0).eq(0)
    out["alert_started"] = started.astype(int)
    incident_now = (
        out["event_anchor"].notna()
        & out["window_ts"].le(out["event_anchor"])
        & out["window_end"].ge(out["event_anchor"])
    )
    out["incident_marker"] = incident_now.astype(int)
    out["alert_state"] = np.select(
        [
            incident_now,
            confirmed,
            score.ge(out["watch_threshold"]),
        ],
        ["incident", "alert", "watch"],
        default="normal",
    )
    out["pred_state"] = np.select(
        [out["alert_state"].isin(["alert", "incident"]), out["alert_state"].eq("watch")],
        ["risk", "warning"],
        default="normal",
    )
    return out


def _add_incident_markers(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["incident_marker"] = 0
    for _, group in out.loc[out["event_anchor"].notna()].groupby("run_id", sort=False):
        anchor = pd.to_datetime(group["event_anchor"], utc=True, errors="coerce").dropna()
        if anchor.empty:
            continue
        target_ts = pd.Timestamp(anchor.iloc[0])
        distance = (pd.to_datetime(group["window_end"], utc=True, errors="coerce") - target_ts).abs()
        if not distance.empty:
            out.loc[distance.idxmin(), "incident_marker"] = 1
    incident = pd.to_numeric(out["incident_marker"], errors="coerce").fillna(0).astype(int).eq(1)
    if "alert_state" in out.columns:
        out.loc[incident, "alert_state"] = "incident"
    if "pred_state" in out.columns:
        out.loc[incident, "pred_state"] = "risk"
    return out


def _add_fusion_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    log_signal = pd.concat(
        [
            pd.to_numeric(out.get("log_warning_score", 0.0), errors="coerce"),
            pd.to_numeric(out.get("log_risk_score", 0.0), errors="coerce"),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    kpi_signal = pd.concat(
        [
            pd.to_numeric(out.get("kpi_warning_score", 0.0), errors="coerce"),
            pd.to_numeric(out.get("kpi_risk_score", 0.0), errors="coerce"),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0).clip(0.0, 1.0)

    log_available = pd.to_numeric(out.get("log_available", 0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    kpi_available = pd.to_numeric(out.get("kpi_available", 0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    log_signal = log_signal * log_available
    kpi_signal = kpi_signal * kpi_available
    denom = (log_signal + kpi_signal).replace(0.0, np.nan)
    out["log_only_score"] = log_signal
    out["kpi_only_score"] = kpi_signal
    both = log_available.gt(0) & kpi_available.gt(0)
    share_available = both & denom.notna()
    out["fusion_log_share"] = np.where(share_available, log_signal / denom, np.nan)
    out["fusion_kpi_share"] = np.where(share_available, kpi_signal / denom, np.nan)
    out["fusion_gate"] = out["fusion_log_share"]
    out["fusion_share_available"] = share_available.astype(int)
    out["fusion_display_mode"] = np.select(
        [
            share_available,
            log_available.gt(0) & kpi_available.gt(0),
            log_available.gt(0),
            kpi_available.gt(0),
        ],
        ["diagnostic log+kpi", "log+kpi; zero auxiliary signal", "log-only fallback", "kpi-only fallback"],
        default="no modality",
    )
    return out


def _mark_demo_cases(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["demo_case_kind"] = ""
    out["demo_case_id"] = ""

    events = _run_events(out)
    early_runs_all = events.loc[
        events["event_type"].eq("early_alert")
        & pd.to_numeric(events.get("n_windows", 0), errors="coerce").fillna(0).ge(3)
        & pd.to_numeric(events.get("lead_time_sec", 0), errors="coerce").fillna(0).ge(20)
    ].copy()
    early_runs_with_context = early_runs_all.loc[
        pd.to_numeric(early_runs_all.get("pre_alert_windows", 0), errors="coerce").fillna(0).ge(1)
    ].copy()
    early_runs = early_runs_with_context if not early_runs_with_context.empty else early_runs_all
    normal_runs = events.loc[events["event_type"].eq("normal_no_alert")].copy()
    used_runs: set[str] = set()

    def assign(kind: str, run_id: str | None) -> bool:
        if not run_id:
            return False
        if str(run_id) in used_runs:
            return False
        mask = out["run_id"].astype(str).eq(str(run_id))
        if mask.any():
            out.loc[mask, "demo_case_kind"] = kind
            out.loc[mask, "demo_case_id"] = f"{kind}::{run_id}"
            used_runs.add(str(run_id))
            return True
        return False

    def assign_first(kind: str, candidates: pd.DataFrame) -> None:
        for run_id in candidates.index.astype(str).tolist():
            if assign(kind, run_id):
                break

    if not early_runs.empty:
        run_scores = out.groupby("run_id").agg(
            max_score=("model_score", "max"),
            log_share=("fusion_log_share", "mean"),
            kpi_share=("fusion_kpi_share", "mean"),
            both=("has_both_modalities", "mean"),
        )
        early_ids = set(early_runs["run_id"].astype(str))
        all_early_ids = set(early_runs_all["run_id"].astype(str))
        candidates = run_scores.loc[run_scores.index.astype(str).isin(early_ids)].copy()
        all_candidates = run_scores.loc[run_scores.index.astype(str).isin(all_early_ids)].copy()
        both = candidates.loc[candidates["both"].ge(0.5)].copy()
        if not both.empty:
            both["_balance"] = 1.0 - (both["log_share"] - 0.5).abs() * 2.0
            assign_first("multimodal_success", both.sort_values(["_balance", "max_score"], ascending=[False, False]))
        kpi_driven = candidates.copy()
        if not kpi_driven.empty:
            assign_first("kpi_driven_success", kpi_driven.sort_values(["kpi_share", "max_score"], ascending=False))
        log_driven = candidates.copy()
        if not log_driven.empty:
            assign_first("log_driven_success", log_driven.sort_values(["log_share", "max_score"], ascending=False))
        if "log_driven_success" not in set(out["demo_case_kind"].astype(str)) and not all_candidates.empty:
            assign_first("log_driven_success", all_candidates.sort_values(["log_share", "max_score"], ascending=False))
        if "kpi_driven_success" not in set(out["demo_case_kind"].astype(str)) and not all_candidates.empty:
            assign_first("kpi_driven_success", all_candidates.sort_values(["kpi_share", "max_score"], ascending=False))

    if not normal_runs.empty:
        normal_score = out.loc[out["run_id"].astype(str).isin(normal_runs["run_id"].astype(str))].groupby("run_id")["model_score"].max()
        for run_id in normal_score.sort_values().index.astype(str).tolist():
            if assign("normal_no_alert", run_id):
                break

    return out


def _run_events(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_id, run_group in frame.sort_values("window_ts").groupby("run_id", sort=False):
        group = run_group.sort_values("window_ts").copy()
        scenario_type = str(group.get("scenario_type", pd.Series("", index=group.index)).fillna("").iloc[0])
        is_failure = scenario_type == "failure" or pd.to_numeric(group.get("y_warning", 0), errors="coerce").fillna(0).max() > 0
        alert_flags = pd.to_numeric(group.get("alert_fired", 0), errors="coerce").fillna(0).astype(int).eq(1)
        alerts = group.loc[alert_flags]
        alert_positions = np.flatnonzero(alert_flags.to_numpy())
        first_alert_window_index = int(alert_positions[0]) if alert_positions.size else -1
        event_anchor = pd.to_datetime(group.get("event_anchor", pd.Series(pd.NaT, index=group.index)), utc=True, errors="coerce").dropna()
        anchor = event_anchor.iloc[0] if not event_anchor.empty else pd.NaT
        first_alert_ts = pd.NaT if alerts.empty else pd.Timestamp(alerts["window_ts"].iloc[0])
        if is_failure and pd.notna(anchor) and pd.notna(first_alert_ts):
            lead = (anchor - first_alert_ts).total_seconds()
            event_type = "early_alert" if lead >= 10 else "late_alert"
        elif is_failure:
            lead = np.nan
            event_type = "missed_failure"
        elif not alerts.empty:
            lead = np.nan
            event_type = "false_alert"
        else:
            lead = np.nan
            event_type = "normal_no_alert"
        rows.append(
            {
                "run_id": str(run_id),
                "workload": str(group.get("workload", pd.Series("", index=group.index)).fillna("").iloc[0]),
                "scenario": str(group.get("scenario", pd.Series("", index=group.index)).fillna("").iloc[0]),
                "scenario_type": scenario_type,
                "event_type": event_type,
                "first_alert_ts": first_alert_ts,
                "event_anchor": anchor,
                "lead_time_sec": lead,
                "n_windows": int(len(group)),
                "first_alert_window_index": first_alert_window_index,
                "pre_alert_windows": int(first_alert_window_index if first_alert_window_index >= 0 else len(group)),
                "alert_windows": int(pd.to_numeric(group.get("alert_fired", 0), errors="coerce").fillna(0).sum()),
                "max_model_score": float(pd.to_numeric(group.get("model_score", 0.0), errors="coerce").fillna(0.0).max()),
            }
        )
    return pd.DataFrame(rows)


def _metrics_summary(frame: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    failure = events["scenario_type"].astype(str).eq("failure")
    normal = ~failure
    early = events["event_type"].eq("early_alert")
    alerts = events["event_type"].isin(["early_alert", "late_alert", "false_alert"])
    failure_run_recall = float((early & failure).sum() / max(failure.sum(), 1))
    false_alert_run_rate = float((events["event_type"].eq("false_alert") & normal).sum() / max(normal.sum(), 1))
    precision = float((early & failure).sum() / max(alerts.sum(), 1))
    f1 = 2 * precision * failure_run_recall / max(precision + failure_run_recall, 1e-9)
    lead = pd.to_numeric(events.loc[early, "lead_time_sec"], errors="coerce").dropna()
    density = float(pd.to_numeric(frame.get("alert_fired", 0), errors="coerce").fillna(0).mean())
    return pd.DataFrame(
        [
            {
                "model": "final-multimodal-strict-intensity-fusion",
                "failure_runs": int(failure.sum()),
                "normal_runs": int(normal.sum()),
                "failure_run_recall": failure_run_recall,
                "run_alert_precision": precision,
                "run_alert_f1": f1,
                "false_alert_run_rate": false_alert_run_rate,
                "alert_density": density,
                "median_lead_time_sec": float(lead.median()) if not lead.empty else np.nan,
                "alert_events": int(alerts.sum()),
            }
        ]
    )


def _series_summary(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p50": float(values.quantile(0.50)),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def _top_fraction_metrics(labels: pd.Series, scores: pd.Series, fraction: float = 0.10) -> dict[str, float]:
    y = pd.to_numeric(labels, errors="coerce").fillna(0).astype(int)
    s = pd.to_numeric(scores, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.notna()
    y = y.loc[valid]
    s = s.loc[valid]
    n = int(len(s))
    positives = int(y.sum())
    if n == 0:
        return {
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "prevalence": np.nan,
            "threshold_value": np.nan,
            "selected_windows": 0.0,
        }
    k = max(1, int(np.ceil(n * fraction)))
    chosen = s.sort_values(ascending=False).head(k).index
    pred = pd.Series(0, index=s.index, dtype=int)
    pred.loc[chosen] = 1
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "prevalence": float(positives / max(n, 1)),
        "threshold_value": float(s.loc[chosen].min()),
        "selected_windows": float(k),
    }


def _fill_row_metrics(table: pd.DataFrame, mask: pd.Series, labels: pd.Series, scores: pd.Series) -> None:
    metrics = _top_fraction_metrics(labels, scores, fraction=0.10)
    for column in ["precision", "recall", "f1", "prevalence", "threshold_value", "selected_windows"]:
        if column not in table.columns:
            table[column] = np.nan
        table.loc[mask, column] = metrics[column]
    table.loc[mask, "threshold_policy"] = "top_10_percent"


def _enrich_baseline_table(table: pd.DataFrame, model_run_dir: Path, artifact_dir: Path) -> pd.DataFrame:
    if table.empty:
        return table
    out = table.copy()
    if "threshold_policy" not in out.columns:
        out["threshold_policy"] = ""

    pred_path = model_run_dir / "ensemble_predictions.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        if {"y_warning", "warning_intensity_score"}.issubset(pred.columns):
            mask = out["target"].astype(str).eq("y_warning") & out["baseline"].astype(str).eq("final-multimodal-strict-intensity-fusion")
            if mask.any():
                _fill_row_metrics(out, mask, pred["y_warning"], pred["warning_intensity_score"])
        if {"y_risk", "state_score"}.issubset(pred.columns):
            mask = out["target"].astype(str).eq("y_risk") & out["baseline"].astype(str).eq("final-multimodal-state-score")
            if mask.any():
                _fill_row_metrics(out, mask, pred["y_risk"], pred["state_score"])

    baseline_pred_path = artifact_dir / "reports" / "baseline_predictions.csv.gz"
    if baseline_pred_path.exists() and {"target", "baseline"}.issubset(out.columns):
        baseline_pred = pd.read_csv(baseline_pred_path)
        required = {"label", "target", "baseline", "score"}
        if required.issubset(baseline_pred.columns):
            split = baseline_pred.get("split", pd.Series("test", index=baseline_pred.index)).fillna("test").astype(str)
            baseline_pred = baseline_pred.loc[split.eq("test")].copy()
            for (target, baseline), group in baseline_pred.groupby(["target", "baseline"], sort=False):
                mask = out["target"].astype(str).eq(str(target)) & out["baseline"].astype(str).eq(str(baseline))
                if mask.any():
                    _fill_row_metrics(out, mask, group["label"], group["score"])

    numeric_cols = [
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "prevalence",
        "threshold_value",
        "selected_windows",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _bundle_quality_report(
    replay: pd.DataFrame,
    validation: pd.DataFrame,
    thresholds: dict[str, Any],
    events: pd.DataFrame,
) -> dict[str, Any]:
    state_counts = replay.get("alert_state", pd.Series(dtype=str)).fillna("normal").astype(str).value_counts().to_dict()
    source_columns = {}
    for key in [
        "kpi_dashboard_cpu_pct_source",
        "kpi_dashboard_memory_pct_source",
        "kpi_dashboard_disk_pct_source",
        "kpi_dashboard_network_mbps_source",
    ]:
        if key in replay.columns:
            value = replay[key].dropna().astype(str)
            source_columns[key] = value.iloc[0] if not value.empty else ""

    return {
        "score_semantics": {
            "model_score_raw": "сырой warning_intensity_score; это не вероятность сбоя",
            "model_score_rank": "ранг сырого score относительно validation; диагностический канал",
            "risk_score_raw": "строгий risk/state канал; не используется как главный alert-score",
            "fusion_balance": "diagnostic auxiliary-head balance; это не причинная атрибуция",
        },
        "score_distribution": {
            "test_model_score_raw": _series_summary(replay.get("model_score_raw", pd.Series(dtype=float))),
            "test_model_score_rank": _series_summary(replay.get("model_score_rank", pd.Series(dtype=float))),
            "test_risk_score_raw": _series_summary(replay.get("risk_score_raw", pd.Series(dtype=float))),
            "validation_warning_intensity_score": _series_summary(validation.get("warning_intensity_score", pd.Series(dtype=float))),
        },
        "alert_layer": thresholds,
        "alert_state_counts": {str(k): int(v) for k, v in state_counts.items()},
        "proxy_windows": int((replay.get("score_source", pd.Series("", index=replay.index)).astype(str) != "model_store").sum()),
        "kpi_sources": source_columns,
        "run_event_counts": events.get("event_type", pd.Series(dtype=str)).fillna("").astype(str).value_counts().to_dict(),
        "warnings": [
            "model_score_rank не является вероятностью сбоя",
            "fusion balance не является causal attribution",
            "alert_state является решением alert-layer, а incident_marker является фактом события",
        ],
    }


def _strict_metric_rows(model_run_dir: Path = MODEL_RUN_DIR, artifact_dir: Path = ARTIFACT_DIR) -> list[dict[str, Any]]:
    ensemble = pd.read_csv(model_run_dir / "ensemble_metrics.csv")
    rows: list[dict[str, Any]] = []
    for _, row in ensemble.iterrows():
        target = str(row["target"])
        score_name = str(row.get("score", ""))
        model_name = str(row.get("model", ""))
        if target == "y_warning" and score_name == "warning_intensity_score":
            rows.append(
                {
                    "target": "y_warning",
                    "baseline": "final-multimodal-strict-intensity-fusion",
                    "score_name": "warning_intensity_score",
                    "average_precision": _safe_float(row["average_precision"]),
                    "roc_auc": _safe_float(row.get("roc_auc")),
                    "precision": _safe_float(row.get("precision_at_0_5")),
                    "recall": _safe_float(row.get("recall_at_0_5")),
                    "f1": _safe_float(row.get("f1_at_0_5")),
                    "prevalence": _safe_float(row.get("prevalence")),
                }
            )
        if target == "y_risk" and score_name == "state_score" and "state_score" in model_name:
            rows.append(
                {
                    "target": "y_risk",
                    "baseline": "final-multimodal-state-score",
                    "score_name": "state_score",
                    "average_precision": _safe_float(row["average_precision"]),
                    "roc_auc": _safe_float(row.get("roc_auc")),
                    "precision": _safe_float(row.get("precision_at_0_5")),
                    "recall": _safe_float(row.get("recall_at_0_5")),
                    "f1": _safe_float(row.get("f1_at_0_5")),
                    "prevalence": _safe_float(row.get("prevalence")),
                }
            )
    baseline = pd.read_csv(model_run_dir / "baseline_comparison.csv")
    keep_cols = [
        "target",
        "baseline",
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "prevalence",
    ]
    keep = baseline.loc[baseline["target"].isin(["y_warning", "y_risk"]), [col for col in keep_cols if col in baseline.columns]].copy()
    rows.extend(keep.to_dict("records"))
    enriched = _enrich_baseline_table(pd.DataFrame(rows), model_run_dir, artifact_dir)
    return enriched.to_dict("records")


def _model_card(report: dict[str, Any], metrics: pd.DataFrame) -> str:
    row = metrics.iloc[0].to_dict() if not metrics.empty else {}
    threshold = report.get("alert_layer", {})
    return "\n".join(
        [
            "# Финальная модель dashboard",
            "",
            "**Финальная модель:** мультимодальная strict-intensity fusion.",
            "",
            "Модель использует логи, KPI и маски доступности модальностей. Основной score для дашборда — `model_score_raw` "
            "на основе `warning_intensity_score`; диагностический strict risk-channel — `risk_score_raw` / `state_score`.",
            "",
            "## Alert-layer",
            "",
            f"- Порог наблюдения: `{threshold.get('watch_threshold', float('nan')):.4f}`",
            f"- Порог тревоги: `{threshold.get('alert_threshold', float('nan')):.4f}`",
            f"- Подтверждение тревоги: `{int(threshold.get('min_consecutive_windows', 2))}` окна подряд",
            "",
            "## Run-level replay metrics",
            "",
            f"- Failure-run recall: `{_safe_float(row.get('failure_run_recall')):.3f}`",
            f"- Precision по run-alert: `{_safe_float(row.get('run_alert_precision')):.3f}`",
            f"- False-alert normal rate: `{_safe_float(row.get('false_alert_run_rate')):.3f}`",
            f"- Alert density: `{_safe_float(row.get('alert_density')):.3f}`",
            f"- Median lead time, сек: `{_safe_float(row.get('median_lead_time_sec')):.1f}`",
            "",
            "## Ограничения",
            "",
            "Это лабораторный replay-прототип. Он не является production-ready системой и не должен трактоваться как идеальный бинарный оракул аварий. "
            "Его назначение — честно показать динамику раннего предупреждения, вклад модальностей и ошибки alert-layer.",
            "",
        ]
    )


def export_bundle(
    model_run_dir: str | Path = MODEL_RUN_DIR,
    contract_path: str | Path = STRICT_CONTRACT_PATH,
    out_dir: str | Path = OUT_DIR,
) -> dict[str, Any]:
    model_run_dir = Path(model_run_dir)
    contract_path = Path(contract_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred, validation = _load_strict_predictions(model_run_dir)
    raw = _load_strict_contract(contract_path)
    thresholds = _choose_thresholds(validation)

    replay = _merge_predictions_with_contract(pred, raw)
    replay = _add_dashboard_kpis(replay)
    replay["binary_warning_score"] = pd.to_numeric(
        replay["warning_score"] if "warning_score" in replay.columns else pd.Series(np.nan, index=replay.index),
        errors="coerce",
    ).clip(0.0, 1.0)
    replay["binary_risk_score"] = pd.to_numeric(
        replay["risk_score"] if "risk_score" in replay.columns else pd.Series(np.nan, index=replay.index),
        errors="coerce",
    ).clip(0.0, 1.0)
    replay["warning_intensity_score_raw"] = pd.to_numeric(replay["warning_intensity_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    replay["model_score_raw"] = replay["warning_intensity_score_raw"]
    replay["model_score_rank"] = _rank_calibrate_score(replay["model_score_raw"], validation["warning_intensity_score"])
    replay["model_score"] = replay["model_score_raw"]
    replay["alert_score"] = replay["model_score_raw"]
    replay["warning_score"] = replay["warning_intensity_score_raw"]
    replay["strict_state_score"] = pd.to_numeric(replay["state_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    replay["risk_score_raw"] = replay["strict_state_score"]
    replay["risk_score"] = replay["risk_score_raw"]
    replay["time_to_failure_sec"] = pd.to_numeric(replay.get("strict_lead_sec", np.nan), errors="coerce")
    replay["model_mode"] = "final_multimodal_strict_intensity"
    replay["model_name"] = "Мультимодальная strict-intensity fusion"
    replay["score_source"] = "model_store"
    replay["source_bundle"] = str(model_run_dir.relative_to(ROOT)) if model_run_dir.is_relative_to(ROOT) else str(model_run_dir)
    replay = _add_fusion_diagnostics(replay)
    replay = _apply_alert_layer(replay, thresholds)
    replay = _add_incident_markers(replay)
    replay = _mark_demo_cases(replay)

    events = _run_events(replay)
    metrics = _metrics_summary(replay, events)
    strict_rows = _strict_metric_rows(model_run_dir, out_dir.parent)
    baseline_table = pd.DataFrame(strict_rows)
    quality_report = _bundle_quality_report(replay, validation, thresholds, events)

    report = {
        "model_name": "Мультимодальная strict-intensity fusion",
        "model_family": "final_multimodal_strict_intensity_fusion",
        "run_id": model_run_dir.name,
        "source_dir": str(model_run_dir.relative_to(ROOT)) if model_run_dir.is_relative_to(ROOT) else str(model_run_dir),
        "contract_path": str(contract_path.relative_to(ROOT)) if contract_path.is_relative_to(ROOT) else str(contract_path),
        "contract_mode": "strict_explicit_only_gap10",
        "primary_dashboard_score": "model_score_raw",
        "diagnostic_risk_channel": "state_score",
        "score_source": "model_store",
        "proxy_fallback_windows": 0,
        "alert_layer": thresholds,
        "score_semantics": quality_report["score_semantics"],
        "bundle_quality_report": quality_report,
        "run_level_metrics": metrics.to_dict("records"),
        "strict_research_metrics": strict_rows,
        "baseline_diagnostics": [],
    }

    replay.to_csv(out_dir / "replay_bundle.csv.gz", index=False, compression="gzip")
    replay.to_csv(out_dir / "dashboard_replay_store.csv.gz", index=False, compression="gzip")
    events.to_csv(out_dir / "alert_events.csv", index=False)
    events.to_csv(out_dir / "product_case_catalog.csv", index=False)
    metrics.to_csv(out_dir / "metrics_summary.csv", index=False)
    metrics.to_csv(out_dir / "alert_run_metrics.csv", index=False)
    baseline_table.to_csv(out_dir / "baseline_comparison.csv", index=False)
    baseline_table.to_csv(out_dir / "alert_model_comparison.csv", index=False)
    pd.DataFrame(strict_rows).to_csv(out_dir / "strict_research_summary.csv", index=False)
    _write_json(out_dir / "dashboard_model_report.json", report)
    _write_json(out_dir / "product_acceptance_report.json", report)
    _write_json(out_dir / "bundle_quality_report.json", quality_report)
    (out_dir / "model_card_product.md").write_text(_model_card(report, metrics), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export final dashboard replay bundle from model artifacts.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR.relative_to(ROOT)))
    parser.add_argument("--model-run-dir", default=None)
    parser.add_argument("--contract-path", default=None)
    args = parser.parse_args()
    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = ROOT / artifact_dir
    model_run_dir = Path(args.model_run_dir) if args.model_run_dir else artifact_dir / "model" / "final_multimodal_strict_intensity_gap10"
    contract_path = Path(args.contract_path) if args.contract_path else artifact_dir / "contract" / "strict_intensity_contract.csv.gz"
    out_dir = artifact_dir / "bundle"
    report = export_bundle(model_run_dir=model_run_dir, contract_path=contract_path, out_dir=out_dir)
    print(json.dumps({"status": "ok", "out_dir": str(out_dir), "model": report["model_name"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
