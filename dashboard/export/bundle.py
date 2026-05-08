from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def _rank_calibrate_score(values: pd.Series, validation_values: pd.Series) -> pd.Series:
    train = pd.to_numeric(validation_values, errors="coerce").dropna().to_numpy(dtype=float)
    if train.size == 0:
        return pd.to_numeric(values, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    train = np.sort(train)
    score = pd.to_numeric(values, errors="coerce").fillna(float(np.nanmedian(train))).to_numpy(dtype=float)
    ranks = np.searchsorted(train, score, side="right") / float(train.size)
    return pd.Series(ranks, index=values.index, dtype=float).clip(0.0, 1.0)


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

    def prefer_long(frame: pd.DataFrame, min_span_sec: float = 120.0) -> pd.DataFrame:
        if frame.empty or "span_sec" not in frame.columns:
            return frame
        long = frame.loc[pd.to_numeric(frame["span_sec"], errors="coerce").fillna(0.0).ge(min_span_sec)].copy()
        return long

    if not early_runs.empty:
        run_scores = out.groupby("run_id").agg(
            max_score=("model_score", "max"),
            log_share=("fusion_log_share", "mean"),
            kpi_share=("fusion_kpi_share", "mean"),
            both=("has_both_modalities", "mean"),
            start=("window_ts", "min"),
            end=("window_ts", "max"),
        )
        run_scores["span_sec"] = (
            pd.to_datetime(run_scores["end"], utc=True, errors="coerce")
            - pd.to_datetime(run_scores["start"], utc=True, errors="coerce")
        ).dt.total_seconds()
        early_ids = set(early_runs["run_id"].astype(str))
        all_early_ids = set(early_runs_all["run_id"].astype(str))
        failure_ids = set(
            events.loc[
                events["scenario_type"].astype(str).eq("failure")
                & pd.to_numeric(events.get("n_windows", 0), errors="coerce").fillna(0).ge(8),
                "run_id",
            ].astype(str)
        )
        candidates = run_scores.loc[run_scores.index.astype(str).isin(early_ids)].copy()
        all_candidates = run_scores.loc[run_scores.index.astype(str).isin(all_early_ids)].copy()
        failure_candidates = run_scores.loc[run_scores.index.astype(str).isin(failure_ids)].copy()
        both_failure = failure_candidates.loc[failure_candidates["both"].ge(0.5)].copy()

        def pick_pool(primary: pd.DataFrame, fallback: pd.DataFrame, min_span_sec: float = 120.0) -> pd.DataFrame:
            primary_long = prefer_long(primary, min_span_sec=min_span_sec)
            if not primary_long.empty:
                return primary_long
            fallback_long = prefer_long(fallback, min_span_sec=min_span_sec)
            if not fallback_long.empty:
                return fallback_long
            return primary if not primary.empty else fallback

        both_primary = candidates.loc[candidates["both"].ge(0.5)].copy()
        both_primary_long = prefer_long(both_primary)
        if not both_primary_long.empty:
            both = both_primary_long
        elif not both_primary.empty:
            both = both_primary
        else:
            both = pick_pool(both_primary, both_failure)
        if not both.empty:
            both["_balance"] = 1.0 - (both["log_share"] - 0.5).abs() * 2.0
            assign_first("multimodal_success", both.sort_values(["span_sec", "_balance", "max_score"], ascending=[False, False, False]))
        kpi_primary = candidates.loc[candidates["kpi_share"].gt(candidates["log_share"])].copy()
        kpi_fallback = failure_candidates.loc[failure_candidates["kpi_share"].gt(failure_candidates["log_share"])].copy()
        kpi_driven = pick_pool(kpi_primary, kpi_fallback if not kpi_fallback.empty else failure_candidates)
        if not kpi_driven.empty:
            assign_first("kpi_driven_success", kpi_driven.sort_values(["span_sec", "kpi_share", "max_score"], ascending=[False, False, False]))
        log_primary = candidates.loc[candidates["log_share"].gt(candidates["kpi_share"])].copy()
        log_fallback = failure_candidates.loc[failure_candidates["log_share"].gt(failure_candidates["kpi_share"])].copy()
        log_driven = pick_pool(log_primary, log_fallback if not log_fallback.empty else failure_candidates)
        if not log_driven.empty:
            assign_first("log_driven_success", log_driven.sort_values(["span_sec", "log_share", "max_score"], ascending=[False, False, False]))
        if "log_driven_success" not in set(out["demo_case_kind"].astype(str)) and not all_candidates.empty:
            fallback = prefer_long(all_candidates)
            if fallback.empty:
                fallback = all_candidates
            assign_first("log_driven_success", fallback.sort_values(["span_sec", "log_share", "max_score"], ascending=[False, False, False]))
        if "kpi_driven_success" not in set(out["demo_case_kind"].astype(str)) and not all_candidates.empty:
            fallback = prefer_long(all_candidates)
            if fallback.empty:
                fallback = all_candidates
            assign_first("kpi_driven_success", fallback.sort_values(["span_sec", "kpi_share", "max_score"], ascending=[False, False, False]))

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
                "model": MODEL_BASELINE_LABEL,
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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = ROOT / "dashboard" / "artifacts"
DEFAULT_PROTOCOL_RUN_DIR = DEFAULT_ARTIFACT_DIR / "model_runs" / "protocol_fixed_hades" / "final_protocol_fixed_hades"
DEFAULT_MODEL_DIR = DEFAULT_ARTIFACT_DIR / "model" / "protocol_fixed_efficient_hades"
DEFAULT_BUNDLE_DIR = DEFAULT_ARTIFACT_DIR / "bundle"
DEFAULT_CONTRACT_PATH = DEFAULT_ARTIFACT_DIR / "contract" / "strict_intensity_contract.csv.gz"
DEFAULT_PROTOCOL_BASELINE_METRICS = DEFAULT_ARTIFACT_DIR / "reports" / "protocol_fixed_baselines" / "baseline_protocol_metrics.csv"
MODEL_DISPLAY_NAME = "Облегченный мультимодальный контур предупреждения"
MODEL_FAMILY = "lightweight_multimodal_warning_contour"
MODEL_BASELINE_LABEL = MODEL_DISPLAY_NAME

DEFAULT_SPLIT_PROTOCOL = "stratified_by_workload"
DEFAULT_VARIANT = "efficient_hades"
DEFAULT_SEED = 43
PRIMARY_TARGET = "horizon_300s"
SECONDARY_TARGET = "horizon_120s"
AGGRESSIVE_TARGET = "horizon_60s"
RISK_TARGET = "horizon_30s"

TARGETS = [
    "y_warning",
    "y_risk",
    "y_anomaly",
    "horizon_30s",
    "horizon_60s",
    "horizon_120s",
    "horizon_300s",
]

COPY_FILES = [
    "comparison_metrics.csv",
    "workload_metrics.csv",
    "alert_level_metrics.csv",
    "feature_manifest.json",
    "latency_and_export_report.json",
    "metrics_summary.json",
    "run_config.json",
    "split_summary.csv",
    "train_manifest.json",
    "EXPERIMENT_CARD.md",
]


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def protocol_run_exists(protocol_run_dir: str | Path = DEFAULT_PROTOCOL_RUN_DIR) -> bool:
    path = Path(protocol_run_dir)
    return (path / "test_predictions.csv.gz").exists() and (path / "validation_predictions.csv.gz").exists()


def _load_contract(contract_path: Path) -> pd.DataFrame:
    if not contract_path.exists():
        raise FileNotFoundError(f"Missing strict contract: {contract_path}")
    frame = pd.read_csv(contract_path, low_memory=False)
    frame["run_id"] = frame["run_id"].astype(str)
    frame["window_ts"] = pd.to_datetime(frame["window_ts"], utc=True, errors="coerce")
    if "window_end" in frame.columns:
        frame["window_end"] = pd.to_datetime(frame["window_end"], utc=True, errors="coerce")
    if "event_anchor" in frame.columns:
        frame["event_anchor"] = pd.to_datetime(frame["event_anchor"], utc=True, errors="coerce")
    return frame.dropna(subset=["run_id", "window_ts"]).drop_duplicates(["run_id", "window_ts"], keep="last")


def _load_protocol_predictions(protocol_run_dir: Path, split_name: str) -> pd.DataFrame:
    path = protocol_run_dir / f"{split_name}_predictions.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing protocol predictions: {path}")
    frame = pd.read_csv(path, low_memory=False)
    frame["run_id"] = frame["run_id"].astype(str)
    frame["window_ts"] = pd.to_datetime(frame["window_ts"], utc=True, errors="coerce")
    frame["window_end"] = pd.to_datetime(frame["window_end"], utc=True, errors="coerce")
    return frame.dropna(subset=["run_id", "window_ts", "target_name"])


def _filter_predictions(
    frame: pd.DataFrame,
    split_protocol: str,
    variant: str,
    seed: int,
) -> pd.DataFrame:
    out = frame.loc[
        frame["split_protocol"].astype(str).eq(split_protocol)
        & frame["variant"].astype(str).eq(variant)
        & pd.to_numeric(frame["seed"], errors="coerce").astype("Int64").eq(seed)
    ].copy()
    if out.empty:
        raise ValueError(
            "No protocol predictions for "
            f"split_protocol={split_protocol!r}, variant={variant!r}, seed={seed!r}"
        )
    return out


def _pivot_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        "row_index",
        "run_id",
        "workload",
        "scenario",
        "scenario_type",
        "failure_family",
        "anchor_policy",
        "window_ts",
        "window_end",
        "split",
        "original_split",
        "split_protocol",
        "variant",
        "seed",
        "pred_split",
    ]
    id_cols = [col for col in id_cols if col in frame.columns]
    values = (
        frame.groupby(id_cols + ["target_name"], dropna=False, as_index=False)
        .agg(score=("score", "last"), target=("target", "last"))
        .copy()
    )
    scores = values.pivot(index=id_cols, columns="target_name", values="score").reset_index()
    labels = values.pivot(index=id_cols, columns="target_name", values="target").reset_index()
    scores.columns = [f"score_{col}" if col not in id_cols else col for col in scores.columns]
    labels.columns = [f"target_{col}" if col not in id_cols else col for col in labels.columns]
    merged = scores.merge(labels, on=id_cols, how="left")
    for target in TARGETS:
        score_col = f"score_{target}"
        label_col = f"target_{target}"
        if score_col not in merged.columns:
            merged[score_col] = np.nan
        if label_col not in merged.columns:
            merged[label_col] = np.nan
        merged[score_col] = pd.to_numeric(merged[score_col], errors="coerce").clip(0.0, 1.0)
        merged[label_col] = pd.to_numeric(merged[label_col], errors="coerce")
    return merged


def _merge_with_contract(predictions: pd.DataFrame, contract: pd.DataFrame) -> pd.DataFrame:
    pred = predictions.rename(
        columns={
            "split": "protocol_prediction_split",
            "original_split": "protocol_original_split",
            "window_end": "protocol_window_end",
            "workload": "protocol_workload",
            "scenario": "protocol_scenario",
            "scenario_type": "protocol_scenario_type",
            "failure_family": "protocol_failure_family",
            "anchor_policy": "protocol_anchor_policy",
        }
    )
    merged = contract.merge(pred, on=["run_id", "window_ts"], how="right")
    for base, fallback in [
        ("workload", "protocol_workload"),
        ("scenario", "protocol_scenario"),
        ("scenario_type", "protocol_scenario_type"),
        ("failure_family", "protocol_failure_family"),
        ("anchor_policy", "protocol_anchor_policy"),
    ]:
        if base not in merged.columns:
            merged[base] = merged.get(fallback, "")
        elif fallback in merged.columns:
            merged[base] = merged[base].fillna(merged[fallback])
    if "window_end" not in merged.columns:
        merged["window_end"] = merged.get("protocol_window_end")
    merged["window_end"] = pd.to_datetime(merged["window_end"], utc=True, errors="coerce")
    protocol_end = pd.to_datetime(merged.get("protocol_window_end"), utc=True, errors="coerce")
    merged["window_end"] = merged["window_end"].fillna(protocol_end)
    merged["window_end"] = merged["window_end"].fillna(merged["window_ts"] + pd.to_timedelta(10, unit="s"))
    if "event_anchor" in merged.columns:
        merged["event_anchor"] = pd.to_datetime(merged["event_anchor"], utc=True, errors="coerce")
    if "split" not in merged.columns:
        merged["split"] = merged.get("pred_split", merged.get("protocol_prediction_split", "test"))
    merged["contract_split"] = merged.get("split", "")
    merged["split"] = merged.get("pred_split", merged.get("protocol_prediction_split", merged["split"]))
    return merged


def _load_selected_metrics(
    protocol_run_dir: Path,
    split_protocol: str,
    variant: str,
    seed: int,
    split: str = "test",
) -> pd.DataFrame:
    path = protocol_run_dir / "comparison_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    metrics = pd.read_csv(path)
    mask = (
        metrics["split_protocol"].astype(str).eq(split_protocol)
        & metrics["variant"].astype(str).eq(variant)
        & pd.to_numeric(metrics["seed"], errors="coerce").astype("Int64").eq(seed)
        & metrics["runtime"].astype(str).eq("torch_fp32")
        & metrics["threshold_policy"].astype(str).eq("validation_best_f1")
    )
    if "split" in metrics.columns:
        mask &= metrics["split"].astype(str).eq(split)
    return metrics.loc[mask].copy()


def _threshold_map(metrics: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if metrics.empty:
        return out
    for _, row in metrics.iterrows():
        target = str(row.get("target", ""))
        value = _safe_float(row.get("threshold"))
        if target and np.isfinite(value):
            out[target] = float(np.clip(value, 0.0, 1.0))
    return out


def _score_or_default(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).clip(0.0, 1.0)


def _add_score_channels(frame: pd.DataFrame, validation: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    out["model_score_raw_300s"] = _score_or_default(out, "score_horizon_300s")
    out["model_score_120s"] = _score_or_default(out, "score_horizon_120s")
    out["model_score_60s"] = _score_or_default(out, "score_horizon_60s")
    out["risk_score_30s"] = _score_or_default(out, "score_horizon_30s")
    out["warning_score"] = _score_or_default(out, "score_y_warning", default=float(out["model_score_raw_300s"].median()))
    out["risk_score_raw"] = out["risk_score_30s"]
    out["risk_score"] = out["risk_score_raw"]
    out["anomaly_score"] = _score_or_default(out, "score_y_anomaly", default=float(out["warning_score"].median()))
    out["model_score_raw"] = out["model_score_raw_300s"]
    out["model_score"] = out["model_score_raw"]
    out["alert_score"] = out["model_score_raw"]
    out["model_score_rank"] = _rank_calibrate_score(out["model_score_raw"], validation["model_score_raw_300s"])
    out["time_to_failure_sec"] = pd.to_numeric(out.get("strict_lead_sec", np.nan), errors="coerce")
    if out["time_to_failure_sec"].isna().all() and "event_anchor" in out.columns:
        anchor = pd.to_datetime(out["event_anchor"], utc=True, errors="coerce")
        window_end = pd.to_datetime(out["window_end"], utc=True, errors="coerce")
        out["time_to_failure_sec"] = (anchor - window_end).dt.total_seconds()

    out["threshold_horizon_300s"] = thresholds.get(PRIMARY_TARGET, 0.5)
    out["threshold_horizon_120s"] = thresholds.get(SECONDARY_TARGET, 0.5)
    out["threshold_horizon_60s"] = thresholds.get(AGGRESSIVE_TARGET, 0.5)
    out["threshold_horizon_30s"] = thresholds.get(RISK_TARGET, thresholds.get("y_risk", 0.5))
    out["threshold_warning"] = thresholds.get("y_warning", 0.5)
    out["threshold_risk"] = thresholds.get("y_risk", out["threshold_horizon_30s"].iloc[0] if len(out) else 0.5)
    out["watch_threshold"] = out["threshold_horizon_300s"]
    out["alert_threshold"] = out["threshold_horizon_300s"]
    out["alert_confirmation_threshold_120s"] = out["threshold_horizon_120s"]
    out["alert_confirmation_threshold_60s"] = out["threshold_horizon_60s"]
    return out


def _apply_protocol_alert_layer(frame: pd.DataFrame, min_consecutive: int = 2, cooldown_windows: int = 3) -> pd.DataFrame:
    out = frame.sort_values(["run_id", "window_ts"]).copy()
    h300 = pd.to_numeric(out["model_score_raw_300s"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    h60 = pd.to_numeric(out["model_score_60s"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    watch_threshold = pd.to_numeric(out["threshold_horizon_300s"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    confirm_threshold = pd.to_numeric(out["threshold_horizon_60s"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    watch = h300.ge(watch_threshold)
    confirm = h60.ge(confirm_threshold)

    confirmed_now = pd.Series(False, index=out.index)
    alert_active = pd.Series(False, index=out.index)
    min_consecutive = max(1, int(min_consecutive))
    cooldown_windows = max(0, int(cooldown_windows))
    for _, idx in out.groupby("run_id", sort=False).groups.items():
        run_confirm = confirm.loc[idx].astype(int)
        rolling = run_confirm.rolling(min_consecutive, min_periods=min_consecutive).sum()
        current = rolling.ge(min_consecutive)
        active_values: list[bool] = []
        remaining = 0
        for flag in current.to_numpy(dtype=bool):
            if flag and remaining == 0:
                remaining = cooldown_windows + 1
            active = remaining > 0
            active_values.append(active)
            if remaining > 0:
                remaining -= 1
        active_series = pd.Series(active_values, index=idx)
        confirmed_now.loc[idx] = current.to_numpy()
        alert_active.loc[idx] = active_series.to_numpy()

    incident_now = pd.Series(False, index=out.index)
    if {"event_anchor", "window_ts", "window_end"}.issubset(out.columns):
        anchor = pd.to_datetime(out["event_anchor"], utc=True, errors="coerce")
        start = pd.to_datetime(out["window_ts"], utc=True, errors="coerce")
        end = pd.to_datetime(out["window_end"], utc=True, errors="coerce")
        incident_now = anchor.notna() & start.le(anchor) & end.ge(anchor)

    out["alert_fired"] = alert_active.astype(int)
    out["alert_confirmed_now"] = confirmed_now.astype(int)
    started = pd.Series(False, index=out.index)
    for _, idx in out.groupby("run_id", sort=False).groups.items():
        fired = alert_active.loc[idx].astype(int)
        started.loc[idx] = fired.eq(1) & fired.shift(1, fill_value=0).eq(0)
    out["alert_started"] = started.astype(int)
    out["incident_marker"] = incident_now.astype(int)
    out["alert_state"] = np.select(
        [incident_now, alert_active, watch],
        ["incident", "alert", "watch"],
        default="normal",
    )
    out["pred_state"] = np.select(
        [out["alert_state"].isin(["alert", "incident"]), out["alert_state"].eq("watch")],
        ["risk", "warning"],
        default="normal",
    )
    return out


def _add_fusion_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    log_signal = (
        0.55 * pd.to_numeric(out.get("macro_error_count", 0), errors="coerce").fillna(0.0)
        + 0.25 * pd.to_numeric(out.get("macro_warn_count", 0), errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(out.get("macro_keyword_count", 0), errors="coerce").fillna(0.0)
    )
    if float(log_signal.max()) > 0:
        log_signal = np.log1p(log_signal) / np.log1p(float(log_signal.max()))
    else:
        log_signal = pd.Series(0.0, index=out.index)
    kpi_signal = pd.concat(
        [
            pd.to_numeric(out.get("model_score_120s", 0), errors="coerce").fillna(0.0),
            pd.to_numeric(out.get("risk_score_30s", 0), errors="coerce").fillna(0.0),
        ],
        axis=1,
    ).max(axis=1)
    log_available = pd.to_numeric(out.get("log_available", 0), errors="coerce").fillna(0.0).clip(0, 1)
    kpi_available = pd.to_numeric(out.get("kpi_available", 0), errors="coerce").fillna(0.0).clip(0, 1)
    log_signal = log_signal * log_available
    kpi_signal = kpi_signal * kpi_available
    denom = (log_signal + kpi_signal).replace(0.0, np.nan)
    share_available = denom.notna() & log_available.gt(0) & kpi_available.gt(0)
    out["log_only_score"] = log_signal.clip(0.0, 1.0)
    out["kpi_only_score"] = kpi_signal.clip(0.0, 1.0)
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
        ["protocol log+kpi diagnostic", "log+kpi; weak branch signal", "log-only fallback", "kpi-only fallback"],
        default="missing modalities",
    )
    return out


def _prepare_replay_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["model_mode"] = MODEL_FAMILY
    out["model_name"] = MODEL_DISPLAY_NAME
    out["score_source"] = "model_store"
    out["source_bundle"] = "protocol_fixed_cuda_h60"
    out["primary_dashboard_score"] = PRIMARY_TARGET
    out["secondary_dashboard_score"] = SECONDARY_TARGET
    out["aggressive_dashboard_score"] = AGGRESSIVE_TARGET
    out["short_risk_score"] = RISK_TARGET
    for col in ["y_warning", "y_risk", "y_anomaly"]:
        if col not in out.columns and f"target_{col}" in out.columns:
            out[col] = pd.to_numeric(out[f"target_{col}"], errors="coerce").fillna(0).astype(int)
    return out


def _metrics_records(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    if metrics.empty:
        return []
    keep = [
        "target",
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "false_positives",
        "missed_anomalies",
        "threshold",
        "prevalence",
        "torch_latency_ms_per_sample",
    ]
    rows = metrics[[col for col in keep if col in metrics.columns]].copy()
    rows["baseline"] = MODEL_BASELINE_LABEL
    rows["score_name"] = rows["target"].map(lambda value: f"score_{value}")
    rows["threshold_policy"] = "validation_best_f1"
    return rows.to_dict("records")


def _baseline_records(path: Path = DEFAULT_PROTOCOL_BASELINE_METRICS) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    required = {"split", "threshold_policy", "target", "baseline", "average_precision", "roc_auc", "precision", "recall", "f1"}
    if not required.issubset(frame.columns):
        return []
    frame = frame.loc[
        frame["split"].astype(str).eq("test")
        & frame["threshold_policy"].astype(str).eq("validation_best_f1")
    ].copy()
    if frame.empty:
        return []
    keep = [
        "target",
        "baseline",
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "false_positives",
        "missed_anomalies",
        "threshold",
        "prevalence",
        "runtime",
        "threshold_policy",
    ]
    out = frame[[col for col in keep if col in frame.columns]].copy()
    out["score_name"] = out["target"].map(lambda value: f"baseline_score_{value}")
    return out.to_dict("records")


def _diagnostics(metrics: pd.DataFrame, protocol_run_dir: Path, split_protocol: str, variant: str, seed: int) -> list[dict[str, Any]]:
    if metrics.empty:
        return []
    current = _load_selected_metrics(protocol_run_dir, "current", variant, seed, split="test")
    rows: list[dict[str, Any]] = []
    for target in ["y_warning", "y_risk", "horizon_120s", "horizon_300s"]:
        central_rows = metrics.loc[metrics["target"].astype(str).eq(target)]
        if central_rows.empty:
            continue
        central = central_rows.iloc[0]
        baseline_rows = current.loc[current["target"].astype(str).eq(target)]
        best = baseline_rows.iloc[0] if not baseline_rows.empty else None
        central_ap = _safe_float(central.get("average_precision"))
        best_ap = _safe_float(best.get("average_precision")) if best is not None else np.nan
        delta = central_ap - best_ap if np.isfinite(central_ap) and np.isfinite(best_ap) else np.nan
        verdict = "лучше baseline" if np.isfinite(delta) and delta > 0.02 else ("хуже baseline" if np.isfinite(delta) and delta < -0.02 else "паритет")
        rows.append(
            {
                "target": target,
                "central_average_precision": central_ap,
                "central_roc_auc": _safe_float(central.get("roc_auc")),
                "central_f1": _safe_float(central.get("f1")),
                "central_recall": _safe_float(central.get("recall")),
                "central_precision": _safe_float(central.get("precision")),
                "random_reference_ap": _safe_float(central.get("prevalence")),
                "best_baseline": "current split / same architecture",
                "best_baseline_average_precision": best_ap,
                "best_baseline_roc_auc": _safe_float(best.get("roc_auc")) if best is not None else np.nan,
                "best_baseline_f1": _safe_float(best.get("f1")) if best is not None else np.nan,
                "best_baseline_recall": _safe_float(best.get("recall")) if best is not None else np.nan,
                "delta_vs_best_baseline_ap": delta,
                "verdict": verdict,
            }
        )
    return rows


def _quality_report(
    replay: pd.DataFrame,
    validation: pd.DataFrame,
    thresholds: dict[str, float],
    events: pd.DataFrame,
    runtime_slots: dict[str, Any],
) -> dict[str, Any]:
    state_counts = replay.get("alert_state", pd.Series(dtype=str)).fillna("normal").astype(str).value_counts().to_dict()
    return {
        "score_semantics": {
            "model_score_raw": "primary horizon_300s score; not a calibrated incident probability",
            "model_score_rank": "rank of horizon_300s score against validation windows",
            "model_score_120s": "working early-warning channel",
            "model_score_60s": "aggressive early channel; noisier",
            "risk_score_30s": "short-horizon risk channel; keep separate from warning/anomaly",
            "alert_state": "alert-layer state; not the raw model score and not the incident fact",
        },
        "score_distribution": {
            "test_horizon_300s": _series_summary(replay.get("model_score_raw_300s", pd.Series(dtype=float))),
            "test_horizon_120s": _series_summary(replay.get("model_score_120s", pd.Series(dtype=float))),
            "test_horizon_60s": _series_summary(replay.get("model_score_60s", pd.Series(dtype=float))),
            "test_horizon_30s": _series_summary(replay.get("risk_score_30s", pd.Series(dtype=float))),
            "validation_horizon_300s": _series_summary(validation.get("model_score_raw_300s", pd.Series(dtype=float))),
        },
        "alert_layer": {
            "watch_target": PRIMARY_TARGET,
            "watch_threshold": thresholds.get(PRIMARY_TARGET),
            "alert_confirmation_target": AGGRESSIVE_TARGET,
            "alert_confirmation_threshold": thresholds.get(AGGRESSIVE_TARGET),
            "min_consecutive_windows": 2,
            "cooldown_windows": 3,
        },
        "alert_state_counts": {str(key): int(value) for key, value in state_counts.items()},
        "runtime_slots": runtime_slots,
        "run_event_counts": events.get("event_type", pd.Series(dtype=str)).fillna("").astype(str).value_counts().to_dict(),
        "warnings": [
            "ONNX/INT8 slots are reserved; selected full checkpoint must be copied before true online inference.",
            "model_score_raw is a ranking/alert score, not an incident probability.",
            "horizon_30s/y_risk is a separate short-horizon task and should not be averaged into warning.",
        ],
    }


def _model_card(report: dict[str, Any], metrics: pd.DataFrame) -> str:
    selected = metrics.loc[metrics["target"].isin([PRIMARY_TARGET, SECONDARY_TARGET, AGGRESSIVE_TARGET, RISK_TARGET])].copy()
    metric_lines = []
    for _, row in selected.iterrows():
        metric_lines.append(
            f"- `{row['target']}`: F1={_safe_float(row.get('f1')):.3f}, "
            f"Precision={_safe_float(row.get('precision')):.3f}, Recall={_safe_float(row.get('recall')):.3f}, "
            f"AP={_safe_float(row.get('average_precision')):.3f}, ROC-AUC={_safe_float(row.get('roc_auc')):.3f}"
        )
    runtime = report.get("runtime", {})
    return "\n".join(
        [
            "# Финальная модель dashboard",
            "",
            f"**Модель:** {MODEL_DISPLAY_NAME}.",
            "",
            "Архитектурный путь сохраняет HADES-like смысл: log sequence modeling, metric time-series segment modeling, cross-modal fusion и multi-head early-warning targets. Название подчеркивает прикладной и облегченный характер реализации, а не заявку на новую фундаментальную архитектуру.",
            "",
            "## Основные каналы",
            "",
            "- `model_score_raw` = `horizon_300s`: главный демонстрационный early-warning score.",
            "- `model_score_120s`: основной рабочий early-warning канал.",
            "- `model_score_60s`: агрессивный ранний канал, ожидаемо шумнее.",
            "- `risk_score_30s`: отдельный short-horizon risk канал.",
            "",
            "## Test metrics выбранного checkpoint",
            "",
            *metric_lines,
            "",
            "## Runtime",
            "",
            f"- Torch checkpoint: `{runtime.get('torch_state', {}).get('status', 'unknown')}`",
            f"- ONNX FP32: `{runtime.get('onnx_fp32', {}).get('status', 'reserved')}`",
            f"- ONNX INT8: `{runtime.get('onnx_int8', {}).get('status', 'reserved')}`",
            "",
            "ONNX/INT8 оставлены как совместимый runtime-слот для слабых CPU-only систем. До переноса полного checkpoint из обучающей машины dashboard работает от сохраненного replay bundle, а не от online forward pass.",
            "",
            "## Ограничения",
            "",
            "Score модели не является вероятностью сбоя. Alert-layer отделен от фактического incident marker. Пороги выбраны по validation и не подбираются по test.",
            "",
        ]
    )


def _copy_support_files(protocol_run_dir: Path, model_out_dir: Path) -> None:
    model_out_dir.mkdir(parents=True, exist_ok=True)
    for name in COPY_FILES:
        src = protocol_run_dir / name
        if src.exists():
            shutil.copy2(src, model_out_dir / name)


def _promote_onnx_exports(protocol_run_dir: Path, model_out_dir: Path, split_protocol: str, variant: str, seed: int) -> dict[str, str]:
    model_out_dir.mkdir(parents=True, exist_ok=True)
    split_name = split_protocol.replace(":", "_")
    export_dirs = [
        protocol_run_dir / "exports" / split_name / f"{variant}_seed{seed}",
        protocol_run_dir / "exports" / f"{variant}_seed{seed}",
    ]
    mapping = {
        "model_fp32.onnx": "model.onnx",
        "model_int8.onnx": "model.int8.onnx",
    }
    promoted: dict[str, str] = {}
    for src_name, dst_name in mapping.items():
        for export_dir in export_dirs:
            src = export_dir / src_name
            if src.exists():
                dst = model_out_dir / dst_name
                shutil.copy2(src, dst)
                promoted[dst_name] = _relative(src)
                break
    return promoted


def _runtime_slots(protocol_run_dir: Path, model_out_dir: Path, split_protocol: str, variant: str, seed: int) -> dict[str, Any]:
    expected_state = protocol_run_dir / "models" / split_protocol / f"{variant}_seed{seed}.pt"
    local_state = model_out_dir / "model.pt"
    onnx_fp32 = model_out_dir / "model.onnx"
    onnx_int8 = model_out_dir / "model.int8.onnx"
    if expected_state.exists():
        shutil.copy2(expected_state, local_state)
        torch_status = "available"
    elif local_state.exists():
        torch_status = "available"
    else:
        torch_status = "missing_checkpoint_copy_required"
    runtime = {
        "torch_state": {
            "path": _relative(local_state),
            "source_expected": _relative(expected_state),
            "status": torch_status,
        },
        "onnx_fp32": {
            "path": _relative(onnx_fp32),
            "status": "available" if onnx_fp32.exists() else "reserved_not_exported",
        },
        "onnx_int8": {
            "path": _relative(onnx_int8),
            "status": "available" if onnx_int8.exists() else "reserved_not_quantized",
            "quantization_policy": "dynamic INT8 for Linear/attention; static INT8 optional with representative calibration set",
        },
    }
    export_dir = protocol_run_dir / "exports" / split_protocol.replace(":", "_") / f"{variant}_seed{seed}"
    if (export_dir / "model_fp32.onnx").exists():
        runtime["onnx_fp32"]["source_expected"] = _relative(export_dir / "model_fp32.onnx")
    if (export_dir / "model_int8.onnx").exists():
        runtime["onnx_int8"]["source_expected"] = _relative(export_dir / "model_int8.onnx")
    return runtime


def export_protocol_fixed_bundle(
    protocol_run_dir: str | Path = DEFAULT_PROTOCOL_RUN_DIR,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    out_dir: str | Path = DEFAULT_BUNDLE_DIR,
    model_out_dir: str | Path = DEFAULT_MODEL_DIR,
    split_protocol: str = DEFAULT_SPLIT_PROTOCOL,
    variant: str = DEFAULT_VARIANT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    protocol_run_dir = Path(protocol_run_dir)
    contract_path = Path(contract_path)
    out_dir = Path(out_dir)
    model_out_dir = Path(model_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_out_dir.mkdir(parents=True, exist_ok=True)

    contract = _load_contract(contract_path)
    test_long = _filter_predictions(_load_protocol_predictions(protocol_run_dir, "test"), split_protocol, variant, seed)
    val_long = _filter_predictions(_load_protocol_predictions(protocol_run_dir, "validation"), split_protocol, variant, seed)
    test = _merge_with_contract(_pivot_predictions(test_long), contract)
    validation = _merge_with_contract(_pivot_predictions(val_long), contract)
    metrics = _load_selected_metrics(protocol_run_dir, split_protocol, variant, seed, split="test")
    thresholds = _threshold_map(metrics)
    validation = _add_score_channels(validation, validation=pd.DataFrame({"model_score_raw_300s": validation.get("score_horizon_300s", pd.Series(dtype=float))}), thresholds=thresholds)
    replay = _add_score_channels(test, validation=validation, thresholds=thresholds)
    replay = _prepare_replay_frame(replay)
    replay = _add_dashboard_kpis(replay)
    replay = _add_fusion_diagnostics(replay)
    replay = _apply_protocol_alert_layer(replay, min_consecutive=2, cooldown_windows=3)
    replay = _mark_demo_cases(replay)

    events = _run_events(replay)
    run_metrics = _metrics_summary(replay, events)
    if not run_metrics.empty:
        run_metrics.loc[:, "model"] = "protocol-fixed-efficient-hades"

    _copy_support_files(protocol_run_dir, model_out_dir)
    promoted_onnx = _promote_onnx_exports(protocol_run_dir, model_out_dir, split_protocol, variant, seed)
    runtime_slots = _runtime_slots(protocol_run_dir, model_out_dir, split_protocol, variant, seed)
    try:
        from dashboard.runtime.live_inference import build_live_inference_slice, copy_live_inference_slice

        live_forward = build_live_inference_slice(
            model_out_dir,
            contract_path=contract_path,
            replay_frame=replay,
            output_dir=model_out_dir,
            force=True,
            demo_only=True,
            input_builder="raw_rebuild",
        )
        copy_live_inference_slice(model_out_dir, out_dir)
    except Exception as exc:
        live_forward = {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    metric_records = _metrics_records(metrics)
    baseline_records = _baseline_records()
    diagnostics = _diagnostics(metrics, protocol_run_dir, split_protocol, variant, seed)
    quality_report = _quality_report(replay, validation, thresholds, events, runtime_slots)

    report = {
        "model_name": MODEL_DISPLAY_NAME,
        "model_family": MODEL_FAMILY,
        "run_id": protocol_run_dir.name,
        "source_dir": _relative(protocol_run_dir),
        "contract_path": _relative(contract_path),
        "contract_mode": "strict_intensity_gap10_protocol_fixed",
        "split_protocol": split_protocol,
        "variant": variant,
        "seed": int(seed),
        "primary_dashboard_score": "model_score_raw",
        "primary_dashboard_target": PRIMARY_TARGET,
        "secondary_dashboard_target": SECONDARY_TARGET,
        "aggressive_dashboard_target": AGGRESSIVE_TARGET,
        "short_risk_target": RISK_TARGET,
        "score_source": "model_store",
        "proxy_fallback_windows": 0,
        "alert_layer": quality_report["alert_layer"],
        "score_semantics": quality_report["score_semantics"],
        "bundle_quality_report": quality_report,
        "run_level_metrics": run_metrics.to_dict("records"),
        "strict_research_metrics": metric_records,
        "protocol_baseline_metrics": baseline_records,
        "baseline_diagnostics": diagnostics,
        "runtime": runtime_slots,
        "promoted_onnx_exports": promoted_onnx,
        "live_forward": live_forward,
        "onnx_int8_plan": {
            "status": runtime_slots["onnx_int8"]["status"],
            "reason": "Reserved for CPU-only deployment; export requires selected Torch checkpoint.",
            "expected_files": {
                "fp32": runtime_slots["onnx_fp32"]["path"],
                "int8": runtime_slots["onnx_int8"]["path"],
            },
        },
        "training_verdict": {
            "message": "Protocol-fixed split and workload-aware metric preprocessing are the winning data protocol; architecture is kept HADES-like.",
        },
    }

    replay.to_csv(out_dir / "replay_bundle.csv.gz", index=False, compression="gzip")
    replay.to_csv(out_dir / "dashboard_replay_store.csv.gz", index=False, compression="gzip")
    events.to_csv(out_dir / "alert_events.csv", index=False)
    run_metrics.to_csv(out_dir / "metrics_summary.csv", index=False)
    baseline_table = pd.DataFrame(metric_records + baseline_records)
    baseline_table.to_csv(out_dir / "baseline_comparison.csv", index=False)
    _write_json(out_dir / "dashboard_model_report.json", report)
    _write_json(out_dir / "bundle_quality_report.json", quality_report)
    (out_dir / "model_card_product.md").write_text(_model_card(report, metrics), encoding="utf-8")

    replay.to_csv(model_out_dir / "replay_bundle.csv.gz", index=False, compression="gzip")
    replay.to_csv(model_out_dir / "dashboard_replay_store.csv.gz", index=False, compression="gzip")
    baseline_table.to_csv(model_out_dir / "baseline_comparison.csv", index=False)
    _write_json(model_out_dir / "dashboard_model_report.json", report)
    _write_json(model_out_dir / "bundle_quality_report.json", quality_report)
    _write_json(model_out_dir / "protocol_fixed_model_manifest.json", report)
    (model_out_dir / "model_card_product.md").write_text(_model_card(report, metrics), encoding="utf-8")
    return report


def export_bundle(
    model_run_dir: str | Path | None = None,
    contract_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    mode: str = "protocol_fixed",
    protocol_run_dir: str | Path | None = None,
    protocol_model_out_dir: str | Path | None = None,
    split_protocol: str = DEFAULT_SPLIT_PROTOCOL,
    variant: str = DEFAULT_VARIANT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if mode not in {"auto", "protocol_fixed"}:
        raise ValueError("The compact final exporter only supports protocol_fixed mode.")
    selected_protocol_run = DEFAULT_PROTOCOL_RUN_DIR if protocol_run_dir is None else protocol_run_dir
    selected_contract = DEFAULT_CONTRACT_PATH if contract_path is None else contract_path
    selected_out = DEFAULT_BUNDLE_DIR if out_dir is None else out_dir
    selected_model = DEFAULT_MODEL_DIR if protocol_model_out_dir is None else protocol_model_out_dir
    return export_protocol_fixed_bundle(
        protocol_run_dir=selected_protocol_run,
        contract_path=selected_contract,
        out_dir=selected_out,
        model_out_dir=selected_model,
        split_protocol=split_protocol,
        variant=variant,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the final protocol-fixed dashboard replay bundle.")
    parser.add_argument("--mode", choices=["auto", "protocol_fixed"], default="protocol_fixed")
    parser.add_argument("--protocol-run-dir", default=str(DEFAULT_PROTOCOL_RUN_DIR.relative_to(ROOT)))
    parser.add_argument("--contract-path", default=str(DEFAULT_CONTRACT_PATH.relative_to(ROOT)))
    parser.add_argument("--out-dir", default=str(DEFAULT_BUNDLE_DIR.relative_to(ROOT)))
    parser.add_argument("--model-out-dir", "--protocol-model-out-dir", dest="model_out_dir", default=str(DEFAULT_MODEL_DIR.relative_to(ROOT)))
    parser.add_argument("--split-protocol", default=DEFAULT_SPLIT_PROTOCOL)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    protocol_run_dir = Path(args.protocol_run_dir)
    contract_path = Path(args.contract_path)
    out_dir = Path(args.out_dir)
    model_out_dir = Path(args.model_out_dir)
    if not protocol_run_dir.is_absolute():
        protocol_run_dir = ROOT / protocol_run_dir
    if not contract_path.is_absolute():
        contract_path = ROOT / contract_path
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if not model_out_dir.is_absolute():
        model_out_dir = ROOT / model_out_dir

    report = export_bundle(
        protocol_run_dir=protocol_run_dir,
        contract_path=contract_path,
        out_dir=out_dir,
        protocol_model_out_dir=model_out_dir,
        mode=args.mode,
        split_protocol=args.split_protocol,
        variant=args.variant,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "mode": "protocol_fixed",
                "out_dir": _relative(out_dir),
                "model_out_dir": _relative(model_out_dir),
                "model": report["model_name"],
                "runtime": report["runtime"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
