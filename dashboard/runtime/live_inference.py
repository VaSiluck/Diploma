from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dashboard.modeling import hades_blocks as eh
from dashboard.modeling.protocol_fixed_hades import (
    DEFAULT_TARGETS,
    MultiHeadCrossModalHades,
    ProtocolConfig,
    ProtocolDataset,
    batch_to_device,
    load_metric_frames,
    make_protocol_frame,
    make_target_matrix,
    transform_metric_segments,
)
from dashboard.runtime.contract import APP_ROOT, RAW_DATA_ROOT, resolve_dashboard_model_dir


SLICE_ARRAY_FILE = "live_inference_slice.npz"
SLICE_INDEX_FILE = "live_inference_index.csv.gz"
SLICE_MANIFEST_FILE = "live_inference_manifest.json"
SCORE_SOURCE = "live_forward"
LONG_DEMO_RUN_ID = "demo/long_test_stream"
LONG_DEMO_CASE_ID = "long_test_stream"
LONG_DEMO_WINDOWS = 180
LONG_DEMO_PREFIX_WINDOWS = 110


@dataclass
class LiveInferenceStatus:
    ready: bool
    model_dir: Path
    checkpoint_path: Path | None
    slice_path: Path
    index_path: Path
    manifest_path: Path
    reason: str
    manifest: dict[str, Any]


@dataclass
class LiveInferenceResult:
    frame: pd.DataFrame
    elapsed_ms: float
    rows_scored: int
    status: LiveInferenceStatus


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


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _path_from_report(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = APP_ROOT / path
    return path.resolve()


def _optional_manifest_path(manifest: dict[str, Any], key: str) -> Path | None:
    value = str(manifest.get(key, "")).strip() if isinstance(manifest, dict) else ""
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = APP_ROOT / path
    return path.resolve()


def _checkpoint_path(model_dir: Path, report: dict[str, Any] | None = None) -> Path | None:
    report = report or _read_json(model_dir / "dashboard_model_report.json")
    local = model_dir / "model.pt"
    if local.exists():
        return local
    runtime = report.get("runtime", {}) if isinstance(report, dict) else {}
    torch_state = runtime.get("torch_state", {}) if isinstance(runtime, dict) else {}
    from_report = _path_from_report(torch_state.get("path")) if isinstance(torch_state, dict) else None
    if from_report is not None and from_report.exists():
        return from_report
    promoted = APP_ROOT / "dashboard" / "artifacts" / "model" / "protocol_fixed_efficient_hades" / "model.pt"
    return promoted if promoted.exists() else None


def inspect_live_inference(model_dir: str | Path | None = None) -> LiveInferenceStatus:
    resolved = resolve_dashboard_model_dir(model_dir)
    manifest_path = resolved / SLICE_MANIFEST_FILE
    manifest = _read_json(manifest_path)
    checkpoint = _checkpoint_path(resolved)
    slice_path = _first_existing(resolved / SLICE_ARRAY_FILE, *([_optional_manifest_path(manifest, "slice_path")] if _optional_manifest_path(manifest, "slice_path") is not None else [])) or (resolved / SLICE_ARRAY_FILE)
    index_path = _first_existing(resolved / SLICE_INDEX_FILE, *([_optional_manifest_path(manifest, "index_path")] if _optional_manifest_path(manifest, "index_path") is not None else [])) or (resolved / SLICE_INDEX_FILE)

    missing = []
    if checkpoint is None:
        missing.append("model.pt")
    if not slice_path.exists():
        missing.append(SLICE_ARRAY_FILE)
    if not index_path.exists():
        missing.append(SLICE_INDEX_FILE)
    if not manifest_path.exists():
        missing.append(SLICE_MANIFEST_FILE)
    ready = not missing
    reason = "ready" if ready else "missing: " + ", ".join(missing)
    return LiveInferenceStatus(
        ready=ready,
        model_dir=resolved,
        checkpoint_path=checkpoint,
        slice_path=slice_path,
        index_path=index_path,
        manifest_path=manifest_path,
        reason=reason,
        manifest=manifest,
    )


def load_live_timeline_skeleton(model_dir: str | Path | None = None) -> pd.DataFrame:
    status = inspect_live_inference(model_dir)
    if not status.index_path.exists():
        return pd.DataFrame()
    out = pd.read_csv(status.index_path, low_memory=False)
    if out.empty:
        return out
    out["run_id"] = out["run_id"].astype(str)
    for col in [
        "window_ts",
        "window_end",
        "run_start",
        "run_end",
        "stream_start",
        "stream_end",
        "event_anchor",
        "fail_start",
        "source_window_ts",
    ]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    score_cols = [
        "score_y_warning",
        "score_y_risk",
        "score_y_anomaly",
        "score_horizon_30s",
        "score_horizon_60s",
        "score_horizon_120s",
        "score_horizon_300s",
        "model_score_raw_300s",
        "model_score_120s",
        "model_score_60s",
        "risk_score_30s",
        "warning_score",
        "risk_score_raw",
        "risk_score",
        "anomaly_score",
        "early_warning_score_raw",
        "alert_score_raw",
        "model_score_raw",
        "model_score",
        "alert_score",
        "model_score_rank",
        "confidence",
    ]
    for col in score_cols:
        if col in out.columns:
            out[col] = np.nan
    out["score_source"] = "live_pending"
    out["alert_state"] = "normal"
    out["pred_state"] = "normal"
    for col in ["alert_fired", "alert_started", "alert_confirmed_now", "incident_marker"]:
        out[col] = 0
    return out.sort_values(["run_id", "window_ts"]).reset_index(drop=True)


def _cfg_from_model_dir(model_dir: Path, contract_path: str | Path | None = None) -> ProtocolConfig:
    artifact_dir = (_checkpoint_path(model_dir) or model_dir / "model.pt").parent
    payload = _read_json(artifact_dir / "run_config.json") or _read_json(model_dir / "run_config.json")
    if not payload:
        raise RuntimeError(f"Cannot build live inference slice: run_config.json was not found for {model_dir}")
    payload = dict(payload)
    payload["root"] = str(APP_ROOT)
    payload["contract_path"] = str(Path(contract_path).resolve() if contract_path is not None else APP_ROOT / "dashboard" / "artifacts" / "contract" / "strict_intensity_contract.csv.gz")
    payload["output_dir"] = str(model_dir / "_live_build_cache")
    payload["device"] = "cpu"
    payload.setdefault("targets", list(DEFAULT_TARGETS))
    return ProtocolConfig(**payload)


def _load_replay_frame(model_dir: Path, replay_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    if replay_frame is not None and not replay_frame.empty:
        out = replay_frame.copy()
    else:
        path = _first_existing(model_dir / "replay_bundle.csv.gz", model_dir / "dashboard_replay_store.csv.gz", model_dir / "replay_bundle.csv")
        if path is None:
            raise RuntimeError(f"Cannot build live inference slice: replay bundle not found in {model_dir}")
        out = pd.read_csv(path, low_memory=False)
    if out.empty:
        raise RuntimeError("Cannot build live inference slice from an empty replay frame.")
    out["run_id"] = out["run_id"].astype(str)
    out["window_ts"] = pd.to_datetime(out["window_ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["run_id", "window_ts"]).sort_values(["run_id", "window_ts"]).reset_index(drop=True)
    out["live_replay_pos"] = np.arange(len(out), dtype=np.int64)
    return out


def _stored_score(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default).clip(0.0, 1.0)
    return pd.Series(default, index=frame.index, dtype=float)


def _make_long_test_demo_stream(replay: pd.DataFrame, max_windows: int = LONG_DEMO_WINDOWS) -> pd.DataFrame:
    """Build one deterministic long stream from held-out replay rows.

    The dashboard uses this frame as a single online demo scenario. The rows
    still come from test replay, but timestamps are rewritten into one
    continuous stream so the UI does not look like short pre-failure clips
    stitched onto an unrelated raw-data span.
    """
    if replay.empty:
        return replay.copy()

    candidates = replay.copy()
    if "split" in candidates.columns:
        test_rows = candidates.loc[candidates["split"].fillna("").astype(str).str.lower().eq("test")].copy()
        if not test_rows.empty:
            candidates = test_rows

    normal = candidates.copy()
    for col in ["y_warning", "y_risk", "y_anomaly", "incident_marker"]:
        if col in normal.columns:
            normal = normal.loc[pd.to_numeric(normal[col], errors="coerce").fillna(0).eq(0)].copy()

    positive_mask = pd.Series(False, index=candidates.index)
    for col in ["y_warning", "y_risk", "y_anomaly"]:
        if col in candidates.columns:
            positive_mask |= pd.to_numeric(candidates[col], errors="coerce").fillna(0).gt(0)
    failure_runs = candidates.loc[positive_mask, "run_id"].dropna().astype(str).unique().tolist() if "run_id" in candidates.columns else []

    prefix_len = min(int(LONG_DEMO_PREFIX_WINDOWS), max(1, int(max_windows) - 30))
    tail_len = max(1, int(max_windows) - prefix_len)

    preferred_calm_runs = [
        "als/normal4",
        "als/normal2",
        "als/normal1",
        "repartition/normal6",
        "svd/resourcemanager_suspended",
        "terasort/net_killed",
        "scan/normal7",
        "rf/normal5",
        "nweight/late_sub",
        "nweight/nodemanager_killed",
        "web_pagerank/cpu_stress",
        "sort/normal2",
        "scan/normal6",
        "graph_pagerank/normal7",
    ]
    calm_parts: list[pd.DataFrame] = []
    if "run_id" in normal.columns:
        for rid in preferred_calm_runs:
            if sum(len(part) for part in calm_parts) >= prefix_len:
                break
            group = normal.loc[normal["run_id"].astype(str).eq(rid)].sort_values("window_ts").copy()
            if group.empty:
                continue
            calm_parts.append(group.head(max(0, prefix_len - sum(len(part) for part in calm_parts))))
    normal_pool = normal.drop(index=pd.concat(calm_parts).index, errors="ignore").copy() if calm_parts else normal.copy()
    if {"kpi_available", "log_available"}.issubset(normal_pool.columns):
        both = normal_pool.loc[
            pd.to_numeric(normal_pool["kpi_available"], errors="coerce").fillna(0).eq(1)
            & pd.to_numeric(normal_pool["log_available"], errors="coerce").fillna(0).eq(1)
        ].copy()
        if len(both) >= max(40, min(prefix_len, len(normal_pool)) // 2):
            normal_pool = both
    normal_pool["_calm_score"] = np.maximum.reduce(
        [
            _stored_score(normal_pool, "model_score_raw", 0.0).to_numpy(),
            _stored_score(normal_pool, "score_horizon_300s", 0.0).to_numpy(),
            _stored_score(normal_pool, "score_horizon_120s", 0.0).to_numpy(),
            _stored_score(normal_pool, "score_horizon_60s", 0.0).to_numpy(),
        ]
    )
    normal_prefix = pd.concat(calm_parts, ignore_index=False) if calm_parts else pd.DataFrame(columns=normal.columns)
    normal_prefix = pd.concat(
        [
            normal_prefix,
            normal_pool.sort_values(["_calm_score", "run_id", "window_ts"])
            .head(max(0, prefix_len - len(normal_prefix)))
            .drop(columns=["_calm_score"], errors="ignore")
            .copy(),
        ],
        ignore_index=False,
    )
    if len(normal_prefix) < prefix_len and len(normal) > len(normal_prefix):
        extra = normal.drop(index=normal_prefix.index, errors="ignore").copy()
        extra["_calm_score"] = np.maximum.reduce(
            [
                _stored_score(extra, "model_score_raw", 0.0).to_numpy(),
                _stored_score(extra, "score_horizon_300s", 0.0).to_numpy(),
                _stored_score(extra, "score_horizon_120s", 0.0).to_numpy(),
                _stored_score(extra, "score_horizon_60s", 0.0).to_numpy(),
            ]
        )
        normal_prefix = pd.concat(
            [normal_prefix, extra.sort_values(["_calm_score", "run_id", "window_ts"]).head(prefix_len - len(normal_prefix)).drop(columns=["_calm_score"], errors="ignore")],
            ignore_index=False,
        )

    tail_parts: list[pd.DataFrame] = []
    preferred_runs = [
        "svd/datanode_suspended",
        "graph_pagerank/nodemanager_suspended",
        "bayes/datanode_suspended",
        "bayes/datanode_killed",
        "join/mq_stress",
        "terasort/worker_suspended",
        "svd/slow_stress",
        "als/net_killed",
        "wordcount/master_killed",
    ]
    ordered_failure_runs = [rid for rid in preferred_runs if rid in failure_runs] + [rid for rid in failure_runs if rid not in set(preferred_runs)]
    for rid in ordered_failure_runs:
        if sum(len(part) for part in tail_parts) >= tail_len:
            break
        group = candidates.loc[candidates["run_id"].astype(str).eq(rid)].sort_values("window_ts").copy()
        if group.empty:
            continue
        tail_parts.append(group.head(max(0, tail_len - sum(len(part) for part in tail_parts))))
    failure_tail = pd.concat(tail_parts, ignore_index=False) if tail_parts else candidates.loc[positive_mask].head(tail_len).copy()

    selected = pd.concat([normal_prefix, failure_tail], ignore_index=False).head(max_windows).copy().reset_index(drop=True)
    if selected.empty:
        return replay.copy()

    selected["source_run_id"] = selected.get("run_id", pd.Series("", index=selected.index)).astype(str)
    selected["source_workload"] = selected.get("workload", pd.Series("", index=selected.index)).astype(str)
    selected["source_scenario"] = selected.get("scenario", pd.Series("", index=selected.index)).astype(str)
    selected["source_window_ts"] = pd.to_datetime(selected.get("window_ts"), utc=True, errors="coerce")

    base_ts = pd.to_datetime(selected["source_window_ts"], utc=True, errors="coerce").dropna()
    start_ts = pd.Timestamp(base_ts.min()).floor("10s") if not base_ts.empty else pd.Timestamp("2024-01-01T12:00:00Z")
    micro_sec = 10
    if "micro_window_sec" in selected.columns:
        valid_micro = pd.to_numeric(selected["micro_window_sec"], errors="coerce").dropna()
        if not valid_micro.empty:
            micro_sec = int(max(1, round(float(valid_micro.median()))))
    step = pd.to_timedelta(micro_sec, unit="s")
    synthetic_ts = pd.date_range(start=start_ts, periods=len(selected), freq=step)
    synthetic_end = synthetic_ts + step

    run_start = synthetic_ts[0]
    run_end = synthetic_end[-1]
    selected["run_id"] = LONG_DEMO_RUN_ID
    selected["group_id"] = LONG_DEMO_RUN_ID
    selected["failure_family"] = np.where(np.arange(len(selected)) < len(normal_prefix), "no_failure", "demo_failure")
    selected["scenario_type"] = np.where(np.arange(len(selected)) < len(normal_prefix), "normal_prefix", "failure_tail")
    selected["split"] = "test"
    selected["window_ts"] = synthetic_ts
    selected["window_end"] = synthetic_end
    selected["window_index"] = np.arange(len(selected), dtype=np.int64)
    selected["run_start"] = run_start
    selected["run_end"] = run_end
    selected["stream_start"] = run_start
    selected["stream_end"] = run_end
    selected["stream_span_source"] = "synthetic_continuous_test_stream"
    transition_idx = min(len(selected) - 1, len(normal_prefix))
    incident_idx = min(len(selected) - 1, transition_idx + max(18, min(36, len(selected) - transition_idx - 1)))
    incident_ts = pd.Timestamp(synthetic_ts[incident_idx])
    selected["event_anchor"] = incident_ts
    selected["fail_start"] = incident_ts
    selected["anchor_policy"] = "explicit_log_signal"
    selected["strict_lead_sec"] = (incident_ts - pd.Series(synthetic_ts, index=selected.index)).dt.total_seconds()
    selected["minutes_before_fail"] = selected["strict_lead_sec"] / 60.0
    selected["y_warning"] = (np.arange(len(selected)) >= transition_idx).astype(int)
    selected["y_anomaly"] = selected["y_warning"]
    selected["y_risk"] = (np.arange(len(selected)) >= max(transition_idx, incident_idx - 8)).astype(int)
    selected["state"] = np.where(np.arange(len(selected)) >= incident_idx, "incident", np.where(selected["y_warning"].astype(int).eq(1), "warning", "normal"))
    selected["incident_marker"] = ((pd.Series(synthetic_ts, index=selected.index).le(incident_ts)) & (pd.Series(synthetic_end, index=selected.index).gt(incident_ts))).astype(int)
    selected["alert_fired"] = 0
    selected["alert_started"] = 0
    selected["alert_confirmed_now"] = 0
    selected["alert_state"] = "normal"
    selected["pred_state"] = "normal"
    selected["demo_case_kind"] = LONG_DEMO_CASE_ID
    selected["demo_case_id"] = LONG_DEMO_CASE_ID
    selected["demo_stream_window_count"] = len(selected)
    selected["demo_transition_window"] = int(transition_idx)
    selected["demo_incident_window"] = int(incident_idx)
    selected["demo_stream_note"] = "single long test stream: calm prefix, then real failure-window tail scored online by live_forward"
    selected["live_replay_pos"] = np.arange(len(selected), dtype=np.int64)
    return selected.reset_index(drop=True)


def _feature_manifest_for_protocol(model_dir: Path, split_protocol: str) -> dict[str, Any]:
    artifact_dir = (_checkpoint_path(model_dir) or model_dir / "model.pt").parent
    payload = _read_json(artifact_dir / "feature_manifest.json") or _read_json(model_dir / "feature_manifest.json")
    if split_protocol in payload:
        return payload[split_protocol]
    if payload:
        first = next(iter(payload.values()))
        return first if isinstance(first, dict) else {}
    return {}


def _build_raw_metric_segments_for_channels(
    frame: pd.DataFrame,
    metric_frames: dict[str, pd.DataFrame],
    channels: list[str],
    segment_sec: int,
    segment_len: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not channels:
        raise RuntimeError("No metric channels are available for live inference slice.")
    segments = np.full((len(frame), int(segment_len), len(channels)), np.nan, dtype=np.float32)
    masks = np.zeros((len(frame), int(segment_len)), dtype=np.float32)
    for row in frame.itertuples(index=False):
        run_frame = metric_frames.get(str(row.run_id), pd.DataFrame())
        segment, mask = eh.segment_from_run_frame(run_frame, row.window_end, channels, segment_sec, segment_len)
        segments[int(row.row_pos)] = segment
        masks[int(row.row_pos)] = mask
    return segments, masks, {
        "source": "raw_data_kpi_files_unscaled",
        "segment_sec": int(segment_sec),
        "segment_len": int(segment_len),
        "channels": list(channels),
        "channel_count": int(len(channels)),
        "runs_loaded": int(sum(not part.empty for part in metric_frames.values())),
        "mean_valid_steps": float(masks.sum(axis=1).mean()) if len(masks) else 0.0,
    }


def _target_matrix_from_replay(replay: pd.DataFrame, target_names: list[str]) -> np.ndarray:
    cols = []
    for name in target_names:
        if name in replay.columns:
            values = pd.to_numeric(replay[name], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=np.float32)
        else:
            values = np.zeros(len(replay), dtype=np.float32)
        cols.append(values)
    return np.stack(cols, axis=1).astype(np.float32) if cols else np.zeros((len(replay), 0), dtype=np.float32)


def _sysstat_file_span(path: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if not path.exists() or path.name.startswith("."):
        return None, None
    ts_idx: int | None = None
    first: float | None = None
    last: float | None = None
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                if parts[0] == "HostName" and "TimeStamp" in parts:
                    ts_idx = parts.index("TimeStamp")
                    continue
                if ts_idx is None or len(parts) <= ts_idx:
                    continue
                value = pd.to_numeric(parts[ts_idx], errors="coerce")
                if not np.isfinite(value):
                    continue
                value_f = float(value)
                if first is None:
                    first = value_f
                last = value_f
    except Exception:
        return None, None
    if first is None or last is None:
        return None, None
    return pd.to_datetime(first, unit="s", utc=True), pd.to_datetime(last, unit="s", utc=True)


def _raw_stream_span(workload: str, scenario: str, cfg: ProtocolConfig) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str]:
    run_root = RAW_DATA_ROOT / str(workload) / "kpis" / str(scenario) / "instruments"
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for node in cfg.metric_nodes:
        for instrument in cfg.metric_instruments:
            start, end = _sysstat_file_span(run_root / str(node) / str(instrument))
            if start is not None and end is not None:
                starts.append(pd.Timestamp(start))
                ends.append(pd.Timestamp(end))
    if starts and ends:
        return min(starts), max(ends), "raw_kpi_sysstat_span"
    return None, None, "contract_span_fallback"


def _add_raw_stream_span_columns(replay: pd.DataFrame, cfg: ProtocolConfig) -> pd.DataFrame:
    out = replay.copy()
    spans: dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None, str]] = {}
    for run_id, group in out.groupby("run_id", sort=False):
        workload = str(group["workload"].iloc[0]) if "workload" in group.columns and len(group) else ""
        scenario = str(group["scenario"].iloc[0]) if "scenario" in group.columns and len(group) else ""
        spans[str(run_id)] = _raw_stream_span(workload, scenario, cfg)
    stream_start = []
    stream_end = []
    stream_source = []
    for row in out.itertuples(index=False):
        existing_source = str(getattr(row, "stream_span_source", "") or "")
        if existing_source == "synthetic_continuous_test_stream":
            stream_start.append(pd.to_datetime(getattr(row, "stream_start", pd.NaT), utc=True, errors="coerce"))
            stream_end.append(pd.to_datetime(getattr(row, "stream_end", pd.NaT), utc=True, errors="coerce"))
            stream_source.append(existing_source)
            continue
        start, end, source = spans.get(str(row.run_id), (None, None, "contract_span_fallback"))
        fallback_start = pd.to_datetime(getattr(row, "run_start", pd.NaT), utc=True, errors="coerce")
        fallback_end = pd.to_datetime(getattr(row, "run_end", pd.NaT), utc=True, errors="coerce")
        window_start = pd.to_datetime(getattr(row, "window_ts", pd.NaT), utc=True, errors="coerce")
        window_end = pd.to_datetime(getattr(row, "window_end", pd.NaT), utc=True, errors="coerce")
        if start is None or pd.isna(start):
            start = fallback_start if pd.notna(fallback_start) else window_start
        if end is None or pd.isna(end):
            end = fallback_end if pd.notna(fallback_end) else window_end
        stream_start.append(pd.Timestamp(start) if pd.notna(start) else pd.NaT)
        stream_end.append(pd.Timestamp(end) if pd.notna(end) else pd.NaT)
        stream_source.append(source)
    out["stream_start"] = stream_start
    out["stream_end"] = stream_end
    out["stream_span_source"] = stream_source
    return out


def _contract_metric_segments_from_replay(
    replay: pd.DataFrame,
    channels: list[str],
    cfg: ProtocolConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n = len(replay)
    segment_len = int(cfg.metric_segment_len)
    raw = np.zeros((n, segment_len, len(channels)), dtype=np.float32)
    mask = np.ones((n, segment_len), dtype=np.float32)
    if "kpi_available" in replay.columns:
        available = pd.to_numeric(replay["kpi_available"], errors="coerce").fillna(1.0).clip(0, 1).to_numpy(dtype=np.float32)
        mask = mask * available[:, None]

    grid = np.linspace(0.0, 1.0, segment_len, dtype=np.float32)
    used = 0
    def series(column: str, default: Any = 0.0) -> pd.Series:
        if column in replay.columns:
            return pd.to_numeric(replay[column], errors="coerce")
        if isinstance(default, pd.Series):
            return default.copy()
        return pd.Series(default, index=replay.index, dtype=float)

    for channel_idx, channel in enumerate(channels):
        prefix = f"kpi_{channel}__mean_dim__"
        mean = series(prefix + "mean", 0.0)
        last = series(prefix + "last", mean)
        delta = series(prefix + "delta", 0.0)
        std = series(prefix + "std", 0.0)
        slope = series(prefix + "slope", 0.0)
        if any((prefix + suffix) in replay.columns for suffix in ["mean", "last", "delta", "std", "slope"]):
            used += 1
        mean_np = mean.fillna(0.0).to_numpy(dtype=np.float32)
        last_np = last.fillna(mean).fillna(0.0).to_numpy(dtype=np.float32)
        delta_np = delta.fillna(0.0).to_numpy(dtype=np.float32)
        std_np = std.fillna(0.0).abs().to_numpy(dtype=np.float32)
        slope_np = slope.fillna(0.0).to_numpy(dtype=np.float32)
        start_np = np.where(np.isfinite(last_np - delta_np), last_np - delta_np, mean_np)
        seq = start_np[:, None] + (last_np - start_np)[:, None] * grid[None, :]
        seq = seq + 0.15 * std_np[:, None] * np.sin(np.pi * grid[None, :]).astype(np.float32)
        seq = seq + 0.05 * slope_np[:, None] * (grid[None, :] - 0.5)
        raw[:, :, channel_idx] = seq.astype(np.float32)
    raw[mask <= 0.5] = 0.0
    return raw, mask, {
        "source": "strict_contract_aggregate_kpi_columns",
        "segment_len": segment_len,
        "channels": list(channels),
        "channel_count": int(len(channels)),
        "channels_with_contract_features": int(used),
        "builder_note": "Lightweight dashboard slice reconstructs a metric segment from strict contract aggregate columns; raw-data exact rebuild is intentionally kept out of Streamlit.",
    }


def _split_log_text(value: Any, max_events: int) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    parts = [part.strip() for part in re.split(r"(?:\n+|\s{2,}|\s+\|\s+)", text) if part.strip()]
    if len(parts) <= 1:
        words = text.split()
        chunk = max(6, min(18, len(words) // max(1, int(max_events // 2)) or 10))
        parts = [" ".join(words[i : i + chunk]) for i in range(0, len(words), chunk)]
    return parts[: int(max_events)]


def _contract_log_arrays_from_replay(
    replay: pd.DataFrame,
    cfg: ProtocolConfig,
    spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    max_events = int(cfg.max_log_events)
    dim = int(cfg.log_embedding_dim)
    mode = "fasttext_hash" if spec.get("log_embedding") == "fasttext_hash" else "static_hash"
    emb = np.zeros((len(replay), max_events, dim), dtype=np.float32)
    mask = np.zeros((len(replay), max_events), dtype=np.float32)
    selected_counts: list[int] = []
    for row_idx, text in enumerate(replay.get("log_text", pd.Series("", index=replay.index))):
        fragments = _split_log_text(text, max_events)
        selected_counts.append(len(fragments))
        for event_idx, fragment in enumerate(fragments[:max_events]):
            emb[row_idx, event_idx] = eh.hashed_subword_embedding(fragment, dim, mode)
            mask[row_idx, event_idx] = 1.0
    return emb, mask, {
        "source": "strict_contract_log_text",
        "backend": mode,
        "max_log_events": max_events,
        "embedding_dim": dim,
        "mean_selected_events": float(np.mean(selected_counts)) if selected_counts else 0.0,
        "empty_window_share": float(np.mean(np.array(selected_counts) <= 0)) if selected_counts else 1.0,
    }


def _write_contract_feature_slice(
    replay: pd.DataFrame,
    cfg: ProtocolConfig,
    spec: dict[str, Any],
    selected_channels: list[str],
    model_dir: Path,
    output_dir: Path,
    split_protocol: str,
    variant: str,
) -> dict[str, Any]:
    slice_path = output_dir / SLICE_ARRAY_FILE
    index_path = output_dir / SLICE_INDEX_FILE
    manifest_path = output_dir / SLICE_MANIFEST_FILE
    replay = replay.copy().reset_index(drop=True)
    replay["live_replay_pos"] = np.arange(len(replay), dtype=np.int64)
    replay["window_ts"] = pd.to_datetime(replay["window_ts"], utc=True, errors="coerce")
    if "window_end" in replay.columns:
        replay["window_end"] = pd.to_datetime(replay["window_end"], utc=True, errors="coerce")
    replay = _add_raw_stream_span_columns(replay, cfg)

    raw_metric_seg, metric_mask, metric_raw_manifest = _contract_metric_segments_from_replay(replay, selected_channels, cfg)
    metric_frame = replay[["run_id", "workload"]].copy()
    metric_frame["split"] = "train"
    metric_seg, metric_transform_manifest = transform_metric_segments(
        metric_frame,
        raw_metric_seg,
        metric_mask,
        train_idx=np.arange(len(replay), dtype=np.int64),
        normalization=cfg.normalization,
        feature_mode=cfg.metric_feature_mode,
    )
    log_emb, log_mask, log_manifest = _contract_log_arrays_from_replay(replay, cfg, spec)
    target_matrix = _target_matrix_from_replay(replay, list(cfg.targets))

    np.savez_compressed(
        slice_path,
        log_emb=log_emb.astype(np.float32),
        log_mask=log_mask.astype(np.float32),
        metric_seg=metric_seg.astype(np.float32),
        metric_mask=metric_mask.astype(np.float32),
        target_matrix=target_matrix.astype(np.float32),
        row_pos=np.arange(len(replay), dtype=np.int64),
        replay_pos=replay["live_replay_pos"].to_numpy(dtype=np.int64),
    )

    replay.to_csv(index_path, index=False, compression="gzip")
    manifest = {
        "status": "ready",
        "mode": "prepared_contract_feature_live_forward",
        "created_at_unix": time.time(),
        "model_dir": str(model_dir),
        "checkpoint_path": str(_checkpoint_path(model_dir) or ""),
        "slice_path": str(slice_path),
        "index_path": str(index_path),
        "rows": int(len(replay)),
        "runs": int(replay["run_id"].nunique()),
        "demo_only": bool("demo_case_kind" in replay.columns and replay["demo_case_kind"].fillna("").astype(str).ne("").any()),
        "demo_run_id": str(replay["run_id"].iloc[0]) if replay["run_id"].nunique() == 1 else "",
        "demo_case_id": str(replay["demo_case_id"].iloc[0]) if "demo_case_id" in replay.columns and replay["demo_case_id"].nunique() == 1 else "",
        "split_protocol": split_protocol,
        "variant": variant,
        "target_names": list(cfg.targets),
        "shapes": {
            "log_emb": list(log_emb.shape),
            "log_mask": list(log_mask.shape),
            "metric_seg": list(metric_seg.shape),
            "metric_mask": list(metric_mask.shape),
            "target_matrix": list(target_matrix.shape),
        },
        "feature_rebuild": {
            "scope": "dashboard replay rows only",
            "metric": {**metric_raw_manifest, **metric_transform_manifest},
            "log": log_manifest,
            "stream_span": "stream_start/stream_end are read from raw KPI sysstat file boundaries when available",
        },
        "semantics": "Dashboard live mode runs model.forward over prepared strict-contract tensors. It demonstrates the inference step; full raw-data preprocessing remains in the training/export pipeline.",
    }
    _write_json(manifest_path, manifest)
    return manifest


def _align_replay_rows(frame: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    keys = replay[["run_id", "window_ts", "live_replay_pos"]].copy()
    keys["window_ts"] = pd.to_datetime(keys["window_ts"], utc=True, errors="coerce")
    source = frame[["run_id", "window_ts", "row_pos"]].copy()
    source["run_id"] = source["run_id"].astype(str)
    source["window_ts"] = pd.to_datetime(source["window_ts"], utc=True, errors="coerce")
    aligned = keys.merge(source, on=["run_id", "window_ts"], how="left")
    missing = aligned["row_pos"].isna()
    if missing.any():
        sample = aligned.loc[missing, ["run_id", "window_ts"]].head(5).to_dict("records")
        raise RuntimeError(f"Cannot align {int(missing.sum())} replay rows to protocol frame; sample={sample}")
    return aligned.sort_values("live_replay_pos").reset_index(drop=True)


def build_live_inference_slice(
    model_dir: str | Path | None = None,
    contract_path: str | Path | None = None,
    replay_frame: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    force: bool = False,
    demo_only: bool = True,
    input_builder: str = "contract_features",
) -> dict[str, Any]:
    model_dir = resolve_dashboard_model_dir(model_dir)
    output_dir = Path(output_dir).resolve() if output_dir is not None else model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    slice_path = output_dir / SLICE_ARRAY_FILE
    index_path = output_dir / SLICE_INDEX_FILE
    manifest_path = output_dir / SLICE_MANIFEST_FILE
    if not force and slice_path.exists() and index_path.exists() and manifest_path.exists():
        return _read_json(manifest_path)

    cfg = _cfg_from_model_dir(model_dir, contract_path)
    split_protocol = str(cfg.split_protocols[0] if cfg.split_protocols else "stratified_by_workload")
    variant = str(cfg.variants[0] if cfg.variants else "modern_metric")
    spec = eh.VARIANT_SPECS.get(variant, eh.VARIANT_SPECS["modern_metric"])
    feature_manifest = _feature_manifest_for_protocol(model_dir, split_protocol)
    metric_manifest = feature_manifest.get("metric", {}) if isinstance(feature_manifest, dict) else {}
    selected_channels = list(metric_manifest.get("selected_channels") or metric_manifest.get("channels") or [])

    replay = _load_replay_frame(model_dir, replay_frame)
    if input_builder == "contract_features" and demo_only:
        replay = _make_long_test_demo_stream(replay, max_windows=LONG_DEMO_WINDOWS)
    elif demo_only and "demo_case_kind" in replay.columns:
        demo_runs = replay.loc[replay["demo_case_kind"].fillna("").astype(str).ne(""), "run_id"].astype(str).unique()
        if len(demo_runs):
            replay = replay.loc[replay["run_id"].astype(str).isin(set(demo_runs))].copy().reset_index(drop=True)
            replay["live_replay_pos"] = np.arange(len(replay), dtype=np.int64)
    if input_builder == "contract_features":
        return _write_contract_feature_slice(
            replay=replay,
            cfg=cfg,
            spec=spec,
            selected_channels=selected_channels,
            model_dir=model_dir,
            output_dir=output_dir,
            split_protocol=split_protocol,
            variant=variant,
        )
    if input_builder != "raw_rebuild":
        raise ValueError("input_builder must be 'contract_features' or 'raw_rebuild'.")
    contract = eh.load_contract(Path(cfg.contract_path), cfg.max_rows, "y_anomaly")
    full_frame = make_protocol_frame(contract, split_protocol, cfg)
    full_frame = full_frame.sort_values(["run_id", "window_ts", "window_index"]).reset_index(drop=True)
    full_frame["window_ts"] = pd.to_datetime(full_frame["window_ts"], utc=True, errors="coerce")
    full_frame["window_end"] = pd.to_datetime(full_frame["window_end"], utc=True, errors="coerce")

    replay_keys = replay[["run_id", "window_ts"]].copy()
    replay_keys["run_id"] = replay_keys["run_id"].astype(str)
    replay_keys["window_ts"] = pd.to_datetime(replay_keys["window_ts"], utc=True, errors="coerce")
    replay_workloads = set(replay["workload"].dropna().astype(str)) if "workload" in replay.columns else set()
    replay_rows = full_frame.merge(replay_keys.drop_duplicates(), on=["run_id", "window_ts"], how="inner")
    if replay_workloads:
        support_rows = full_frame.loc[
            full_frame["split"].astype(str).eq("train") & full_frame["workload"].astype(str).isin(replay_workloads)
        ].copy()
    else:
        support_rows = full_frame.loc[full_frame["split"].astype(str).eq("train")].copy()
    frame = (
        pd.concat([support_rows, replay_rows], ignore_index=True)
        .drop_duplicates(["run_id", "window_ts"], keep="last")
        .sort_values(["run_id", "window_ts", "window_index"])
        .reset_index(drop=True)
    )
    frame["row_pos"] = np.arange(len(frame), dtype=np.int64)
    train_idx = frame.index[frame["split"].astype(str).eq("train")].to_numpy(dtype=np.int64)
    if len(train_idx) == 0:
        raise RuntimeError("Cannot build live inference slice without train support rows for normalization.")
    aligned = _align_replay_rows(frame, replay)
    row_pos = aligned["row_pos"].to_numpy(dtype=np.int64)

    cache_dir = output_dir / "_live_build_cache"
    metric_frames = load_metric_frames(
        frame,
        raw_root=RAW_DATA_ROOT,
        nodes=cfg.metric_nodes,
        instruments=cfg.metric_instruments,
        cache_dir=cache_dir / "raw_metrics",
    )
    raw_metric_seg, metric_mask, metric_raw_manifest = _build_raw_metric_segments_for_channels(
        frame,
        metric_frames,
        selected_channels,
        segment_sec=cfg.metric_segment_sec,
        segment_len=cfg.metric_segment_len,
    )
    metric_seg, metric_transform_manifest = transform_metric_segments(
        frame,
        raw_metric_seg,
        metric_mask,
        train_idx=train_idx,
        normalization=cfg.normalization,
        feature_mode=cfg.metric_feature_mode,
    )
    log_emb, log_mask, log_manifest = eh.build_log_arrays(
        frame,
        raw_root=RAW_DATA_ROOT,
        train_idx=train_idx,
        max_events=cfg.max_log_events,
        context_sec=cfg.log_context_sec,
        backend=spec["log_embedding"],
        dim=cfg.log_embedding_dim,
        output_dir=output_dir,
        embeddinggemma_model=cfg.embeddinggemma_model,
        cache_dir=cache_dir / "raw_logs",
    )
    target_matrix, target_manifest = make_target_matrix(frame, cfg.targets)

    np.savez_compressed(
        slice_path,
        log_emb=log_emb[row_pos].astype(np.float32),
        log_mask=log_mask[row_pos].astype(np.float32),
        metric_seg=metric_seg[row_pos].astype(np.float32),
        metric_mask=metric_mask[row_pos].astype(np.float32),
        target_matrix=target_matrix[row_pos].astype(np.float32),
        row_pos=row_pos.astype(np.int64),
        replay_pos=aligned["live_replay_pos"].to_numpy(dtype=np.int64),
    )

    keep_cols = [
        "live_replay_pos",
        "run_id",
        "window_ts",
        "window_end",
        "workload",
        "scenario",
        "split",
        "window_index",
        "demo_case_kind",
        "event_anchor",
        "anchor_policy",
        "strict_lead_sec",
        "y_warning",
        "y_risk",
        "y_anomaly",
        "score_y_warning",
        "score_y_risk",
        "score_y_anomaly",
        "score_horizon_30s",
        "score_horizon_60s",
        "score_horizon_120s",
        "score_horizon_300s",
        "model_score_raw",
        "model_score_60s",
        "model_score_120s",
        "risk_score_30s",
        "alert_state",
        "alert_fired",
        "incident_marker",
    ]
    index_frame = replay[[col for col in keep_cols if col in replay.columns]].copy()
    index_frame.to_csv(index_path, index=False, compression="gzip")

    manifest = {
        "status": "ready",
        "mode": "prepared_tensor_live_forward",
        "created_at_unix": time.time(),
        "model_dir": str(model_dir),
        "checkpoint_path": str(_checkpoint_path(model_dir) or ""),
        "slice_path": str(slice_path),
        "index_path": str(index_path),
        "rows": int(len(row_pos)),
        "runs": int(replay["run_id"].nunique()),
        "demo_only": bool(demo_only),
        "support_rows": int(len(frame)),
        "support_runs": int(frame["run_id"].nunique()),
        "support_workloads": sorted(frame["workload"].dropna().astype(str).unique().tolist()),
        "split_protocol": split_protocol,
        "variant": variant,
        "target_names": list(cfg.targets),
        "shapes": {
            "log_emb": list(log_emb[row_pos].shape),
            "log_mask": list(log_mask[row_pos].shape),
            "metric_seg": list(metric_seg[row_pos].shape),
            "metric_mask": list(metric_mask[row_pos].shape),
            "target_matrix": list(target_matrix[row_pos].shape),
        },
        "feature_rebuild": {
            "scope": "train support rows for displayed workloads plus saved replay rows; dashboard scores only saved replay rows",
            "metric": {**metric_raw_manifest, **metric_transform_manifest},
            "log": log_manifest,
            "targets": target_manifest,
        },
        "semantics": "Dashboard live mode loads these prepared tensors, runs model.forward, and applies fixed validation thresholds/alert-layer. It does not train the model or choose thresholds.",
    }
    _write_json(manifest_path, manifest)
    return manifest


def copy_live_inference_slice(src_dir: str | Path, dst_dir: str | Path) -> None:
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for name in [SLICE_ARRAY_FILE, SLICE_INDEX_FILE, SLICE_MANIFEST_FILE]:
        path = src / name
        if path.exists():
            shutil.copy2(path, dst / name)


def _load_checkpoint(status: LiveInferenceStatus) -> dict[str, Any]:
    if status.checkpoint_path is None:
        raise RuntimeError("Live inference checkpoint is not available.")
    payload = torch.load(status.checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError(f"Unsupported checkpoint payload: {status.checkpoint_path}")
    return payload


def _model_from_checkpoint(payload: dict[str, Any], metric_input_dim: int, log_input_dim: int) -> MultiHeadCrossModalHades:
    config = payload.get("config", {}) if isinstance(payload.get("config", {}), dict) else {}
    spec = payload.get("spec", {}) if isinstance(payload.get("spec", {}), dict) else {}
    target_names = list(payload.get("target_names") or config.get("targets") or DEFAULT_TARGETS)
    model = MultiHeadCrossModalHades(
        log_input_dim=int(log_input_dim),
        metric_input_dim=int(metric_input_dim),
        d_model=int(config.get("d_model", 128)),
        heads=int(config.get("heads", 4)),
        log_layers=int(config.get("log_layers", 1)),
        metric_layers=int(config.get("metric_layers", 2)),
        metric_encoder=str(spec.get("metric_encoder", "tiny_time_mixer")),
        target_names=target_names,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def _scores_from_arrays(status: LiveInferenceStatus, batch_size: int = 128) -> tuple[pd.DataFrame, float]:
    payload = _load_checkpoint(status)
    arrays_npz = np.load(status.slice_path)
    arrays = {
        "log_emb": arrays_npz["log_emb"].astype(np.float32),
        "log_mask": arrays_npz["log_mask"].astype(np.float32),
        "metric_seg": arrays_npz["metric_seg"].astype(np.float32),
        "metric_mask": arrays_npz["metric_mask"].astype(np.float32),
        "target_matrix": arrays_npz["target_matrix"].astype(np.float32),
    }
    model = _model_from_checkpoint(
        payload,
        metric_input_dim=int(arrays["metric_seg"].shape[-1]),
        log_input_dim=int(arrays["log_emb"].shape[-1]),
    )
    loader = DataLoader(ProtocolDataset(arrays, np.arange(len(arrays["target_matrix"]))), batch_size=int(batch_size), shuffle=False, num_workers=0)
    scores: list[np.ndarray] = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, torch.device("cpu"))
            logits = model(batch["log_emb"], batch["log_mask"], batch["metric_seg"], batch["metric_mask"])
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    matrix = np.concatenate(scores, axis=0).astype(np.float32) if scores else np.empty((0, 0), dtype=np.float32)
    target_names = list(payload.get("target_names") or DEFAULT_TARGETS)
    frame = pd.DataFrame(matrix, columns=[f"score_{name}" for name in target_names])
    frame["live_replay_pos"] = arrays_npz["replay_pos"].astype(np.int64)
    return frame, elapsed_ms


def _thresholds_from_checkpoint(payload: dict[str, Any], policy: str = "best_f1") -> dict[str, float]:
    raw = payload.get("thresholds", {})
    out: dict[str, float] = {}
    if not isinstance(raw, dict):
        return out
    for target, item in raw.items():
        if isinstance(item, dict):
            value = item.get(policy, item.get("best_f1", 0.5))
        else:
            value = item
        try:
            value_f = float(value)
        except Exception:
            continue
        if np.isfinite(value_f):
            out[str(target)] = float(np.clip(value_f, 0.0, 1.0))
    return out


def _score_or_default(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).clip(0.0, 1.0)


def _add_live_score_channels(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
    alert_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    alert_thresholds = alert_thresholds or thresholds
    out = frame.copy()
    out["model_score_raw_300s"] = _score_or_default(out, "score_horizon_300s")
    out["model_score_120s"] = _score_or_default(out, "score_horizon_120s")
    out["model_score_60s"] = _score_or_default(out, "score_horizon_60s")
    out["risk_score_30s"] = _score_or_default(out, "score_horizon_30s")
    out["warning_score"] = _score_or_default(out, "score_y_warning", default=float(out["model_score_raw_300s"].median()) if len(out) else 0.0)
    out["risk_score_raw"] = _score_or_default(out, "score_y_risk", default=float(out["risk_score_30s"].median()) if len(out) else 0.0)
    out["risk_score"] = out["risk_score_raw"]
    out["anomaly_score"] = _score_or_default(out, "score_y_anomaly", default=float(out["warning_score"].median()) if len(out) else 0.0)
    out["model_score_raw"] = out["model_score_raw_300s"]
    out["early_warning_score_raw"] = out["model_score_raw_300s"]
    out["alert_score_raw"] = out["model_score_raw_300s"]
    out["model_score"] = out["model_score_raw"]
    out["alert_score"] = out["model_score_raw"]
    out["model_score_rank"] = out["model_score_raw"].rank(pct=True).astype(float)
    out["threshold_horizon_300s"] = thresholds.get("horizon_300s", 0.5)
    out["threshold_horizon_120s"] = thresholds.get("horizon_120s", 0.5)
    out["threshold_horizon_60s"] = thresholds.get("horizon_60s", 0.5)
    out["threshold_horizon_30s"] = thresholds.get("horizon_30s", thresholds.get("y_risk", 0.5))
    out["threshold_warning"] = thresholds.get("y_warning", 0.5)
    out["threshold_risk"] = thresholds.get("y_risk", out["threshold_horizon_30s"].iloc[0] if len(out) else 0.5)
    out["watch_threshold"] = out["threshold_horizon_300s"]
    fp_alert = alert_thresholds.get("horizon_300s", max(0.75, thresholds.get("horizon_300s", 0.5)))
    out["alert_threshold"] = min(float(fp_alert), max(0.72, thresholds.get("horizon_300s", 0.5) + 0.12))
    out["alert_confirmation_threshold_120s"] = alert_thresholds.get("horizon_120s", thresholds.get("horizon_120s", 0.5))
    out["alert_confirmation_threshold_60s"] = alert_thresholds.get("horizon_60s", thresholds.get("horizon_60s", 0.5))
    return out


def _apply_alert_layer(frame: pd.DataFrame, min_consecutive: int = 2, cooldown_windows: int = 3) -> pd.DataFrame:
    out = frame.sort_values(["run_id", "window_ts"]).copy()
    h300 = pd.to_numeric(out["model_score_raw_300s"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    h120 = pd.to_numeric(out["model_score_120s"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    h60 = pd.to_numeric(out["model_score_60s"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    watch_threshold = pd.to_numeric(out["threshold_horizon_300s"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    alert_threshold = pd.to_numeric(out["alert_threshold"], errors="coerce").fillna(0.8).clip(0.0, 1.0)
    confirm_threshold_120 = pd.to_numeric(out["alert_confirmation_threshold_120s"], errors="coerce").fillna(0.8).clip(0.0, 1.0)
    confirm_threshold_60 = pd.to_numeric(out["alert_confirmation_threshold_60s"], errors="coerce").fillna(0.8).clip(0.0, 1.0)

    evidence_raw = (0.55 * h300 + 0.25 * h120 + 0.20 * h60).clip(0.0, 1.0)
    smoothed = pd.Series(np.nan, index=out.index, dtype=float)
    for _, idx in out.groupby("run_id", sort=False).groups.items():
        smoothed.loc[idx] = evidence_raw.loc[idx].ewm(alpha=0.22, adjust=False).mean().to_numpy()
    smoothed = smoothed.fillna(evidence_raw).clip(0.0, 1.0)
    out["alert_score_raw"] = evidence_raw
    out["alert_score"] = smoothed
    out["model_score"] = smoothed
    out["confidence"] = smoothed
    out["model_score_rank"] = smoothed.rank(pct=True).astype(float)

    watch = smoothed.ge(watch_threshold)
    strong_primary = smoothed.ge(alert_threshold)
    confirmed_by_near_horizon = h120.ge(confirm_threshold_120) | h60.ge(confirm_threshold_60)
    sustained_early = h300.groupby(out["run_id"], sort=False).rolling(3, min_periods=2).mean().reset_index(level=0, drop=True).ge(alert_threshold)
    confirm = strong_primary & (confirmed_by_near_horizon | sustained_early)

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
        confirmed_now.loc[idx] = current.to_numpy()
        alert_active.loc[idx] = np.array(active_values, dtype=bool)

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
    out["alert_state"] = np.select([incident_now, alert_active, watch], ["incident", "alert", "watch"], default="normal")
    out["pred_state"] = np.select([out["alert_state"].isin(["alert", "incident"]), out["alert_state"].eq("watch")], ["risk", "warning"], default="normal")
    return out.sort_values("live_replay_pos").reset_index(drop=True)


def score_live_inference_slice(model_dir: str | Path | None = None, batch_size: int = 128) -> LiveInferenceResult:
    status = inspect_live_inference(model_dir)
    if not status.ready:
        raise RuntimeError(f"Live inference is not ready: {status.reason}")
    payload = _load_checkpoint(status)
    score_frame, elapsed_ms = _scores_from_arrays(status, batch_size=batch_size)
    index = pd.read_csv(status.index_path, low_memory=False)
    index["window_ts"] = pd.to_datetime(index["window_ts"], utc=True, errors="coerce")
    if "window_end" in index.columns:
        index["window_end"] = pd.to_datetime(index["window_end"], utc=True, errors="coerce")
    if "event_anchor" in index.columns:
        index["event_anchor"] = pd.to_datetime(index["event_anchor"], utc=True, errors="coerce")
    for col in ["run_start", "run_end", "stream_start", "stream_end"]:
        if col in index.columns:
            index[col] = pd.to_datetime(index[col], utc=True, errors="coerce")
    out = index.merge(score_frame, on="live_replay_pos", how="left", suffixes=("_stored", ""))
    thresholds = _thresholds_from_checkpoint(payload, policy="best_f1")
    alert_thresholds = _thresholds_from_checkpoint(payload, policy="fp_budget")
    out = _add_live_score_channels(out, thresholds, alert_thresholds=alert_thresholds)
    out = _apply_alert_layer(out, min_consecutive=2, cooldown_windows=3)
    out["score_source"] = SCORE_SOURCE
    out["live_forward_latency_ms"] = float(elapsed_ms)
    out["live_forward_rows"] = int(len(out))
    return LiveInferenceResult(frame=out, elapsed_ms=float(elapsed_ms), rows_scored=int(len(out)), status=status)


def score_live_run_prefix(
    model_dir: str | Path | None,
    run_id: str,
    max_windows: int,
    batch_size: int = 128,
) -> LiveInferenceResult:
    status = inspect_live_inference(model_dir)
    if not status.ready:
        raise RuntimeError(f"Live inference is not ready: {status.reason}")
    payload = _load_checkpoint(status)
    index = pd.read_csv(status.index_path, low_memory=False)
    index["run_id"] = index["run_id"].astype(str)
    index["window_ts"] = pd.to_datetime(index["window_ts"], utc=True, errors="coerce")
    if "window_end" in index.columns:
        index["window_end"] = pd.to_datetime(index["window_end"], utc=True, errors="coerce")
    if "event_anchor" in index.columns:
        index["event_anchor"] = pd.to_datetime(index["event_anchor"], utc=True, errors="coerce")
    for col in ["run_start", "run_end", "stream_start", "stream_end"]:
        if col in index.columns:
            index[col] = pd.to_datetime(index[col], utc=True, errors="coerce")
    selected = (
        index.loc[index["run_id"].eq(str(run_id))]
        .sort_values("window_ts")
        .head(max(1, int(max_windows)))
        .copy()
        .reset_index(drop=True)
    )
    if selected.empty:
        return LiveInferenceResult(frame=pd.DataFrame(), elapsed_ms=0.0, rows_scored=0, status=status)

    arrays_npz = np.load(status.slice_path)
    replay_pos = arrays_npz["replay_pos"].astype(np.int64)
    pos_by_replay = {int(value): idx for idx, value in enumerate(replay_pos.tolist())}
    array_idx = np.array([pos_by_replay[int(value)] for value in selected["live_replay_pos"].to_numpy(dtype=np.int64)], dtype=np.int64)
    arrays = {
        "log_emb": arrays_npz["log_emb"][array_idx].astype(np.float32),
        "log_mask": arrays_npz["log_mask"][array_idx].astype(np.float32),
        "metric_seg": arrays_npz["metric_seg"][array_idx].astype(np.float32),
        "metric_mask": arrays_npz["metric_mask"][array_idx].astype(np.float32),
        "target_matrix": arrays_npz["target_matrix"][array_idx].astype(np.float32),
    }
    model = _model_from_checkpoint(
        payload,
        metric_input_dim=int(arrays["metric_seg"].shape[-1]),
        log_input_dim=int(arrays["log_emb"].shape[-1]),
    )
    loader = DataLoader(ProtocolDataset(arrays, np.arange(len(arrays["target_matrix"]))), batch_size=int(batch_size), shuffle=False, num_workers=0)
    scores: list[np.ndarray] = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, torch.device("cpu"))
            logits = model(batch["log_emb"], batch["log_mask"], batch["metric_seg"], batch["metric_mask"])
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    matrix = np.concatenate(scores, axis=0).astype(np.float32) if scores else np.empty((0, 0), dtype=np.float32)
    target_names = list(payload.get("target_names") or DEFAULT_TARGETS)
    score_frame = pd.DataFrame(matrix, columns=[f"score_{name}" for name in target_names])
    score_frame["live_replay_pos"] = selected["live_replay_pos"].to_numpy(dtype=np.int64)

    thresholds = _thresholds_from_checkpoint(payload, policy="best_f1")
    alert_thresholds = _thresholds_from_checkpoint(payload, policy="fp_budget")
    out = selected.merge(score_frame, on="live_replay_pos", how="left", suffixes=("_stored", ""))
    out = _add_live_score_channels(out, thresholds, alert_thresholds=alert_thresholds)
    out = _apply_alert_layer(out, min_consecutive=2, cooldown_windows=3)
    out["score_source"] = SCORE_SOURCE
    out["live_forward_latency_ms"] = float(elapsed_ms)
    out["live_forward_rows"] = int(len(out))
    return LiveInferenceResult(frame=out, elapsed_ms=float(elapsed_ms), rows_scored=int(len(out)), status=status)


def apply_live_scores_to_timeline(timeline: pd.DataFrame, live_frame: pd.DataFrame) -> pd.DataFrame:
    if timeline.empty or live_frame.empty:
        return timeline.copy()
    out = timeline.copy()
    out["run_id"] = out["run_id"].astype(str)
    out["window_ts"] = pd.to_datetime(out["window_ts"], utc=True, errors="coerce")
    live = live_frame.copy()
    live["run_id"] = live["run_id"].astype(str)
    live["window_ts"] = pd.to_datetime(live["window_ts"], utc=True, errors="coerce")
    live = live.drop_duplicates(["run_id", "window_ts"], keep="last")
    live_keys = live[["run_id", "window_ts"]].dropna().copy()
    if not live_keys.empty:
        existing = out[["run_id", "window_ts"]].dropna().drop_duplicates()
        missing = live_keys.merge(existing, on=["run_id", "window_ts"], how="left", indicator=True)
        missing = missing.loc[missing["_merge"].eq("left_only"), ["run_id", "window_ts"]]
        if not missing.empty:
            append = live.merge(missing, on=["run_id", "window_ts"], how="inner")
            out = pd.concat([out, append], ignore_index=True, sort=False)
    replace_cols = [
        "score_y_warning",
        "score_y_risk",
        "score_y_anomaly",
        "score_horizon_30s",
        "score_horizon_60s",
        "score_horizon_120s",
        "score_horizon_300s",
        "model_score_raw_300s",
        "model_score_120s",
        "model_score_60s",
        "risk_score_30s",
        "warning_score",
        "risk_score_raw",
        "risk_score",
        "anomaly_score",
        "early_warning_score_raw",
        "alert_score_raw",
        "model_score_raw",
        "model_score",
        "alert_score",
        "model_score_rank",
        "confidence",
        "threshold_horizon_300s",
        "threshold_horizon_120s",
        "threshold_horizon_60s",
        "threshold_horizon_30s",
        "threshold_warning",
        "threshold_risk",
        "watch_threshold",
        "alert_threshold",
        "alert_confirmation_threshold_120s",
        "alert_confirmation_threshold_60s",
        "alert_fired",
        "alert_confirmed_now",
        "alert_started",
        "incident_marker",
        "alert_state",
        "pred_state",
        "score_source",
        "live_forward_latency_ms",
        "live_forward_rows",
        "stream_start",
        "stream_end",
        "stream_span_source",
    ]
    live_cols = ["run_id", "window_ts"] + [col for col in replace_cols if col in live.columns]
    merged = out.merge(live[live_cols], on=["run_id", "window_ts"], how="left", suffixes=("", "__live"))
    matched = merged.get("score_source__live", pd.Series("", index=merged.index)).fillna("").eq(SCORE_SOURCE)
    for col in replace_cols:
        live_col = f"{col}__live"
        if live_col not in merged.columns:
            continue
        if col not in merged.columns:
            merged[col] = np.nan
        merged.loc[matched, col] = merged.loc[matched, live_col]
        merged = merged.drop(columns=[live_col])
    return merged


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build or test prepared live Efficient HADES inference slice.")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--contract-path", default=None)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--all-replay", action="store_true")
    parser.add_argument("--input-builder", default="contract_features", choices=["contract_features", "raw_rebuild"])
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.build:
        manifest = build_live_inference_slice(
            args.model_dir,
            contract_path=args.contract_path,
            force=args.force,
            demo_only=not args.all_replay,
            input_builder=args.input_builder,
        )
        print(json.dumps({"built": True, "rows": manifest.get("rows"), "runs": manifest.get("runs")}, ensure_ascii=False, indent=2))
    if args.score:
        result = score_live_inference_slice(args.model_dir)
        print(json.dumps({"scored": True, "rows": result.rows_scored, "elapsed_ms": result.elapsed_ms}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
