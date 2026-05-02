from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = APP_ROOT / "outputs" / "final_alert_system"

STATE_LABELS = {
    "normal": "НОРМА",
    "watch": "НАБЛЮДЕНИЕ",
    "alert": "ТРЕВОГА",
    "incident": "ИНЦИДЕНТ",
}

EVENT_LABELS = {
    "early_alert": "тревога до сбоя",
    "late_alert": "поздняя тревога",
    "false_alert": "ложная тревога",
    "missed_failure": "сбой пропущен",
    "normal_no_alert": "норма без тревоги",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def first_existing(model_dir: Path, names: list[str]) -> Path:
    for name in names:
        path = model_dir / name
        if path.exists():
            return path
    return model_dir / names[0]


def apply_confirmed_alerts(frame: pd.DataFrame, score_col: str, threshold: float, min_consecutive: int) -> pd.Series:
    work = frame[["run_id", "window_ts", score_col]].copy()
    work["_order"] = np.arange(len(work))
    work = work.sort_values(["run_id", "window_ts"])
    candidate = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0).ge(float(threshold)).astype(int)
    if min_consecutive <= 1:
        alert = candidate.eq(1)
    else:
        alert = (
            candidate.groupby(work["run_id"], group_keys=False)
            .rolling(min_consecutive, min_periods=min_consecutive)
            .sum()
            .reset_index(level=0, drop=True)
            .ge(min_consecutive)
        )
    work["alert_confirmed"] = alert.astype(int)
    return work.sort_values("_order")["alert_confirmed"].reset_index(drop=True).astype(int)


def add_incident_markers(frame: pd.DataFrame) -> pd.Series:
    marker = pd.Series(0, index=frame.index, dtype=int)
    for _, group in frame.groupby("run_id", sort=False):
        scenario_type = group.get("scenario_type", pd.Series("", index=group.index)).fillna("").astype(str).str.lower()
        anchor_policy = group.get("anchor_policy", pd.Series("", index=group.index)).fillna("").astype(str)
        if bool(scenario_type.eq("normal").any() or anchor_policy.eq("normal_no_failure").any()):
            continue
        anchors = pd.to_datetime(group.get("event_anchor"), utc=True, errors="coerce").dropna()
        if anchors.empty:
            continue
        anchor_ts = pd.Timestamp(anchors.iloc[0])
        starts = pd.to_datetime(group["window_ts"], utc=True, errors="coerce")
        ends = pd.to_datetime(group.get("window_end"), utc=True, errors="coerce")
        contains = starts.le(anchor_ts) & ends.ge(anchor_ts)
        if contains.any():
            marker.loc[contains[contains].index] = 1
            continue
        distance = (starts - anchor_ts).abs()
        if distance.notna().any():
            marker.loc[distance.idxmin()] = 1
    return marker


def build_replay_bundle(model_dir: Path, report: dict[str, Any]) -> pd.DataFrame:
    source_path = first_existing(model_dir, ["dashboard_replay_store.csv.gz", "dashboard_replay_store.csv", "alert_predictions.csv"])
    replay = read_csv(source_path)
    if replay.empty:
        raise RuntimeError(f"Empty replay source: {source_path}")

    for col in ["window_ts", "window_end", "event_anchor", "fail_start", "run_start", "run_end"]:
        if col in replay.columns:
            replay[col] = pd.to_datetime(replay[col], utc=True, errors="coerce")
    replay["run_id"] = replay["run_id"].astype(str)
    replay = replay.sort_values(["run_id", "window_ts"]).reset_index(drop=True)

    thresholds = report.get("thresholds", {}) if isinstance(report.get("thresholds"), dict) else {}
    alert_threshold = safe_float(thresholds.get("fused"), safe_float(replay.get("alert_threshold", pd.Series([0.6])).median(), 0.6))
    watch_threshold = max(0.45, float(alert_threshold) - 0.15)
    config = report.get("config", {}) if isinstance(report.get("config"), dict) else {}
    min_consecutive = int(safe_float(config.get("min_consecutive_windows"), 3) or 3)

    score_source_col = "alert_score" if "alert_score" in replay.columns else ("risk_score" if "risk_score" in replay.columns else "fused_score")
    replay["model_score"] = pd.to_numeric(replay[score_source_col], errors="coerce").fillna(0.0).clip(0, 1)
    replay["risk_score"] = replay["model_score"]
    replay["warning_score"] = replay["model_score"]
    replay["alert_score"] = replay["model_score"]
    replay["model_mode"] = "fused"
    replay["watch_threshold"] = watch_threshold
    replay["alert_threshold"] = float(alert_threshold)
    replay["threshold_risk"] = float(alert_threshold)
    replay["threshold_warning"] = float(alert_threshold)
    replay["time_to_failure_sec"] = (
        pd.to_datetime(replay.get("event_anchor"), utc=True, errors="coerce")
        - pd.to_datetime(replay.get("window_end"), utc=True, errors="coerce")
    ).dt.total_seconds()

    replay["alert_confirmed"] = apply_confirmed_alerts(replay, "model_score", float(alert_threshold), min_consecutive)
    replay["alert_fired"] = replay["alert_confirmed"]
    replay["incident_marker"] = add_incident_markers(replay)
    replay["alert_state"] = np.select(
        [
            replay["incident_marker"].eq(1),
            replay["alert_confirmed"].eq(1),
            replay["model_score"].ge(watch_threshold),
        ],
        ["incident", "alert", "watch"],
        default="normal",
    )
    replay["alert_state_label"] = replay["alert_state"].map(STATE_LABELS).fillna("НОРМА")
    log_available = pd.to_numeric(replay.get("log_available", 0), errors="coerce").fillna(0).astype(int).eq(1)
    kpi_available = pd.to_numeric(replay.get("kpi_available", 0), errors="coerce").fillna(0).astype(int).eq(1)
    has_log_score = "log_only_score" in replay.columns
    has_kpi_score = "kpi_only_score" in replay.columns
    log_score = (
        pd.to_numeric(replay["log_only_score"], errors="coerce").clip(0, 1)
        if has_log_score
        else pd.Series(np.nan, index=replay.index)
    )
    kpi_score = (
        pd.to_numeric(replay["kpi_only_score"], errors="coerce").clip(0, 1)
        if has_kpi_score
        else pd.Series(np.nan, index=replay.index)
    )
    log_strength = (log_score - watch_threshold).clip(lower=0.0)
    kpi_strength = (kpi_score - watch_threshold).clip(lower=0.0)
    no_active_signal = (log_strength + kpi_strength).le(1e-9)
    log_strength = log_strength.where(~no_active_signal, log_score)
    kpi_strength = kpi_strength.where(~no_active_signal, kpi_score)
    log_strength = log_strength.where(log_available, 0.0)
    kpi_strength = kpi_strength.where(kpi_available, 0.0)
    denom = (log_strength + kpi_strength).replace(0.0, np.nan)
    replay["fusion_log_share"] = (log_strength / denom).clip(0, 1)
    replay["fusion_kpi_share"] = (kpi_strength / denom).clip(0, 1)
    replay.loc[log_available & ~kpi_available, ["fusion_log_share", "fusion_kpi_share"]] = [1.0, 0.0]
    replay.loc[~log_available & kpi_available, ["fusion_log_share", "fusion_kpi_share"]] = [0.0, 1.0]
    share_available = (
        has_log_score
        & has_kpi_score
        & (log_available | kpi_available)
        & replay["fusion_log_share"].notna()
        & replay["fusion_kpi_share"].notna()
    )
    replay["fusion_share_available"] = share_available.astype(int)
    replay["fusion_share_method"] = np.where(
        share_available,
        "normalized_unimodal_scores",
        "unavailable",
    )
    replay["fusion_display_mode"] = np.select(
        [log_available & kpi_available, log_available & ~kpi_available, ~log_available & kpi_available],
        ["log+kpi", "log_only_fallback", "kpi_only_fallback"],
        default="no_modalities",
    )
    replay["pred_warning"] = replay["alert_state"].isin(["watch", "alert", "incident"]).astype(int)
    replay["pred_risk"] = replay["alert_state"].isin(["alert", "incident"]).astype(int)
    replay["pred_state"] = np.select(
        [replay["alert_state"].isin(["alert", "incident"]), replay["alert_state"].eq("watch")],
        ["risk", "warning"],
        default="normal",
    )
    replay["score_source"] = replay.get("score_source", "model_store")
    replay["score_source"] = replay["score_source"].fillna("model_store").replace("", "model_store")

    return replay


def classify_event(row: pd.Series, target_lead_min_sec: float) -> str:
    is_failure = int(safe_float(row.get("is_failure_run"), 0) or 0) == 1
    is_normal = int(safe_float(row.get("is_normal_run"), 0) or 0) == 1
    has_alert = int(safe_float(row.get("has_alert"), 0) or 0) == 1
    lead = safe_float(row.get("lead_time_sec"))
    if is_failure:
        if not has_alert:
            return "missed_failure"
        if lead is None or lead < target_lead_min_sec:
            return "late_alert"
        return "early_alert"
    if is_normal and has_alert:
        return "false_alert"
    return "normal_no_alert"


def build_alert_events(model_dir: Path, replay: pd.DataFrame, report: dict[str, Any]) -> pd.DataFrame:
    run_metrics = read_csv(model_dir / "alert_run_metrics.csv")
    if run_metrics.empty:
        return pd.DataFrame()
    config = report.get("config", {}) if isinstance(report.get("config"), dict) else {}
    target_lead_min_sec = float(safe_float(config.get("target_lead_min_sec"), 10.0) or 10.0)
    window_sec = float(safe_float(replay.get("micro_window_sec", pd.Series([10])).median(), 10.0) or 10.0)

    rows: list[dict[str, Any]] = []
    for _, row in run_metrics.iterrows():
        event_type = classify_event(row, target_lead_min_sec)
        first_alert = pd.to_datetime(row.get("first_alert_ts"), utc=True, errors="coerce")
        anchor = pd.to_datetime(row.get("event_anchor"), utc=True, errors="coerce")
        alert_windows = int(safe_float(row.get("alert_windows"), 0) or 0)
        duration = float(alert_windows * window_sec) if alert_windows > 0 else 0.0
        event_start = first_alert if pd.notna(first_alert) else anchor
        event_end = event_start + pd.Timedelta(seconds=duration) if pd.notna(event_start) and duration > 0 else event_start
        rows.append(
            {
                "model": str(row.get("model", "fused")),
                "run_id": str(row.get("run_id", "")),
                "workload": str(row.get("workload", "")),
                "scenario": str(row.get("scenario", "")),
                "event_type": event_type,
                "event_label": EVENT_LABELS.get(event_type, event_type),
                "event_start_ts": event_start.isoformat() if pd.notna(event_start) else "",
                "event_end_ts": event_end.isoformat() if pd.notna(event_end) else "",
                "duration_sec": duration,
                "max_score": safe_float(row.get("max_score"), 0.0),
                "lead_time_sec": safe_float(row.get("lead_time_sec")),
                "is_false_alert": int(event_type == "false_alert"),
                "is_late": int(event_type == "late_alert"),
                "is_missed": int(event_type == "missed_failure"),
                "has_alert": int(safe_float(row.get("has_alert"), 0) or 0),
                "anchor_policy": str(row.get("anchor_policy", "")),
            }
        )
    return pd.DataFrame(rows)


def merge_fused_event_summary(replay: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return replay
    fused = events.loc[events["model"].astype(str).eq("fused")].copy()
    if fused.empty:
        return replay
    fused = fused.rename(
        columns={
            "event_type": "run_event_type",
            "event_label": "run_event_label",
            "event_start_ts": "run_event_start_ts",
            "event_end_ts": "run_event_end_ts",
            "duration_sec": "run_alert_duration_sec",
            "lead_time_sec": "run_event_lead_time_sec",
        }
    )
    keep = [
        "run_id",
        "run_event_type",
        "run_event_label",
        "run_event_start_ts",
        "run_event_end_ts",
        "run_alert_duration_sec",
        "run_event_lead_time_sec",
    ]
    replay = replay.drop(columns=[col for col in keep if col != "run_id" and col in replay.columns], errors="ignore")
    return replay.merge(fused[keep], on="run_id", how="left")


def build_metrics_summary(model_dir: Path, events: pd.DataFrame) -> pd.DataFrame:
    comparison = read_csv(model_dir / "alert_model_comparison.csv")
    if comparison.empty:
        comparison = read_csv(model_dir / "baseline_comparison.csv")
    if comparison.empty:
        return pd.DataFrame()
    out = comparison.copy()
    if not events.empty and "model" in out.columns:
        event_counts = events.groupby("model").agg(
            alert_events=("has_alert", "sum"),
            early_alert_events=("event_type", lambda s: int((s == "early_alert").sum())),
            late_alert_events=("event_type", lambda s: int((s == "late_alert").sum())),
            false_alert_events=("event_type", lambda s: int((s == "false_alert").sum())),
            missed_failure_events=("event_type", lambda s: int((s == "missed_failure").sum())),
        )
        out = out.merge(event_counts, left_on="model", right_index=True, how="left")
    out.insert(0, "summary_scope", "run_level_alert_layer")
    return out


def build_strict_research_summary(app_root: Path) -> pd.DataFrame:
    strict_dir = app_root / "outputs" / "hades_like_v2" / "hades_v2_strict_intensity_gap10"
    ensemble_path = strict_dir / "ensemble_metrics.csv"
    baseline_path = strict_dir / "baseline_comparison.csv"
    ensemble = read_csv(ensemble_path)
    baseline = read_csv(baseline_path)
    if ensemble.empty or baseline.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for target, group in ensemble.groupby("target"):
        model_row = group.sort_values("average_precision", ascending=False).iloc[0]
        base_group = baseline.loc[baseline["target"].astype(str).eq(str(target))].copy()
        if base_group.empty:
            continue
        base_row = base_group.sort_values("average_precision", ascending=False).iloc[0]
        model_ap = safe_float(model_row.get("average_precision"), 0.0) or 0.0
        base_ap = safe_float(base_row.get("average_precision"), 0.0) or 0.0
        rows.append(
            {
                "protocol": "strict_explicit_gap10_intensity",
                "target": str(target),
                "model": str(model_row.get("model", "hades_like")),
                "score": str(model_row.get("score", "")),
                "model_average_precision": model_ap,
                "model_roc_auc": safe_float(model_row.get("roc_auc")),
                "best_baseline": str(base_row.get("baseline", "baseline")),
                "best_baseline_average_precision": base_ap,
                "best_baseline_roc_auc": safe_float(base_row.get("roc_auc")),
                "delta_ap": model_ap - base_ap,
                "note": "strict setup controls explicit failure anchors and uses gap10 forecast boundary",
            }
        )
    return pd.DataFrame(rows)


def _demo_case_row(kind: str, row: pd.Series) -> dict[str, Any]:
    lead = safe_float(row.get("time_to_failure_sec"))
    score = safe_float(row.get("model_score"), 0.0) or 0.0
    log_share = safe_float(row.get("fusion_log_share"))
    kpi_share = safe_float(row.get("fusion_kpi_share"))
    return {
        "case_id": f"{kind}::{row.get('run_id')}",
        "case_kind": kind,
        "run_id": str(row.get("run_id", "")),
        "focus_window_ts": pd.Timestamp(row.get("window_ts")).isoformat() if pd.notna(row.get("window_ts")) else "",
        "lead_time_sec": lead,
        "alert_score": score,
        "fusion_log_share": log_share,
        "fusion_kpi_share": kpi_share,
        "note": f"{kind}: score={score:.3f}, lead={'n/a' if lead is None else f'{lead:.0f}s'}",
    }


def build_product_case_catalog(replay: pd.DataFrame) -> pd.DataFrame:
    if replay.empty:
        return pd.DataFrame()

    work = replay.copy()
    for col in ["model_score", "log_only_score", "kpi_only_score", "fusion_log_share", "fusion_kpi_share", "time_to_failure_sec"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    both = (
        pd.to_numeric(work.get("log_available", 0), errors="coerce").fillna(0).astype(int).eq(1)
        & pd.to_numeric(work.get("kpi_available", 0), errors="coerce").fillna(0).astype(int).eq(1)
        & work.get("fusion_share_available", pd.Series(0, index=work.index)).astype(int).eq(1)
    )
    alert_before = (
        work.get("alert_state", pd.Series("", index=work.index)).fillna("").astype(str).eq("alert")
        & work.get("time_to_failure_sec", pd.Series(np.nan, index=work.index)).ge(10)
    )
    candidates = work.loc[both & alert_before].copy()
    if candidates.empty:
        return pd.DataFrame()

    candidates["_balance"] = 1.0 - (candidates["fusion_log_share"].fillna(0.5) - 0.5).abs() * 2.0
    selected: list[dict[str, Any]] = []
    used_runs: set[str] = set()

    def pick(kind: str, frame: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> None:
        nonlocal selected
        if frame.empty:
            return
        local = frame.loc[~frame["run_id"].astype(str).isin(used_runs)].copy()
        if local.empty:
            local = frame.copy()
        local = local.sort_values(sort_cols, ascending=ascending)
        row = local.iloc[0]
        used_runs.add(str(row["run_id"]))
        selected.append(_demo_case_row(kind, row))

    pick(
        "log_driven_success",
        candidates.loc[candidates["fusion_log_share"].ge(0.55)],
        ["fusion_log_share", "log_only_score", "model_score"],
        [False, False, False],
    )
    if not any(row["case_kind"] == "log_driven_success" for row in selected):
        pick("log_driven_success", candidates, ["fusion_log_share", "log_only_score", "model_score"], [False, False, False])

    pick(
        "kpi_driven_success",
        candidates.loc[candidates["fusion_kpi_share"].ge(0.70)],
        ["fusion_kpi_share", "kpi_only_score", "model_score"],
        [False, False, False],
    )
    if not any(row["case_kind"] == "kpi_driven_success" for row in selected):
        pick("kpi_driven_success", candidates, ["fusion_kpi_share", "kpi_only_score", "model_score"], [False, False, False])

    pick(
        "multimodal_success",
        candidates.loc[candidates["fusion_log_share"].between(0.35, 0.65)],
        ["_balance", "model_score"],
        [False, False],
    )
    if not any(row["case_kind"] == "multimodal_success" for row in selected):
        pick("multimodal_success", candidates, ["_balance", "model_score"], [False, False])

    normal_mask = (
        work.get("scenario_type", pd.Series("", index=work.index)).fillna("").astype(str).str.lower().eq("normal")
        | work.get("anchor_policy", pd.Series("", index=work.index)).fillna("").astype(str).eq("normal_no_failure")
        | pd.to_numeric(work.get("is_normal_run", pd.Series(0, index=work.index)), errors="coerce").fillna(0).astype(int).eq(1)
    )
    alert_by_run = work.get("alert_state", pd.Series("", index=work.index)).fillna("").astype(str).eq("alert").groupby(work["run_id"]).transform("any")
    normal = work.loc[normal_mask & ~alert_by_run].copy()
    if not normal.empty:
        normal["_both"] = both.loc[normal.index].astype(int)
        normal = normal.sort_values(["_both", "model_score"], ascending=[False, True])
        selected.append(_demo_case_row("normal_no_alert", normal.iloc[len(normal) // 2 if len(normal) < 4 else 0]))

    order = {"multimodal_success": 0, "log_driven_success": 1, "kpi_driven_success": 2, "normal_no_alert": 3}
    out = pd.DataFrame(selected).drop_duplicates("case_kind", keep="first")
    if out.empty:
        return out
    out["_order"] = out["case_kind"].map(order).fillna(99)
    return out.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def stamp_demo_cases(replay: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    out = replay.copy()
    out["demo_case_kind"] = ""
    out["demo_case_id"] = ""
    if cases.empty:
        return out
    for _, case in cases.iterrows():
        mask = out["run_id"].astype(str).eq(str(case["run_id"]))
        out.loc[mask, "demo_case_kind"] = str(case["case_kind"])
        out.loc[mask, "demo_case_id"] = str(case["case_id"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    report_path = first_existing(model_dir, ["dashboard_model_report.json", "model_report.json"])
    report = read_json(report_path)

    replay = build_replay_bundle(model_dir, report)
    events = build_alert_events(model_dir, replay, report)
    replay = merge_fused_event_summary(replay, events)
    demo_cases = build_product_case_catalog(replay)
    replay = stamp_demo_cases(replay, demo_cases)
    metrics = build_metrics_summary(model_dir, events)
    strict_summary = build_strict_research_summary(APP_ROOT)

    replay.to_csv(model_dir / "replay_bundle.csv.gz", index=False, compression="gzip")
    replay.to_csv(model_dir / "replay_bundle.csv", index=False)
    events.to_csv(model_dir / "alert_events.csv", index=False)
    metrics.to_csv(model_dir / "metrics_summary.csv", index=False)
    strict_summary.to_csv(model_dir / "strict_research_summary.csv", index=False)
    demo_cases.to_csv(model_dir / "product_case_catalog.csv", index=False)

    thresholds = report.get("thresholds", {}) if isinstance(report.get("thresholds"), dict) else {}
    alert_threshold = safe_float(thresholds.get("fused"), safe_float(replay["alert_threshold"].median(), 0.6))
    alert_layer = {
        "schema_version": 1,
        "primary_score": "model_score",
        "state_column": "alert_state",
        "state_labels": STATE_LABELS,
        "alert_threshold": alert_threshold,
        "watch_threshold": max(0.45, float(alert_threshold) - 0.15),
        "min_consecutive_windows": int(safe_float(report.get("config", {}).get("min_consecutive_windows"), 3) or 3),
        "files": {
            "replay_bundle": "replay_bundle.csv.gz",
            "alert_events": "alert_events.csv",
            "metrics_summary": "metrics_summary.csv",
            "strict_research_summary": "strict_research_summary.csv",
        },
        "note": "Alert-layer separates model score, confirmed alert decision and incident marker.",
    }
    report["alert_layer"] = alert_layer
    dashboard_contract = report.get("dashboard_contract", {}) if isinstance(report.get("dashboard_contract"), dict) else {}
    dashboard_contract.update(
        {
            "primary_score": "model_score",
            "state_column": "alert_state",
            "score_source": "model_store",
            "replay_bundle": "replay_bundle.csv.gz",
            "events_file": "alert_events.csv",
        }
    )
    report["dashboard_contract"] = dashboard_contract
    if not demo_cases.empty:
        report["demo_cases"] = demo_cases.to_dict("records")
    if not strict_summary.empty:
        report["strict_research_summary"] = strict_summary.to_dict("records")
    write_json(model_dir / "dashboard_model_report.json", report)

    print(f"Wrote {model_dir / 'replay_bundle.csv.gz'}")
    print(f"Wrote {model_dir / 'alert_events.csv'}")
    print(f"Wrote {model_dir / 'metrics_summary.csv'}")
    print(f"Wrote {model_dir / 'strict_research_summary.csv'}")
    print(f"Wrote {model_dir / 'product_case_catalog.csv'}")


if __name__ == "__main__":
    main()
