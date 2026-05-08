from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dashboard.runtime.contract import ContractBundle, contract_stats
from dashboard.runtime.inference import InferenceBundleStatus


STATE_LABELS = {
    "normal": "Норма",
    "watch": "Наблюдение",
    "alert": "Тревога",
    "incident": "Инцидент",
    "warning": "Предупреждение",
    "risk": "Риск",
}

CASE_LABELS = {
    "long_test_stream": "длинный сбойный test-stream",
    "multimodal_success": "мультимодальный кейс",
    "log_driven_success": "логовый кейс",
    "kpi_driven_success": "KPI-кейс",
    "normal_no_alert": "норма без тревоги",
    "early_alert": "тревога до сбоя",
    "late_alert": "поздняя тревога",
    "false_alert": "ложная тревога",
    "missed_failure": "сбой пропущен",
    "risk_escalation": "рост риска перед сбоем",
    "early_warning": "раннее предупреждение",
    "degraded_fallback": "неполные модальности",
    "normal_reference": "нормальный эталон",
}

CASE_ORDER = [
    "long_test_stream",
    "multimodal_success",
    "log_driven_success",
    "kpi_driven_success",
    "normal_no_alert",
    "early_alert",
    "late_alert",
    "false_alert",
    "missed_failure",
    "risk_escalation",
    "early_warning",
    "degraded_fallback",
    "normal_reference",
]

LOG_PATTERNS = [
    (r"blk_update_request|buffer i/o|i/o error", "I/O error / storage"),
    (r"task blocked|blocked for more than", "task blocked / задержка"),
    (r"connection.*timeout|timeout", "connection timeout"),
    (r"failed|failure|exception|killed", "failure / exception"),
    (r"memory|memorystore|oom|out of memory", "memory pressure"),
    (r"remote fetch|shuffle|fetch", "shuffle / remote fetch"),
    (r"resourcemanager|nodemanager|applicationmaster", "YARN coordination"),
    (r"launching container|container", "container lifecycle"),
    (r"shutdown|stopped|blockmanager|executor lost|lost executor", "service lifecycle / shutdown"),
    (r"heartbeat|lost|expired|unreachable", "lost heartbeat / unreachable"),
]

LOG_PATTERN_MEANING = {
    "I/O error / storage": "прямой storage-сигнал",
    "task blocked / задержка": "задержка выполнения",
    "connection timeout": "сетевой таймаут",
    "failure / exception": "ошибка приложения",
    "memory pressure": "давление на память",
    "shuffle / remote fetch": "shuffle / data exchange",
    "YARN coordination": "координация кластера",
    "container lifecycle": "контейнеры",
    "service lifecycle / shutdown": "остановка сервисов",
    "lost heartbeat / unreachable": "потеря связи",
}

FAMILY_LABELS = {
    "cpustat": "CPU",
    "vmstat": "VM",
    "iostat": "Disk",
    "memstat": "Memory",
    "netstat": "Network",
}

METRIC_LABELS = {
    "user": "user CPU",
    "system": "system CPU",
    "iowait": "I/O wait",
    "wa": "I/O wait",
    "util": "utilization",
    "r_s": "read ops/s",
    "w_s": "write ops/s",
    "rkb_s": "read KB/s",
    "wkb_s": "write KB/s",
    "r_await": "read await",
    "w_await": "write await",
    "aqu_sz": "queue depth",
    "memused": "memory used",
    "kbmemused": "memory used",
    "kbavail": "available memory",
    "commit": "commit",
    "swpused": "swap used",
    "rxpck_s": "RX packets/s",
    "txpck_s": "TX packets/s",
    "rxkb_s": "RX KB/s",
    "txkb_s": "TX KB/s",
    "ifutil": "interface util",
    "idle": "idle",
    "user": "user",
    "system": "system",
    "free": "free memory",
}


@dataclass
class DashboardView:
    timeline: pd.DataFrame
    cases: pd.DataFrame
    stats: dict[str, Any]
    mode: str
    mode_note: str
    kpi_columns: list[str]
    baseline_table: pd.DataFrame
    dashboard_report: dict[str, Any]
    calibration_bins: pd.DataFrame
    seed_metrics: pd.DataFrame


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def _numeric_column(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        source = frame[column]
    else:
        source = pd.Series(default, index=frame.index)
    return pd.to_numeric(source, errors="coerce").fillna(default)


def _target_proxy_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    warning = pd.to_numeric(out["y_warning"], errors="coerce").fillna(0).astype(int)
    risk = pd.to_numeric(out["y_risk"], errors="coerce").fillna(0).astype(int)

    log_pressure = (
        _numeric_column(out, "macro_error_count", 0)
        + 0.45 * _numeric_column(out, "macro_warn_count", 0)
        + 0.18 * _numeric_column(out, "macro_keyword_count", 0)
    )
    if float(log_pressure.max()) > 0:
        log_pressure = np.log1p(log_pressure) / np.log1p(float(log_pressure.max()))
    else:
        log_pressure = pd.Series(0.0, index=out.index)

    coverage_penalty = 0.08 * (1.0 - _numeric_column(out, "kpi_avail_ratio", 1.0).clip(0, 1))
    base = 0.10 + 0.16 * log_pressure + coverage_penalty
    out["warning_score"] = np.clip(np.where(warning.eq(1), 0.58, base + 0.10 * log_pressure), 0.0, 1.0)
    out["risk_score"] = np.clip(np.where(risk.eq(1), 0.82, np.where(warning.eq(1), 0.54, base)), 0.0, 1.0)
    out["pred_state"] = np.select([risk.eq(1), warning.eq(1)], ["risk", "warning"], default="normal")

    log_available = _numeric_column(out, "log_available", 0).clip(0, 1)
    kpi_available = _numeric_column(out, "kpi_available", 0).clip(0, 1)
    both = _numeric_column(out, "has_both_modalities", 0).clip(0, 1)
    out["fusion_gate"] = np.where(
        (log_available.eq(1) & kpi_available.eq(1)),
        0.62 - 0.12 * both,
        np.where(log_available.eq(1), 1.0, np.where(kpi_available.eq(1), 0.0, 0.5)),
    )
    out["confidence"] = np.nan
    out["score_source"] = "proxy"
    return out


def _apply_model_scores(df: pd.DataFrame, inference: InferenceBundleStatus) -> pd.DataFrame:
    if {"model_score", "alert_state"}.issubset(df.columns):
        out = df.copy()
        alert_score = pd.to_numeric(out.get("alert_score", out.get("model_score_raw", out["model_score"])), errors="coerce")
        model_score = pd.to_numeric(out.get("model_score_raw", out["model_score"]), errors="coerce").fillna(alert_score)
        alert_score = alert_score.fillna(model_score).fillna(0.0).clip(0, 1)
        model_score = model_score.fillna(alert_score).fillna(0.0).clip(0, 1)
        out["model_score_raw"] = model_score
        out["model_score"] = model_score
        out["alert_score"] = alert_score
        if "warning_score" in out.columns:
            out["warning_score"] = pd.to_numeric(out["warning_score"], errors="coerce").fillna(model_score).clip(0, 1)
        else:
            out["warning_score"] = model_score
        if "risk_score_raw" in out.columns:
            out["risk_score"] = pd.to_numeric(out["risk_score_raw"], errors="coerce").fillna(0.0).clip(0, 1)
        elif "risk_score" in out.columns:
            out["risk_score"] = pd.to_numeric(out["risk_score"], errors="coerce").fillna(0.0).clip(0, 1)
        else:
            out["risk_score"] = model_score
        if "model_score_rank" in out.columns:
            out["model_score_rank"] = pd.to_numeric(out["model_score_rank"], errors="coerce").clip(0, 1)
        alert_state = out["alert_state"].fillna("normal").astype(str).str.lower()
        out["alert_state"] = alert_state.where(alert_state.isin(["normal", "watch", "alert", "incident"]), "normal")
        out["alert_state_label"] = out["alert_state"].map(STATE_LABELS).fillna("Норма")
        out["pred_state"] = np.select(
            [out["alert_state"].isin(["alert", "incident"]), out["alert_state"].eq("watch")],
            ["risk", "warning"],
            default="normal",
        )
        out["score_source"] = out.get("score_source", "model_store")
        out["score_source"] = out["score_source"].fillna("model_store").replace("", "model_store")
        out["confidence"] = alert_score
        return out

    out = _target_proxy_scores(df)
    store = inference.prediction_store.copy()
    required = {"run_id", "window_ts", "warning_score", "risk_score"}
    if store.empty or not required.issubset(store.columns):
        out["score_source"] = "proxy_fallback"
        return out

    store["run_id"] = store["run_id"].astype(str)
    store["window_ts"] = pd.to_datetime(store["window_ts"], utc=True, errors="coerce")
    store = store.dropna(subset=["run_id", "window_ts"]).drop_duplicates(["run_id", "window_ts"], keep="last")

    model_cols = [
        "run_id",
        "window_ts",
        "warning_score",
        "risk_score",
        "fusion_gate",
        "pred_warning",
        "pred_risk",
        "pred_state",
        "threshold_warning",
        "threshold_risk",
        "baseline_warning_score",
        "baseline_warning_model",
        "baseline_risk_score",
        "baseline_risk_model",
        "label_anchor",
        "anchor_reliability",
        "score_source",
    ]
    model_cols = [col for col in model_cols if col in store.columns]
    model = store[model_cols].rename(
        columns={
            "warning_score": "model_warning_score",
            "risk_score": "model_risk_score",
            "fusion_gate": "model_fusion_gate",
            "pred_warning": "model_pred_warning",
            "pred_risk": "model_pred_risk",
            "pred_state": "model_pred_state",
            "threshold_warning": "model_threshold_warning",
            "threshold_risk": "model_threshold_risk",
            "score_source": "model_score_source",
        }
    )

    out = out.merge(model, on=["run_id", "window_ts"], how="left")
    has_model = out["model_risk_score"].notna()
    out.loc[has_model, "warning_score"] = pd.to_numeric(out.loc[has_model, "model_warning_score"], errors="coerce")
    out.loc[has_model, "risk_score"] = pd.to_numeric(out.loc[has_model, "model_risk_score"], errors="coerce")
    if "model_fusion_gate" in out.columns:
        out.loc[has_model & out["model_fusion_gate"].notna(), "fusion_gate"] = pd.to_numeric(
            out.loc[has_model & out["model_fusion_gate"].notna(), "model_fusion_gate"],
            errors="coerce",
        )

    if "model_pred_state" in out.columns:
        state = out["model_pred_state"].fillna("").astype(str).str.lower()
        valid_state = has_model & state.isin(["normal", "warning", "risk"])
        out.loc[valid_state, "pred_state"] = state.loc[valid_state]

    if "model_pred_risk" in out.columns and "model_pred_warning" in out.columns:
        pred_risk = pd.to_numeric(out["model_pred_risk"], errors="coerce").fillna(0).astype(int)
        pred_warning = pd.to_numeric(out["model_pred_warning"], errors="coerce").fillna(0).astype(int)
        derived = np.select([pred_risk.eq(1), pred_warning.eq(1)], ["risk", "warning"], default="normal")
        out.loc[has_model, "pred_state"] = derived[has_model]

    out["score_source"] = np.where(has_model, "model_store", "proxy_fallback")
    if "model_score_source" in out.columns:
        source = out["model_score_source"].fillna("").astype(str)
        out.loc[has_model & source.ne(""), "score_source"] = source.loc[has_model & source.ne("")]
    out["confidence"] = np.where(has_model, np.maximum(out["warning_score"], out["risk_score"]), np.nan)
    return out


def _build_baseline_table(bundle: ContractBundle, inference: InferenceBundleStatus) -> pd.DataFrame:
    if inference.ready and not inference.baseline_comparison.empty:
        return inference.baseline_comparison.copy()

    rows = bundle.baseline_summary.get("baseline_metrics", [])
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    if "status" in table.columns:
        table = table.loc[table["status"].fillna("ok").eq("ok")].copy()
    keep = [
        "target",
        "baseline",
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "n",
        "positives",
    ]
    return table[[col for col in keep if col in table.columns]].reset_index(drop=True)


def build_dashboard_view(bundle: ContractBundle, inference: InferenceBundleStatus) -> DashboardView:
    if inference.ready:
        timeline = _apply_model_scores(bundle.timeline, inference)
        model_rows = int(timeline["score_source"].eq("model_store").sum())
        mode_note = (
            f"Режим scoring: используется финальная мультимодальная модель "
            f"и alert-слой для {model_rows} replay-окон."
        )
    else:
        timeline = _target_proxy_scores(bundle.timeline)
        mode_note = (
            "Preview-режим: compact model bundle отсутствует, поэтому график использует proxy-разметку "
            "из aligned contract и не выдаёт модельные prediction claims."
        )

    cases = choose_demo_cases(timeline)
    return DashboardView(
        timeline=timeline,
        cases=cases,
        stats=contract_stats(bundle),
        mode=inference.mode,
        mode_note=mode_note,
        kpi_columns=bundle.kpi_columns,
        baseline_table=_build_baseline_table(bundle, inference),
        dashboard_report=inference.dashboard_report,
        calibration_bins=inference.calibration_bins,
        seed_metrics=inference.seed_metrics,
    )


def choose_demo_cases(timeline: pd.DataFrame) -> pd.DataFrame:
    if {"demo_case_kind", "demo_case_id"}.issubset(timeline.columns):
        demo_rows = timeline.loc[timeline["demo_case_kind"].fillna("").astype(str).ne("")].copy()
        if not demo_rows.empty:
            rows: list[dict[str, Any]] = []
            for case_id, group in demo_rows.groupby("demo_case_id", sort=False):
                group = group.sort_values("window_ts")
                kind_values = group["demo_case_kind"].fillna("").astype(str)
                kind = str(kind_values.loc[kind_values.ne("")].iloc[0]) if kind_values.ne("").any() else "normal_no_alert"
                alert_rows = group.loc[group.get("alert_state", pd.Series("", index=group.index)).fillna("").astype(str).eq("alert")]
                if alert_rows.empty:
                    alert_rows = group.loc[_numeric_column(group, "alert_fired", 0).astype(int).eq(1)]
                focus_pool = alert_rows if not alert_rows.empty else group
                pre_alert_focus: pd.Series | None = None
                if kind in {"multimodal_success", "log_driven_success", "kpi_driven_success"} and not alert_rows.empty:
                    first_alert_ts = pd.to_datetime(alert_rows["window_ts"], utc=True, errors="coerce").dropna().min()
                    ordered = group.sort_values("window_ts").reset_index(drop=True)
                    ordered_ts = pd.to_datetime(ordered["window_ts"], utc=True, errors="coerce")
                    positions = np.flatnonzero(ordered_ts.eq(first_alert_ts).to_numpy()) if pd.notna(first_alert_ts) else np.array([], dtype=int)
                    if positions.size and positions[0] > 0:
                        pre_alert_focus = ordered.iloc[max(0, int(positions[0]) - 1)]
                if kind in {"multimodal_success", "log_driven_success", "kpi_driven_success"} and "time_to_failure_sec" in focus_pool.columns:
                    early_pool = focus_pool.loc[pd.to_numeric(focus_pool["time_to_failure_sec"], errors="coerce").ge(10)]
                    if not early_pool.empty:
                        focus_pool = early_pool
                if pre_alert_focus is not None:
                    focus = pre_alert_focus
                elif kind == "log_driven_success" and {"fusion_log_share", "log_only_score"}.issubset(focus_pool.columns):
                    focus = focus_pool.sort_values(["fusion_log_share", "log_only_score", "model_score" if "model_score" in focus_pool.columns else "risk_score"], ascending=[False, False, False]).iloc[0]
                elif kind == "kpi_driven_success" and {"fusion_kpi_share", "kpi_only_score"}.issubset(focus_pool.columns):
                    focus = focus_pool.sort_values(["fusion_kpi_share", "kpi_only_score", "model_score" if "model_score" in focus_pool.columns else "risk_score"], ascending=[False, False, False]).iloc[0]
                elif kind == "multimodal_success" and {"fusion_log_share", "fusion_kpi_share"}.issubset(focus_pool.columns):
                    focus_pool = focus_pool.copy()
                    focus_pool["_balance"] = 1.0 - (pd.to_numeric(focus_pool["fusion_log_share"], errors="coerce").fillna(0.5) - 0.5).abs() * 2.0
                    focus = focus_pool.sort_values(["_balance", "model_score" if "model_score" in focus_pool.columns else "risk_score"], ascending=[False, False]).iloc[0]
                elif kind == "multimodal_success" and {"log_only_score", "kpi_only_score"}.issubset(focus_pool.columns):
                    focus_pool = focus_pool.copy()
                    focus_pool["_both_signal"] = pd.to_numeric(focus_pool["log_only_score"], errors="coerce").fillna(0.0).clip(0, 1) * pd.to_numeric(focus_pool["kpi_only_score"], errors="coerce").fillna(0.0).clip(0, 1)
                    focus = focus_pool.sort_values(["_both_signal", "model_score" if "model_score" in focus_pool.columns else "risk_score"], ascending=[False, False]).iloc[0]
                elif kind == "normal_no_alert" and "model_score" in focus_pool.columns:
                    focus = focus_pool.sort_values("model_score", ascending=True).iloc[0]
                else:
                    focus = focus_pool.iloc[len(focus_pool) // 2]
                score = _safe_float(focus.get("model_score", focus.get("alert_score", focus.get("risk_score"))))
                lead = _safe_float(focus.get("time_to_failure_sec", focus.get("run_event_lead_time_sec", focus.get("lead_time_sec"))), np.nan)
                lead_text = "n/a" if not np.isfinite(lead) else f"{lead:.0f} сек"
                if kind == "long_test_stream":
                    note = (
                        "Длинный live-сценарий: спокойный test-префикс, затем реальный сбойный хвост. "
                        "Он нужен, чтобы проверить не только пик score, но и поведение alert-слоя во времени."
                    )
                else:
                    note = f"{CASE_LABELS.get(kind, kind)}: оценка модели={score:.2f}; lead={lead_text}."
                rows.append(
                    {
                        "case_id": str(case_id),
                        "kind": kind,
                        "label": CASE_LABELS.get(kind, kind),
                        "run_id": str(focus["run_id"]),
                        "focus_window_ts": pd.Timestamp(focus["window_ts"]),
                        "risk_score": score,
                        "note": note,
                    }
                )
            out = pd.DataFrame(rows)
            order = {kind: idx for idx, kind in enumerate(CASE_ORDER)}
            out["order"] = out["kind"].map(order).fillna(99).astype(int)
            out = out.sort_values(["order", "risk_score"], ascending=[True, False]).drop(columns=["order"]).reset_index(drop=True)
            out["run_number"] = out["run_id"].map({rid: idx + 1 for idx, rid in enumerate(sorted(timeline["run_id"].unique()))})
            return out

    rows: list[dict[str, Any]] = []
    used: set[tuple[str, pd.Timestamp]] = set()
    used_runs: set[str] = set()

    risk_score = _numeric_column(timeline, "risk_score", 0.0)
    warning_score = pd.to_numeric(timeline["warning_score"], errors="coerce").fillna(risk_score) if "warning_score" in timeline.columns else risk_score
    log_available = _numeric_column(timeline, "log_available", 0).astype(int).eq(1)
    kpi_available = _numeric_column(timeline, "kpi_available", 0).astype(int).eq(1)
    both_modalities = log_available & kpi_available
    minutes_before_fail = _numeric_column(timeline, "minutes_before_fail", np.nan)
    pred_state = timeline.get("pred_state", pd.Series("normal", index=timeline.index)).fillna("normal").astype(str)

    def case_note(kind: str, row: pd.Series) -> str:
        workload = str(row.get("workload", "") or "").strip()
        scenario = str(row.get("scenario", "") or "").strip()
        context = f"{workload}/{scenario}".strip("/") or str(row.get("run_id", "run"))
        if kind == "risk_escalation":
            return (
                f"{context}: фрагмент с высоким риском и обеими модальностями. "
                "Подходит для показа роста score, лог-паттернов и KPI-сигналов перед сбоем."
            )
        if kind == "early_warning":
            return (
                f"{context}: окно в зоне предупреждения до критического риска. "
                "Нужно для демонстрации раннего сигнала, когда система еще не перешла в аварийную область."
            )
        if kind == "degraded_fallback":
            return (
                f"{context}: одна из модальностей недоступна, поэтому replay явно показывает degraded/fallback context. "
                "Сценарий проверяет, что интерфейс не скрывает неполное покрытие данных."
            )
        return (
            f"{context}: спокойный участок без target warning/risk и без выраженных лог-ошибок. "
            "Это эталон для сравнения масштаба шума с предупреждением и риском."
        )

    def add_case(kind: str, frame: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> None:
        if frame.empty:
            return
        existing_sort_cols = [col for col in sort_cols if col in frame.columns]
        existing_ascending = ascending[: len(existing_sort_cols)]
        if existing_sort_cols:
            frame = frame.sort_values(existing_sort_cols, ascending=existing_ascending).copy()
        chosen: pd.Series | None = None
        for _, candidate in frame.iterrows():
            run_id = str(candidate["run_id"])
            key = (run_id, pd.Timestamp(candidate["window_ts"]))
            if key not in used and run_id not in used_runs:
                chosen = candidate
                break
        if chosen is None:
            for _, candidate in frame.iterrows():
                key = (str(candidate["run_id"]), pd.Timestamp(candidate["window_ts"]))
                if key not in used:
                    chosen = candidate
                    break
        if chosen is None:
            return
        row = chosen
        key = (str(row["run_id"]), pd.Timestamp(row["window_ts"]))
        used.add(key)
        used_runs.add(key[0])
        rows.append(
            {
                "case_id": f"{kind}::{key[0]}::{key[1].strftime('%Y%m%d%H%M%S')}",
                "kind": kind,
                "label": CASE_LABELS[kind],
                "run_id": str(row["run_id"]),
                "focus_window_ts": pd.Timestamp(row["window_ts"]),
                "risk_score": _safe_float(row.get("risk_score")),
                "note": case_note(kind, row),
            }
        )

    work = timeline.copy()
    work["_risk_score"] = risk_score
    work["_warning_score"] = warning_score
    work["_both_modalities"] = both_modalities.astype(int)
    work["_minutes_before_fail"] = minutes_before_fail.fillna(-1.0)
    work["_run_gain"] = work.groupby("run_id")["_risk_score"].transform(lambda series: float(series.max() - series.min()))
    dashboard_kpi_columns = [col for col in work.columns if col.startswith("kpi_") and "__" in col]
    cpu_idle = _aggregate_kpi_series(
        work,
        _select_kpi_columns(dashboard_kpi_columns, "cpustat", ["idle"], stats=["last", "mean"], dims=["max_dim", "mean_dim"]),
        how="min",
    )
    work["_cpu_usage"] = (100.0 - cpu_idle).clip(lower=0.0, upper=100.0)

    if "score_source" in work.columns and work["score_source"].eq("model_store").any():
        split_series = work["split"] if "split" in work.columns else pd.Series("test", index=work.index)
        demo_pool = work.loc[work["score_source"].eq("model_store") & split_series.eq("test")].copy()
    elif "split" in work.columns and work["split"].eq("test").any():
        demo_pool = work.loc[work["split"].eq("test")].copy()
    else:
        demo_pool = work.copy()
    if demo_pool.empty:
        demo_pool = work.copy()
    for column, default in {
        "log_available": 0,
        "kpi_available": 0,
        "macro_error_count": 0,
        "macro_warn_count": 0,
        "y_warning": 0,
        "y_risk": 0,
    }.items():
        if column not in demo_pool.columns:
            demo_pool[column] = default
        demo_pool[column] = pd.to_numeric(demo_pool[column], errors="coerce").fillna(default)
    anchor_available = (
        demo_pool["event_anchor"].notna()
        if "event_anchor" in demo_pool.columns
        else demo_pool.get("fail_start", pd.Series(pd.NaT, index=demo_pool.index)).notna()
    )

    add_case(
        "risk_escalation",
        demo_pool.loc[
            demo_pool["_risk_score"].ge(0.70)
            & demo_pool["_both_modalities"].eq(1)
            & anchor_available
        ],
        ["_run_gain", "_risk_score", "_minutes_before_fail", "macro_error_count", "macro_warn_count"],
        [False, False, True, False, False],
    )
    add_case(
        "early_warning",
        demo_pool.loc[
            demo_pool["_risk_score"].between(0.30, 0.70, inclusive="left")
            & demo_pool["_both_modalities"].eq(1)
            & (
                demo_pool.get("pred_state", pd.Series("normal", index=demo_pool.index)).fillna("normal").astype(str).eq("warning")
                | demo_pool["y_warning"].eq(1)
            )
            & anchor_available
        ],
        ["_minutes_before_fail", "_warning_score", "macro_warn_count", "window_ts"],
        [False, False, False, True],
    )
    add_case(
        "degraded_fallback",
        demo_pool.loc[
            demo_pool.get("has_both_modalities", pd.Series(0, index=demo_pool.index)).eq(0)
        ],
        ["_risk_score", "macro_error_count", "macro_warn_count", "window_ts"],
        [False, False, False, True],
    )

    run_stats = demo_pool.groupby("run_id").agg(
        n=("run_id", "size"),
        max_risk=("_risk_score", "max"),
        mean_risk=("_risk_score", "mean"),
        log_coverage=("log_available", "mean"),
        kpi_coverage=("kpi_available", "mean"),
        warn_targets=("y_warning", "sum"),
        risk_targets=("y_risk", "sum"),
        max_errors=("macro_error_count", "max"),
        max_warnings=("macro_warn_count", "max"),
        median_cpu=("_cpu_usage", "median"),
    )
    normal_runs = run_stats.loc[
        run_stats["n"].ge(5)
        & run_stats["max_risk"].le(0.15)
        & run_stats["warn_targets"].eq(0)
        & run_stats["risk_targets"].eq(0)
        & run_stats["max_errors"].fillna(0).eq(0)
        & run_stats["max_warnings"].fillna(0).eq(0)
        & run_stats["log_coverage"].ge(0.5)
        & run_stats["kpi_coverage"].ge(0.5)
        & (run_stats["median_cpu"].isna() | run_stats["median_cpu"].le(35.0))
    ].sort_values(["median_cpu", "n", "kpi_coverage", "log_coverage", "mean_risk"], ascending=[True, False, False, False, True])
    normal_frame = pd.DataFrame()
    if not normal_runs.empty:
        normal_run_id = str(normal_runs.index[0])
        normal_run = demo_pool.loc[demo_pool["run_id"].eq(normal_run_id)].sort_values("window_ts")
        normal_frame = normal_run.iloc[[len(normal_run) // 2]].copy()
    else:
        normal_frame = demo_pool.loc[
            demo_pool["_risk_score"].le(0.15)
            & demo_pool["y_warning"].eq(0)
            & demo_pool["y_risk"].eq(0)
            & demo_pool["log_available"].eq(1)
            & demo_pool["kpi_available"].eq(1)
        ]
    add_case(
        "normal_reference",
        normal_frame,
        ["_risk_score", "window_ts"],
        [True, True],
    )

    if not rows:
        first = timeline.iloc[0]
        rows.append(
            {
                "case_id": f"normal_reference::{first['run_id']}::{pd.Timestamp(first['window_ts']).strftime('%Y%m%d%H%M%S')}",
                "kind": "normal_reference",
                "label": CASE_LABELS["normal_reference"],
                "run_id": str(first["run_id"]),
                "focus_window_ts": pd.Timestamp(first["window_ts"]),
                "risk_score": _safe_float(first.get("risk_score")),
                "note": "Fallback-кейс: выбран первый доступный run.",
            }
        )

    out = pd.DataFrame(rows)
    order = {kind: idx for idx, kind in enumerate(CASE_ORDER)}
    out["order"] = out["kind"].map(order).fillna(99).astype(int)
    out = out.sort_values(["order", "risk_score"], ascending=[True, False]).drop(columns=["order"]).reset_index(drop=True)
    out["run_number"] = out["run_id"].map({rid: idx + 1 for idx, rid in enumerate(sorted(timeline["run_id"].unique()))})
    return out


def run_frame_for_case(timeline: pd.DataFrame, run_id: str, max_steps: int = 180) -> pd.DataFrame:
    run_df = timeline.loc[timeline["run_id"].eq(str(run_id))].sort_values("window_ts").reset_index(drop=True)
    if run_df.empty:
        return run_df
    if len(run_df) <= max_steps:
        return run_df.copy()
    positions = set(np.linspace(0, len(run_df) - 1, max_steps).round().astype(int).tolist())
    important = pd.Series(False, index=run_df.index)
    for col in ["alert_fired", "incident_marker"]:
        if col in run_df.columns:
            important |= pd.to_numeric(run_df[col], errors="coerce").fillna(0).astype(int).eq(1)
    if "alert_state" in run_df.columns:
        important |= run_df["alert_state"].fillna("").astype(str).str.lower().isin(["alert", "incident"])
    for pos in np.flatnonzero(important.to_numpy()):
        for neighbor in [pos - 1, pos, pos + 1]:
            if 0 <= neighbor < len(run_df):
                positions.add(int(neighbor))
    positions = np.array(sorted(positions), dtype=int)
    sampled = run_df.iloc[positions].copy().reset_index(drop=True)
    sampled["_source_pos"] = positions[: len(sampled)]
    return sampled


def nearest_step(run_df: pd.DataFrame, focus_ts: pd.Timestamp) -> int:
    if run_df.empty:
        return 0
    ts = pd.to_datetime(run_df["window_ts"], utc=True, errors="coerce")
    focus = pd.to_datetime(focus_ts, utc=True, errors="coerce")
    if pd.isna(focus):
        return 0
    return int((ts - pd.Timestamp(focus)).abs().argmin())


def build_log_evidence(run_df: pd.DataFrame, current_idx: int) -> pd.DataFrame:
    current_idx = int(np.clip(current_idx, 0, max(len(run_df) - 1, 0)))
    scope = run_df.iloc[: current_idx + 1].copy()
    recent = scope.tail(min(16, max(4, len(scope)))).copy()
    current = run_df.iloc[[current_idx]].copy()
    baseline = run_df.head(max(8, len(run_df) // 4)).copy()

    rows: list[dict[str, Any]] = []
    current_log_available = bool(int(_safe_float(run_df.iloc[current_idx].get("log_available", 0), 0)))
    if not current_log_available:
        rows.append(
            {
                "Сигнал": "логи текущего окна недоступны",
                "Сейчас": 0,
                "Последние окна": int(_numeric_column(recent, "log_available", 0).clip(0, 1).sum()),
                "Префикс": int(_numeric_column(scope, "log_available", 0).clip(0, 1).sum()),
                "Δ к baseline": 0.0,
                "Смысл": "текущий score читаем как KPI/fallback",
            }
        )
    for pattern, label in LOG_PATTERNS:
        regex = re.compile(pattern, flags=re.IGNORECASE)
        current_hits = 0
        recent_hits = 0
        base_hits = 0
        prefix_hits = 0
        for _, row in current.iterrows():
            if regex.search(str(row.get("log_text", ""))):
                current_hits += 1
        for _, row in recent.iterrows():
            if regex.search(str(row.get("log_text", ""))):
                recent_hits += 1
        for _, row in scope.iterrows():
            if regex.search(str(row.get("log_text", ""))):
                prefix_hits += 1
        for _, row in baseline.iterrows():
            if regex.search(str(row.get("log_text", ""))):
                base_hits += 1
        if recent_hits <= 0 and current_hits <= 0:
            continue
        active_rate = recent_hits / max(len(recent), 1)
        base_rate = base_hits / max(len(baseline), 1)
        delta = ((active_rate + 1e-6) / (base_rate + 1e-6) - 1.0) * 100.0
        rows.append(
            {
                "Сигнал": label,
                "Сейчас": current_hits,
                "Последние окна": recent_hits,
                "Префикс": prefix_hits,
                "Δ к baseline": delta,
                "Смысл": LOG_PATTERN_MEANING.get(label, "логовый паттерн"),
            }
        )

    if not rows:
        current_text = str(run_df.iloc[current_idx].get("log_text", "") or "").strip()
        if not current_log_available or not current_text or current_text.lower() == "nan":
            rows = [
                {
                    "Сигнал": "логи текущего окна недоступны",
                    "Сейчас": 0,
                    "Последние окна": int(_numeric_column(recent, "log_available", 0).sum()),
                    "Префикс": int(_numeric_column(scope, "log_available", 0).sum()),
                    "Δ к baseline": 0.0,
                    "Смысл": "модель опирается на KPI/fallback",
                }
            ]
        else:
            rows = [
                {
                    "Сигнал": "критических логовых паттернов нет",
                    "Сейчас": 0,
                    "Последние окна": 0,
                    "Префикс": 0,
                    "Δ к baseline": 0.0,
                    "Смысл": "логи есть, но без явной аварийной лексики",
                }
            ]

    out = pd.DataFrame(rows)
    out["_priority"] = out["Сигнал"].astype(str).eq("логи текущего окна недоступны").astype(int)
    out = (
        out.sort_values(["_priority", "Сейчас", "Последние окна", "Δ к baseline"], ascending=[False, False, False, False])
        .drop(columns=["_priority"])
        .head(5)
        .reset_index(drop=True)
    )
    contrib = np.maximum(out["Сейчас"].astype(float) * 1.8 + out["Последние окна"].astype(float), 0.0)
    contrib = contrib * (1.0 + np.maximum(out["Δ к baseline"].astype(float), -90.0) / 100.0)
    denom = float(contrib.sum()) if float(contrib.sum()) > 0 else 1.0
    out["Вклад"] = contrib / denom
    out.insert(0, "Ранг", np.arange(1, len(out) + 1))
    return out


def _parse_kpi_column(column: str) -> dict[str, str]:
    parts = column.removeprefix("kpi_").split("__")
    payload = {
        "host": parts[0] if len(parts) > 0 else "",
        "family": parts[1] if len(parts) > 1 else "",
        "metric": parts[2] if len(parts) > 2 else "",
        "dim": parts[3] if len(parts) > 3 else "",
        "stat": parts[-1] if len(parts) else "",
    }
    return payload


def kpi_label(column: str) -> str:
    info = _parse_kpi_column(column)
    family = FAMILY_LABELS.get(info["family"], info["family"].upper())
    metric = METRIC_LABELS.get(info["metric"], info["metric"])
    stat = info["stat"]
    host = info["host"]
    return f"{family} · {metric} · {stat} · {host}"


def _select_kpi_columns(
    kpi_columns: list[str],
    family: str,
    metrics: list[str],
    stats: list[str] | None = None,
    dims: list[str] | None = None,
) -> list[str]:
    stats = stats or ["last", "mean"]
    dims = dims or ["max_dim", "mean_dim"]
    metric_rank = {metric: idx for idx, metric in enumerate(metrics)}
    stat_rank = {stat: idx for idx, stat in enumerate(stats)}
    dim_rank = {dim: idx for idx, dim in enumerate(dims)}
    ranked: list[tuple[tuple[int, int, int, str], str]] = []
    for col in kpi_columns:
        info = _parse_kpi_column(col)
        if info["family"] != family:
            continue
        if info["metric"] not in metric_rank or info["stat"] not in stat_rank or info["dim"] not in dim_rank:
            continue
        ranked.append(
            (
                (
                    metric_rank[info["metric"]],
                    stat_rank[info["stat"]],
                    dim_rank[info["dim"]],
                    col,
                ),
                col,
            )
        )
    return [col for _, col in sorted(ranked)]


def _aggregate_kpi_series(run_df: pd.DataFrame, columns: list[str], how: str = "max") -> pd.Series:
    present = [col for col in columns if col in run_df.columns]
    if not present:
        return pd.Series(np.nan, index=run_df.index, dtype=float)
    values = run_df[present].apply(pd.to_numeric, errors="coerce")
    if how == "min":
        out = values.min(axis=1, skipna=True)
    elif how == "sum":
        out = values.sum(axis=1, min_count=1)
    elif how == "median":
        out = values.median(axis=1, skipna=True)
    else:
        out = values.max(axis=1, skipna=True)
    return out.astype(float)


def _sum_kpi_series(left: pd.Series, right: pd.Series) -> pd.Series:
    mask = left.notna() | right.notna()
    out = left.fillna(0.0) + right.fillna(0.0)
    out.loc[~mask] = np.nan
    return out.astype(float)


def _format_metric_value(value: float, unit: str) -> str:
    if not np.isfinite(value):
        return "n/a"
    if unit == "%":
        return f"{value:.0f}%"
    if unit == "MB/s":
        return f"{value:.1f} MB/s" if abs(value) < 100 else f"{value:.0f} MB/s"
    return f"{value:.3g}"


def _make_kpi_overview_item(
    *,
    title: str,
    label: str,
    y: pd.Series,
    current_idx: int,
    unit: str,
    accent: str,
    subtitle: str,
    columns: list[str],
    run_df: pd.DataFrame,
) -> dict[str, Any]:
    current = y.iloc[current_idx] if 0 <= current_idx < len(y) else np.nan
    if pd.isna(current):
        current_value = np.nan
        baseline = np.nan
        delta = np.nan
        score = 0.0
        delta_text = "n/a"
        signal = "Нет данных"
    else:
        current_value = float(current)
        baseline_mask = run_df.index < max(3, len(run_df) // 4)
        base_values = y.loc[baseline_mask].dropna()
        if base_values.empty:
            base_values = y.dropna()
        baseline = float(base_values.median()) if not base_values.empty else current_value
        std = float(base_values.std(ddof=0)) if len(base_values) > 1 else 0.0
        delta = current_value - baseline
        score = abs(delta) / max(std, 1e-6) if std > 0 else (0.0 if abs(delta) < 1e-9 else abs(delta))
        if unit == "%":
            delta_text = f"{delta:+.1f} п.п."
        elif unit == "MB/s":
            delta_text = f"{delta:+.1f} MB/s"
        else:
            delta_text = f"{delta:+.3g}"
        signal = "Высокий" if score >= 2.0 else ("Средний" if score >= 1.0 else "Низкий")
    return {
        "column": "|".join(columns[:4]),
        "title": title,
        "label": label,
        "value": float(current_value),
        "value_text": _format_metric_value(float(current_value), unit),
        "baseline": baseline,
        "delta": delta,
        "delta_pct": delta,
        "delta_text": delta_text,
        "score": float(score),
        "signal": signal,
        "subtitle": subtitle,
        "unit": unit,
        "accent": accent,
        "x": pd.to_datetime(run_df["window_ts"], utc=True, errors="coerce"),
        "y": y,
    }


def _short_source_text(value: Any, limit: int = 78) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 1)] + "…"


def build_kpi_evidence(run_df: pd.DataFrame, kpi_columns: list[str], current_idx: int, top_n: int = 6) -> list[dict[str, Any]]:
    if run_df.empty or not kpi_columns:
        return []
    current_idx = int(np.clip(current_idx, 0, max(len(run_df) - 1, 0)))
    items: list[dict[str, Any]] = []
    compact_specs = [
        ("kpi_dashboard_cpu_pct", "CPU", "CPU", "%", "#2563eb", "агрегированная загрузка CPU"),
        ("kpi_dashboard_memory_pct", "Память", "Memory", "%", "#7c3aed", "агрегированная занятость памяти"),
        ("kpi_dashboard_disk_pct", "Диск", "Disk", "%", "#f59e0b", "агрегированная активность диска"),
        ("kpi_dashboard_network_mbps", "Сеть", "Network", "MB/s", "#0f766e", "агрегированный сетевой поток"),
    ]
    if any(col in run_df.columns for col, *_ in compact_specs):
        for col, title, label, unit, accent, subtitle in compact_specs:
            if col not in run_df.columns:
                continue
            source_text = _short_source_text(run_df.iloc[current_idx].get(f"{col}_source", ""))
            item_subtitle = f"{subtitle} · источник: {source_text}" if source_text else subtitle
            item = _make_kpi_overview_item(
                title=title,
                label=label,
                y=pd.to_numeric(run_df[col], errors="coerce"),
                current_idx=current_idx,
                unit=unit,
                accent=accent,
                subtitle=item_subtitle,
                columns=[col],
                run_df=run_df,
            )
            if item:
                items.append(item)
        if items:
            return items[:top_n]

    cpu_idle = _aggregate_kpi_series(
        run_df,
        _select_kpi_columns(kpi_columns, "cpustat", ["idle"], stats=["last", "mean"], dims=["max_dim", "mean_dim"]),
        how="min",
    )
    cpu_usage = (100.0 - cpu_idle).clip(lower=0.0, upper=100.0)
    cpu_iowait = _aggregate_kpi_series(
        run_df,
        _select_kpi_columns(kpi_columns, "cpustat", ["iowait"], stats=["last", "mean"], dims=["max_dim", "mean_dim"]),
        how="max",
    ).clip(lower=0.0, upper=100.0)
    if cpu_usage.notna().sum() < 2:
        cpu_user = _aggregate_kpi_series(
            run_df,
            _select_kpi_columns(kpi_columns, "cpustat", ["user"], stats=["last", "mean"], dims=["max_dim", "mean_dim"]),
            how="max",
        )
        cpu_system = _aggregate_kpi_series(
            run_df,
            _select_kpi_columns(kpi_columns, "cpustat", ["system"], stats=["last", "mean"], dims=["max_dim", "mean_dim"]),
            how="max",
        )
        cpu_usage = _sum_kpi_series(_sum_kpi_series(cpu_user, cpu_system), cpu_iowait).clip(lower=0.0, upper=100.0)
    cpu_wait_current = cpu_iowait.iloc[current_idx] if cpu_iowait.notna().any() else np.nan
    cpu_cols = _select_kpi_columns(kpi_columns, "cpustat", ["idle", "user", "system", "iowait"], stats=["last", "mean"], dims=["max_dim", "mean_dim"])
    item = _make_kpi_overview_item(
        title="CPU",
        label="CPU",
        y=cpu_usage,
        current_idx=current_idx,
        unit="%",
        accent="#2563eb",
        subtitle="пиковая загрузка узла" + (f" · I/O wait {cpu_wait_current:.1f}%" if np.isfinite(cpu_wait_current) else ""),
        columns=cpu_cols,
        run_df=run_df,
    )
    if item:
        items.append(item)

    mem_used = _aggregate_kpi_series(
        run_df,
        _select_kpi_columns(kpi_columns, "memstat", ["memused"], stats=["last", "mean"], dims=["max_dim", "mean_dim"]),
        how="max",
    ).clip(lower=0.0, upper=100.0)
    mem_cols = _select_kpi_columns(kpi_columns, "memstat", ["memused", "kbmemused", "commit"], stats=["last", "mean"], dims=["max_dim", "mean_dim"])
    item = _make_kpi_overview_item(
        title="Память",
        label="Memory",
        y=mem_used,
        current_idx=current_idx,
        unit="%",
        accent="#7c3aed",
        subtitle="максимальная занятость памяти среди узлов",
        columns=mem_cols,
        run_df=run_df,
    )
    if item:
        items.append(item)

    disk_util = _aggregate_kpi_series(
        run_df,
        _select_kpi_columns(kpi_columns, "iostat", ["util"], stats=["last", "mean"], dims=["max_dim", "mean_dim"]),
        how="max",
    ).clip(lower=0.0, upper=100.0)
    disk_read = _aggregate_kpi_series(
        run_df,
        _select_kpi_columns(kpi_columns, "iostat", ["rkb_s"], stats=["last", "mean"], dims=["max_dim", "mean_dim"]),
        how="max",
    )
    disk_write = _aggregate_kpi_series(
        run_df,
        _select_kpi_columns(kpi_columns, "iostat", ["wkb_s"], stats=["last", "mean"], dims=["max_dim", "mean_dim"]),
        how="max",
    )
    disk_throughput = _sum_kpi_series(disk_read, disk_write) / 1024.0
    disk_flow_current = disk_throughput.iloc[current_idx] if disk_throughput.notna().any() else np.nan
    disk_cols = _select_kpi_columns(kpi_columns, "iostat", ["util", "rkb_s", "wkb_s", "r_await", "w_await"], stats=["last", "mean"], dims=["max_dim", "mean_dim"])
    item = _make_kpi_overview_item(
        title="Диск",
        label="Disk",
        y=disk_util,
        current_idx=current_idx,
        unit="%",
        accent="#f59e0b",
        subtitle="активное время" + (f" · чтение+запись {disk_flow_current:.1f} MB/s" if np.isfinite(disk_flow_current) else ""),
        columns=disk_cols,
        run_df=run_df,
    )
    if item:
        items.append(item)

    net_rx = _aggregate_kpi_series(
        run_df,
        _select_kpi_columns(kpi_columns, "netstat", ["rxkb_s"], stats=["last", "mean"], dims=["max_dim", "mean_dim"]),
        how="max",
    )
    net_tx = _aggregate_kpi_series(
        run_df,
        _select_kpi_columns(kpi_columns, "netstat", ["txkb_s"], stats=["last", "mean"], dims=["max_dim", "mean_dim"]),
        how="max",
    )
    net_throughput = _sum_kpi_series(net_rx, net_tx) / 1024.0
    net_util = _aggregate_kpi_series(
        run_df,
        _select_kpi_columns(kpi_columns, "netstat", ["ifutil"], stats=["last", "mean"], dims=["max_dim", "mean_dim"]),
        how="max",
    ).clip(lower=0.0, upper=100.0)
    net_util_current = net_util.iloc[current_idx] if net_util.notna().any() else np.nan
    net_cols = _select_kpi_columns(kpi_columns, "netstat", ["rxkb_s", "txkb_s", "ifutil"], stats=["last", "mean"], dims=["max_dim", "mean_dim"])
    item = _make_kpi_overview_item(
        title="Сеть",
        label="Network",
        y=net_throughput,
        current_idx=current_idx,
        unit="MB/s",
        accent="#0f766e",
        subtitle="прием+передача" + (f" · util {net_util_current:.1f}%" if np.isfinite(net_util_current) else ""),
        columns=net_cols,
        run_df=run_df,
    )
    if item:
        items.append(item)

    items.sort(
        key=lambda item: (
            {"CPU": 0, "Память": 1, "Диск": 2, "Сеть": 3}.get(str(item["title"]), 99),
            -float(item["score"]),
        )
    )
    return items[:top_n]
