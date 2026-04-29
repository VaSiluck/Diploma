from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dashboard.contract_pipeline import ContractBundle, contract_stats
from dashboard.inference import InferenceBundleStatus


STATE_LABELS = {
    "normal": "Норма",
    "warning": "Предупреждение",
    "risk": "Риск",
}

CASE_LABELS = {
    "multimodal_risk": "мультимодальная аномалия",
    "warning_only": "только предупреждение",
    "kpi_missing": "KPI отсутствуют",
    "normal": "норма",
}

CASE_ORDER = ["multimodal_risk", "warning_only", "kpi_missing", "normal"]

LOG_PATTERNS = [
    (r"blk_update_request|buffer i/o|i/o error", "I/O error / storage"),
    (r"task blocked|blocked for more than", "task blocked / задержка"),
    (r"connection.*timeout|timeout", "connection timeout"),
    (r"failed|failure|exception|killed", "failure / exception"),
    (r"memory|memorystore|oom|out of memory", "memory pressure"),
    (r"remote fetch|shuffle|fetch", "shuffle / remote fetch"),
    (r"resourcemanager|nodemanager|applicationmaster", "YARN coordination"),
    (r"launching container|container", "container lifecycle"),
]

FAMILY_LABELS = {
    "cpustat": "CPU",
    "vmstat": "VM",
    "iostat": "Disk",
    "memstat": "Memory",
    "netstat": "Network",
}

METRIC_LABELS = {
    "iowait": "I/O wait",
    "wa": "I/O wait",
    "util": "utilization",
    "r_await": "read await",
    "w_await": "write await",
    "memused": "memory used",
    "kbmemused": "memory used",
    "commit": "commit",
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def _target_proxy_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    warning = pd.to_numeric(out["y_warning"], errors="coerce").fillna(0).astype(int)
    risk = pd.to_numeric(out["y_risk"], errors="coerce").fillna(0).astype(int)

    log_pressure = (
        pd.to_numeric(out.get("macro_error_count", 0), errors="coerce").fillna(0)
        + 0.45 * pd.to_numeric(out.get("macro_warn_count", 0), errors="coerce").fillna(0)
        + 0.18 * pd.to_numeric(out.get("macro_keyword_count", 0), errors="coerce").fillna(0)
    )
    if float(log_pressure.max()) > 0:
        log_pressure = np.log1p(log_pressure) / np.log1p(float(log_pressure.max()))
    else:
        log_pressure = pd.Series(0.0, index=out.index)

    coverage_penalty = 0.08 * (1.0 - pd.to_numeric(out.get("kpi_avail_ratio", 1.0), errors="coerce").fillna(1.0).clip(0, 1))
    base = 0.10 + 0.16 * log_pressure + coverage_penalty
    out["warning_score"] = np.clip(np.where(warning.eq(1), 0.58, base + 0.10 * log_pressure), 0.0, 1.0)
    out["risk_score"] = np.clip(np.where(risk.eq(1), 0.82, np.where(warning.eq(1), 0.54, base)), 0.0, 1.0)
    out["pred_state"] = np.select([risk.eq(1), warning.eq(1)], ["risk", "warning"], default="normal")

    log_available = pd.to_numeric(out.get("log_available", 0), errors="coerce").fillna(0).clip(0, 1)
    kpi_available = pd.to_numeric(out.get("kpi_available", 0), errors="coerce").fillna(0).clip(0, 1)
    both = pd.to_numeric(out.get("has_both_modalities", 0), errors="coerce").fillna(0).clip(0, 1)
    out["fusion_gate"] = np.where(
        (log_available.eq(1) & kpi_available.eq(1)),
        0.62 - 0.12 * both,
        np.where(log_available.eq(1), 1.0, np.where(kpi_available.eq(1), 0.0, 0.5)),
    )
    out["confidence"] = np.nan
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
        # The project currently has no compact bundle in the snapshot. The UI still
        # keeps a separate mode so real scoring can be wired without changing layout.
        timeline = _target_proxy_scores(bundle.timeline)
        mode_note = "Compact bundle найден; scoring-адаптер готов к подключению в следующем срезе."
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
    )


def choose_demo_cases(timeline: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    used: set[tuple[str, pd.Timestamp]] = set()

    def add_case(kind: str, frame: pd.DataFrame, sort_cols: list[str], ascending: list[bool], note: str) -> None:
        if frame.empty:
            return
        frame = frame.sort_values(sort_cols, ascending=ascending).copy()
        row = frame.iloc[0]
        key = (str(row["run_id"]), pd.Timestamp(row["window_ts"]))
        if key in used:
            return
        used.add(key)
        rows.append(
            {
                "case_id": f"{kind}::{key[0]}::{key[1].strftime('%Y%m%d%H%M%S')}",
                "kind": kind,
                "label": CASE_LABELS[kind],
                "run_id": str(row["run_id"]),
                "focus_window_ts": pd.Timestamp(row["window_ts"]),
                "risk_score": _safe_float(row.get("risk_score")),
                "note": note,
            }
        )

    add_case(
        "multimodal_risk",
        timeline.loc[
            timeline["y_risk"].eq(1)
            & timeline["log_available"].eq(1)
            & timeline["kpi_available"].eq(1)
        ],
        ["risk_score", "macro_error_count", "macro_warn_count"],
        [False, False, False],
        "Окно риска с доступными логами и KPI; удобно для демонстрации multimodal evidence.",
    )
    add_case(
        "warning_only",
        timeline.loc[timeline["y_warning"].eq(1) & timeline["y_risk"].eq(0)],
        ["warning_score", "macro_warn_count", "window_ts"],
        [False, False, True],
        "Окно предупреждения до зоны риска; показывает раннюю динамику деградации.",
    )
    add_case(
        "kpi_missing",
        timeline.loc[timeline["kpi_available"].eq(0) | timeline["has_both_modalities"].eq(0)],
        ["kpi_available", "risk_score", "window_ts"],
        [True, False, True],
        "Окно с неполной модальностью; интерфейс должен явно показывать fallback/degraded context.",
    )
    add_case(
        "normal",
        timeline.loc[timeline["y_warning"].eq(0) & timeline["y_risk"].eq(0)],
        ["risk_score", "window_ts"],
        [True, True],
        "Нормальный фрагмент для сравнения с предупреждением и риском.",
    )

    if not rows:
        first = timeline.iloc[0]
        rows.append(
            {
                "case_id": f"normal::{first['run_id']}::{pd.Timestamp(first['window_ts']).strftime('%Y%m%d%H%M%S')}",
                "kind": "normal",
                "label": CASE_LABELS["normal"],
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


def run_frame_for_case(timeline: pd.DataFrame, run_id: str, max_steps: int = 36) -> pd.DataFrame:
    run_df = timeline.loc[timeline["run_id"].eq(str(run_id))].sort_values("window_ts").reset_index(drop=True)
    if run_df.empty:
        return run_df
    if len(run_df) <= max_steps:
        return run_df.copy()
    positions = np.linspace(0, len(run_df) - 1, max_steps).round().astype(int)
    positions = np.unique(positions)
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
    baseline = run_df.head(max(3, len(run_df) // 4)).copy()

    rows: list[dict[str, Any]] = []
    for pattern, label in LOG_PATTERNS:
        regex = re.compile(pattern, flags=re.IGNORECASE)
        active = 0
        base = 0
        for _, row in scope.iterrows():
            if regex.search(str(row.get("log_text", ""))):
                active += max(1, int(_safe_float(row.get("micro_log_count", row.get("macro_log_count", 1)), 1)))
        for _, row in baseline.iterrows():
            if regex.search(str(row.get("log_text", ""))):
                base += max(1, int(_safe_float(row.get("micro_log_count", row.get("macro_log_count", 1)), 1)))
        if active <= 0:
            continue
        active_rate = active / max(len(scope), 1)
        base_rate = base / max(len(baseline), 1)
        delta = ((active_rate + 1e-6) / (base_rate + 1e-6) - 1.0) * 100.0
        rows.append({"Шаблон": label, "Частота": active, "Δ к baseline": delta})

    if not rows:
        rows = [{"Шаблон": "нет выраженных логовых паттернов", "Частота": 0, "Δ к baseline": 0.0}]

    out = pd.DataFrame(rows).sort_values(["Частота", "Δ к baseline"], ascending=[False, False]).head(5).reset_index(drop=True)
    contrib = np.maximum(out["Частота"].astype(float), 0.0) * (1.0 + np.maximum(out["Δ к baseline"].astype(float), -90.0) / 100.0)
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


def build_kpi_evidence(run_df: pd.DataFrame, kpi_columns: list[str], current_idx: int, top_n: int = 6) -> list[dict[str, Any]]:
    if run_df.empty or not kpi_columns:
        return []
    current_idx = int(np.clip(current_idx, 0, max(len(run_df) - 1, 0)))
    baseline_mask = run_df.index < max(3, len(run_df) // 4)
    items: list[dict[str, Any]] = []
    for col in kpi_columns:
        if col not in run_df.columns:
            continue
        values = pd.to_numeric(run_df[col], errors="coerce")
        if int(values.notna().sum()) < 2:
            continue
        current = values.iloc[current_idx]
        if pd.isna(current):
            continue
        base_values = values.loc[baseline_mask].dropna()
        if base_values.empty:
            base_values = values.dropna()
        baseline = float(base_values.median()) if not base_values.empty else float(current)
        std = float(base_values.std(ddof=0)) if len(base_values) > 1 else 0.0
        score = abs(float(current) - baseline) / max(std, 1e-6)
        delta_pct = ((float(current) - baseline) / max(abs(baseline), 1e-6)) * 100.0
        items.append(
            {
                "column": col,
                "label": kpi_label(col),
                "value": float(current),
                "baseline": baseline,
                "delta_pct": float(delta_pct),
                "score": float(score),
                "signal": "Высокий" if score >= 2.0 else ("Средний" if score >= 1.0 else "Низкий"),
                "x": pd.to_datetime(run_df["window_ts"], utc=True, errors="coerce"),
                "y": values,
            }
        )
    items.sort(key=lambda item: item["score"], reverse=True)
    return items[:top_n]

