from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET_SCORE_COLUMNS = {
    "y_warning": "warning_score",
    "y_risk": "risk_score",
}

TARGET_SHORT_NAMES = {
    "y_warning": "warning",
    "y_risk": "risk",
}

CENTRAL_MODEL_NAME = "central-fusion-model"
MIN_MEAN_AP_IMPROVEMENT = 0.02


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def anchor_reliability_label(anchor_policy: str) -> str:
    policy = str(anchor_policy or "").strip()
    if policy == "explicit_raw_log_keyword_or_severity":
        return "явный лог-якорь"
    if policy in {"run_end_proxy", "run_end_proxy_no_log_events"}:
        return "proxy по концу run"
    if policy == "normal_no_failure":
        return "нормальный запуск"
    return "неизвестный якорь"


def event_anchor_label(row: pd.Series) -> str:
    policy = str(row.get("anchor_policy", ""))
    if policy == "normal_no_failure" or str(row.get("scenario_type", "")) == "normal":
        return "Нет сбоя"
    return "Якорь события"


def build_label_provenance(timeline: pd.DataFrame) -> dict[str, Any]:
    if timeline.empty or "anchor_policy" not in timeline.columns:
        return {
            "available": False,
            "warning": "anchor_policy отсутствует; точность event anchor не может быть оценена.",
        }

    runs = timeline.drop_duplicates("run_id").copy() if "run_id" in timeline.columns else timeline.copy()
    counts = runs["anchor_policy"].fillna("unknown").astype(str).value_counts().to_dict()
    total = max(int(len(runs)), 1)
    explicit = int(counts.get("explicit_raw_log_keyword_or_severity", 0))
    proxy = int(counts.get("run_end_proxy", 0)) + int(counts.get("run_end_proxy_no_log_events", 0))
    normal = int(counts.get("normal_no_failure", 0))
    return {
        "available": True,
        "unit": "runs",
        "total_runs": total,
        "counts": {key: int(value) for key, value in counts.items()},
        "explicit_share": explicit / total,
        "run_end_proxy_share": proxy / total,
        "normal_share": normal / total,
        "warnings": [
            "run_end_proxy означает приближение по концу run, а не точное начало сбоя.",
            "Внешнего fault-onset ground truth в текущем контракте нет.",
        ],
    }


def build_baseline_diagnostics(baseline_comparison: pd.DataFrame) -> list[dict[str, Any]]:
    if baseline_comparison.empty:
        return []

    table = baseline_comparison.copy()
    if "status" in table.columns:
        table = table.loc[table["status"].fillna("ok").eq("ok")].copy()
    if "slice" in table.columns:
        table = table.loc[table["slice"].fillna("all").eq("all")].copy()

    rows: list[dict[str, Any]] = []
    for target in TARGET_SCORE_COLUMNS:
        target_rows = table.loc[table["target"].astype(str).eq(target)].copy()
        if target_rows.empty:
            continue

        central_rows = target_rows.loc[target_rows["baseline"].astype(str).eq(CENTRAL_MODEL_NAME)].copy()
        non_central = target_rows.loc[~target_rows["baseline"].astype(str).eq(CENTRAL_MODEL_NAME)].copy()
        if central_rows.empty:
            continue

        central = central_rows.sort_values("average_precision", ascending=False).iloc[0]
        if non_central.empty:
            best_baseline = None
            best_ap = None
            best_name = None
        else:
            best_baseline = non_central.sort_values("average_precision", ascending=False).iloc[0]
            best_ap = _safe_float(best_baseline.get("average_precision"))
            best_name = str(best_baseline.get("baseline", "baseline"))

        central_ap = _safe_float(central.get("average_precision"))
        prevalence = _safe_float(central.get("prevalence"), 0.0)
        delta_vs_best = None if central_ap is None or best_ap is None else central_ap - best_ap
        delta_vs_random = None if central_ap is None or prevalence is None else central_ap - prevalence
        if delta_vs_best is None:
            verdict = "нет baseline"
        elif delta_vs_best >= MIN_MEAN_AP_IMPROVEMENT:
            verdict = "лучше baseline"
        elif delta_vs_best <= -MIN_MEAN_AP_IMPROVEMENT:
            verdict = "хуже baseline"
        else:
            verdict = "паритет"

        rows.append(
            {
                "target": target,
                "target_short": TARGET_SHORT_NAMES[target],
                "central_model": CENTRAL_MODEL_NAME,
                "central_average_precision": central_ap,
                "central_roc_auc": _safe_float(central.get("roc_auc")),
                "central_f1": _safe_float(central.get("f1")),
                "central_precision": _safe_float(central.get("precision")),
                "central_recall": _safe_float(central.get("recall")),
                "central_threshold": _safe_float(central.get("threshold")),
                "random_reference_ap": prevalence,
                "best_baseline": best_name,
                "best_baseline_average_precision": best_ap,
                "best_baseline_roc_auc": _safe_float(best_baseline.get("roc_auc")) if best_baseline is not None else None,
                "best_baseline_f1": _safe_float(best_baseline.get("f1")) if best_baseline is not None else None,
                "best_baseline_recall": _safe_float(best_baseline.get("recall")) if best_baseline is not None else None,
                "delta_vs_random_ap": delta_vs_random,
                "delta_vs_best_baseline_ap": delta_vs_best,
                "verdict": verdict,
            }
        )
    return rows


def build_calibration_bins(prediction_store: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if prediction_store.empty:
        return pd.DataFrame()

    for target, score_col in TARGET_SCORE_COLUMNS.items():
        if target not in prediction_store.columns or score_col not in prediction_store.columns:
            continue
        y_true = pd.to_numeric(prediction_store[target], errors="coerce")
        score = pd.to_numeric(prediction_store[score_col], errors="coerce").clip(0, 1)
        valid = y_true.notna() & score.notna()
        if not bool(valid.any()):
            continue

        frame = pd.DataFrame({"y_true": y_true[valid].astype(int), "score": score[valid]})
        frame["bin"] = pd.cut(
            frame["score"],
            bins=np.linspace(0.0, 1.0, bins + 1),
            include_lowest=True,
            duplicates="drop",
        )
        grouped = frame.groupby("bin", observed=False)
        for interval, group in grouped:
            if group.empty:
                continue
            rows.append(
                {
                    "target": target,
                    "bin_left": float(interval.left),
                    "bin_right": float(interval.right),
                    "n": int(len(group)),
                    "mean_score": float(group["score"].mean()),
                    "event_rate": float(group["y_true"].mean()),
                    "abs_calibration_error": float(abs(group["score"].mean() - group["y_true"].mean())),
                }
            )
    return pd.DataFrame(rows)


def _best_baseline_names(baseline_comparison: pd.DataFrame) -> dict[str, str]:
    best: dict[str, str] = {}
    if baseline_comparison.empty:
        return best
    table = baseline_comparison.copy()
    if "status" in table.columns:
        table = table.loc[table["status"].fillna("ok").eq("ok")].copy()
    if "slice" in table.columns:
        table = table.loc[table["slice"].fillna("all").eq("all")].copy()
    table = table.loc[~table["baseline"].astype(str).eq(CENTRAL_MODEL_NAME)].copy()
    for target in TARGET_SCORE_COLUMNS:
        rows = table.loc[table["target"].astype(str).eq(target)].copy()
        if rows.empty:
            continue
        best[target] = str(rows.sort_values("average_precision", ascending=False).iloc[0]["baseline"])
    return best


def _pivot_baseline_scores(
    baseline_predictions: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
) -> pd.DataFrame:
    if baseline_predictions.empty:
        return pd.DataFrame()

    best_names = _best_baseline_names(baseline_comparison)
    frames = []
    for target, score_col in TARGET_SCORE_COLUMNS.items():
        baseline_name = best_names.get(target)
        if not baseline_name:
            continue
        rows = baseline_predictions.loc[
            baseline_predictions["target"].astype(str).eq(target)
            & baseline_predictions["baseline"].astype(str).eq(baseline_name)
        ].copy()
        if rows.empty:
            continue
        short = TARGET_SHORT_NAMES[target]
        rows["run_id"] = rows["run_id"].astype(str)
        rows["window_ts"] = pd.to_datetime(rows["window_ts"], utc=True, errors="coerce")
        frames.append(
            rows[["run_id", "window_ts", "score"]]
            .rename(columns={"score": f"baseline_{short}_score"})
            .assign(**{f"baseline_{short}_model": baseline_name})
        )

    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=["run_id", "window_ts"], how="outer")
    return out


def _aggregate_series(frame: pd.DataFrame, include: list[str], how: str) -> pd.Series:
    columns = [col for col in frame.columns if all(token in col for token in include)]
    if not columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    if how == "min":
        return values.min(axis=1, skipna=True).astype(float)
    if how == "sum":
        return values.sum(axis=1, min_count=1).astype(float)
    return values.max(axis=1, skipna=True).astype(float)


def add_dashboard_kpi_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    cpu_idle = _aggregate_series(out, ["__cpustat__", "__idle__", "__last"], "min")
    out["kpi_dashboard_cpu_pct"] = (100.0 - cpu_idle).clip(lower=0.0, upper=100.0)
    out["kpi_dashboard_memory_pct"] = _aggregate_series(out, ["__memstat__", "__memused__", "__last"], "max").clip(0, 100)
    out["kpi_dashboard_disk_pct"] = _aggregate_series(out, ["__iostat__", "__util__", "__last"], "max").clip(0, 100)
    net_rx = _aggregate_series(out, ["__netstat__", "__rxkb_s__", "__last"], "max")
    net_tx = _aggregate_series(out, ["__netstat__", "__txkb_s__", "__last"], "max")
    out["kpi_dashboard_network_mbps"] = (net_rx.fillna(0.0) + net_tx.fillna(0.0)) / 1024.0
    out.loc[net_rx.isna() & net_tx.isna(), "kpi_dashboard_network_mbps"] = np.nan
    return out


def build_dashboard_replay_store(
    timeline: pd.DataFrame,
    prediction_store: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
) -> pd.DataFrame:
    if prediction_store.empty:
        return pd.DataFrame()

    pred = prediction_store.copy()
    pred["run_id"] = pred["run_id"].astype(str)
    pred["window_ts"] = pd.to_datetime(pred["window_ts"], utc=True, errors="coerce")
    pred = pred.dropna(subset=["run_id", "window_ts"]).drop_duplicates(["run_id", "window_ts"], keep="last")
    if "split" in pred.columns:
        pred = pred.loc[pred["split"].astype(str).eq("test")].copy()

    context_cols = [
        "run_id",
        "window_ts",
        "window_end",
        "run_start",
        "run_end",
        "event_anchor",
        "fail_start",
        "anchor_policy",
        "minutes_before_fail",
        "log_text",
        "macro_log_count",
        "macro_warn_count",
        "macro_error_count",
        "macro_keyword_count",
        "micro_log_count",
        "micro_warn_count",
        "micro_error_count",
        "micro_keyword_count",
    ]
    context_cols = [col for col in context_cols if col in timeline.columns]
    kpi_signal_cols = [
        col
        for col in timeline.columns
        if any(token in col for token in ["__cpustat__", "__memstat__", "__iostat__", "__netstat__"])
    ]
    context = timeline[context_cols + kpi_signal_cols].copy()
    context["run_id"] = context["run_id"].astype(str)
    context["window_ts"] = pd.to_datetime(context["window_ts"], utc=True, errors="coerce")
    context = add_dashboard_kpi_aggregates(context)
    kpi_dashboard_cols = [
        "kpi_dashboard_cpu_pct",
        "kpi_dashboard_memory_pct",
        "kpi_dashboard_disk_pct",
        "kpi_dashboard_network_mbps",
    ]
    context = context[context_cols + [col for col in kpi_dashboard_cols if col in context.columns]]

    keep_pred = [
        "run_id",
        "window_ts",
        "split",
        "workload",
        "scenario",
        "scenario_type",
        "failure_family",
        "anchor_policy",
        "log_available",
        "kpi_available",
        "has_both_modalities",
        "kpi_avail_ratio",
        "y_warning",
        "y_risk",
        "warning_score",
        "risk_score",
        "fusion_gate",
        "threshold_warning",
        "threshold_risk",
        "pred_warning",
        "pred_risk",
        "pred_state",
    ]
    keep_pred = [col for col in keep_pred if col in pred.columns]
    out = pred[keep_pred].merge(context, on=["run_id", "window_ts"], how="left", suffixes=("", "_contract"))
    if "anchor_policy_contract" in out.columns:
        out["anchor_policy"] = out["anchor_policy"].fillna(out["anchor_policy_contract"])
        out = out.drop(columns=["anchor_policy_contract"])

    baseline_scores = _pivot_baseline_scores(baseline_predictions, baseline_comparison)
    if not baseline_scores.empty:
        out = out.merge(baseline_scores, on=["run_id", "window_ts"], how="left")

    out["label_anchor"] = out.apply(event_anchor_label, axis=1)
    out["anchor_reliability"] = out["anchor_policy"].map(anchor_reliability_label)
    out["score_source"] = "model_store"
    return out.sort_values(["run_id", "window_ts"]).reset_index(drop=True)


def build_training_verdict(baseline_diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [
        row["delta_vs_best_baseline_ap"]
        for row in baseline_diagnostics
        if row.get("delta_vs_best_baseline_ap") is not None
    ]
    if not deltas:
        return {"status": "unknown", "message": "Нет достаточных baseline-сравнений."}
    min_delta = min(float(delta) for delta in deltas)
    max_delta = max(float(delta) for delta in deltas)
    if min_delta >= MIN_MEAN_AP_IMPROVEMENT:
        status = "better_than_baseline"
        message = "central fusion устойчиво превосходит лучший baseline по AP."
    elif max_delta <= -MIN_MEAN_AP_IMPROVEMENT:
        status = "worse_than_baseline"
        message = "central fusion не улучшает сильный baseline на текущем контракте."
    else:
        status = "mixed_or_parity"
        message = "central fusion находится в зоне паритета/смешанных результатов относительно baseline."
    return {
        "status": status,
        "message": message,
        "min_delta_vs_best_baseline_ap": min_delta,
        "max_delta_vs_best_baseline_ap": max_delta,
        "ap_margin": MIN_MEAN_AP_IMPROVEMENT,
    }


def build_seed_metrics(metrics_payload: dict[str, Any], existing_seed_metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    if existing_seed_metrics is not None and not existing_seed_metrics.empty and "seed" in existing_seed_metrics.columns:
        return existing_seed_metrics.copy()

    rows = []
    for row in metrics_payload.get("primary_metrics", []):
        if row.get("slice") != "all":
            continue
        rows.append(
            {
                "seed": metrics_payload.get("seed", 42),
                "status": "current_single_seed",
                "target": row.get("target"),
                "average_precision": row.get("average_precision"),
                "roc_auc": row.get("roc_auc"),
                "f1": row.get("f1"),
                "precision": row.get("precision"),
                "recall": row.get("recall"),
            }
        )
    if not rows:
        return pd.DataFrame([{"seed": 42, "status": "pending_clean_3_seed_rerun"}])
    return pd.DataFrame(rows)


def build_dashboard_model_report(
    timeline: pd.DataFrame,
    prediction_store: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    metrics_payload: dict[str, Any],
    seed_metrics: pd.DataFrame,
) -> dict[str, Any]:
    split_summary = []
    if "split" in timeline.columns:
        for split, group in timeline.groupby("split"):
            split_summary.append(
                {
                    "split": str(split),
                    "rows": int(len(group)),
                    "runs": int(group["run_id"].nunique()) if "run_id" in group.columns else None,
                    "warning_prevalence": _safe_float(pd.to_numeric(group.get("y_warning"), errors="coerce").mean()),
                    "risk_prevalence": _safe_float(pd.to_numeric(group.get("y_risk"), errors="coerce").mean()),
                }
            )

    baseline_diagnostics = build_baseline_diagnostics(baseline_comparison)
    seed_status = "pending_clean_3_seed_rerun"
    if not seed_metrics.empty and "seed" in seed_metrics.columns and seed_metrics["seed"].nunique() >= 3:
        seed_status = "complete_3_seed"

    warnings = [
        "Score не является откалиброванной вероятностью.",
        "Dashboard claims строятся по test split и baseline deltas, без победных формулировок при проигрыше baseline.",
    ]
    warnings.extend(build_label_provenance(timeline).get("warnings", []))

    return {
        "schema_version": 1,
        "split_summary": split_summary,
        "label_provenance": build_label_provenance(timeline),
        "baseline_diagnostics": baseline_diagnostics,
        "training_verdict": build_training_verdict(baseline_diagnostics),
        "seed_protocol": {
            "expected_seeds": [42, 43, 44],
            "status": seed_status,
            "observed_seeds": sorted(seed_metrics["seed"].dropna().astype(int).unique().tolist()) if "seed" in seed_metrics.columns else [],
        },
        "thresholds": {
            "warning": metrics_payload.get("threshold_warning"),
            "risk": metrics_payload.get("threshold_risk"),
        },
        "warnings": warnings,
    }


def write_dashboard_diagnostics(
    *,
    timeline: pd.DataFrame,
    prediction_store: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    metrics_payload: dict[str, Any],
    output_dir: Path,
    existing_seed_metrics: pd.DataFrame | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics = build_seed_metrics(metrics_payload, existing_seed_metrics)
    calibration_bins = build_calibration_bins(prediction_store)
    replay_store = build_dashboard_replay_store(
        timeline=timeline,
        prediction_store=prediction_store,
        baseline_predictions=baseline_predictions,
        baseline_comparison=baseline_comparison,
    )
    report = build_dashboard_model_report(
        timeline=timeline,
        prediction_store=prediction_store,
        baseline_comparison=baseline_comparison,
        metrics_payload=metrics_payload,
        seed_metrics=seed_metrics,
    )

    seed_metrics.to_csv(output_dir / "seed_metrics.csv", index=False)
    calibration_bins.to_csv(output_dir / "calibration_bins.csv", index=False)
    replay_store.to_csv(output_dir / "dashboard_replay_store.csv", index=False)
    write_json(output_dir / "dashboard_model_report.json", report)

    return {
        "dashboard_model_report": str(output_dir / "dashboard_model_report.json"),
        "dashboard_replay_store": str(output_dir / "dashboard_replay_store.csv"),
        "seed_metrics": str(output_dir / "seed_metrics.csv"),
        "calibration_bins": str(output_dir / "calibration_bins.csv"),
    }
