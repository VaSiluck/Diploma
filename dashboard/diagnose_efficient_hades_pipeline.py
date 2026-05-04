"""
Data/pipeline diagnostics for Efficient HADES hypothesis outputs.

The goal is to decide whether weak precision comes from the model itself or from
target/window/preprocessing alignment. This module reads an existing
efficient_hades_hypothesis run and writes diagnostic artifacts without retraining
the neural model.

Typical run:
    python -m dashboard.diagnose_efficient_hades_pipeline --run-dir hypothesis_results/efficient_hades/efficient_hades_cuda_full
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = ["y_anomaly", "y_warning", "y_risk"]
TOP_FRACS = [0.05, 0.10]
KPI_EXCLUDE = {"kpi_available", "kpi_avail_ratio", "kpi_feature_count", "kpi_recency_sec", "kpi_series_available"}


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat() if pd.notna(obj) else None
    try:
        if pd.isna(obj) and not isinstance(obj, str):
            return None
    except Exception:
        pass
    return obj


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def latest_run(base: Path) -> Path:
    runs = [p for p in base.iterdir() if p.is_dir() and p.name != "_cache"]
    if not runs:
        raise FileNotFoundError(f"No run directories found under {base}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def load_contract(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["window_ts", "window_end"])
    frame["split"] = frame["split"].astype(str).str.lower().replace({"valid": "validation", "val": "validation"})
    if "y_anomaly" not in frame.columns:
        frame["y_anomaly"] = np.maximum(frame["y_warning"].astype(int), frame["y_risk"].astype(int))
    if "window_index" not in frame.columns:
        frame["window_index"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.sort_values(["run_id", "window_ts", "window_index"]).reset_index(drop=True)
    frame["row_index"] = np.arange(len(frame), dtype=np.int64)
    return frame


def load_predictions(run_dir: Path, contract: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for name in ["validation_predictions.csv.gz", "test_predictions.csv.gz"]:
        path = run_dir / name
        if path.exists():
            parts.append(pd.read_csv(path, parse_dates=["window_ts"]))
    if not parts:
        raise FileNotFoundError(f"No prediction files found in {run_dir}")
    pred = pd.concat(parts, ignore_index=True)
    pred["split"] = pred.get("split_x", pred.get("split", pred.get("split_y", ""))).astype(str)
    pred["window_ts"] = pd.to_datetime(pred["window_ts"], utc=True, errors="coerce")

    meta_cols = [
        "row_index",
        "run_id",
        "workload",
        "scenario",
        "scenario_type",
        "failure_family",
        "anchor_policy",
        "window_ts",
        "window_end",
        "event_anchor",
        "fail_start",
        "strict_lead_sec",
        "strict_lead_gap_sec",
        "strict_warning_horizon_sec",
        "strict_risk_horizon_sec",
        "macro_log_count",
        "macro_warn_count",
        "macro_error_count",
        "macro_keyword_count",
        "micro_log_count",
        "micro_warn_count",
        "micro_error_count",
        "micro_keyword_count",
        *TARGETS,
    ]
    meta_cols = [col for col in meta_cols if col in contract.columns]
    meta = contract[meta_cols].copy()
    meta["window_ts"] = pd.to_datetime(meta["window_ts"], utc=True, errors="coerce")

    drop_cols = [col for col in TARGETS + ["label"] if col in pred.columns]
    pred = pred.drop(columns=drop_cols, errors="ignore")
    merged = pred.merge(meta, on=["run_id", "window_ts"], how="left", suffixes=("", "_contract"))
    if "row_index_contract" in merged.columns:
        merged["row_index"] = merged["row_index_contract"].fillna(merged.get("row_index")).astype("Int64")
        merged = merged.drop(columns=["row_index_contract"])
    return merged


def safe_ap(y, score):
    y = np.asarray(y).astype(int)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, score))


def safe_roc(y, score):
    y = np.asarray(y).astype(int)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def best_f1_threshold(y, score) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    finite = np.isfinite(score)
    if not finite.any() or len(np.unique(y[finite])) < 2:
        return 0.5
    candidates = np.unique(np.quantile(score[finite], np.linspace(0.01, 0.99, 99)))
    candidates = np.unique(np.concatenate([[0.5], candidates]))
    best = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        pred = (score >= float(threshold)).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        if float(f1) > best_f1:
            best_f1 = float(f1)
            best = float(threshold)
    return best


def metrics_at_threshold(y, score, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pred = (score >= float(threshold)).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positives": int(((pred == 1) & (y == 0)).sum()),
        "missed_anomalies": int(((pred == 0) & (y == 1)).sum()),
        "threshold": float(threshold),
    }


def lift_at_fraction(y, score, fraction: float) -> dict:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    if len(y) == 0:
        return {"top_fraction": fraction, "top_n": 0, "precision_at_top": None, "lift": None}
    k = max(1, int(math.ceil(len(y) * float(fraction))))
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf))[:k]
    precision = float(y[order].mean())
    prevalence = float(y.mean()) if len(y) else 0.0
    lift = precision / prevalence if prevalence > 0 else None
    return {"top_fraction": float(fraction), "top_n": int(k), "precision_at_top": precision, "lift": lift}


def score_separation(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        if target not in pred.columns:
            continue
        tmp = pred.loc[pred[target].notna()].copy()
        tmp["target_name"] = target
        for key, model_part in tmp.groupby(["variant", "seed", "target_name"], dropna=False):
            variant, seed, target_name = key
            threshold_source = "validation"
            source_part = model_part.loc[model_part["split"].astype(str).eq("validation")]
            if source_part.empty or source_part[target_name].astype(int).nunique() < 2:
                threshold_source = "train"
                source_part = model_part.loc[model_part["split"].astype(str).eq("train")]
            if source_part.empty or source_part[target_name].astype(int).nunique() < 2:
                threshold_source = "all_available"
                source_part = model_part
            threshold = best_f1_threshold(
                source_part[target_name].astype(int).to_numpy(),
                pd.to_numeric(source_part["score"], errors="coerce").to_numpy(dtype=float),
            )
            for split, part in model_part.groupby("split", dropna=False):
                y = part[target_name].astype(int).to_numpy()
                score = pd.to_numeric(part["score"], errors="coerce").to_numpy(dtype=float)
                pos = score[y == 1]
                neg = score[y == 0]
                row = {
                    "split": split,
                    "variant": variant,
                    "seed": int(seed),
                    "target": target_name,
                    "threshold_policy": "validation_best_f1_applied",
                    "threshold_source": threshold_source,
                    "n": int(len(part)),
                    "positives": int(y.sum()),
                    "prevalence": float(y.mean()) if len(y) else None,
                    "average_precision": safe_ap(y, score),
                    "roc_auc": safe_roc(y, score),
                    "score_mean_positive": float(np.nanmean(pos)) if len(pos) else None,
                    "score_median_positive": float(np.nanmedian(pos)) if len(pos) else None,
                    "score_mean_negative": float(np.nanmean(neg)) if len(neg) else None,
                    "score_median_negative": float(np.nanmedian(neg)) if len(neg) else None,
                    "score_mean_gap": float(np.nanmean(pos) - np.nanmean(neg)) if len(pos) and len(neg) else None,
                    **metrics_at_threshold(y, score, threshold),
                }
                for frac in TOP_FRACS:
                    lift = lift_at_fraction(y, score, frac)
                    row[f"precision_at_top_{int(frac * 100)}pct"] = lift["precision_at_top"]
                    row[f"lift_at_top_{int(frac * 100)}pct"] = lift["lift"]
                rows.append(row)
    return pd.DataFrame(rows)


def choose_best_model(target_df: pd.DataFrame, target: str = "y_anomaly") -> dict:
    part = target_df.loc[
        target_df["split"].astype(str).eq("test")
        & target_df["target"].astype(str).eq(target)
        & target_df["variant"].notna()
    ].copy()
    if part.empty:
        return {}
    part = part.sort_values(["f1", "average_precision"], ascending=[False, False])
    row = part.iloc[0]
    return {"variant": row["variant"], "seed": int(row["seed"]), "target": target}


def write_score_plot(pred: pd.DataFrame, best: dict, output_path: Path) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"matplotlib unavailable: {type(exc).__name__}: {exc}"

    if not best:
        return "no best model selected"
    part = pred.loc[
        pred["split"].astype(str).eq("test")
        & pred["variant"].astype(str).eq(str(best["variant"]))
        & pred["seed"].astype(int).eq(int(best["seed"]))
    ].copy()
    if part.empty:
        return "selected model has no test predictions"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, target in zip(axes, TARGETS):
        y = part[target].astype(int).to_numpy()
        scores = pd.to_numeric(part["score"], errors="coerce").to_numpy(dtype=float)
        ax.hist(scores[y == 0], bins=30, alpha=0.65, label="negative", density=True)
        ax.hist(scores[y == 1], bins=30, alpha=0.65, label="positive", density=True)
        ax.set_title(f"{target}: {best['variant']} seed={best['seed']}")
        ax.set_xlabel("score")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("density")
    axes[-1].legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return "ok"


def top_fp_fn_examples(pred: pd.DataFrame, target_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    best = choose_best_model(target_df, "y_anomaly")
    if not best:
        return pd.DataFrame()
    target_rows = target_df.loc[
        target_df["split"].astype(str).eq("test")
        & target_df["target"].astype(str).eq("y_anomaly")
        & target_df["variant"].astype(str).eq(str(best["variant"]))
        & target_df["seed"].astype(int).eq(int(best["seed"]))
    ]
    threshold = float(target_rows.iloc[0]["threshold"]) if not target_rows.empty else 0.5
    part = pred.loc[
        pred["split"].astype(str).eq("test")
        & pred["variant"].astype(str).eq(str(best["variant"]))
        & pred["seed"].astype(int).eq(int(best["seed"]))
    ].copy()
    if part.empty:
        return pd.DataFrame()
    part["score"] = pd.to_numeric(part["score"], errors="coerce")
    part["pred_label"] = (part["score"] >= threshold).astype(int)
    part["case_type"] = np.where(
        (part["pred_label"] == 1) & (part["y_anomaly"].astype(int) == 0),
        "false_positive",
        np.where((part["pred_label"] == 0) & (part["y_anomaly"].astype(int) == 1), "false_negative", "other"),
    )
    fps = part.loc[part["case_type"].eq("false_positive")].sort_values("score", ascending=False).head(top_n)
    fns = part.loc[part["case_type"].eq("false_negative")].sort_values("score", ascending=True).head(top_n)
    examples = pd.concat([fps, fns], ignore_index=True)
    cols = [
        "case_type",
        "variant",
        "seed",
        "score",
        "pred_label",
        "y_anomaly",
        "y_warning",
        "y_risk",
        "run_id",
        "workload",
        "scenario",
        "scenario_type",
        "failure_family",
        "anchor_policy",
        "window_ts",
        "window_end",
        "event_anchor",
        "strict_lead_sec",
        "macro_log_count",
        "macro_warn_count",
        "macro_error_count",
        "macro_keyword_count",
        "micro_log_count",
        "micro_warn_count",
        "micro_error_count",
        "micro_keyword_count",
    ]
    cols = [col for col in cols if col in examples.columns]
    return examples[cols]


def group_score_report(pred: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    best = choose_best_model(target_df, "y_anomaly")
    if not best:
        return pd.DataFrame()
    target_rows = target_df.loc[
        target_df["split"].astype(str).eq("test")
        & target_df["target"].astype(str).eq("y_anomaly")
        & target_df["variant"].astype(str).eq(str(best["variant"]))
        & target_df["seed"].astype(int).eq(int(best["seed"]))
    ]
    threshold = float(target_rows.iloc[0]["threshold"]) if not target_rows.empty else 0.5
    part = pred.loc[
        pred["split"].astype(str).eq("test")
        & pred["variant"].astype(str).eq(str(best["variant"]))
        & pred["seed"].astype(int).eq(int(best["seed"]))
    ].copy()
    if part.empty:
        return pd.DataFrame()
    part["score"] = pd.to_numeric(part["score"], errors="coerce")
    part["y"] = part["y_anomaly"].astype(int)
    part["pred_label"] = (part["score"] >= threshold).astype(int)
    rows = []
    group_cols = ["workload", "scenario_type", "scenario", "run_id"]
    group_cols = [col for col in group_cols if col in part.columns]
    for key, group in part.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        pred_label = group["pred_label"].to_numpy(dtype=int)
        y = group["y"].to_numpy(dtype=int)
        row.update(
            {
                "variant": best["variant"],
                "seed": int(best["seed"]),
                "threshold": threshold,
                "n": int(len(group)),
                "positives": int(y.sum()),
                "prevalence": float(y.mean()) if len(y) else None,
                "score_mean": float(group["score"].mean()),
                "score_median": float(group["score"].median()),
                "score_min": float(group["score"].min()),
                "score_max": float(group["score"].max()),
                "predicted_positive_rate": float(pred_label.mean()) if len(pred_label) else None,
                "false_positives": int(((pred_label == 1) & (y == 0)).sum()),
                "missed_anomalies": int(((pred_label == 0) & (y == 1)).sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["false_positives", "score_mean", "missed_anomalies"],
        ascending=[False, False, False],
    )


def numeric_kpi_columns(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in frame.columns
        if col.startswith("kpi_") and col not in KPI_EXCLUDE and pd.api.types.is_numeric_dtype(frame[col])
    ]


def select_high_variance_columns(frame: pd.DataFrame, train_idx: np.ndarray, max_features: int) -> list[str]:
    cols = numeric_kpi_columns(frame)
    if not cols:
        return []
    train = frame.iloc[train_idx][cols].apply(pd.to_numeric, errors="coerce")
    variances = train.var(axis=0, skipna=True).fillna(0.0).sort_values(ascending=False)
    return variances.head(int(max_features)).index.tolist()


def rolling_deviation_features(frame: pd.DataFrame, cols: list[str], window: int) -> pd.DataFrame:
    base = frame[["run_id", "window_ts"]].copy()
    values = frame[cols].apply(pd.to_numeric, errors="coerce")
    out_parts = []
    order = base.assign(_row=np.arange(len(frame))).sort_values(["run_id", "window_ts", "_row"])
    for _, group in order.groupby("run_id", sort=False):
        idx = group["_row"].to_numpy(dtype=int)
        part = values.iloc[idx]
        prev_mean = part.rolling(int(window), min_periods=3).mean().shift(1)
        prev_std = part.rolling(int(window), min_periods=3).std().shift(1).replace(0.0, np.nan)
        z = ((part - prev_mean) / prev_std).replace([np.inf, -np.inf], np.nan)
        delta = part.diff()
        z.columns = [f"{col}__rolling_z" for col in cols]
        delta.columns = [f"{col}__delta_prev" for col in cols]
        block = pd.concat([z, delta], axis=1)
        block["_row"] = idx
        out_parts.append(block)
    out = pd.concat(out_parts, ignore_index=True).sort_values("_row").drop(columns=["_row"]).reset_index(drop=True)
    return out


def evaluate_feature_set(x: pd.DataFrame, y: pd.Series, splits: pd.Series, target: str, feature_set: str) -> list[dict]:
    train_idx = np.flatnonzero(splits.astype(str).eq("train").to_numpy())
    val_idx = np.flatnonzero(splits.astype(str).eq("validation").to_numpy())
    test_idx = np.flatnonzero(splits.astype(str).eq("test").to_numpy())
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0 or y.iloc[train_idx].nunique() < 2:
        return []
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")),
        ]
    )
    model.fit(x.iloc[train_idx], y.iloc[train_idx].astype(int))
    val_score = model.predict_proba(x.iloc[val_idx])[:, 1]
    threshold = best_f1_threshold(y.iloc[val_idx], val_score)
    rows = []
    for split, idx in [("validation", val_idx), ("test", test_idx)]:
        score = model.predict_proba(x.iloc[idx])[:, 1]
        rows.append(
            {
                "target": target,
                "feature_set": feature_set,
                "split": split,
                "features": int(x.shape[1]),
                "average_precision": safe_ap(y.iloc[idx], score),
                "roc_auc": safe_roc(y.iloc[idx], score),
                **metrics_at_threshold(y.iloc[idx], score, threshold),
            }
        )
    return rows


def feature_ablation_raw_vs_deviation(contract: pd.DataFrame, max_features: int, rolling_window: int) -> pd.DataFrame:
    train_idx = np.flatnonzero(contract["split"].astype(str).eq("train").to_numpy())
    cols = select_high_variance_columns(contract, train_idx, max_features=max_features)
    if not cols:
        return pd.DataFrame()
    raw = contract[cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    dev = rolling_deviation_features(contract, cols, window=rolling_window)
    raw_plus = pd.concat([raw, dev], axis=1)
    rows = []
    for target in TARGETS:
        y = contract[target].astype(int).reset_index(drop=True)
        rows.extend(evaluate_feature_set(raw, y, contract["split"], target, "contract_raw_selected"))
        rows.extend(evaluate_feature_set(dev, y, contract["split"], target, "rolling_z_delta"))
        rows.extend(evaluate_feature_set(raw_plus, y, contract["split"], target, "raw_plus_rolling_z_delta"))
    return pd.DataFrame(rows)


def horizon_target_metrics(pred: pd.DataFrame, best: dict, horizons: list[int]) -> pd.DataFrame:
    if not best or "strict_lead_sec" not in pred.columns:
        return pd.DataFrame()
    part = pred.loc[
        pred["split"].astype(str).eq("test")
        & pred["variant"].astype(str).eq(str(best["variant"]))
        & pred["seed"].astype(int).eq(int(best["seed"]))
    ].copy()
    if part.empty:
        return pd.DataFrame()
    rows = []
    score = pd.to_numeric(part["score"], errors="coerce").to_numpy(dtype=float)
    scenario_failure = part["scenario_type"].astype(str).str.lower().eq("failure").to_numpy()
    lead = pd.to_numeric(part["strict_lead_sec"], errors="coerce").to_numpy(dtype=float)
    gap = pd.to_numeric(part.get("strict_lead_gap_sec", pd.Series(10, index=part.index)), errors="coerce").fillna(10).to_numpy(dtype=float)
    for horizon in horizons:
        y = (scenario_failure & np.isfinite(lead) & (lead >= gap) & (lead <= float(horizon))).astype(int)
        threshold = best_f1_threshold(y, score)
        rows.append(
            {
                "variant": best["variant"],
                "seed": int(best["seed"]),
                "synthetic_target": f"failure_lead_{horizon}s",
                "horizon_sec": int(horizon),
                "positives": int(y.sum()),
                "average_precision": safe_ap(y, score),
                "roc_auc": safe_roc(y, score),
                **metrics_at_threshold(y, score, threshold),
            }
        )
    return pd.DataFrame(rows)


def alignment_report(contract: pd.DataFrame, pred: pd.DataFrame, best: dict, run_config: dict, output_path: Path) -> pd.DataFrame:
    metric_sec = int(run_config.get("metric_segment_sec", 120))
    log_sec = int(run_config.get("log_context_sec", 120))
    df = contract.copy()
    for col in ["window_ts", "window_end", "event_anchor", "fail_start"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    if "window_end" in df.columns:
        df["metric_feature_start"] = df["window_end"] - pd.to_timedelta(metric_sec, unit="s")
        df["log_feature_start"] = df["window_end"] - pd.to_timedelta(log_sec, unit="s")
    lines = [
        "# Alignment Report",
        "",
        f"- metric_segment_sec: `{metric_sec}`",
        f"- log_context_sec: `{log_sec}`",
        f"- rows: `{len(df)}`",
        "",
        "## Split/Target Counts",
        "",
        markdown_table(
            df.groupby("split")[TARGETS].agg(["sum", "mean"]).reset_index(),
            max_rows=20,
        ),
        "",
    ]
    if {"split", "workload"}.issubset(df.columns):
        workload_counts = df.groupby(["split", "workload"])[TARGETS].agg(["count", "sum", "mean"]).reset_index()
        lines.extend(
            [
                "## Split/Workload Target Counts",
                "",
                markdown_table(workload_counts, max_rows=80),
                "",
            ]
        )
    if "strict_lead_sec" in df.columns:
        lead = pd.to_numeric(df["strict_lead_sec"], errors="coerce")
        lines.extend(
            [
                "## Lead-Time Summary",
                "",
                markdown_table(
                    pd.DataFrame(
                        [
                            {
                                "target": target,
                                "positive_rows": int(df[target].sum()),
                                "lead_min": float(lead[df[target].astype(int).eq(1)].min()),
                                "lead_median": float(lead[df[target].astype(int).eq(1)].median()),
                                "lead_max": float(lead[df[target].astype(int).eq(1)].max()),
                            }
                            for target in TARGETS
                            if target in df.columns and int(df[target].sum()) > 0
                        ]
                    )
                ),
                "",
            ]
        )
    if {"metric_feature_start", "window_end", "event_anchor"}.issubset(df.columns):
        has_anchor = df["event_anchor"].notna()
        crosses_anchor = has_anchor & (df["metric_feature_start"] <= df["event_anchor"]) & (df["window_end"] > df["event_anchor"])
        after_anchor = has_anchor & (df["window_end"] > df["event_anchor"])
        lines.extend(
            [
                "## Leakage Checks",
                "",
                f"- metric windows crossing `event_anchor`: `{int(crosses_anchor.sum())}`",
                f"- rows with `window_end > event_anchor`: `{int(after_anchor.sum())}`",
                "",
            ]
        )
    horizon_df = horizon_target_metrics(pred, best, horizons=[30, 60, 120, 300])
    if not horizon_df.empty:
        lines.extend(["## Horizon Target Diagnostics", "", markdown_table(horizon_df), ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return horizon_df


def markdown_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "_empty_"
    view = df.head(max_rows).copy()
    view.columns = ["_".join(map(str, col)).strip("_") if isinstance(col, tuple) else str(col) for col in view.columns]
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    rows = ["| " + " | ".join(str(col) for col in view.columns) + " |"]
    rows.append("| " + " | ".join(["---"] * len(view.columns)) + " |")
    for _, row in view.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
    if len(df) > max_rows:
        rows.append(f"\n_Showing first {max_rows} rows out of {len(df)}._")
    return "\n".join(rows)


def write_main_report(
    output_path: Path,
    run_dir: Path,
    target_df: pd.DataFrame,
    examples: pd.DataFrame,
    feature_ablation: pd.DataFrame,
    horizon_df: pd.DataFrame,
    group_scores: pd.DataFrame,
    plot_status: str,
) -> None:
    test = target_df.loc[target_df["split"].astype(str).eq("test")].copy()
    best_anomaly = choose_best_model(target_df, "y_anomaly")
    summary_cols = [
        "variant",
        "seed",
        "target",
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "false_positives",
        "missed_anomalies",
        "score_mean_positive",
        "score_mean_negative",
        "score_mean_gap",
        "lift_at_top_5pct",
        "lift_at_top_10pct",
    ]
    summary_cols = [col for col in summary_cols if col in test.columns]
    best_rows = (
        test.sort_values(["target", "f1", "average_precision"], ascending=[True, False, False])
        .groupby("target", as_index=False)
        .head(5)
    )
    lines = [
        "# Data Pipeline Diagnostic",
        "",
        f"- run_dir: `{run_dir}`",
        f"- selected_model_for_distributions: `{best_anomaly}`",
        f"- score_distribution_plot: `{plot_status}`",
        "",
        "## Score Separation",
        "",
        markdown_table(best_rows[summary_cols]),
        "",
        "## Interpretation",
        "",
        "- If positive/negative score means and medians are close, threshold tuning cannot fix precision by itself.",
        "- If `y_risk` scores separate better than `y_warning`, use separate heads or report risk-channel quality separately.",
        "- If rolling/deviation feature ablations beat raw selected features, preprocessing should add deviation/residual signal before changing the architecture.",
        "- If a normal workload/scenario has high median score and many false positives, treat it as split/domain shift before changing the model.",
        "",
        "## Ambiguous Examples",
        "",
        markdown_table(examples.head(20)) if not examples.empty else "_No FP/FN examples generated._",
        "",
        "## Workload/Scenario Score Shift",
        "",
        markdown_table(group_scores.head(30)) if not group_scores.empty else "_No group score report generated._",
        "",
        "## Feature Ablation",
        "",
        markdown_table(feature_ablation.loc[feature_ablation["split"].astype(str).eq("test")]) if not feature_ablation.empty else "_No feature ablation generated._",
        "",
        "## Horizon Diagnostics",
        "",
        markdown_table(horizon_df) if not horizon_df.empty else "_No horizon diagnostics generated._",
        "",
        "## Recommendation Logic",
        "",
        "1. If `average_precision`/`roc_auc` are weak across all targets, first improve preprocessing/target alignment.",
        "2. If `y_risk` is materially better than `y_warning`, split targets into multi-head warning/risk scoring.",
        "3. If deviation features outperform raw contract proxies, add rolling z-score, delta, slope and residual channels to metric preprocessing.",
        "4. If horizon targets differ strongly, rebuild chunk labels around the horizon that actually separates.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--contract-path", default="dashboard/artifacts/contract/strict_intensity_contract.csv.gz")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-ablation-features", type=int, default=80)
    parser.add_argument("--rolling-window", type=int, default=6)
    parser.add_argument("--top-examples", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_run(Path("hypothesis_results/efficient_hades"))
    contract_path = Path(args.contract_path)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "data_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config_path = run_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else {}
    contract = load_contract(contract_path)
    pred = load_predictions(run_dir, contract)

    target_df = score_separation(pred)
    target_df.to_csv(output_dir / "target_comparison.csv", index=False)

    best = choose_best_model(target_df, "y_anomaly")
    plot_status = write_score_plot(pred, best, output_dir / "score_distributions.png")

    examples = top_fp_fn_examples(pred, target_df, top_n=int(args.top_examples))
    examples.to_csv(output_dir / "top_fp_top_fn_examples.csv", index=False)

    group_scores = group_score_report(pred, target_df)
    group_scores.to_csv(output_dir / "group_score_report.csv", index=False)

    feature_ablation = feature_ablation_raw_vs_deviation(
        contract,
        max_features=int(args.max_ablation_features),
        rolling_window=int(args.rolling_window),
    )
    feature_ablation.to_csv(output_dir / "feature_ablation_raw_vs_deviation.csv", index=False)

    horizon_df = alignment_report(contract, pred, best, run_config, output_dir / "alignment_report.md")
    write_main_report(
        output_dir / "data_pipeline_diagnostic.md",
        run_dir,
        target_df,
        examples,
        feature_ablation,
        horizon_df,
        group_scores,
        plot_status,
    )
    write_json(
        output_dir / "diagnostic_summary.json",
        {
            "run_dir": str(run_dir),
            "contract_path": str(contract_path),
            "output_dir": str(output_dir),
            "best_model": best,
            "plot_status": plot_status,
            "rows": {
                "contract": int(len(contract)),
                "predictions": int(len(pred)),
                "target_comparison": int(len(target_df)),
                "examples": int(len(examples)),
                "group_scores": int(len(group_scores)),
                "feature_ablation": int(len(feature_ablation)),
            },
        },
    )
    print(f"Diagnostics written to: {output_dir}")
    print(target_df.loc[target_df["split"].astype(str).eq("test")].sort_values(["target", "f1"], ascending=[True, False]).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
