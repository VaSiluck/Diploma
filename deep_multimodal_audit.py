"""
Deep forensic audit for the multimodal AIOps pipeline.

This script is intentionally diagnostic, not another model candidate.
It checks whether the current results are caused by model weakness,
split/domain shift, label-policy shortcuts, temporal leakage around the
failure anchor, or modality preprocessing losses.

Typical run:
    python deep_multimodal_audit.py --run-id audit_full

Faster local smoke:
    python deep_multimodal_audit.py --run-id audit_smoke --max-tfidf-features 512 --random-seeds 42 --skip-fused-random
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


RANDOM_SEED = 42

META_COLUMNS = {
    "workload",
    "scenario",
    "run_id",
    "group_id",
    "failure_family",
    "scenario_type",
    "window_ts",
    "window_end",
    "run_start",
    "run_end",
    "fail_start",
    "event_anchor",
    "anchor_policy",
    "minutes_before_fail",
    "macro_window_sec",
    "micro_window_sec",
    "state",
    "y_warning",
    "y_risk",
    "y_anomaly",
    "target_available",
    "target_policy",
    "log_text",
    "parsed_label_available",
    "strict_path_pair",
    "window_index",
    "risk_horizon_sec",
    "warning_horizon_sec",
    "split",
    "split_group",
}

AVAILABILITY_FEATURES = [
    "log_available",
    "kpi_available",
    "has_both_modalities",
    "kpi_avail_ratio",
    "kpi_recency_sec",
    "kpi_series_available",
    "kpi_feature_count",
]

LOG_NUMERIC_FEATURES = [
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

LOG_NUMERIC_NO_SEVERITY_FEATURES = [
    "macro_log_count",
    "macro_unique_components",
    "macro_log_rate",
    "micro_log_count",
    "micro_unique_components",
    "micro_log_rate",
]

SHORTCUT_TERMS = [
    "error",
    "errors",
    "exception",
    "failed",
    "failure",
    "fatal",
    "critical",
    "warn",
    "warning",
    "killed",
    "kill",
    "lost",
    "timeout",
    "timed",
    "suspended",
    "blacklist",
    "retry",
    "unhealthy",
    "refused",
    "denied",
    "namenode",
    "datanode",
    "nodemanager",
    "resourcemanager",
    "worker",
    "master",
    "stress",
    "slow",
    "delay",
]


@dataclass
class AuditConfig:
    run_id: str
    root: str
    output_dir: str
    contract_path: str
    baseline_predictions_path: str | None
    baseline_summary_path: str | None
    max_tfidf_features: int
    random_seeds: list[int]
    skip_fused_random: bool


class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column: str = "log_text", redacted: bool = False):
        self.column = column
        self.redacted = redacted

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        values = X[self.column].fillna("").astype(str).to_numpy()
        if self.redacted:
            values = np.array([redact_log_text(v) for v in values], dtype=object)
        return values


def first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if pd.isna(obj) if not isinstance(obj, (list, tuple, dict, np.ndarray)) else False:
        return None
    return obj


def safe_ap(y_true, score) -> float | None:
    y = pd.Series(y_true).astype(int)
    if y.nunique() < 2:
        return None
    return float(average_precision_score(y, score))


def safe_roc(y_true, score) -> float | None:
    y = pd.Series(y_true).astype(int)
    if y.nunique() < 2:
        return None
    return float(roc_auc_score(y, score))


def threshold_metrics(y_true, score, threshold: float = 0.5) -> dict:
    pred = np.asarray(score) >= threshold
    p, r, f1, _ = precision_recall_fscore_support(
        np.asarray(y_true).astype(int),
        pred.astype(int),
        average="binary",
        zero_division=0,
    )
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "threshold": float(threshold)}


def metric_row(y_true, score, extra: dict | None = None) -> dict:
    y = pd.Series(y_true).astype(int)
    s = pd.Series(score).astype(float)
    row = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else None,
        "average_precision": safe_ap(y, s),
        "roc_auc": safe_roc(y, s),
    }
    row.update(threshold_metrics(y, s, 0.5))
    if extra:
        row.update(extra)
    return row


def redact_log_text(text: str) -> str:
    value = str(text).lower()
    for term in SHORTCUT_TERMS:
        value = re.sub(rf"\b{re.escape(term)}\b", "<shortcut>", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def load_contract(path: Path) -> pd.DataFrame:
    parse_dates = ["window_ts", "window_end", "run_start", "run_end", "fail_start", "event_anchor"]
    frame = pd.read_csv(path, parse_dates=[c for c in parse_dates if c in pd.read_csv(path, nrows=0).columns])
    frame["split"] = frame["split"].astype(str).str.lower().replace({"valid": "validation", "val": "validation"})
    frame["log_text"] = frame.get("log_text", "").fillna("").astype(str)
    for col in AVAILABILITY_FEATURES + LOG_NUMERIC_FEATURES:
        if col not in frame.columns:
            frame[col] = 0.0
    return frame.sort_values(["run_id", "window_ts", "window_index"]).reset_index(drop=True)


def feature_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    availability = [c for c in AVAILABILITY_FEATURES if c in frame.columns and c not in META_COLUMNS]
    log_numeric = [c for c in LOG_NUMERIC_FEATURES if c in frame.columns and c not in META_COLUMNS]
    log_numeric_no_severity = [c for c in LOG_NUMERIC_NO_SEVERITY_FEATURES if c in frame.columns and c not in META_COLUMNS]
    kpi_numeric = [
        c
        for c in frame.columns
        if c.startswith("kpi_") and "__" in c and c not in META_COLUMNS and frame[c].notna().any()
    ]
    return {
        "availability": availability,
        "log_numeric": log_numeric,
        "log_numeric_no_severity": log_numeric_no_severity,
        "kpi_numeric": kpi_numeric,
    }


def make_model(frame: pd.DataFrame, model_name: str, groups: dict[str, list[str]], max_tfidf_features: int) -> Pipeline:
    model_specs = {
        "availability-only": {"numeric": groups["availability"], "text": False, "redacted": False},
        "log-count-only": {"numeric": groups["log_numeric"], "text": False, "redacted": False},
        "log-count-no-severity": {"numeric": groups["log_numeric_no_severity"], "text": False, "redacted": False},
        "log-text-only": {"numeric": [], "text": True, "redacted": False},
        "log-text-redacted": {"numeric": [], "text": True, "redacted": True},
        "log-only": {"numeric": groups["log_numeric"], "text": True, "redacted": False},
        "log-only-redacted": {"numeric": groups["log_numeric_no_severity"], "text": True, "redacted": True},
        "kpi-only": {"numeric": groups["availability"] + groups["kpi_numeric"], "text": False, "redacted": False},
        "fused": {"numeric": groups["availability"] + groups["log_numeric"] + groups["kpi_numeric"], "text": True, "redacted": False},
    }
    if model_name not in model_specs:
        raise KeyError(f"Unknown model_name: {model_name}")
    spec = model_specs[model_name]
    transformers = []
    if spec["numeric"]:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                spec["numeric"],
            )
        )
    if spec["text"]:
        transformers.append(
            (
                "text",
                Pipeline(
                    [
                        ("selector", TextSelector("log_text", redacted=bool(spec["redacted"]))),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                max_features=int(max_tfidf_features),
                                min_df=2,
                                ngram_range=(1, 2),
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
                ["log_text"],
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.35)
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_SEED,
    )
    return Pipeline([("features", preprocessor), ("clf", clf)])


def predict_scores(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model.named_steps["clf"], "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def evaluate_model(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target: str,
    model_name: str,
    groups: dict[str, list[str]],
    max_tfidf_features: int,
    protocol: str,
    seed: int | None = None,
) -> tuple[dict, Pipeline | None]:
    y_train = frame.iloc[train_idx][target].astype(int)
    y_test = frame.iloc[test_idx][target].astype(int)
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        row = {
            "protocol": protocol,
            "seed": seed,
            "target": target,
            "model": model_name,
            "status": "skipped_single_class",
            "n": int(len(test_idx)),
            "positives": int(y_test.sum()),
            "prevalence": float(y_test.mean()) if len(y_test) else None,
        }
        return row, None
    try:
        model = make_model(frame, model_name, groups, max_tfidf_features)
        model.fit(frame.iloc[train_idx], y_train)
        score = predict_scores(model, frame.iloc[test_idx])
        row = metric_row(
            y_test,
            score,
            {
                "protocol": protocol,
                "seed": seed,
                "target": target,
                "model": model_name,
                "status": "ok",
                "train_n": int(len(train_idx)),
                "train_positives": int(y_train.sum()),
                "train_prevalence": float(y_train.mean()),
            },
        )
        return row, model
    except Exception as exc:
        return {
            "protocol": protocol,
            "seed": seed,
            "target": target,
            "model": model_name,
            "status": f"error: {type(exc).__name__}: {exc}",
        }, None


def evaluate_predictions_by_slices(
    pred: pd.DataFrame,
    contract: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    pred = pred.copy()
    pred["window_ts"] = pd.to_datetime(pred["window_ts"], utc=True, errors="coerce")
    keys = [
        "run_id",
        "window_ts",
        "workload",
        "scenario",
        "failure_family",
        "scenario_type",
        "anchor_policy",
        "log_available",
        "kpi_available",
        "has_both_modalities",
        "y_warning",
        "y_risk",
    ]
    key = contract[keys].copy()
    key["window_ts"] = pd.to_datetime(key["window_ts"], utc=True, errors="coerce")
    merged = pred.merge(key, on=["run_id", "window_ts", "workload", "scenario", "scenario_type", "failure_family", "anchor_policy"], how="left")
    rows = []
    slice_specs: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
        ("all", lambda d: pd.Series(True, index=d.index)),
        ("both_modalities", lambda d: d["has_both_modalities"].astype(float).gt(0.5)),
        ("kpi_only", lambda d: d["kpi_available"].astype(float).gt(0.5) & d["log_available"].astype(float).le(0.5)),
        ("log_available", lambda d: d["log_available"].astype(float).gt(0.5)),
        ("explicit_anchor", lambda d: d["anchor_policy"].astype(str).eq("explicit_raw_log_keyword_or_severity")),
        ("run_end_proxy", lambda d: d["anchor_policy"].astype(str).eq("run_end_proxy")),
        ("workload=svd", lambda d: d["workload"].astype(str).eq("svd")),
    ]
    for target in ["y_warning", "y_risk"]:
        target_df = merged.loc[merged["target"].astype(str).eq(target)].copy()
        y_col = target
        for baseline in sorted(target_df["baseline"].astype(str).unique()):
            base_df = target_df.loc[target_df["baseline"].astype(str).eq(baseline)]
            for slice_name, mask_fn in slice_specs:
                mask = mask_fn(base_df)
                part = base_df.loc[mask].copy()
                if part.empty or part[y_col].isna().all():
                    continue
                rows.append(
                    metric_row(
                        part[y_col].astype(int),
                        part["score"].astype(float),
                        {
                            "target": target,
                            "baseline": baseline,
                            "slice": slice_name,
                        },
                    )
                )
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "baseline_prediction_slice_metrics.csv", index=False)
    return out


def contract_overview(frame: pd.DataFrame, groups: dict[str, list[str]], output_dir: Path) -> dict:
    overview = {
        "rows": int(len(frame)),
        "runs": int(frame["run_id"].nunique()),
        "workloads": int(frame["workload"].nunique()),
        "feature_counts": {k: len(v) for k, v in groups.items()},
        "split_counts": frame["split"].value_counts(dropna=False).to_dict(),
        "target_prevalence": {
            "y_warning": float(frame["y_warning"].mean()),
            "y_risk": float(frame["y_risk"].mean()),
        },
    }
    split_summary = (
        frame.groupby("split")
        .agg(
            rows=("run_id", "size"),
            runs=("run_id", "nunique"),
            workloads=("workload", "nunique"),
            warning_prevalence=("y_warning", "mean"),
            risk_prevalence=("y_risk", "mean"),
            log_available=("log_available", "mean"),
            kpi_available=("kpi_available", "mean"),
            both_modalities=("has_both_modalities", "mean"),
        )
        .reset_index()
    )
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    pd.crosstab(frame["workload"], frame["split"]).to_csv(output_dir / "workload_split_matrix.csv")
    pd.crosstab(frame["failure_family"], frame["split"]).to_csv(output_dir / "failure_family_split_matrix.csv")
    write_json(output_dir / "contract_overview.json", overview)
    return overview


def label_policy_audit(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for cols in [["anchor_policy"], ["anchor_policy", "split"], ["failure_family"], ["workload"]]:
        grouped = frame.groupby(cols, dropna=False)
        for key, part in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            row = {col: value for col, value in zip(cols, key)}
            row.update(
                {
                    "section": "+".join(cols),
                    "rows": int(len(part)),
                    "runs": int(part["run_id"].nunique()),
                    "warning_prevalence": float(part["y_warning"].mean()),
                    "risk_prevalence": float(part["y_risk"].mean()),
                    "log_available": float(part["log_available"].mean()),
                    "both_modalities": float(part["has_both_modalities"].mean()),
                }
            )
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "label_policy_slices.csv", index=False)
    return out


def temporal_anchor_audit(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    df = frame.copy()
    for col in ["window_ts", "window_end", "event_anchor", "fail_start"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    rows = []
    valid_anchor = df["event_anchor"].notna()
    if not valid_anchor.any():
        out = pd.DataFrame()
        out.to_csv(output_dir / "temporal_anchor_audit.csv", index=False)
        return out
    df["sec_anchor_minus_window_end"] = (df["event_anchor"] - df["window_end"]).dt.total_seconds()
    df["sec_anchor_minus_window_start"] = (df["event_anchor"] - df["window_ts"]).dt.total_seconds()
    macro_sec = df.get("macro_window_sec", pd.Series(60, index=df.index)).fillna(60).astype(float)
    df["macro_start"] = df["window_end"] - pd.to_timedelta(macro_sec, unit="s")
    df["anchor_inside_micro_window"] = (df["event_anchor"] >= df["window_ts"]) & (df["event_anchor"] < df["window_end"])
    df["anchor_inside_macro_window"] = (df["event_anchor"] >= df["macro_start"]) & (df["event_anchor"] < df["window_end"])
    for target in ["y_warning", "y_risk"]:
        for policy, part in df.loc[valid_anchor & df[target].astype(int).eq(1)].groupby("anchor_policy", dropna=False):
            if part.empty:
                continue
            rows.append(
                {
                    "target": target,
                    "anchor_policy": policy,
                    "positive_windows": int(len(part)),
                    "runs": int(part["run_id"].nunique()),
                    "anchor_inside_micro_share": float(part["anchor_inside_micro_window"].mean()),
                    "anchor_inside_macro_share": float(part["anchor_inside_macro_window"].mean()),
                    "window_end_after_anchor_share": float(part["sec_anchor_minus_window_end"].lt(0).mean()),
                    "sec_anchor_minus_window_end_p05": float(part["sec_anchor_minus_window_end"].quantile(0.05)),
                    "sec_anchor_minus_window_end_p50": float(part["sec_anchor_minus_window_end"].quantile(0.50)),
                    "sec_anchor_minus_window_end_p95": float(part["sec_anchor_minus_window_end"].quantile(0.95)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "temporal_anchor_audit.csv", index=False)
    return out


def feature_shift_audit(frame: pd.DataFrame, groups: dict[str, list[str]], output_dir: Path) -> pd.DataFrame:
    train = frame.loc[frame["split"].eq("train")]
    test = frame.loc[frame["split"].eq("test")]
    numeric_cols = groups["availability"] + groups["log_numeric"] + groups["kpi_numeric"]
    rows = []
    for col in numeric_cols:
        a = pd.to_numeric(train[col], errors="coerce")
        b = pd.to_numeric(test[col], errors="coerce")
        if a.notna().sum() < 5 or b.notna().sum() < 5:
            continue
        am, bm = float(a.mean()), float(b.mean())
        av, bv = float(a.std(ddof=0)), float(b.std(ddof=0))
        pooled = math.sqrt((av * av + bv * bv) / 2.0) if math.isfinite(av) and math.isfinite(bv) else 0.0
        smd = (bm - am) / pooled if pooled > 1e-12 else 0.0
        qdiff = float(abs(b.quantile(0.5) - a.quantile(0.5)))
        missing_diff = float(b.isna().mean() - a.isna().mean())
        rows.append(
            {
                "feature": col,
                "group": "availability" if col in groups["availability"] else "log_numeric" if col in groups["log_numeric"] else "kpi_numeric",
                "train_mean": am,
                "test_mean": bm,
                "train_std": av,
                "test_std": bv,
                "standardized_mean_diff": float(smd),
                "abs_standardized_mean_diff": float(abs(smd)),
                "median_abs_diff": qdiff,
                "missing_diff_test_minus_train": missing_diff,
            }
        )
    out = pd.DataFrame(rows).sort_values("abs_standardized_mean_diff", ascending=False)
    out.to_csv(output_dir / "feature_shift_top.csv", index=False)
    return out


def split_predictability_audit(frame: pd.DataFrame, groups: dict[str, list[str]], output_dir: Path) -> pd.DataFrame:
    rows = []
    y = frame["split"].astype(str).eq("test").astype(int)
    cv = GroupShuffleSplit(n_splits=3, test_size=0.25, random_state=RANDOM_SEED)
    feature_sets = {
        "availability": groups["availability"],
        "log_numeric": groups["log_numeric"],
        "availability+log_numeric": groups["availability"] + groups["log_numeric"],
        "availability+log_numeric+kpi": groups["availability"] + groups["log_numeric"] + groups["kpi_numeric"],
    }
    for name, cols in feature_sets.items():
        if not cols:
            continue
        fold = 0
        for train_idx, test_idx in cv.split(frame, y, groups=frame["run_id"].astype(str)):
            fold += 1
            model = Pipeline(
                [
                    (
                        "features",
                        ColumnTransformer(
                            [
                                (
                                    "num",
                                    Pipeline(
                                        [
                                            ("imputer", SimpleImputer(strategy="median")),
                                            ("scaler", StandardScaler(with_mean=False)),
                                        ]
                                    ),
                                    cols,
                                )
                            ],
                            sparse_threshold=0.35,
                        ),
                    ),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            solver="liblinear",
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            )
            model.fit(frame.iloc[train_idx], y.iloc[train_idx])
            score = model.predict_proba(frame.iloc[test_idx])[:, 1]
            rows.append(
                metric_row(
                    y.iloc[test_idx],
                    score,
                    {"feature_set": name, "fold": fold, "task": "predict_official_test_split"},
                )
            )
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "split_predictability.csv", index=False)
    return out


def split_indices(frame: pd.DataFrame, protocol: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if protocol == "official_workload_holdout":
        return (
            frame.index[frame["split"].eq("train")].to_numpy(),
            frame.index[frame["split"].eq("test")].to_numpy(),
        )
    if protocol == "random_run_holdout":
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_idx, test_idx = next(splitter.split(frame, groups=frame["run_id"].astype(str)))
        return train_idx, test_idx
    if protocol == "train_workload_validation":
        train_frame = frame.loc[frame["split"].eq("train")].copy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        inner_train, inner_val = next(splitter.split(train_frame, groups=train_frame["workload"].astype(str)))
        index = train_frame.index.to_numpy()
        return index[inner_train], index[inner_val]
    raise ValueError(protocol)


def run_model_audit(frame: pd.DataFrame, groups: dict[str, list[str]], cfg: AuditConfig, output_dir: Path) -> pd.DataFrame:
    model_names = [
        "availability-only",
        "log-count-only",
        "log-count-no-severity",
        "log-text-only",
        "log-text-redacted",
        "log-only",
        "log-only-redacted",
        "kpi-only",
        "fused",
    ]
    rows = []
    protocols: list[tuple[str, int | None]] = [("official_workload_holdout", None)]
    protocols.extend(("random_run_holdout", seed) for seed in cfg.random_seeds)
    protocols.extend(("train_workload_validation", seed) for seed in cfg.random_seeds)
    for protocol, seed in protocols:
        train_idx, test_idx = split_indices(frame, protocol, seed or RANDOM_SEED)
        for target in ["y_warning", "y_risk"]:
            for model_name in model_names:
                if cfg.skip_fused_random and model_name == "fused" and protocol != "official_workload_holdout":
                    continue
                print(f"[model-audit] {protocol} seed={seed} target={target} model={model_name}")
                row, _ = evaluate_model(
                    frame,
                    train_idx,
                    test_idx,
                    target,
                    model_name,
                    groups,
                    cfg.max_tfidf_features,
                    protocol,
                    seed,
                )
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "retrained_model_audit_metrics.csv", index=False)
    return out


def token_audit(frame: pd.DataFrame, groups: dict[str, list[str]], cfg: AuditConfig, output_dir: Path) -> pd.DataFrame:
    rows = []
    train_idx, test_idx = split_indices(frame, "official_workload_holdout", RANDOM_SEED)
    for target in ["y_warning", "y_risk"]:
        for redacted in [False, True]:
            name = "text_redacted" if redacted else "text_raw"
            y_train = frame.iloc[train_idx][target].astype(int)
            if y_train.nunique() < 2:
                continue
            vectorizer = TfidfVectorizer(
                max_features=int(cfg.max_tfidf_features),
                min_df=2,
                ngram_range=(1, 2),
                sublinear_tf=True,
                preprocessor=redact_log_text if redacted else None,
            )
            x_train = vectorizer.fit_transform(frame.iloc[train_idx]["log_text"].fillna("").astype(str))
            x_test = vectorizer.transform(frame.iloc[test_idx]["log_text"].fillna("").astype(str))
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=RANDOM_SEED)
            clf.fit(x_train, y_train)
            score = clf.predict_proba(x_test)[:, 1]
            rows.append(metric_row(frame.iloc[test_idx][target].astype(int), score, {"target": target, "variant": name, "token": "__MODEL_METRIC__", "coef": None}))
            terms = np.asarray(vectorizer.get_feature_names_out())
            coef = clf.coef_[0]
            order = np.argsort(coef)
            for rank, idx in enumerate(order[:40], start=1):
                rows.append({"target": target, "variant": name, "direction": "negative", "rank": rank, "token": terms[idx], "coef": float(coef[idx])})
            for rank, idx in enumerate(order[-40:][::-1], start=1):
                rows.append({"target": target, "variant": name, "direction": "positive", "rank": rank, "token": terms[idx], "coef": float(coef[idx])})
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "text_token_audit.csv", index=False)
    return out


def hades_v2_result_audit(root: Path, output_dir: Path) -> pd.DataFrame:
    base = root / "outputs" / "hades_like_v2"
    rows = []
    if not base.exists():
        out = pd.DataFrame()
        out.to_csv(output_dir / "hades_v2_result_audit.csv", index=False)
        return out
    for run_dir in sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime):
        metrics_path = run_dir / "ensemble_metrics.csv"
        seed_path = run_dir / "seed_metrics.csv"
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            for row in df.to_dict("records"):
                row.update({"run_id": run_dir.name, "source": "ensemble_metrics"})
                rows.append(row)
        if seed_path.exists():
            df = pd.read_csv(seed_path)
            summary = (
                df.groupby("target")
                .agg(
                    average_precision_mean=("average_precision", "mean"),
                    average_precision_std=("average_precision", "std"),
                    roc_auc_mean=("roc_auc", "mean"),
                    recall_mean=("recall", "mean"),
                    best_epoch_mean=("best_epoch", "mean"),
                )
                .reset_index()
            )
            for row in summary.to_dict("records"):
                row.update({"run_id": run_dir.name, "source": "seed_summary"})
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "hades_v2_result_audit.csv", index=False)
    return out


def baseline_file_audit(baseline_path: Path | None, frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if baseline_path is None or not baseline_path.exists():
        out = pd.DataFrame()
        out.to_csv(output_dir / "baseline_prediction_slice_metrics.csv", index=False)
        return out
    pred = pd.read_csv(baseline_path)
    return evaluate_predictions_by_slices(pred, frame, output_dir)


def build_summary_markdown(
    cfg: AuditConfig,
    overview: dict,
    label_slices: pd.DataFrame,
    temporal: pd.DataFrame,
    model_metrics: pd.DataFrame,
    split_predictability: pd.DataFrame,
    output_dir: Path,
) -> None:
    def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
        if df.empty:
            return "_empty_"
        view = df.head(max_rows).copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        headers = [str(c) for c in view.columns]
        rows = []
        rows.append("| " + " | ".join(headers) + " |")
        rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in view.iterrows():
            rows.append("| " + " | ".join(str(row[c]) for c in view.columns) + " |")
        if len(df) > max_rows:
            rows.append(f"\n_Показаны первые {max_rows} строк из {len(df)}._")
        return "\n".join(rows)

    lines = []
    lines.append(f"# Deep multimodal audit: `{cfg.run_id}`")
    lines.append("")
    lines.append("## What This Audit Checks")
    lines.append("")
    lines.append("- whether official test is a different workload domain rather than a normal validation split;")
    lines.append("- whether log-only baseline benefits from target construction based on raw-log severity/keywords;")
    lines.append("- whether log/KPI/fusion results change under random run split versus workload holdout;")
    lines.append("- whether positive windows include the explicit anchor inside the feature window;")
    lines.append("- whether redacting obvious failure words collapses the log baseline.")
    lines.append("")
    lines.append("## Contract")
    lines.append("")
    lines.append(f"- rows: `{overview.get('rows')}`")
    lines.append(f"- runs: `{overview.get('runs')}`")
    lines.append(f"- workloads: `{overview.get('workloads')}`")
    lines.append(f"- feature counts: `{overview.get('feature_counts')}`")
    lines.append(f"- split counts: `{overview.get('split_counts')}`")
    lines.append(f"- target prevalence: `{overview.get('target_prevalence')}`")
    lines.append("")
    if not model_metrics.empty:
        lines.append("## Main Model-Audit Table")
        lines.append("")
        pivot = model_metrics.loc[model_metrics["status"].eq("ok")].copy()
        pivot = pivot[pivot["protocol"].isin(["official_workload_holdout", "random_run_holdout", "train_workload_validation"])]
        summary = (
            pivot.groupby(["protocol", "target", "model"])
            .agg(ap_mean=("average_precision", "mean"), ap_std=("average_precision", "std"), roc_mean=("roc_auc", "mean"), n=("n", "mean"))
            .reset_index()
            .sort_values(["target", "protocol", "ap_mean"], ascending=[True, True, False])
        )
        summary.to_csv(output_dir / "retrained_model_audit_summary.csv", index=False)
        lines.append(markdown_table(summary))
        lines.append("")
    if not temporal.empty:
        lines.append("## Temporal Anchor Risk")
        lines.append("")
        lines.append(markdown_table(temporal))
        lines.append("")
    if not split_predictability.empty:
        lines.append("## Split Predictability")
        lines.append("")
        sp = (
            split_predictability.groupby("feature_set")
            .agg(ap_mean=("average_precision", "mean"), roc_mean=("roc_auc", "mean"))
            .reset_index()
            .sort_values("roc_mean", ascending=False)
        )
        lines.append(markdown_table(sp))
        lines.append("")
    lines.append("## Files")
    lines.append("")
    for path in sorted(output_dir.glob("*.csv")):
        lines.append(f"- `{path.name}`")
    lines.append("- `contract_overview.json`")
    lines.append("- `audit_config.json`")
    lines.append("")
    (output_dir / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"deep_audit_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="outputs/deep_multimodal_audit")
    parser.add_argument("--contract-path", default=None)
    parser.add_argument("--baseline-predictions-path", default=None)
    parser.add_argument("--baseline-summary-path", default=None)
    parser.add_argument("--max-tfidf-features", type=int, default=2048)
    parser.add_argument("--random-seeds", default="42,43,44")
    parser.add_argument("--skip-fused-random", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    contract_path = Path(args.contract_path) if args.contract_path else first_existing(
        root / "outputs/raw_multirate/aligned_multirate_windows.csv.gz",
        root / "outputs/raw_multirate/aligned_multirate_windows.csv",
    )
    if contract_path is None:
        raise FileNotFoundError("Cannot find aligned_multirate_windows.csv(.gz)")
    baseline_predictions_path = Path(args.baseline_predictions_path) if args.baseline_predictions_path else first_existing(
        root / "outputs/raw_multirate/baseline_predictions.csv.gz",
        root / "outputs/raw_multirate/baseline_predictions.csv",
    )
    baseline_summary_path = Path(args.baseline_summary_path) if args.baseline_summary_path else first_existing(
        root / "outputs/raw_multirate/baseline_summary_metrics.json",
    )
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = AuditConfig(
        run_id=args.run_id,
        root=str(root),
        output_dir=str(output_dir),
        contract_path=str(contract_path),
        baseline_predictions_path=str(baseline_predictions_path) if baseline_predictions_path else None,
        baseline_summary_path=str(baseline_summary_path) if baseline_summary_path else None,
        max_tfidf_features=int(args.max_tfidf_features),
        random_seeds=[int(s.strip()) for s in str(args.random_seeds).split(",") if s.strip()],
        skip_fused_random=bool(args.skip_fused_random),
    )
    write_json(output_dir / "audit_config.json", asdict(cfg))

    print(f"[1/9] Loading contract: {contract_path}")
    frame = load_contract(contract_path)
    groups = feature_groups(frame)

    print("[2/9] Contract and split overview")
    overview = contract_overview(frame, groups, output_dir)

    print("[3/9] Label policy audit")
    label_slices = label_policy_audit(frame, output_dir)

    print("[4/9] Temporal anchor audit")
    temporal = temporal_anchor_audit(frame, output_dir)

    print("[5/9] Feature shift audit")
    feature_shift_audit(frame, groups, output_dir)

    print("[6/9] Split predictability audit")
    split_pred = split_predictability_audit(frame, groups, output_dir)

    print("[7/9] Saved baseline prediction slice audit")
    baseline_file_audit(baseline_predictions_path, frame, output_dir)

    print("[8/9] Re-training diagnostic baselines under multiple split protocols")
    model_metrics = run_model_audit(frame, groups, cfg, output_dir)

    print("[9/9] Token and HADES-v2 result audit")
    token_audit(frame, groups, cfg, output_dir)
    hades_v2_result_audit(root, output_dir)
    build_summary_markdown(cfg, overview, label_slices, temporal, model_metrics, split_pred, output_dir)
    print(f"Done. Audit outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
