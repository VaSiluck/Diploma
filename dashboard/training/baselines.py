from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dashboard.modeling.hades_blocks import best_f1_threshold, fp_budget_threshold
from dashboard.modeling.protocol_fixed_hades import (
    DEFAULT_TARGETS,
    make_protocol_frame,
    make_target_matrix,
    parse_csv_list,
    parse_target_names,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = ROOT / "dashboard" / "artifacts"
DEFAULT_CONTRACT_PATH = DEFAULT_ARTIFACT_DIR / "contract" / "strict_intensity_contract.csv.gz"
DEFAULT_OUTPUT_DIR = DEFAULT_ARTIFACT_DIR / "reports" / "protocol_fixed_baselines"

LOG_NUMERIC_COLUMNS = [
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
MASK_COLUMNS = ["log_available", "kpi_available", "kpi_series_available", "has_both_modalities", "kpi_avail_ratio", "kpi_recency_sec"]


@dataclass
class BaselineConfig:
    contract_path: str
    output_dir: str
    split_protocol: str
    targets: list[str]
    seed: int
    validation_fraction: float
    validation_seed: int
    fp_budget: float
    max_tfidf_features: int
    max_iter: int


def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    if np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def _safe_roc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    if np.unique(y).size < 2 or np.unique(score).size < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def _evaluate(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pred = score >= float(threshold)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred.astype(int), average="binary", zero_division=0)
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else np.nan,
        "average_precision": _safe_ap(y, score),
        "roc_auc": _safe_roc(y, score),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positives": int(((pred == 1) & (y == 0)).sum()),
        "missed_anomalies": int(((pred == 0) & (y == 1)).sum()),
        "threshold": float(threshold),
    }


def _load_contract(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing contract: {path}")
    frame = pd.read_csv(path, low_memory=False)
    frame["window_ts"] = pd.to_datetime(frame["window_ts"], utc=True, errors="coerce")
    frame["window_end"] = pd.to_datetime(frame["window_end"], utc=True, errors="coerce")
    frame["run_id"] = frame["run_id"].astype(str)
    frame["log_text"] = frame.get("log_text", "").fillna("").astype(str)
    if "y_anomaly" not in frame.columns:
        frame["y_anomaly"] = np.maximum(frame["y_warning"].astype(int), frame["y_risk"].astype(int))
    frame = frame.sort_values(["run_id", "window_ts", "window_index" if "window_index" in frame.columns else "window_ts"]).reset_index(drop=True)
    frame["row_pos"] = np.arange(len(frame), dtype=np.int64)
    return frame


def _feature_sets(frame: pd.DataFrame) -> dict[str, tuple[list[str], bool]]:
    kpi_cols = [col for col in frame.columns if col.startswith("kpi_") and "__" in col]
    deviation_cols = [col for col in kpi_cols if col.endswith("__delta") or col.endswith("__slope") or col.endswith("__std")]
    raw_kpi_cols = [col for col in kpi_cols if col not in deviation_cols]
    log_numeric = [col for col in LOG_NUMERIC_COLUMNS if col in frame.columns]
    masks = [col for col in MASK_COLUMNS if col in frame.columns]
    return {
        "log_tfidf": (log_numeric + ["log_available"], True),
        "metric_raw": (raw_kpi_cols + masks, False),
        "metric_raw_plus_deviation": (kpi_cols + masks, False),
        "fused_log_metric": (log_numeric + kpi_cols + masks, True),
        "availability_only": (masks, False),
    }


def _make_model(baseline: str, numeric_cols: list[str], use_text: bool, cfg: BaselineConfig) -> Pipeline:
    if use_text:
        transformer = ColumnTransformer(
            transformers=[
                ("text", TfidfVectorizer(max_features=int(cfg.max_tfidf_features), min_df=2, ngram_range=(1, 2)), "log_text"),
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))]), numeric_cols),
            ],
            remainder="drop",
        )
        features_step: Any = transformer
    else:
        features_step = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))])
    clf = LogisticRegression(
        max_iter=int(cfg.max_iter),
        class_weight="balanced",
        solver="liblinear",
        random_state=int(cfg.seed),
    )
    return Pipeline([("features", features_step), ("clf", clf)])


def _fit_predict(train: pd.DataFrame, pred: pd.DataFrame, target: str, baseline: str, numeric_cols: list[str], use_text: bool, cfg: BaselineConfig) -> np.ndarray:
    y_train = train[target].astype(int)
    if y_train.nunique() < 2:
        return np.full(len(pred), float(y_train.mean()) if len(y_train) else 0.0)
    model = _make_model(baseline, numeric_cols, use_text, cfg)
    features = ["log_text"] + numeric_cols if use_text else numeric_cols
    if not features:
        return np.full(len(pred), float(y_train.mean()))
    model.fit(train[features], y_train)
    return model.predict_proba(pred[features])[:, 1]


def _score_all_splits(frame: pd.DataFrame, baseline: str, numeric_cols: list[str], use_text: bool, target: str, cfg: BaselineConfig) -> dict[str, np.ndarray]:
    train = frame.loc[frame["split"].eq("train")].copy()
    if train.empty:
        raise RuntimeError("Empty train split for protocol baseline.")
    out = {}
    for split in ["validation", "test"]:
        part = frame.loc[frame["split"].eq(split)].copy()
        out[split] = _fit_predict(train, part, target, baseline, numeric_cols, use_text, cfg) if not part.empty else np.empty(0)
    return out


def _prediction_frame(frame: pd.DataFrame, split: str, target: str, baseline: str, score: np.ndarray, split_protocol: str) -> pd.DataFrame:
    part = frame.loc[frame["split"].eq(split)].copy()
    cols = ["row_pos", "run_id", "workload", "scenario", "scenario_type", "failure_family", "anchor_policy", "window_ts", "window_end", "split", "original_split"]
    cols = [col for col in cols if col in part.columns]
    out = part[cols].rename(columns={"row_pos": "row_index"}).reset_index(drop=True)
    out["target_name"] = target
    out["target"] = part[target].astype(int).to_numpy()
    out["score"] = score.astype(float)
    out["split_protocol"] = split_protocol
    out["baseline"] = baseline
    out["pred_split"] = split
    return out


def _workload_metrics(pred: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    threshold_table = metrics.loc[
        metrics["split"].astype(str).eq("validation")
        & metrics["threshold_policy"].astype(str).eq("validation_best_f1")
    ]
    for key, group in pred.groupby(["split_protocol", "baseline", "target_name", "workload"], dropna=False):
        split_protocol, baseline, target, workload = key
        thr = threshold_table.loc[
            threshold_table["split_protocol"].astype(str).eq(str(split_protocol))
            & threshold_table["baseline"].astype(str).eq(str(baseline))
            & threshold_table["target"].astype(str).eq(str(target))
        ]
        threshold = float(thr.iloc[0]["threshold"]) if not thr.empty else 0.5
        rows.append(
            {
                "split_protocol": split_protocol,
                "baseline": baseline,
                "target": target,
                "workload": workload,
                **_evaluate(group["target"].to_numpy(), group["score"].to_numpy(), threshold),
            }
        )
    return pd.DataFrame(rows)


def train_baselines(cfg: BaselineConfig) -> dict[str, Any]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = _load_contract(Path(cfg.contract_path))
    protocol_ns = argparse.Namespace(validation_seed=cfg.validation_seed, validation_fraction=cfg.validation_fraction)
    frame = make_protocol_frame(base, cfg.split_protocol, protocol_ns)
    frame["original_split"] = frame.get("original_split", frame["split"]).astype(str)
    target_matrix, target_manifest = make_target_matrix(frame, cfg.targets)
    for idx, target in enumerate(cfg.targets):
        frame[target] = target_matrix[:, idx].astype(int)

    feature_sets = _feature_sets(frame)
    predictions = []
    metric_rows = []
    thresholds: dict[str, dict[str, dict[str, float]]] = {}
    for target in cfg.targets:
        thresholds[target] = {}
        for baseline, (numeric_cols, use_text) in feature_sets.items():
            if not numeric_cols and not use_text:
                continue
            scores = _score_all_splits(frame, baseline, numeric_cols, use_text, target, cfg)
            val_frame = frame.loc[frame["split"].eq("validation")]
            y_val = val_frame[target].astype(int).to_numpy()
            s_val = scores["validation"]
            threshold_best = best_f1_threshold(y_val, s_val) if len(y_val) else 0.5
            threshold_fp = fp_budget_threshold(y_val, s_val, cfg.fp_budget) if len(y_val) else 0.5
            thresholds[target][baseline] = {"best_f1": float(threshold_best), "fp_budget": float(threshold_fp)}
            for split in ["validation", "test"]:
                split_frame = frame.loc[frame["split"].eq(split)]
                y = split_frame[target].astype(int).to_numpy()
                s = scores[split]
                for policy, threshold in [
                    ("validation_best_f1", threshold_best),
                    (f"validation_fp_budget_{cfg.fp_budget}", threshold_fp),
                ]:
                    metric_rows.append(
                        {
                            "split_protocol": cfg.split_protocol,
                            "baseline": baseline,
                            "runtime": "sklearn_cpu",
                            "split": split,
                            "target": target,
                            "threshold_policy": policy,
                            **_evaluate(y, s, threshold),
                        }
                    )
                predictions.append(_prediction_frame(frame, split, target, baseline, s, cfg.split_protocol))

    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    test_pred = pred_df.loc[pred_df["pred_split"].astype(str).eq("test")].copy()
    workload_df = _workload_metrics(test_pred, metrics_df) if not test_pred.empty else pd.DataFrame()
    split_summary = frame.groupby("split").agg(rows=("run_id", "size"), runs=("run_id", "nunique"), workloads=("workload", "nunique")).reset_index()

    metrics_df.to_csv(output_dir / "baseline_protocol_metrics.csv", index=False)
    pred_df.to_csv(output_dir / "baseline_protocol_predictions.csv.gz", index=False, compression="gzip")
    workload_df.to_csv(output_dir / "baseline_protocol_workload_metrics.csv", index=False)
    split_summary.to_csv(output_dir / "baseline_protocol_split_summary.csv", index=False)
    manifest = {
        "source": "dashboard.training.baselines",
        "config": asdict(cfg),
        "target_manifest": target_manifest,
        "feature_sets": {name: {"numeric_columns": len(cols), "uses_text": use_text} for name, (cols, use_text) in feature_sets.items()},
        "thresholds": thresholds,
        "written_files": {
            "metrics": _safe_rel(output_dir / "baseline_protocol_metrics.csv"),
            "predictions": _safe_rel(output_dir / "baseline_protocol_predictions.csv.gz"),
            "workload_metrics": _safe_rel(output_dir / "baseline_protocol_workload_metrics.csv"),
            "split_summary": _safe_rel(output_dir / "baseline_protocol_split_summary.csv"),
            "manifest": _safe_rel(output_dir / "baseline_protocol_manifest.json"),
        },
    }
    _write_json(output_dir / "baseline_protocol_manifest.json", manifest)
    return manifest


def parse_args() -> BaselineConfig:
    parser = argparse.ArgumentParser(description="Train protocol-fixed baselines for notebook 02.")
    parser.add_argument("--contract-path", default=str(DEFAULT_CONTRACT_PATH.relative_to(ROOT)))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR.relative_to(ROOT)))
    parser.add_argument("--split-protocol", default="stratified_by_workload")
    parser.add_argument("--targets", default="default")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--validation-seed", type=int, default=43)
    parser.add_argument("--fp-budget", type=float, default=0.10)
    parser.add_argument("--max-tfidf-features", type=int, default=2200)
    parser.add_argument("--max-iter", type=int, default=700)
    args = parser.parse_args()
    contract_path = Path(args.contract_path)
    output_dir = Path(args.output_dir)
    if not contract_path.is_absolute():
        contract_path = ROOT / contract_path
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    return BaselineConfig(
        contract_path=str(contract_path),
        output_dir=str(output_dir),
        split_protocol=str(args.split_protocol),
        targets=parse_target_names(args.targets),
        seed=int(args.seed),
        validation_fraction=float(args.validation_fraction),
        validation_seed=int(args.validation_seed),
        fp_budget=float(args.fp_budget),
        max_tfidf_features=int(args.max_tfidf_features),
        max_iter=int(args.max_iter),
    )


def main() -> None:
    cfg = parse_args()
    manifest = train_baselines(cfg)
    print(json.dumps({"status": "ok", **manifest["written_files"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
