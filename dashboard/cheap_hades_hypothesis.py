"""
Cheap HADES-like hypothesis experiment for CPU-only AIOps anomaly detection.

This module is intentionally separate from the final training path. It reads the
strict contract, rebuilds cheap log-template and metric-stat features, trains a
small PyTorch model, compares ablations, and optionally exports the best model
to ONNX plus ONNX Runtime INT8.

Typical smoke run:
    python -m dashboard.cheap_hades_hypothesis --max-rows 900 --epochs 2 --seeds 42

Report-style run on CPU:
    python -m dashboard.cheap_hades_hypothesis --seeds 42,43,44 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


RAW_LOG_RE = re.compile(
    r"^(?P<date>\d{2}/\d{2}/\d{2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<level>[A-Z]+)\s+"
    r"(?P<component>[^:]+):\s*"
    r"(?P<message>.*)$"
)

KEYWORDS = [
    "error",
    "exception",
    "failed",
    "failure",
    "killed",
    "lost",
    "timeout",
    "suspended",
    "blacklist",
    "retry",
    "critical",
    "fatal",
    "unhealthy",
    "refused",
    "denied",
    "warn",
]

SEVERITY_WEIGHT = {
    "FATAL": 5.0,
    "ERROR": 4.0,
    "WARN": 2.0,
    "WARNING": 2.0,
    "INFO": 0.2,
    "DEBUG": 0.0,
    "TRACE": 0.0,
}

KPI_EXCLUDE = {
    "kpi_available",
    "kpi_avail_ratio",
    "kpi_feature_count",
    "kpi_recency_sec",
    "kpi_series_available",
}

METRIC_STAT_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "p95",
    "p99",
    "delta",
    "slope",
    "spike_count",
    "missing_count",
    "robust_zscore_max",
]


@dataclass
class CheapHadesConfig:
    run_id: str
    root: str
    contract_path: str
    output_dir: str
    target: str
    variants: list[str]
    seeds: list[int]
    max_rows: int | None
    max_log_events: int
    log_context_sec: int
    max_templates: int
    min_template_count: int
    log_encoder: str
    embedding_dim: int
    repr_dim: int
    metric_context_steps: int
    max_metric_series: int
    metric_hidden_dim: int
    batch_size: int
    epochs: int
    patience: int
    lr: float
    weight_decay: float
    spike_z: float
    validation_strategy: str
    validation_fraction: float
    validation_seed: int
    device: str
    compare_existing: bool
    baseline_predictions_path: str | None
    existing_model_dir: str | None
    export_onnx: bool
    quantization: str
    calibration_size: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def parse_csv_list(value: str, cast=str) -> list:
    return [cast(v.strip()) for v in str(value).split(",") if v.strip()]


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


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))).astype(np.float32)


def best_f1_threshold(y, score) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    finite = np.isfinite(score)
    if not finite.any() or len(np.unique(y[finite])) < 2:
        return 0.5
    values = np.unique(np.quantile(score[finite], np.linspace(0.02, 0.98, 97)))
    values = np.unique(np.concatenate([[0.5], values]))
    best_thr = 0.5
    best_f1 = -1.0
    for threshold in values:
        pred = (score >= float(threshold)).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        if float(f1) > best_f1:
            best_f1 = float(f1)
            best_thr = float(threshold)
    return best_thr


def evaluate_binary(y, score, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pred = (score >= float(threshold)).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else None,
        "average_precision": safe_ap(y, score),
        "roc_auc": safe_roc(y, score),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positives": fp,
        "missed_anomalies": fn,
        "true_positives": tp,
        "true_negatives": tn,
        "threshold": float(threshold),
    }


def load_contract(path: Path, max_rows: int | None, target: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["window_ts", "window_end"])
    if "split" not in frame.columns:
        raise RuntimeError("Contract must contain a split column.")
    split_map = {"valid": "validation", "val": "validation", "dev": "validation"}
    frame["split"] = frame["split"].astype(str).str.strip().str.lower().replace(split_map)
    if "window_index" not in frame.columns:
        frame["window_index"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.sort_values(["run_id", "window_ts", "window_index"]).reset_index(drop=True)
    if "y_anomaly" not in frame.columns:
        frame["y_anomaly"] = np.maximum(frame["y_warning"].astype(int), frame["y_risk"].astype(int))
    if target not in frame.columns:
        raise RuntimeError(f"Target column {target!r} is not present in the contract.")

    if max_rows:
        rng = np.random.default_rng(42)
        parts = []
        per_split = max(1, int(max_rows) // max(frame["split"].nunique(), 1))
        for _, part in frame.groupby("split", sort=False):
            keep: list[int] = []
            for value in [0, 1]:
                cand = part.index[part[target].astype(int).eq(value)].to_numpy()
                if len(cand):
                    keep.extend(rng.choice(cand, size=min(50, len(cand)), replace=False).tolist())
            rest = np.setdiff1d(part.index.to_numpy(), np.array(sorted(set(keep)), dtype=int))
            need = max(0, min(per_split, len(part)) - len(set(keep)))
            if need and len(rest):
                keep.extend(rng.choice(rest, size=min(need, len(rest)), replace=False).tolist())
            parts.append(frame.loc[sorted(set(keep))])
        frame = pd.concat(parts).sort_values(["run_id", "window_ts", "window_index"]).reset_index(drop=True)

    frame["row_pos"] = np.arange(len(frame), dtype=np.int64)
    frame["log_text"] = frame.get("log_text", "").fillna("").astype(str)
    for col in ["log_available", "kpi_available", "has_both_modalities", "kpi_avail_ratio"]:
        if col not in frame.columns:
            frame[col] = 0.0
    return frame


def apply_validation_split(frame: pd.DataFrame, strategy: str, fraction: float, seed: int) -> pd.DataFrame:
    frame = frame.copy()
    split_values = set(frame["split"].astype(str))
    if strategy == "auto":
        if "validation" in split_values:
            return frame
        strategy = "run_id_holdout"
    if strategy == "existing":
        if "validation" not in split_values:
            raise RuntimeError("validation_strategy=existing requested, but the contract has no validation split.")
        return frame

    train_mask = frame["split"].astype(str).eq("train")
    train_frame = frame.loc[train_mask].copy()
    if train_frame.empty:
        raise RuntimeError("Cannot derive validation split: train split is empty.")
    group_col = "workload" if strategy == "workload_holdout" else "run_id"
    if group_col not in train_frame.columns:
        raise RuntimeError(f"Cannot derive validation split: missing {group_col!r}.")
    if train_frame[group_col].astype(str).nunique() < 2:
        group_col = "run_id"
    if train_frame[group_col].astype(str).nunique() < 2:
        raise RuntimeError(f"Cannot derive validation split: {group_col} has fewer than two groups.")

    splitter = GroupShuffleSplit(n_splits=64, test_size=float(fraction), random_state=int(seed))
    groups = train_frame[group_col].astype(str)
    train_index = train_frame.index.to_numpy()
    best = None
    best_score = -1e9
    for _, val_rel in splitter.split(train_frame, groups=groups):
        val = train_frame.iloc[val_rel]
        score = 0.0
        for target in ["y_anomaly", "y_warning", "y_risk"]:
            if target in val.columns and val[target].nunique() == 2:
                score += 1.0
            if target in val.columns:
                score += min(int(val[target].sum()), 20) / 20.0
        score -= abs((len(val) / max(len(train_frame), 1)) - float(fraction))
        if score > best_score:
            best_score = score
            best = val_rel
    if best is None:
        raise RuntimeError("Cannot derive validation split.")
    frame.loc[train_index[best], "split"] = "validation"
    frame["validation_group_column"] = group_col
    frame["validation_strategy"] = strategy
    return frame


def split_indices(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    result = {}
    for split in ["train", "validation", "test"]:
        result[split] = frame.index[frame["split"].astype(str).eq(split)].to_numpy(dtype=np.int64)
    if not all(len(v) for v in result.values()):
        raise RuntimeError(f"Empty split detected: { {k: len(v) for k, v in result.items()} }")
    return result


def split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, part in frame.groupby("split", sort=False):
        rows.append(
            {
                "split": split,
                "rows": int(len(part)),
                "runs": int(part["run_id"].nunique()) if "run_id" in part else None,
                "workloads": int(part["workload"].nunique()) if "workload" in part else None,
                "anomaly_positives": int(part["y_anomaly"].sum()) if "y_anomaly" in part else None,
                "warning_positives": int(part["y_warning"].sum()) if "y_warning" in part else None,
                "risk_positives": int(part["y_risk"].sum()) if "y_risk" in part else None,
                "log_available": float(part["log_available"].mean()) if "log_available" in part else None,
                "kpi_available": float(part["kpi_available"].mean()) if "kpi_available" in part else None,
            }
        )
    return pd.DataFrame(rows)


class DrainLikeParser:
    """Deterministic Drain-like template normalizer plus fixed train vocabulary."""

    pad_id = 0
    unk_id = 1

    def __init__(self) -> None:
        self.template_to_id: dict[str, int] = {"__PAD__": self.pad_id, "__UNK__": self.unk_id}
        self.id_to_template: dict[int, str] = {self.pad_id: "__PAD__", self.unk_id: "__UNK__"}
        self.template_counts: Counter[str] = Counter()
        self.rare_threshold = 2

    @staticmethod
    def normalize_message(message: str) -> str:
        text = str(message).strip().lower()
        replacements = [
            (r"application_\d+_\d+", "<*>"),
            (r"container_[a-z0-9_]+", "<*>"),
            (r"attempt_[a-z0-9_]+", "<*>"),
            (r"job_\d+_\d+", "<*>"),
            (r"task_[a-z0-9_]+", "<*>"),
            (r"[0-9a-f]{8}-[0-9a-f-]{27,}", "<*>"),
            (r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<*>"),
            (r"https?://\S+|hdfs://\S+|file:/\S+|/\S+", "<*>"),
            (r"\b0x[0-9a-f]+\b", "<*>"),
            (r"\b\d+(?:\.\d+)?(?:ms|s|kb|mb|gb|%)?\b", "<*>"),
            (r"\s+", " "),
        ]
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text)
        return text.strip() or "__EMPTY__"

    def template_from_parts(self, level: str, component: str, message: str) -> str:
        level = str(level or "INFO").upper()
        component = re.sub(r"\s+", " ", str(component or "unknown")).strip().lower()
        return f"{level} {component}: {self.normalize_message(message)}"

    def parse_line(self, line: str, line_id: int = 0) -> dict:
        text = str(line).strip()
        match = RAW_LOG_RE.match(text)
        if match:
            ts = pd.to_datetime(
                f"{match.group('date')} {match.group('time')}",
                format="%y/%m/%d %H:%M:%S",
                utc=True,
                errors="coerce",
            )
            level = match.group("level")
            component = match.group("component")
            message = match.group("message")
        else:
            ts = pd.NaT
            level = "INFO"
            component = "raw"
            message = text
        template = self.template_from_parts(level, component, message)
        return {
            "timestamp": ts,
            "level": str(level).upper(),
            "component": str(component),
            "message": str(message),
            "template": template,
            "template_id": self.template_to_id.get(template, self.unk_id),
            "severity": float(SEVERITY_WEIGHT.get(str(level).upper(), 0.0)),
            "line_id": int(line_id),
        }

    def fit(self, templates: Iterable[str], max_templates: int, min_template_count: int) -> None:
        self.template_counts = Counter(str(t) for t in templates)
        self.rare_threshold = max(2, int(min_template_count) + 1)
        kept = [
            (template, count)
            for template, count in self.template_counts.most_common()
            if count >= int(min_template_count)
        ][: max(0, int(max_templates) - 2)]
        self.template_to_id = {"__PAD__": self.pad_id, "__UNK__": self.unk_id}
        for template, _ in kept:
            self.template_to_id[template] = len(self.template_to_id)
        self.id_to_template = {idx: template for template, idx in self.template_to_id.items()}

    def encode_template(self, template: str) -> int:
        return int(self.template_to_id.get(str(template), self.unk_id))

    def template_is_rare(self, template: str) -> bool:
        return self.template_counts.get(str(template), 0) <= self.rare_threshold

    def encode_raw_logs(self, raw_logs: Iterable[str], max_events: int) -> tuple[np.ndarray, dict]:
        rows = [self.parse_line(line, i) for i, line in enumerate(raw_logs)]
        events = pd.DataFrame(rows)
        selected = select_log_events(events, self, max_events=max_events)
        ids = np.zeros(int(max_events), dtype=np.int64)
        for j, row in enumerate(selected.itertuples(index=False)):
            ids[j] = self.encode_template(row.template)
        diag = log_reason_summary(selected, self)
        return ids, diag

    def to_manifest(self) -> dict:
        return {
            "type": "drain_like_regex",
            "vocab_size": int(len(self.template_to_id)),
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "rare_threshold": int(self.rare_threshold),
            "top_templates": [
                {"template": template, "count": int(count)}
                for template, count in self.template_counts.most_common(50)
            ],
        }

    def to_preprocessor_state(self) -> dict:
        return {
            "template_to_id": self.template_to_id,
            "template_counts": dict(self.template_counts),
            "rare_threshold": int(self.rare_threshold),
        }


def parse_raw_log_file(path: Path, parser: DrainLikeParser) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=["timestamp", "level", "component", "message", "template", "template_id", "severity", "line_id"]
        )
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_id, line in enumerate(handle, start=1):
            row = parser.parse_line(line, line_id=line_id)
            if pd.isna(row["timestamp"]):
                continue
            rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "level", "component", "message", "template", "template_id", "severity", "line_id"]
        )
    return pd.DataFrame(rows).sort_values(["timestamp", "line_id"]).reset_index(drop=True)


def fallback_events_from_text(text: str, parser: DrainLikeParser, timestamp: pd.Timestamp) -> pd.DataFrame:
    value = str(text or "").strip()
    if not value:
        return pd.DataFrame(
            columns=["timestamp", "level", "component", "message", "template", "template_id", "severity", "line_id"]
        )
    chunks = [chunk.strip() for chunk in re.split(r"[\n\r]+", value) if chunk.strip()]
    if not chunks:
        chunks = [value[:420]]
    rows = []
    for line_id, chunk in enumerate(chunks[:128], start=1):
        row = parser.parse_line(chunk, line_id=line_id)
        row["timestamp"] = timestamp
        rows.append(row)
    return pd.DataFrame(rows)


def select_log_events(events: pd.DataFrame, parser: DrainLikeParser, max_events: int) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    part = events.copy()
    level = part["level"].astype(str).str.upper()
    keyword_hit = part["template"].astype(str).str.lower().apply(lambda text: any(kw in text for kw in KEYWORDS))
    severity = level.map(SEVERITY_WEIGHT).fillna(0.0).astype(float)
    rare = part["template"].astype(str).apply(parser.template_is_rare).astype(float)
    unknown = ~part["template"].astype(str).isin(parser.template_to_id)
    recency = np.linspace(0.0, 1.0, len(part), dtype=np.float32)
    part["_select_score"] = severity + 1.5 * keyword_hit.astype(float) + 1.0 * rare + 1.0 * unknown.astype(float) + recency
    if len(part) <= int(max_events):
        return part

    keep = set(part.tail(max(4, int(max_events) // 3)).index.tolist())
    for idx in part.sort_values("_select_score", ascending=False).index:
        keep.add(idx)
        if len(keep) >= int(max_events):
            break
    selected = part.loc[sorted(keep)].sort_values(["timestamp", "line_id"], na_position="last").head(int(max_events))
    return selected


def log_reason_summary(events: pd.DataFrame, parser: DrainLikeParser, top_k: int = 5) -> dict:
    if events.empty:
        return {
            "error_warn_count": 0,
            "new_template_count": 0,
            "rare_template_count": 0,
            "top_rare_templates": [],
        }
    levels = events["level"].astype(str).str.upper()
    error_warn = int(levels.isin(["ERROR", "WARN", "WARNING", "FATAL"]).sum())
    template_counts = Counter(events["template"].astype(str))
    rare = [
        {"template": template, "count": int(count), "train_count": int(parser.template_counts.get(template, 0))}
        for template, count in template_counts.items()
        if parser.template_is_rare(template)
    ]
    rare = sorted(rare, key=lambda item: (item["train_count"], -item["count"], item["template"]))[:top_k]
    new_count = int(sum(count for template, count in template_counts.items() if template not in parser.template_to_id))
    return {
        "error_warn_count": error_warn,
        "new_template_count": new_count,
        "rare_template_count": int(sum(item["count"] for item in rare)),
        "top_rare_templates": rare,
    }


def build_log_arrays(
    frame: pd.DataFrame,
    raw_root: Path,
    train_idx: np.ndarray,
    max_events: int,
    context_sec: int,
    max_templates: int,
    min_template_count: int,
) -> tuple[np.ndarray, DrainLikeParser, pd.DataFrame, dict]:
    parser = DrainLikeParser()
    events_by_run: dict[str, pd.DataFrame] = {}
    train_runs = set(frame.iloc[train_idx]["run_id"].astype(str))
    train_templates: list[str] = []

    for run_id, group in frame.groupby("run_id", sort=False):
        workload = str(group["workload"].iloc[0])
        scenario = str(group["scenario"].iloc[0])
        log_path = raw_root / workload / "mapred_logs" / f"{scenario}.log"
        events = parse_raw_log_file(log_path, parser)
        events_by_run[str(run_id)] = events
        if str(run_id) in train_runs and not events.empty:
            train_templates.extend(events["template"].astype(str).tolist())

    if not train_templates:
        train_templates = ["__NO_LOG__"]
    parser.fit(train_templates, max_templates=max_templates, min_template_count=min_template_count)

    n = len(frame)
    log_ids = np.zeros((n, int(max_events)), dtype=np.int64)
    diagnostics = []

    for row in frame.itertuples(index=False):
        run_id = str(row.run_id)
        events = events_by_run.get(run_id)
        window_end = getattr(row, "window_end")
        if events is None or events.empty:
            events = fallback_events_from_text(getattr(row, "log_text", ""), parser, window_end)
        if events.empty or pd.isna(window_end):
            selected = events.iloc[0:0].copy() if events is not None else pd.DataFrame()
        else:
            start = window_end - pd.Timedelta(seconds=int(context_sec))
            part = events.loc[(events["timestamp"] >= start) & (events["timestamp"] < window_end)].copy()
            if part.empty:
                part = fallback_events_from_text(getattr(row, "log_text", ""), parser, window_end)
            selected = select_log_events(part, parser, max_events=max_events)
        for j, event in enumerate(selected.itertuples(index=False)):
            log_ids[int(row.row_pos), j] = parser.encode_template(event.template)
        reason = log_reason_summary(selected, parser)
        reason.update({"row_pos": int(row.row_pos), "selected_log_events": int(len(selected))})
        diagnostics.append(reason)

    diag_df = pd.DataFrame(diagnostics)
    manifest = {
        **parser.to_manifest(),
        "max_log_events": int(max_events),
        "context_sec": int(context_sec),
        "runs_parsed": int(len(events_by_run)),
        "mean_selected_events": float(diag_df["selected_log_events"].mean()) if not diag_df.empty else 0.0,
        "empty_window_share": float((diag_df["selected_log_events"] <= 0).mean()) if not diag_df.empty else 1.0,
    }
    return log_ids, parser, diag_df, manifest


def is_kpi_numeric_column(frame: pd.DataFrame, column: str) -> bool:
    if not column.startswith("kpi_") or column in KPI_EXCLUDE:
        return False
    return pd.api.types.is_numeric_dtype(frame[column])


def split_kpi_base(column: str) -> tuple[str, str]:
    for suffix in ["__last", "__mean", "__std", "__delta", "__slope", "__min", "__max", "__p95", "__p99"]:
        if column.endswith(suffix):
            return column[: -len(suffix)], suffix[2:]
    for suffix in ["_last", "_mean", "_std", "_delta", "_slope", "_min", "_max", "_p95", "_p99"]:
        if column.endswith(suffix):
            return column[: -len(suffix)], suffix[1:]
    return column, "value"


def metric_aspect(name: str) -> str:
    value = name.lower()
    if any(token in value for token in ["await", "lat", "error", "fail", "time"]):
        return "latency_errors"
    if any(token in value for token in ["cpu", "cpustat", "load"]):
        return "cpu"
    if any(token in value for token in ["mem", "swap"]):
        return "memory"
    if any(token in value for token in ["disk", "iostat", "io_", "__io", "await", "util"]):
        return "disk_io"
    if any(token in value for token in ["net", "rx", "tx", "packet"]):
        return "network"
    return "other"


def source_metric_columns(frame: pd.DataFrame) -> dict[str, str]:
    priority = {"last": 0, "mean": 1, "max": 2, "p95": 3, "p99": 4, "value": 5, "std": 6, "delta": 7, "slope": 8}
    candidates: dict[str, tuple[int, str]] = {}
    for col in frame.columns:
        if not is_kpi_numeric_column(frame, col):
            continue
        base, stat = split_kpi_base(col)
        rank = priority.get(stat, 20)
        if base not in candidates or rank < candidates[base][0]:
            candidates[base] = (rank, col)
    return {base: col for base, (_, col) in candidates.items()}


def select_metric_series(frame: pd.DataFrame, train_idx: np.ndarray, target: str, max_metric_series: int) -> list[tuple[str, str]]:
    sources = source_metric_columns(frame)
    if not sources:
        raise RuntimeError("No KPI numeric columns found in the contract.")
    y = frame[target].iloc[train_idx].astype(float).to_numpy()
    rows = []
    for base, col in sources.items():
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float32)
        train = values[train_idx]
        finite = np.isfinite(train)
        if finite.sum() < 3:
            continue
        variance = float(np.nanvar(train))
        corr = 0.0
        if np.nanstd(train) > 1e-8 and len(np.unique(y[finite].astype(int))) > 1:
            candidate = np.corrcoef(train[finite], y[finite])[0, 1]
            corr = abs(float(candidate)) if np.isfinite(candidate) else 0.0
        missing_share = float((~finite).mean())
        rows.append((base, col, corr, variance, missing_share))
    rows = sorted(rows, key=lambda item: (item[2], item[3], -item[4]), reverse=True)
    return [(base, col) for base, col, _, _, _ in rows[: int(max_metric_series)]]


def robust_train_stats(values: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = values[train_idx]
    median = np.nanmedian(train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    mad = np.nanmedian(np.abs(train - median[None, :]), axis=0)
    scale = 1.4826 * np.where(np.isfinite(mad), mad, 0.0)
    std = np.nanstd(train, axis=0)
    scale = np.where(scale > 1e-6, scale, np.where(std > 1e-6, std, 1.0))
    return median.astype(np.float32), scale.astype(np.float32)


def first_last_delta(window: np.ndarray, finite: np.ndarray) -> np.ndarray:
    out = np.zeros(window.shape[1], dtype=np.float32)
    for j in range(window.shape[1]):
        idx = np.flatnonzero(finite[:, j])
        if len(idx):
            out[j] = float(window[idx[-1], j] - window[idx[0], j])
    return out


def slope_per_column(window: np.ndarray, finite: np.ndarray) -> np.ndarray:
    length, width = window.shape
    if length <= 1:
        return np.zeros(width, dtype=np.float32)
    x = np.arange(length, dtype=np.float32)[:, None]
    mask = finite.astype(np.float32)
    count = mask.sum(axis=0)
    safe_count = np.maximum(count, 1.0)
    x_mean = (x * mask).sum(axis=0) / safe_count
    y_safe = np.where(finite, window, 0.0)
    y_mean = y_safe.sum(axis=0) / safe_count
    x_center = (x - x_mean[None, :]) * mask
    y_center = (y_safe - y_mean[None, :]) * mask
    denom = (x_center * x_center).sum(axis=0)
    numer = (x_center * y_center).sum(axis=0)
    slope = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 1e-8)
    slope = np.where(count >= 2, slope, 0.0)
    return slope.astype(np.float32)


def compute_metric_stats(window: np.ndarray, median: np.ndarray, scale: np.ndarray, spike_z: float) -> np.ndarray:
    width = window.shape[1]
    finite = np.isfinite(window)
    count = finite.sum(axis=0)
    values = np.where(finite, window, np.nan)

    def nan_reduced(fn) -> np.ndarray:
        out = np.zeros(width, dtype=np.float32)
        for j in range(width):
            col = values[:, j]
            col = col[np.isfinite(col)]
            if len(col):
                out[j] = float(fn(col))
        return out

    mean = nan_reduced(np.mean)
    std = nan_reduced(np.std)
    min_v = nan_reduced(np.min)
    max_v = nan_reduced(np.max)
    p95 = nan_reduced(lambda col: np.percentile(col, 95))
    p99 = nan_reduced(lambda col: np.percentile(col, 99))
    delta = first_last_delta(window, finite)
    slope = slope_per_column(window, finite)
    z = np.abs((window - median[None, :]) / scale[None, :])
    z = np.where(finite, z, np.nan)
    robust_max = nan_reduced(np.max)
    # nan_reduced reads from values, so compute robust max separately.
    robust_max = np.zeros(width, dtype=np.float32)
    for j in range(width):
        col = z[:, j]
        col = col[np.isfinite(col)]
        if len(col):
            robust_max[j] = float(np.max(col))
    spike_count = np.where(finite & (z > float(spike_z)), 1, 0).sum(axis=0).astype(np.float32)
    missing_count = (window.shape[0] - count).astype(np.float32)
    return np.concatenate(
        [mean, std, min_v, max_v, p95, p99, delta, slope, spike_count, missing_count, robust_max]
    ).astype(np.float32)


class MetricFeatureBuilder:
    def __init__(
        self,
        selected_series: list[tuple[str, str]],
        context_steps: int,
        spike_z: float,
        train_median: np.ndarray,
        train_scale: np.ndarray,
        impute_values: np.ndarray,
        norm_mean: np.ndarray,
        norm_scale: np.ndarray,
    ) -> None:
        self.selected_series = selected_series
        self.context_steps = int(context_steps)
        self.spike_z = float(spike_z)
        self.train_median = train_median.astype(np.float32)
        self.train_scale = train_scale.astype(np.float32)
        self.impute_values = impute_values.astype(np.float32)
        self.norm_mean = norm_mean.astype(np.float32)
        self.norm_scale = norm_scale.astype(np.float32)
        self.feature_names = [
            f"{base}__cheap_{stat}"
            for stat in METRIC_STAT_NAMES
            for base, _ in self.selected_series
        ]

    @classmethod
    def fit_from_contract(
        cls,
        frame: pd.DataFrame,
        train_idx: np.ndarray,
        target: str,
        max_metric_series: int,
        context_steps: int,
        spike_z: float,
    ) -> tuple["MetricFeatureBuilder", np.ndarray, np.ndarray]:
        selected = select_metric_series(frame, train_idx, target, max_metric_series=max_metric_series)
        source_cols = [col for _, col in selected]
        raw_values = frame[source_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        train_median, train_scale = robust_train_stats(raw_values, train_idx)

        raw_features = build_metric_feature_matrix(
            frame,
            raw_values,
            train_median,
            train_scale,
            context_steps=context_steps,
            spike_z=spike_z,
        )
        impute_values = np.nanmedian(raw_features[train_idx], axis=0)
        impute_values = np.where(np.isfinite(impute_values), impute_values, 0.0).astype(np.float32)
        filled = np.where(np.isfinite(raw_features), raw_features, impute_values[None, :]).astype(np.float32)
        norm_mean = filled[train_idx].mean(axis=0).astype(np.float32)
        norm_scale = filled[train_idx].std(axis=0).astype(np.float32)
        norm_scale = np.where(norm_scale > 1e-6, norm_scale, 1.0).astype(np.float32)
        scaled = ((filled - norm_mean[None, :]) / norm_scale[None, :]).astype(np.float32)
        builder = cls(selected, context_steps, spike_z, train_median, train_scale, impute_values, norm_mean, norm_scale)
        return builder, scaled, raw_features

    def transform_raw_window(self, raw_metrics: dict[str, Iterable[float]]) -> tuple[np.ndarray, dict]:
        width = len(self.selected_series)
        series_values: list[np.ndarray] = []
        max_len = 1
        for base, col in self.selected_series:
            values = raw_metrics.get(col, raw_metrics.get(base, []))
            arr = np.asarray(list(values), dtype=np.float32)
            if arr.size == 0:
                arr = np.array([np.nan], dtype=np.float32)
            series_values.append(arr)
            max_len = max(max_len, len(arr))
        max_len = min(max_len, self.context_steps)
        window = np.full((max_len, width), np.nan, dtype=np.float32)
        for j, arr in enumerate(series_values):
            tail = arr[-max_len:]
            window[-len(tail) :, j] = tail
        raw = compute_metric_stats(window, self.train_median, self.train_scale, self.spike_z)
        filled = np.where(np.isfinite(raw), raw, self.impute_values).astype(np.float32)
        scaled = ((filled - self.norm_mean) / self.norm_scale).astype(np.float32)
        return scaled, self.explain_raw_features(raw)

    def explain_raw_features(self, raw_features: np.ndarray, top_k: int = 5) -> dict:
        raw = np.asarray(raw_features, dtype=np.float32)
        n_series = len(self.selected_series)
        by_stat = {}
        for i, stat in enumerate(METRIC_STAT_NAMES):
            by_stat[stat] = raw[i * n_series : (i + 1) * n_series]

        def top_abs(stat: str) -> list[dict]:
            values = by_stat[stat]
            order = np.argsort(-np.abs(np.nan_to_num(values, nan=0.0)))[:top_k]
            return [
                {"metric": self.selected_series[int(i)][0], "value": float(values[int(i)])}
                for i in order
                if np.isfinite(values[int(i)]) and abs(float(values[int(i)])) > 0.0
            ]

        return {
            "top_metric_zscores": top_abs("robust_zscore_max"),
            "metric_spikes": top_abs("spike_count"),
            "metric_slopes": top_abs("slope"),
            "missing_metrics": top_abs("missing_count"),
        }

    def to_manifest(self) -> dict:
        aspect_counts = Counter(metric_aspect(base) for base, _ in self.selected_series)
        return {
            "selected_series": [{"base": base, "source_column": col, "aspect": metric_aspect(base)} for base, col in self.selected_series],
            "feature_count": int(len(self.feature_names)),
            "context_steps": int(self.context_steps),
            "stats": METRIC_STAT_NAMES,
            "aspect_counts": dict(aspect_counts),
            "spike_z": float(self.spike_z),
        }

    def to_preprocessor_state(self) -> dict:
        return {
            "selected_series": self.selected_series,
            "context_steps": int(self.context_steps),
            "spike_z": float(self.spike_z),
            "train_median": self.train_median.tolist(),
            "train_scale": self.train_scale.tolist(),
            "impute_values": self.impute_values.tolist(),
            "norm_mean": self.norm_mean.tolist(),
            "norm_scale": self.norm_scale.tolist(),
            "feature_names": self.feature_names,
        }


def build_metric_feature_matrix(
    frame: pd.DataFrame,
    raw_values: np.ndarray,
    train_median: np.ndarray,
    train_scale: np.ndarray,
    context_steps: int,
    spike_z: float,
) -> np.ndarray:
    n, width = raw_values.shape
    out = np.zeros((n, width * len(METRIC_STAT_NAMES)), dtype=np.float32)
    order = frame[["run_id", "window_ts", "row_pos"]].copy()
    for _, group in order.sort_values(["run_id", "window_ts", "row_pos"]).groupby("run_id", sort=False):
        rows = group["row_pos"].to_numpy(dtype=np.int64)
        values = raw_values[rows]
        for pos, row_idx in enumerate(rows):
            start = max(0, pos - int(context_steps) + 1)
            window = values[start : pos + 1]
            out[row_idx] = compute_metric_stats(window, train_median, train_scale, spike_z)
    return out


class CheapHadesDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: Iterable[int]):
        self.arrays = arrays
        self.indices = np.asarray(list(indices), dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[i])
        return {
            "log_ids": torch.from_numpy(self.arrays["log_ids"][idx]).long(),
            "metric_features": torch.from_numpy(self.arrays["metric_features"][idx]).float(),
            "target": torch.tensor(self.arrays["target"][idx], dtype=torch.float32),
            "row_index": torch.tensor(idx, dtype=torch.long),
        }


class CheapLogEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        repr_dim: int = 64,
        max_log_events: int = 32,
        encoder: str = "pool",
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.padding_idx = int(padding_idx)
        self.encoder = str(encoder)
        self.embedding = nn.Embedding(int(vocab_size), int(embedding_dim), padding_idx=self.padding_idx)
        if self.encoder == "conv":
            self.conv = nn.Sequential(
                nn.Conv1d(int(embedding_dim), int(repr_dim), kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(int(repr_dim), int(repr_dim), kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.proj = nn.Sequential(nn.Linear(int(repr_dim) * 2, int(repr_dim)), nn.ReLU(), nn.LayerNorm(int(repr_dim)))
        else:
            self.conv = None
            self.proj = nn.Sequential(
                nn.Linear(int(embedding_dim) * 2, int(repr_dim)),
                nn.ReLU(),
                nn.LayerNorm(int(repr_dim)),
            )
        self.max_log_events = int(max_log_events)

    @staticmethod
    def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.float().unsqueeze(-1)
        return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def masked_max(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = tokens.masked_fill(~mask.unsqueeze(-1), -1e9)
        values = masked.max(dim=1).values
        no_token = ~mask.any(dim=1)
        return torch.where(no_token.unsqueeze(1), torch.zeros_like(values), values)

    def forward(self, log_ids: torch.Tensor) -> torch.Tensor:
        mask = log_ids.ne(self.padding_idx)
        emb = self.embedding(log_ids)
        if self.encoder == "conv":
            tokens = self.conv(emb.transpose(1, 2)).transpose(1, 2)
        else:
            tokens = emb
        pooled = torch.cat([self.masked_mean(tokens, mask), self.masked_max(tokens, mask)], dim=1)
        return self.proj(pooled)


class CheapMetricEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, repr_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(repr_dim)),
            nn.ReLU(),
            nn.LayerNorm(int(repr_dim)),
        )

    def forward(self, metric_features: torch.Tensor) -> torch.Tensor:
        return self.net(metric_features)


class CheapFusion(nn.Module):
    def __init__(self, repr_dim: int = 64, mode: str = "product_diff") -> None:
        super().__init__()
        self.repr_dim = int(repr_dim)
        self.mode = str(mode)
        if self.mode == "metric_only":
            self.output_dim = self.repr_dim
        elif self.mode == "log_only":
            self.output_dim = self.repr_dim
        elif self.mode == "concat":
            self.output_dim = self.repr_dim * 2
        elif self.mode == "gated":
            self.gate = nn.Linear(self.repr_dim * 2, self.repr_dim)
            self.output_dim = self.repr_dim
        elif self.mode == "product_diff":
            self.output_dim = self.repr_dim * 4
        else:
            raise ValueError(f"Unsupported fusion mode: {self.mode}")

    def forward(self, log_repr: torch.Tensor, metric_repr: torch.Tensor) -> torch.Tensor:
        if self.mode == "metric_only":
            return metric_repr
        if self.mode == "log_only":
            return log_repr
        if self.mode == "concat":
            return torch.cat([log_repr, metric_repr], dim=1)
        if self.mode == "gated":
            gate = torch.sigmoid(self.gate(torch.cat([log_repr, metric_repr], dim=1)))
            return gate * log_repr + (1.0 - gate) * metric_repr
        return torch.cat([log_repr, metric_repr, log_repr * metric_repr, torch.abs(log_repr - metric_repr)], dim=1)


class CheapHadesA(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        metric_dim: int,
        max_log_events: int = 32,
        embedding_dim: int = 32,
        repr_dim: int = 64,
        metric_hidden_dim: int = 64,
        log_encoder: str = "pool",
        fusion: str = "product_diff",
    ) -> None:
        super().__init__()
        self.fusion_mode = str(fusion)
        self.repr_dim = int(repr_dim)
        self.log_encoder = CheapLogEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            repr_dim=repr_dim,
            max_log_events=max_log_events,
            encoder=log_encoder,
        )
        self.metric_encoder = CheapMetricEncoder(metric_dim, hidden_dim=metric_hidden_dim, repr_dim=repr_dim)
        self.fusion = CheapFusion(repr_dim=repr_dim, mode=fusion)
        self.head = nn.Sequential(
            nn.Linear(self.fusion.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, log_ids: torch.Tensor, metric_features: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "metric_only":
            metric_repr = self.metric_encoder(metric_features)
            log_repr = torch.zeros_like(metric_repr)
        else:
            log_repr = self.log_encoder(log_ids)
        if self.fusion_mode == "log_only":
            metric_repr = torch.zeros_like(log_repr)
        elif self.fusion_mode != "metric_only":
            metric_repr = self.metric_encoder(metric_features)
        fused = self.fusion(log_repr, metric_repr)
        return self.head(fused).squeeze(1)


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def predict_scores(model: CheapHadesA, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            logits = model(batch["log_ids"], batch["metric_features"])
            scores = torch.sigmoid(logits).detach().cpu().numpy()
            for row_idx, target, score in zip(
                batch["row_index"].detach().cpu().numpy(),
                batch["target"].detach().cpu().numpy(),
                scores,
            ):
                rows.append({"row_index": int(row_idx), "target": int(target), "score": float(score)})
    return pd.DataFrame(rows)


def pos_weight_value(y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    return float(neg / pos)


def train_loop(
    model: CheapHadesA,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_y: np.ndarray,
    cfg: CheapHadesConfig,
    device: torch.device,
) -> tuple[CheapHadesA, dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value(train_y), dtype=torch.float32, device=device))
    best_state = None
    best_score = -1.0
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["log_ids"], batch["metric_features"])
            loss = criterion(logits, batch["target"])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_pred = predict_scores(model, val_loader, device)
        val_ap = safe_ap(val_pred["target"], val_pred["score"])
        val_thr = best_f1_threshold(val_pred["target"], val_pred["score"])
        val_metrics = evaluate_binary(val_pred["target"], val_pred["score"], val_thr)
        monitor = val_ap if val_ap is not None else val_metrics["f1"]
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)) if losses else None,
                "val_average_precision": val_ap,
                "val_f1": val_metrics["f1"],
                "val_threshold": val_thr,
            }
        )
        if float(monitor or 0.0) > best_score:
            best_score = float(monitor or 0.0)
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg.patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_monitor": best_score, "history": history}


def train_one_variant_seed(
    arrays: dict[str, np.ndarray],
    idx: dict[str, np.ndarray],
    cfg: CheapHadesConfig,
    variant: str,
    seed: int,
    model_args: dict,
    output_dir: Path,
) -> dict:
    set_seed(seed)
    device = torch.device(cfg.device)
    train_ds = CheapHadesDataset(arrays, idx["train"])
    val_ds = CheapHadesDataset(arrays, idx["validation"])
    test_ds = CheapHadesDataset(arrays, idx["test"])
    train_loader = DataLoader(train_ds, batch_size=int(cfg.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)

    model = CheapHadesA(**model_args, fusion=variant).to(device)
    model, train_info = train_loop(model, train_loader, val_loader, arrays["target"][idx["train"]], cfg, device)

    val_pred = predict_scores(model, val_loader, device)
    threshold = best_f1_threshold(val_pred["target"], val_pred["score"])
    test_pred = predict_scores(model, test_loader, device)
    val_metrics = evaluate_binary(val_pred["target"], val_pred["score"], threshold)
    test_metrics = evaluate_binary(test_pred["target"], test_pred["score"], threshold)

    state_path = output_dir / f"cheap_hades_{variant}_seed{seed}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_args": {**model_args, "fusion": variant},
            "threshold": float(threshold),
            "target": cfg.target,
            "variant": variant,
            "seed": int(seed),
        },
        state_path,
    )
    val_pred["variant"] = variant
    val_pred["seed"] = int(seed)
    val_pred["split"] = "validation"
    test_pred["variant"] = variant
    test_pred["seed"] = int(seed)
    test_pred["split"] = "test"
    metric_rows = []
    for split, metrics in [("validation", val_metrics), ("test", test_metrics)]:
        metric_rows.append(
            {
                "source": "cheap_hades",
                "model": variant,
                "seed": int(seed),
                "split": split,
                "target": cfg.target,
                **metrics,
                "best_epoch": train_info["best_epoch"],
                "state_path": str(state_path),
                "threshold_source": "validation_best_f1",
            }
        )
    return {
        "model": model,
        "state_path": state_path,
        "threshold": float(threshold),
        "val_pred": val_pred,
        "test_pred": test_pred,
        "metrics": pd.DataFrame(metric_rows),
        "train_info": train_info,
    }


def add_contract_columns(pred: pd.DataFrame, frame: pd.DataFrame, target: str) -> pd.DataFrame:
    cols = [
        "row_pos",
        "run_id",
        "workload",
        "scenario",
        "scenario_type",
        "failure_family",
        "anchor_policy",
        "window_ts",
        "split",
        "log_available",
        "kpi_available",
        "has_both_modalities",
        target,
    ]
    cols = [col for col in cols if col in frame.columns]
    meta = frame[cols].rename(columns={"row_pos": "row_index", target: "label"})
    return pred.merge(meta, on="row_index", how="left")


def metrics_from_existing_baselines(frame: pd.DataFrame, baseline_path: Path, target: str) -> pd.DataFrame:
    if not baseline_path.exists():
        return pd.DataFrame()
    pred = pd.read_csv(baseline_path)
    required = {"run_id", "window_ts", "target", "baseline", "score"}
    if not required.issubset(pred.columns):
        return pd.DataFrame()
    pred["window_ts"] = pd.to_datetime(pred["window_ts"], utc=True, errors="coerce")
    key = frame[["run_id", "window_ts", "split", target]].copy()
    key["window_ts"] = pd.to_datetime(key["window_ts"], utc=True, errors="coerce")

    rows = []
    for baseline, part in pred.groupby("baseline", sort=False):
        if target in set(part["target"].astype(str)):
            scores = part.loc[part["target"].astype(str).eq(target), ["run_id", "window_ts", "score"]].copy()
        elif target == "y_anomaly":
            pivot = (
                part.loc[part["target"].astype(str).isin(["y_warning", "y_risk"]), ["run_id", "window_ts", "target", "score"]]
                .pivot_table(index=["run_id", "window_ts"], columns="target", values="score", aggfunc="mean")
                .reset_index()
            )
            if pivot.empty:
                continue
            score_cols = [col for col in ["y_warning", "y_risk"] if col in pivot.columns]
            pivot["score"] = pivot[score_cols].max(axis=1)
            scores = pivot[["run_id", "window_ts", "score"]].copy()
        else:
            continue
        merged = key.merge(scores.drop_duplicates(["run_id", "window_ts"]), on=["run_id", "window_ts"], how="inner")
        available_splits = set(merged["split"].astype(str))
        if merged.empty or "test" not in available_splits:
            continue
        val = merged.loc[merged["split"].astype(str).eq("validation")]
        test = merged.loc[merged["split"].astype(str).eq("test")]
        if "validation" in available_splits and not val.empty:
            threshold = best_f1_threshold(val[target], val["score"])
            threshold_source = "validation_best_f1"
            split_parts = [("validation", val), ("test", test)]
        else:
            threshold = 0.5
            threshold_source = "default_0.5_no_validation_predictions"
            split_parts = [("test", test)]
        for split, part2 in split_parts:
            rows.append(
                {
                    "source": "existing_baseline_predictions",
                    "model": str(baseline),
                    "seed": None,
                    "split": split,
                    "target": target,
                    **evaluate_binary(part2[target], part2["score"], threshold),
                    "best_epoch": None,
                    "state_path": None,
                    "threshold_source": threshold_source,
                }
            )
    return pd.DataFrame(rows)


def score_column_for_target(pred: pd.DataFrame, target: str) -> pd.Series | None:
    if target == "y_warning" and "warning_score" in pred.columns:
        return pred["warning_score"]
    if target == "y_risk" and "risk_score" in pred.columns:
        return pred["risk_score"]
    if target == "y_anomaly":
        if "anomaly_score" in pred.columns:
            return pred["anomaly_score"]
        cols = [col for col in ["warning_score", "risk_score"] if col in pred.columns]
        if cols:
            return pred[cols].max(axis=1)
    return None


def align_existing_final_predictions(pred: pd.DataFrame, frame: pd.DataFrame, target: str, split: str) -> pd.DataFrame:
    score = score_column_for_target(pred, target)
    if score is None or "run_id" not in pred.columns or "window_ts" not in pred.columns:
        return pd.DataFrame()
    scores = pred[["run_id", "window_ts"]].copy()
    scores["window_ts"] = pd.to_datetime(scores["window_ts"], utc=True, errors="coerce")
    scores["score"] = pd.to_numeric(score, errors="coerce")
    key = frame.loc[frame["split"].astype(str).eq(split), ["run_id", "window_ts", target]].copy()
    key["window_ts"] = pd.to_datetime(key["window_ts"], utc=True, errors="coerce")
    return key.merge(scores.drop_duplicates(["run_id", "window_ts"]), on=["run_id", "window_ts"], how="inner")


def metrics_from_existing_final_model(model_dir: Path, target: str, frame: pd.DataFrame) -> pd.DataFrame:
    val_path = model_dir / "validation_ensemble_predictions.csv"
    test_path = model_dir / "ensemble_predictions.csv"
    if not val_path.exists() or not test_path.exists():
        return pd.DataFrame()
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    for df in [val, test]:
        if "y_anomaly" not in df.columns and {"y_warning", "y_risk"}.issubset(df.columns):
            df["y_anomaly"] = np.maximum(df["y_warning"].astype(int), df["y_risk"].astype(int))
    if target not in val.columns or target not in test.columns:
        return pd.DataFrame()
    val_aligned = align_existing_final_predictions(val, frame, target, "validation")
    test_aligned = align_existing_final_predictions(test, frame, target, "test")
    if val_aligned.empty or test_aligned.empty:
        return pd.DataFrame()
    threshold = best_f1_threshold(val_aligned[target], val_aligned["score"])
    rows = []
    model_name = f"final_multimodal:{model_dir.name}"
    for split, part in [("validation", val_aligned), ("test", test_aligned)]:
        rows.append(
            {
                "source": "existing_final_model",
                "model": model_name,
                "seed": None,
                "split": split,
                "target": target,
                **evaluate_binary(part[target], part["score"], threshold),
                "best_epoch": None,
                "state_path": str(model_dir),
                "threshold_source": "validation_best_f1",
            }
        )
    return pd.DataFrame(rows)


def load_existing_comparisons(cfg: CheapHadesConfig, root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline_path = Path(cfg.baseline_predictions_path) if cfg.baseline_predictions_path else root / "dashboard" / "artifacts" / "reports" / "baseline_predictions.csv.gz"
    if not baseline_path.is_absolute():
        baseline_path = root / baseline_path
    if baseline_path.exists():
        rows.append(metrics_from_existing_baselines(frame, baseline_path, cfg.target))

    if cfg.existing_model_dir:
        model_dirs = [Path(cfg.existing_model_dir)]
    else:
        model_root = root / "dashboard" / "artifacts" / "model"
        model_dirs = [path for path in model_root.iterdir() if path.is_dir()] if model_root.exists() else []
    for model_dir in model_dirs:
        if not model_dir.is_absolute():
            model_dir = root / model_dir
        rows.append(metrics_from_existing_final_model(model_dir, cfg.target, frame))
    rows = [df for df in rows if df is not None and not df.empty]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    numeric = [
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "false_positives",
        "missed_anomalies",
        "threshold",
    ]
    available = [col for col in numeric if col in metrics.columns]
    return (
        metrics.groupby(["source", "model", "split", "target"], dropna=False)[available]
        .agg(["mean", "std"])
        .reset_index()
    )


def export_onnx(model: CheapHadesA, metric_dim: int, max_log_events: int, path: Path) -> None:
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy_log_ids = torch.zeros(1, int(max_log_events), dtype=torch.long)
    dummy_metric = torch.zeros(1, int(metric_dim), dtype=torch.float32)
    torch.onnx.export(
        model.cpu(),
        (dummy_log_ids, dummy_metric),
        str(path),
        input_names=["log_ids", "metric_features"],
        output_names=["logit"],
        dynamic_axes={
            "log_ids": {0: "batch"},
            "metric_features": {0: "batch"},
            "logit": {0: "batch"},
        },
        opset_version=13,
    )


class NumpyCalibrationDataReader:
    def __init__(self, log_ids: np.ndarray, metric_features: np.ndarray, batch_size: int = 32):
        from onnxruntime.quantization import CalibrationDataReader

        if not issubclass(type(self), CalibrationDataReader):
            # The runtime checks for the duck-typed get_next method; this import
            # only makes missing onnxruntime fail early and clearly.
            pass
        self.log_ids = log_ids.astype(np.int64)
        self.metric_features = metric_features.astype(np.float32)
        self.batch_size = int(batch_size)
        self.pos = 0

    def get_next(self):
        if self.pos >= len(self.log_ids):
            return None
        end = min(len(self.log_ids), self.pos + self.batch_size)
        batch = {
            "log_ids": self.log_ids[self.pos : end],
            "metric_features": self.metric_features[self.pos : end],
        }
        self.pos = end
        return batch


def quantize_onnx(onnx_path: Path, int8_path: Path, mode: str, calibration_arrays: dict[str, np.ndarray] | None = None) -> dict:
    if mode == "none":
        return {"status": "skipped", "mode": mode}
    try:
        if mode == "dynamic":
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
        elif mode == "static":
            if calibration_arrays is None:
                raise RuntimeError("Static quantization requires calibration arrays.")
            from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

            reader = NumpyCalibrationDataReader(
                calibration_arrays["log_ids"],
                calibration_arrays["metric_features"],
                batch_size=32,
            )
            quantize_static(
                str(onnx_path),
                str(int8_path),
                reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
            )
        else:
            raise ValueError(f"Unknown quantization mode: {mode}")
    except Exception as exc:
        return {"status": "failed", "mode": mode, "error": f"{type(exc).__name__}: {exc}"}
    return {"status": "ok", "mode": mode, "path": str(int8_path)}


def predict(
    model: CheapHadesA,
    parser: DrainLikeParser,
    metric_builder: MetricFeatureBuilder,
    raw_logs: Iterable[str],
    raw_metrics: dict[str, Iterable[float]],
    threshold: float,
    max_log_events: int,
    device: str = "cpu",
) -> dict:
    log_ids, log_diag = parser.encode_raw_logs(raw_logs, max_events=max_log_events)
    metric_features, metric_diag = metric_builder.transform_raw_window(raw_metrics)
    model = model.to(torch.device(device))
    model.eval()
    with torch.no_grad():
        log_tensor = torch.from_numpy(log_ids[None, :]).long().to(torch.device(device))
        metric_tensor = torch.from_numpy(metric_features[None, :]).float().to(torch.device(device))
        score = float(torch.sigmoid(model(log_tensor, metric_tensor)).detach().cpu().item())
    return {
        "score": score,
        "is_anomaly": bool(score >= float(threshold)),
        "log_reasons": [
            {"type": "error_warn_count", "value": log_diag["error_warn_count"]},
            {"type": "new_template_count", "value": log_diag["new_template_count"]},
            {"type": "top_rare_templates", "value": log_diag["top_rare_templates"]},
        ],
        "metric_reasons": [
            {"type": "top_metric_zscores", "value": metric_diag["top_metric_zscores"]},
            {"type": "metric_spikes", "value": metric_diag["metric_spikes"]},
            {"type": "metric_slopes", "value": metric_diag["metric_slopes"]},
            {"type": "missing_metrics", "value": metric_diag["missing_metrics"]},
        ],
    }


def write_experiment_card(output_dir: Path, cfg: CheapHadesConfig, metrics: pd.DataFrame) -> None:
    lines = [
        f"# Cheap HADES Hypothesis: `{cfg.run_id}`",
        "",
        "Purpose: test a CPU-only HADES-like multimodal detector without online LLM or large Transformer inference.",
        "",
        "Online path: Drain-like template parser -> template_id sequence -> nn.Embedding -> pooling/Conv1D; "
        "KPI window statistics -> normalization -> tiny MLP; cheap fusion -> sigmoid anomaly score.",
        "",
        "This is not the final production training contour and does not replace `dashboard.train_multimodal_fusion`.",
        "",
        "## Primary test metrics",
        "",
    ]
    view = metrics.loc[metrics["split"].astype(str).eq("test")].copy() if not metrics.empty else pd.DataFrame()
    if view.empty:
        lines.append("_No test metrics were produced._")
    else:
        cols = ["source", "model", "seed", "target", "average_precision", "roc_auc", "precision", "recall", "f1", "false_positives", "missed_anomalies"]
        cols = [col for col in cols if col in view.columns]
        lines.append(markdown_table(view[cols]))
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `comparison_metrics.csv`: per-seed validation/test metrics for cheap variants and imported existing models.",
            "- `comparison_metrics_aggregated.csv`: mean/std by model and split.",
            "- `test_predictions.csv.gz`, `validation_predictions.csv.gz`: cheap model scores.",
            "- `feature_manifest.json`, `preprocessor_state.json`: reproducible feature contract for this hypothesis.",
            "- `exports/`: ONNX and INT8 ONNX for the best validation-F1 cheap variant when export succeeds.",
        ]
    )
    (output_dir / "EXPERIMENT_CARD.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "_empty_"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    rows = []
    rows.append("| " + " | ".join(str(c) for c in view.columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(view.columns)) + " |")
    for _, row in view.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in view.columns) + " |")
    if len(df) > max_rows:
        rows.append(f"\n_Showing first {max_rows} rows out of {len(df)}._")
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"cheap_hades_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--root", default=".")
    parser.add_argument("--contract-path", default="dashboard/artifacts/contract/strict_intensity_contract.csv.gz")
    parser.add_argument("--output-dir", default="hypothesis_results/cheap_hades")
    parser.add_argument("--target", choices=["y_anomaly", "y_warning", "y_risk"], default="y_anomaly")
    parser.add_argument("--variants", default="metric_only,log_only,concat,product_diff,gated")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-log-events", type=int, default=32)
    parser.add_argument("--log-context-sec", type=int, default=120)
    parser.add_argument("--max-templates", type=int, default=4096)
    parser.add_argument("--min-template-count", type=int, default=1)
    parser.add_argument("--log-encoder", choices=["pool", "conv"], default="pool")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--repr-dim", type=int, default=64)
    parser.add_argument("--metric-context-steps", type=int, default=12)
    parser.add_argument("--max-metric-series", type=int, default=64)
    parser.add_argument("--metric-hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--spike-z", type=float, default=3.5)
    parser.add_argument("--validation-strategy", choices=["auto", "existing", "run_id_holdout", "workload_holdout"], default="auto")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-existing-comparison", action="store_true")
    parser.add_argument("--baseline-predictions-path", default=None)
    parser.add_argument("--existing-model-dir", default=None)
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--quantization", choices=["dynamic", "static", "none"], default="dynamic")
    parser.add_argument("--calibration-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    contract_path = Path(args.contract_path)
    if not contract_path.is_absolute():
        contract_path = root / contract_path
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = root / output_root
    output_dir = output_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    cfg = CheapHadesConfig(
        run_id=args.run_id,
        root=str(root),
        contract_path=str(contract_path),
        output_dir=str(output_dir),
        target=args.target,
        variants=parse_csv_list(args.variants, str),
        seeds=parse_csv_list(args.seeds, int),
        max_rows=args.max_rows,
        max_log_events=args.max_log_events,
        log_context_sec=args.log_context_sec,
        max_templates=args.max_templates,
        min_template_count=args.min_template_count,
        log_encoder=args.log_encoder,
        embedding_dim=args.embedding_dim,
        repr_dim=args.repr_dim,
        metric_context_steps=args.metric_context_steps,
        max_metric_series=args.max_metric_series,
        metric_hidden_dim=args.metric_hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        spike_z=args.spike_z,
        validation_strategy=args.validation_strategy,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
        device=args.device,
        compare_existing=not bool(args.skip_existing_comparison),
        baseline_predictions_path=args.baseline_predictions_path,
        existing_model_dir=args.existing_model_dir,
        export_onnx=not bool(args.skip_onnx),
        quantization=args.quantization,
        calibration_size=args.calibration_size,
    )
    write_json(output_dir / "run_config.json", asdict(cfg))

    print(f"[1/7] Loading strict contract: {contract_path}")
    frame = load_contract(contract_path, cfg.max_rows, cfg.target)
    frame = apply_validation_split(frame, cfg.validation_strategy, cfg.validation_fraction, cfg.validation_seed)
    idx = split_indices(frame)
    split_summary(frame).to_csv(output_dir / "split_summary.csv", index=False)
    print(split_summary(frame).to_string(index=False))

    print("[2/7] Building Drain-like log template sequences")
    log_ids, parser, log_diag, log_manifest = build_log_arrays(
        frame,
        raw_root=root / "raw_data",
        train_idx=idx["train"],
        max_events=cfg.max_log_events,
        context_sec=cfg.log_context_sec,
        max_templates=cfg.max_templates,
        min_template_count=cfg.min_template_count,
    )
    log_diag.to_csv(output_dir / "log_window_diagnostics.csv", index=False)

    print("[3/7] Building cheap metric statistics")
    metric_builder, metric_features, metric_raw_features = MetricFeatureBuilder.fit_from_contract(
        frame,
        idx["train"],
        target=cfg.target,
        max_metric_series=cfg.max_metric_series,
        context_steps=cfg.metric_context_steps,
        spike_z=cfg.spike_z,
    )
    np.savez_compressed(
        output_dir / "feature_arrays_sample.npz",
        log_ids=log_ids[: min(len(log_ids), 1000)],
        metric_features=metric_features[: min(len(metric_features), 1000)],
        target=frame[cfg.target].astype(np.float32).to_numpy()[: min(len(frame), 1000)],
    )

    arrays = {
        "log_ids": log_ids.astype(np.int64),
        "metric_features": metric_features.astype(np.float32),
        "target": frame[cfg.target].astype(np.float32).to_numpy(),
    }
    feature_manifest = {
        "log": log_manifest,
        "metric": metric_builder.to_manifest(),
        "arrays": {key: list(value.shape) for key, value in arrays.items()},
        "target": cfg.target,
        "architecture": {
            "log_encoder": cfg.log_encoder,
            "embedding_dim": int(cfg.embedding_dim),
            "repr_dim": int(cfg.repr_dim),
            "fusion_variants": cfg.variants,
            "online_llm_inference": False,
        },
    }
    write_json(output_dir / "feature_manifest.json", feature_manifest)
    write_json(
        output_dir / "preprocessor_state.json",
        {
            "parser": parser.to_preprocessor_state(),
            "metric_builder": metric_builder.to_preprocessor_state(),
        },
    )

    model_args = {
        "vocab_size": int(len(parser.template_to_id)),
        "metric_dim": int(metric_features.shape[1]),
        "max_log_events": int(cfg.max_log_events),
        "embedding_dim": int(cfg.embedding_dim),
        "repr_dim": int(cfg.repr_dim),
        "metric_hidden_dim": int(cfg.metric_hidden_dim),
        "log_encoder": cfg.log_encoder,
    }

    print("[4/7] Training cheap HADES ablations")
    results = []
    for variant in cfg.variants:
        if variant not in {"metric_only", "log_only", "concat", "product_diff", "gated"}:
            raise ValueError(f"Unknown variant: {variant}")
        for seed in cfg.seeds:
            print(f"  - variant={variant} seed={seed}")
            results.append(train_one_variant_seed(arrays, idx, cfg, variant, seed, model_args, models_dir))

    print("[5/7] Writing predictions and cheap metrics")
    val_predictions = pd.concat([item["val_pred"] for item in results], ignore_index=True)
    test_predictions = pd.concat([item["test_pred"] for item in results], ignore_index=True)
    val_predictions = add_contract_columns(val_predictions, frame, cfg.target)
    test_predictions = add_contract_columns(test_predictions, frame, cfg.target)
    val_predictions.to_csv(output_dir / "validation_predictions.csv.gz", index=False, compression="gzip")
    test_predictions.to_csv(output_dir / "test_predictions.csv.gz", index=False, compression="gzip")
    cheap_metrics = pd.concat([item["metrics"] for item in results], ignore_index=True)

    print("[6/7] Loading existing comparisons")
    existing_metrics = pd.DataFrame()
    if cfg.compare_existing:
        existing_metrics = load_existing_comparisons(cfg, root, frame)
    all_metrics = pd.concat([cheap_metrics, existing_metrics], ignore_index=True) if not existing_metrics.empty else cheap_metrics
    all_metrics.to_csv(output_dir / "comparison_metrics.csv", index=False)
    aggregate = aggregate_metrics(all_metrics)
    aggregate.to_csv(output_dir / "comparison_metrics_aggregated.csv", index=False)

    print("[7/7] Exporting ONNX/INT8 for best cheap variant")
    export_report = {"status": "skipped", "reason": "export disabled"}
    if cfg.export_onnx and results:
        best = max(
            results,
            key=lambda item: float(
                item["metrics"]
                .loc[item["metrics"]["split"].astype(str).eq("validation"), "f1"]
                .fillna(0.0)
                .iloc[0]
            ),
        )
        checkpoint = torch.load(best["state_path"], map_location="cpu")
        export_model = CheapHadesA(**checkpoint["model_args"])
        export_model.load_state_dict(checkpoint["state_dict"])
        export_dir = output_dir / "exports"
        onnx_path = export_dir / f"cheap_hades_{checkpoint['variant']}_seed{checkpoint['seed']}.onnx"
        int8_path = export_dir / f"cheap_hades_{checkpoint['variant']}_seed{checkpoint['seed']}.int8.onnx"
        try:
            export_onnx(export_model, metric_dim=metric_features.shape[1], max_log_events=cfg.max_log_events, path=onnx_path)
            calibration_arrays = None
            if cfg.quantization == "static":
                cal_idx = idx["train"][: min(len(idx["train"]), int(cfg.calibration_size))]
                calibration_arrays = {
                    "log_ids": arrays["log_ids"][cal_idx],
                    "metric_features": arrays["metric_features"][cal_idx],
                }
            q_report = quantize_onnx(onnx_path, int8_path, cfg.quantization, calibration_arrays)
            export_report = {
                "status": "ok",
                "best_variant": checkpoint["variant"],
                "best_seed": int(checkpoint["seed"]),
                "threshold": float(checkpoint["threshold"]),
                "onnx_path": str(onnx_path),
                "quantization": q_report,
            }
        except Exception as exc:
            export_report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    write_json(output_dir / "export_report.json", export_report)

    summary = {
        "run_id": cfg.run_id,
        "output_dir": str(output_dir),
        "split_summary": split_summary(frame).to_dict("records"),
        "feature_manifest": feature_manifest,
        "comparison_metrics": all_metrics.to_dict("records"),
        "export_report": export_report,
    }
    write_json(output_dir / "metrics_summary.json", summary)
    write_experiment_card(output_dir, cfg, all_metrics)

    print(f"Done. Hypothesis outputs written to: {output_dir}")
    print(all_metrics.loc[all_metrics["split"].astype(str).eq("test")].to_string(index=False))


if __name__ == "__main__":
    main()
