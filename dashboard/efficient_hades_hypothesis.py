"""
Efficient HADES-like hypothesis experiment.

This is a separate test module. It does not replace the final training pipeline.

Preserved HADES path:
    logs -> parser -> cached template embeddings -> log sequence encoder
    raw KPI files -> metric time-series segments -> metric sequence encoder
    log tokens <-> metric tokens via cross-modal attention -> classifier

Typical smoke run:
    python -m dashboard.efficient_hades_hypothesis --run-id smoke_eff_hades --max-rows 300 --epochs 2 --variants original,efficient_hades --skip-onnx

Full CPU hypothesis run:
    python -m dashboard.efficient_hades_hypothesis --run-id efficient_hades_cpu --seeds 42,43,44 --epochs 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import time
import warnings
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


RAW_LOG_RE = re.compile(
    r"^(?P<date>\d{2}/\d{2}/\d{2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<level>[A-Z]+)\s+"
    r"(?P<component>[^:]+):\s*"
    r"(?P<message>.*)$"
)

SEVERITY_WEIGHT = {
    "FATAL": 5.0,
    "ERROR": 4.0,
    "WARN": 2.5,
    "WARNING": 2.5,
    "INFO": 0.2,
    "DEBUG": 0.0,
    "TRACE": 0.0,
}

LOG_KEYWORDS = [
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

DEFAULT_INSTRUMENTS = ["cpustat", "iostat", "memstat", "netstat", "vmstat"]
DEFAULT_NODES = ["master3", "worker1", "worker2", "worker3"]


@dataclass
class EfficientHadesConfig:
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
    log_embedding_dim: int
    modern_log_backend: str
    embeddinggemma_model: str
    metric_segment_sec: int
    metric_segment_len: int
    max_metric_channels: int
    metric_nodes: list[str]
    metric_instruments: list[str]
    d_model: int
    heads: int
    log_layers: int
    metric_layers: int
    batch_size: int
    epochs: int
    patience: int
    lr: float
    weight_decay: float
    pseudo_labeling: bool
    pseudo_start_epoch: int
    pseudo_low_quantile: float
    pseudo_high_quantile: float
    pseudo_weight: float
    fp_budget: float
    train_monitor: str
    validation_strategy: str
    validation_fraction: float
    validation_seed: int
    device: str
    export_onnx: bool
    quantization: str
    calibration_size: int
    benchmark_batches: int


VARIANT_SPECS = {
    "original": {
        "log_embedding": "fasttext_hash",
        "log_encoder": "transformer",
        "metric_encoder": "causal_conv",
        "runtime": "fp32",
        "description": "FastText-like cached subword embeddings + Transformer logs + causal conv metrics + cross-modal attention.",
    },
    "modern_log": {
        "log_embedding": "modern",
        "log_encoder": "transformer",
        "metric_encoder": "causal_conv",
        "runtime": "fp32",
        "description": "StaticEmbedding/EmbeddingGemma cached templates + original metric encoder + cross-modal attention.",
    },
    "modern_metric": {
        "log_embedding": "fasttext_hash",
        "log_encoder": "transformer",
        "metric_encoder": "tiny_time_mixer",
        "runtime": "fp32",
        "description": "FastText-like logs + TinyTimeMixer-lite metric segment encoder + cross-modal attention.",
    },
    "modern_runtime": {
        "log_embedding": "fasttext_hash",
        "log_encoder": "transformer",
        "metric_encoder": "causal_conv",
        "runtime": "onnx_int8",
        "description": "Original-style model exported to ONNX INT8 for runtime comparison.",
    },
    "efficient_hades": {
        "log_embedding": "modern",
        "log_encoder": "transformer",
        "metric_encoder": "tiny_time_mixer",
        "runtime": "onnx_int8",
        "description": "Cached modern embeddings + TinyTimeMixer-lite metrics + cross-modal attention + ONNX INT8.",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def stable_hash(value: str) -> int:
    return int(hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:16], 16)


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
    best_thr = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        pred = (score >= float(threshold)).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        if float(f1) > best_f1:
            best_f1 = float(f1)
            best_thr = float(threshold)
    return best_thr


def fp_budget_threshold(y, score, fp_budget: float) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    negatives = int((y == 0).sum())
    if negatives <= 0:
        return best_f1_threshold(y, score)
    allowed_fp = max(0, int(math.floor(float(fp_budget) * negatives)))
    finite_scores = np.unique(np.nan_to_num(score, nan=-np.inf))
    candidates = np.unique(np.concatenate([finite_scores, [0.5, 1.0 + 1e-6]]))
    best = candidates[-1]
    best_recall = -1.0
    for threshold in candidates:
        pred = (score >= float(threshold)).astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        if fp <= allowed_fp:
            recall = int(((pred == 1) & (y == 1)).sum()) / max(int((y == 1).sum()), 1)
            if recall > best_recall:
                best_recall = recall
                best = float(threshold)
    return float(best)


def evaluate_binary(y, score, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pred = (score >= float(threshold)).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
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
        "threshold": float(threshold),
    }


def load_contract(path: Path, max_rows: int | None, target: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["window_ts", "window_end"])
    if "split" not in frame.columns:
        raise RuntimeError("Contract must contain split column.")
    frame["split"] = frame["split"].astype(str).str.strip().str.lower().replace(
        {"valid": "validation", "val": "validation", "dev": "validation"}
    )
    if "window_index" not in frame.columns:
        frame["window_index"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.sort_values(["run_id", "window_ts", "window_index"]).reset_index(drop=True)
    if "y_anomaly" not in frame.columns:
        frame["y_anomaly"] = np.maximum(frame["y_warning"].astype(int), frame["y_risk"].astype(int))
    if target not in frame.columns:
        raise RuntimeError(f"Missing target column: {target}")
    if max_rows:
        rng = np.random.default_rng(42)
        parts = []
        per_split = max(1, int(max_rows) // max(frame["split"].nunique(), 1))
        for _, part in frame.groupby("split", sort=False):
            keep = []
            per_class = max(1, min(40, per_split // 3))
            for label in [0, 1]:
                candidates = part.index[part[target].astype(int).eq(label)].to_numpy()
                if len(candidates):
                    keep.extend(rng.choice(candidates, size=min(per_class, len(candidates)), replace=False).tolist())
            rest = np.setdiff1d(part.index.to_numpy(), np.array(sorted(set(keep)), dtype=int))
            need = max(0, min(per_split, len(part)) - len(set(keep)))
            if need and len(rest):
                keep.extend(rng.choice(rest, size=min(need, len(rest)), replace=False).tolist())
            parts.append(frame.loc[sorted(set(keep))])
        frame = pd.concat(parts).sort_values(["run_id", "window_ts", "window_index"]).reset_index(drop=True)
    frame["row_pos"] = np.arange(len(frame), dtype=np.int64)
    frame["log_text"] = frame.get("log_text", "").fillna("").astype(str)
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
            raise RuntimeError("validation_strategy=existing requested, but no validation split is present.")
        return frame

    train = frame.loc[frame["split"].astype(str).eq("train")].copy()
    if train.empty:
        raise RuntimeError("Cannot derive validation split from empty train split.")
    group_col = "workload" if strategy == "workload_holdout" else "run_id"
    if group_col not in train.columns or train[group_col].astype(str).nunique() < 2:
        group_col = "run_id"
    if train[group_col].astype(str).nunique() < 2:
        raise RuntimeError("Cannot derive validation split: fewer than two groups.")

    splitter = GroupShuffleSplit(n_splits=64, test_size=float(fraction), random_state=int(seed))
    groups = train[group_col].astype(str)
    train_index = train.index.to_numpy()
    best_rel = None
    best_score = -1e9
    for _, val_rel in splitter.split(train, groups=groups):
        val = train.iloc[val_rel]
        score = 0.0
        for target in ["y_anomaly", "y_warning", "y_risk"]:
            if target in val.columns:
                score += 2.0 if val[target].nunique() == 2 else 0.0
                score += min(int(val[target].sum()), 20) / 20.0
        score -= abs((len(val) / max(len(train), 1)) - float(fraction))
        if score > best_score:
            best_score = score
            best_rel = val_rel
    if best_rel is None:
        raise RuntimeError("GroupShuffleSplit did not produce a validation split.")
    frame.loc[train_index[best_rel], "split"] = "validation"
    frame["validation_group_column"] = group_col
    return frame


def split_indices(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    result = {}
    for split in ["train", "validation", "test"]:
        result[split] = frame.index[frame["split"].astype(str).eq(split)].to_numpy(dtype=np.int64)
    if not all(len(value) for value in result.values()):
        raise RuntimeError(f"Empty split detected: { {k: len(v) for k, v in result.items()} }")
    return result


def split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, part in frame.groupby("split", sort=False):
        rows.append(
            {
                "split": split,
                "rows": int(len(part)),
                "runs": int(part["run_id"].nunique()),
                "workloads": int(part["workload"].nunique()),
                "anomaly_positives": int(part["y_anomaly"].sum()) if "y_anomaly" in part else None,
                "warning_positives": int(part["y_warning"].sum()) if "y_warning" in part else None,
                "risk_positives": int(part["y_risk"].sum()) if "y_risk" in part else None,
            }
        )
    return pd.DataFrame(rows)


class DrainLikeParser:
    def __init__(self) -> None:
        self.template_counts: Counter[str] = Counter()

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
        return text.strip() or "__empty__"

    def parse_line(self, line: str, line_id: int) -> dict | None:
        text = str(line).strip()
        match = RAW_LOG_RE.match(text)
        if not match:
            return None
        ts = pd.to_datetime(
            f"{match.group('date')} {match.group('time')}",
            format="%y/%m/%d %H:%M:%S",
            utc=True,
            errors="coerce",
        )
        if pd.isna(ts):
            return None
        level = match.group("level").upper()
        component = re.sub(r"\s+", " ", match.group("component").strip().lower())
        template = f"{level} {component}: {self.normalize_message(match.group('message'))}"
        lower = template.lower()
        keyword_score = sum(1.0 for keyword in LOG_KEYWORDS if keyword in lower)
        return {
            "timestamp": ts,
            "level": level,
            "component": component,
            "template": template,
            "severity": float(SEVERITY_WEIGHT.get(level, 0.0) + keyword_score),
            "line_id": int(line_id),
        }

    def fit(self, templates: Iterable[str]) -> None:
        self.template_counts = Counter(str(template) for template in templates)

    def is_rare(self, template: str) -> bool:
        return self.template_counts.get(str(template), 0) <= 2


def log_cache_key(path: Path) -> str:
    if not path.exists():
        return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:20]
    stat = path.stat()
    return hashlib.sha1(f"{path.resolve()}:{stat.st_size}:{int(stat.st_mtime)}".encode("utf-8")).hexdigest()[:20]


def parse_raw_log_file(path: Path, parser: DrainLikeParser, cache_dir: Path | None = None) -> pd.DataFrame:
    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"raw_log_events_{log_cache_key(path)}.pkl"
        if cache_file.exists():
            return pd.read_pickle(cache_file)
    rows = []
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "level", "component", "template", "severity", "line_id"])
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_id, line in enumerate(handle, start=1):
            row = parser.parse_line(line, line_id)
            if row is not None:
                rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "level", "component", "template", "severity", "line_id"])
    df = pd.DataFrame(rows).sort_values(["timestamp", "line_id"]).reset_index(drop=True)
    if cache_file is not None:
        df.to_pickle(cache_file)
    return df


def select_log_events(events: pd.DataFrame, parser: DrainLikeParser, max_events: int) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    part = events.copy()
    recency = np.linspace(0.0, 1.0, len(part), dtype=np.float32)
    rarity = part["template"].astype(str).apply(lambda value: 1.0 if parser.is_rare(value) else 0.0)
    part["_score"] = part["severity"].astype(float) + rarity.astype(float) + recency
    if len(part) <= int(max_events):
        return part
    keep = set(part.tail(max(4, int(max_events) // 3)).index.tolist())
    for idx in part.sort_values("_score", ascending=False).index:
        keep.add(idx)
        if len(keep) >= int(max_events):
            break
    return part.loc[sorted(keep)].sort_values(["timestamp", "line_id"]).head(int(max_events))


def hashed_subword_embedding(text: str, dim: int, mode: str) -> np.ndarray:
    value = str(text).lower()
    tokens: list[str] = []
    if mode == "fasttext_hash":
        clean = re.sub(r"\s+", " ", value)
        for word in clean.split():
            word = f"<{word}>"
            for n in [3, 4, 5]:
                tokens.extend(word[i : i + n] for i in range(max(1, len(word) - n + 1)))
    else:
        tokens = re.findall(r"[a-z0-9_*<>]+", value)
        tokens.extend([f"component:{part}" for part in value.split(":")[:1]])
    if not tokens:
        tokens = ["__empty__"]
    vec = np.zeros(int(dim), dtype=np.float32)
    for token in tokens:
        h = stable_hash(f"{mode}:{token}")
        idx = h % int(dim)
        sign = 1.0 if ((h >> 8) & 1) else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec /= norm
    return vec.astype(np.float32)


def embeddinggemma_cached(texts: list[str], dim: int, model_name: str, cache_file: Path) -> tuple[np.ndarray, dict]:
    if cache_file.exists():
        payload = np.load(cache_file, allow_pickle=True)
        cached_texts = payload["texts"].astype(str).tolist()
        cached = payload["embeddings"].astype(np.float32)
        if cached_texts == texts and cached.shape[1] == int(dim):
            return cached, {"backend": "embeddinggemma", "cache_loaded": True, "cache_file": str(cache_file)}
    try:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        try:
            embeddings = encoder.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
                truncate_dim=int(dim),
            ).astype(np.float32)
        except TypeError:
            embeddings = encoder.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)[:, : int(dim)]
        if embeddings.shape[1] < int(dim):
            embeddings = np.pad(embeddings, ((0, 0), (0, int(dim) - embeddings.shape[1]))).astype(np.float32)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, texts=np.array(texts, dtype=object), embeddings=embeddings)
        return embeddings, {"backend": "embeddinggemma", "cache_loaded": False, "cache_file": str(cache_file)}
    except Exception as exc:
        fallback = np.vstack([hashed_subword_embedding(text, dim, "static_hash") for text in texts]).astype(np.float32)
        return fallback, {
            "backend": "static_hash_fallback",
            "embeddinggemma_error": f"{type(exc).__name__}: {exc}",
            "cache_loaded": False,
        }


def build_template_embeddings(
    templates: list[str],
    backend: str,
    dim: int,
    output_dir: Path,
    embeddinggemma_model: str,
) -> tuple[dict[str, np.ndarray], dict]:
    unique = sorted(set(str(template) for template in templates))
    if "__pad__" not in unique:
        unique.insert(0, "__pad__")
    if backend == "embeddinggemma":
        digest = hashlib.sha1("\n".join(unique).encode("utf-8")).hexdigest()[:16]
        arr, manifest = embeddinggemma_cached(unique, dim, embeddinggemma_model, output_dir / "cache" / f"embeddinggemma_{digest}.npz")
    else:
        mode = "fasttext_hash" if backend == "fasttext_hash" else "static_hash"
        arr = np.vstack([hashed_subword_embedding(text, dim, mode) for text in unique]).astype(np.float32)
        manifest = {"backend": mode, "cache_loaded": False}
    mapping = {text: arr[i] for i, text in enumerate(unique)}
    manifest.update({"unique_templates": int(len(unique)), "embedding_dim": int(dim)})
    return mapping, manifest


def build_log_arrays(
    frame: pd.DataFrame,
    raw_root: Path,
    train_idx: np.ndarray,
    max_events: int,
    context_sec: int,
    backend: str,
    dim: int,
    output_dir: Path,
    embeddinggemma_model: str,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    parser = DrainLikeParser()
    events_by_run: dict[str, pd.DataFrame] = {}
    train_runs = set(frame.iloc[train_idx]["run_id"].astype(str))
    train_templates: list[str] = []
    for run_id, group in frame.groupby("run_id", sort=False):
        workload = str(group["workload"].iloc[0])
        scenario = str(group["scenario"].iloc[0])
        path = raw_root / workload / "mapred_logs" / f"{scenario}.log"
        events = parse_raw_log_file(path, parser, cache_dir=cache_dir)
        events_by_run[str(run_id)] = events
        if str(run_id) in train_runs and not events.empty:
            train_templates.extend(events["template"].astype(str).tolist())
    parser.fit(train_templates)

    window_templates: list[list[str]] = []
    masks = np.zeros((len(frame), int(max_events)), dtype=np.float32)
    all_selected_templates: list[str] = ["__pad__"]
    diagnostics = []
    for row in frame.itertuples(index=False):
        events = events_by_run.get(str(row.run_id), pd.DataFrame())
        if events.empty or pd.isna(row.window_end):
            selected = events.iloc[0:0].copy() if not events.empty else pd.DataFrame()
        else:
            start = row.window_end - pd.Timedelta(seconds=int(context_sec))
            part = events.loc[(events["timestamp"] >= start) & (events["timestamp"] < row.window_end)].copy()
            selected = select_log_events(part, parser, max_events=max_events)
        templates = selected["template"].astype(str).tolist()[: int(max_events)] if not selected.empty else []
        all_selected_templates.extend(templates)
        padded = templates + ["__pad__"] * (int(max_events) - len(templates))
        window_templates.append(padded[: int(max_events)])
        masks[int(row.row_pos), : len(templates)] = 1.0
        diagnostics.append(
            {
                "row_pos": int(row.row_pos),
                "selected_log_events": int(len(templates)),
                "rare_templates": int(sum(parser.is_rare(template) for template in templates)),
            }
        )

    embedding_map, embedding_manifest = build_template_embeddings(
        all_selected_templates,
        backend=backend,
        dim=dim,
        output_dir=output_dir,
        embeddinggemma_model=embeddinggemma_model,
    )
    log_embeddings = np.zeros((len(frame), int(max_events), int(dim)), dtype=np.float32)
    for i, templates in enumerate(window_templates):
        for j, template in enumerate(templates):
            log_embeddings[i, j] = embedding_map.get(template, embedding_map["__pad__"])
    log_embeddings[masks <= 0.5] = 0.0
    diag = pd.DataFrame(diagnostics)
    manifest = {
        "parser": "drain_like_regex",
        "embedding": embedding_manifest,
        "max_log_events": int(max_events),
        "context_sec": int(context_sec),
        "mean_selected_events": float(diag["selected_log_events"].mean()) if not diag.empty else 0.0,
        "empty_window_share": float((diag["selected_log_events"] <= 0).mean()) if not diag.empty else 1.0,
    }
    return log_embeddings, masks, manifest


def sanitize_metric_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value).strip().replace("%", "pct")).strip("_").lower()


def parse_sysstat_file(path: Path, prefix: str) -> pd.DataFrame:
    if not path.exists() or path.name.startswith("."):
        return pd.DataFrame()
    rows = []
    header: list[str] | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            if parts[0] == "HostName" and "TimeStamp" in parts:
                header = parts
                continue
            if header is None or len(parts) < len(header):
                continue
            record = dict(zip(header, parts[: len(header)]))
            timestamp = pd.to_numeric(record.get("TimeStamp"), errors="coerce")
            if not np.isfinite(timestamp):
                continue
            out = {"timestamp": pd.to_datetime(float(timestamp), unit="s", utc=True)}
            for key, raw in record.items():
                if key in {"HostName", "TimeStamp", "Device:", "IFACE"}:
                    continue
                value = pd.to_numeric(raw, errors="coerce")
                if np.isfinite(value):
                    out[f"{prefix}__{sanitize_metric_name(key)}"] = float(value)
            rows.append(out)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    numeric_cols = [col for col in df.columns if col != "timestamp"]
    if not numeric_cols:
        return pd.DataFrame()
    return df.groupby("timestamp", as_index=False)[numeric_cols].mean().sort_values("timestamp")


def metric_cache_key(run_root: Path, nodes: list[str], instruments: list[str]) -> str:
    parts = [str(run_root.resolve()), ",".join(nodes), ",".join(instruments)]
    for node in nodes:
        for instrument in instruments:
            path = run_root / "instruments" / node / instrument
            if path.exists():
                stat = path.stat()
                parts.append(f"{path}:{stat.st_size}:{int(stat.st_mtime)}")
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:20]


def load_run_metrics(run_root: Path, nodes: list[str], instruments: list[str], cache_dir: Path | None = None) -> pd.DataFrame:
    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"raw_metric_frame_{metric_cache_key(run_root, nodes, instruments)}.pkl"
        if cache_file.exists():
            return pd.read_pickle(cache_file)
    frames = []
    for node in nodes:
        for instrument in instruments:
            path = run_root / "instruments" / node / instrument
            prefix = f"{sanitize_metric_name(node)}__{sanitize_metric_name(instrument)}"
            part = parse_sysstat_file(path, prefix)
            if not part.empty:
                frames.append(part)
    if not frames:
        return pd.DataFrame()
    merged = frames[0]
    for part in frames[1:]:
        merged = merged.merge(part, on="timestamp", how="outer")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    if cache_file is not None:
        merged.to_pickle(cache_file)
    return merged


def select_metric_channels(metric_frames: dict[str, pd.DataFrame], train_runs: set[str], max_channels: int) -> list[str]:
    stats = []
    all_cols = sorted({col for frame in metric_frames.values() for col in frame.columns if col != "timestamp"})
    for col in all_cols:
        values = []
        present_runs = 0
        for run_id, frame in metric_frames.items():
            if run_id not in train_runs or col not in frame.columns:
                continue
            arr = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float32)
            finite = arr[np.isfinite(arr)]
            if len(finite):
                present_runs += 1
                values.append(finite)
        if not values:
            continue
        flat = np.concatenate(values)
        variance = float(np.nanvar(flat))
        coverage = float(present_runs / max(len(train_runs), 1))
        stats.append((col, coverage, variance))
    stats = sorted(stats, key=lambda item: (item[1], item[2]), reverse=True)
    return [col for col, _, _ in stats[: int(max_channels)]]


def segment_from_run_frame(frame: pd.DataFrame, window_end: pd.Timestamp, cols: list[str], segment_sec: int, segment_len: int) -> tuple[np.ndarray, np.ndarray]:
    segment = np.full((int(segment_len), len(cols)), np.nan, dtype=np.float32)
    mask = np.zeros(int(segment_len), dtype=np.float32)
    if frame.empty or pd.isna(window_end):
        return segment, mask
    start = window_end - pd.Timedelta(seconds=int(segment_sec))
    part = frame.loc[(frame["timestamp"] >= start) & (frame["timestamp"] < window_end)].copy()
    if part.empty:
        return segment, mask
    available = [col for col in cols if col in part.columns]
    if not available:
        return segment, mask
    values = part[available].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if len(values) > int(segment_len):
        take_idx = np.linspace(0, len(values) - 1, int(segment_len)).round().astype(int)
        values = values[take_idx]
    offset = int(segment_len) - len(values)
    col_positions = [cols.index(col) for col in available]
    segment[offset:, col_positions] = values
    mask[offset:] = 1.0
    return segment, mask


def build_metric_segments(
    frame: pd.DataFrame,
    raw_root: Path,
    train_idx: np.ndarray,
    nodes: list[str],
    instruments: list[str],
    max_channels: int,
    segment_sec: int,
    segment_len: int,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    metric_frames: dict[str, pd.DataFrame] = {}
    for run_id, group in frame.groupby("run_id", sort=False):
        workload = str(group["workload"].iloc[0])
        scenario = str(group["scenario"].iloc[0])
        run_root = raw_root / workload / "kpis" / scenario
        metric_frames[str(run_id)] = load_run_metrics(run_root, nodes=nodes, instruments=instruments, cache_dir=cache_dir)

    train_runs = set(frame.iloc[train_idx]["run_id"].astype(str))
    selected_cols = select_metric_channels(metric_frames, train_runs, max_channels=max_channels)
    if not selected_cols:
        raise RuntimeError("No raw KPI metric channels were loaded from raw_data.")

    segments = np.full((len(frame), int(segment_len), len(selected_cols)), np.nan, dtype=np.float32)
    masks = np.zeros((len(frame), int(segment_len)), dtype=np.float32)
    for row in frame.itertuples(index=False):
        run_frame = metric_frames.get(str(row.run_id), pd.DataFrame())
        segment, mask = segment_from_run_frame(run_frame, row.window_end, selected_cols, segment_sec, segment_len)
        segments[int(row.row_pos)] = segment
        masks[int(row.row_pos)] = mask

    train_values = segments[train_idx].reshape(-1, len(selected_cols))
    median = np.nanmedian(train_values, axis=0)
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
    filled = np.where(np.isfinite(segments), segments, median[None, None, :]).astype(np.float32)
    mean = filled[train_idx].reshape(-1, len(selected_cols)).mean(axis=0).astype(np.float32)
    std = filled[train_idx].reshape(-1, len(selected_cols)).std(axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    scaled = ((filled - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    scaled[masks <= 0.5] = 0.0
    manifest = {
        "source": "raw_data_kpi_files",
        "nodes": nodes,
        "instruments": instruments,
        "segment_sec": int(segment_sec),
        "segment_len": int(segment_len),
        "channels": selected_cols,
        "channel_count": int(len(selected_cols)),
        "runs_loaded": int(sum(not frame.empty for frame in metric_frames.values())),
        "mean_valid_steps": float(masks.sum(axis=1).mean()) if len(masks) else 0.0,
    }
    return scaled, masks, manifest


class HadesDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: Iterable[int], pseudo: dict[str, np.ndarray] | None = None):
        self.arrays = arrays
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.pseudo = pseudo

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        target = float(self.arrays["target"][idx])
        weight = 1.0
        if self.pseudo is not None and idx in self.pseudo["target_by_idx"]:
            target = float(self.pseudo["target_by_idx"][idx])
            weight = float(self.pseudo["weight_by_idx"][idx])
        return {
            "log_emb": torch.from_numpy(self.arrays["log_emb"][idx]).float(),
            "log_mask": torch.from_numpy(self.arrays["log_mask"][idx]).float(),
            "metric_seg": torch.from_numpy(self.arrays["metric_seg"][idx]).float(),
            "metric_mask": torch.from_numpy(self.arrays["metric_mask"][idx]).float(),
            "target": torch.tensor(target, dtype=torch.float32),
            "weight": torch.tensor(weight, dtype=torch.float32),
            "row_index": torch.tensor(idx, dtype=torch.long),
        }


class LogTransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, heads: int, layers: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(int(input_dim), int(d_model)), nn.LayerNorm(int(d_model)))
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(heads),
            dim_feedforward=int(d_model) * 3,
            dropout=0.10,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(layers))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(x)
        pad = mask <= 0.5
        no_token = pad.all(dim=1, keepdim=True)
        first_col = torch.zeros_like(pad, dtype=torch.bool)
        first_col[:, :1] = True
        # Keep at least one unmasked token in all-empty rows without tensor->python branching.
        pad = pad & ~(no_token & first_col)
        tokens = self.encoder(tokens, src_key_padding_mask=pad)
        return tokens * mask.unsqueeze(-1)


class CausalConvMetricEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, layers: int):
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(d_model))
        self.blocks = nn.ModuleList()
        for _ in range(int(layers)):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(int(d_model), int(d_model), kernel_size=3),
                    nn.GELU(),
                    nn.Conv1d(int(d_model), int(d_model), kernel_size=1),
                    nn.GELU(),
                )
            )
        self.norm = nn.LayerNorm(int(d_model))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = self.input_proj(x)
        y = tokens.transpose(1, 2)
        for block in self.blocks:
            residual = y
            y = F.pad(y, (2, 0))
            y = block(y)
            y = y + residual
        tokens = self.norm(y.transpose(1, 2))
        return tokens * mask.unsqueeze(-1)


class TinyTimeMixerBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.temporal = nn.Conv1d(int(d_model), int(d_model), kernel_size=5, padding=2, groups=int(d_model))
        self.channel = nn.Sequential(
            nn.Linear(int(d_model), int(d_model) * 2),
            nn.GELU(),
            nn.Linear(int(d_model) * 2, int(d_model)),
        )
        self.norm1 = nn.LayerNorm(int(d_model))
        self.norm2 = nn.LayerNorm(int(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.temporal(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = x + y
        x = x + self.channel(self.norm2(x))
        return x


class TinyTimeMixerMetricEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, layers: int):
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(d_model))
        self.blocks = nn.ModuleList([TinyTimeMixerBlock(d_model) for _ in range(int(layers))])
        self.norm = nn.LayerNorm(int(d_model))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = self.input_proj(x)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens) * mask.unsqueeze(-1)


class CrossModalHades(nn.Module):
    def __init__(
        self,
        log_input_dim: int,
        metric_input_dim: int,
        d_model: int,
        heads: int,
        log_layers: int,
        metric_layers: int,
        metric_encoder: str,
    ):
        super().__init__()
        self.log_encoder = LogTransformerEncoder(log_input_dim, d_model, heads, log_layers)
        if metric_encoder == "tiny_time_mixer":
            self.metric_encoder = TinyTimeMixerMetricEncoder(metric_input_dim, d_model, metric_layers)
        elif metric_encoder == "causal_conv":
            self.metric_encoder = CausalConvMetricEncoder(metric_input_dim, d_model, metric_layers)
        else:
            raise ValueError(f"Unsupported metric encoder: {metric_encoder}")
        self.log_to_metric = nn.MultiheadAttention(d_model, heads, dropout=0.05, batch_first=True)
        self.metric_to_log = nn.MultiheadAttention(d_model, heads, dropout=0.05, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    @staticmethod
    def safe_pad(mask: torch.Tensor) -> torch.Tensor:
        pad = mask <= 0.5
        no_token = pad.all(dim=1, keepdim=True)
        first_col = torch.zeros_like(pad, dtype=torch.bool)
        first_col[:, :1] = True
        pad = pad & ~(no_token & first_col)
        return pad

    @staticmethod
    def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.float().unsqueeze(-1)
        return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def forward(self, log_emb: torch.Tensor, log_mask: torch.Tensor, metric_seg: torch.Tensor, metric_mask: torch.Tensor) -> torch.Tensor:
        log_tokens = self.log_encoder(log_emb, log_mask)
        metric_tokens = self.metric_encoder(metric_seg, metric_mask)
        log_pad = self.safe_pad(log_mask)
        metric_pad = self.safe_pad(metric_mask)
        log_cross, _ = self.log_to_metric(
            log_tokens,
            metric_tokens,
            metric_tokens,
            key_padding_mask=metric_pad,
            need_weights=False,
        )
        metric_cross, _ = self.metric_to_log(
            metric_tokens,
            log_tokens,
            log_tokens,
            key_padding_mask=log_pad,
            need_weights=False,
        )
        log_repr = self.masked_mean(log_tokens + log_cross, log_mask)
        metric_repr = self.masked_mean(metric_tokens + metric_cross, metric_mask)
        fused = torch.cat([log_repr, metric_repr, torch.abs(log_repr - metric_repr), log_repr * metric_repr], dim=1)
        return self.classifier(fused).squeeze(1)


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def weighted_bce_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    return (raw * weight).sum() / weight.sum().clamp_min(1.0)


def pos_weight_value(y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    return float(neg / pos)


def predict_scores(model: CrossModalHades, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            logits = model(batch["log_emb"], batch["log_mask"], batch["metric_seg"], batch["metric_mask"])
            scores = torch.sigmoid(logits).detach().cpu().numpy()
            for row_idx, target, score in zip(
                batch["row_index"].detach().cpu().numpy(),
                batch["target"].detach().cpu().numpy(),
                scores,
            ):
                rows.append({"row_index": int(row_idx), "target": int(target), "score": float(score)})
    return pd.DataFrame(rows)


def make_pseudo_labels(model: CrossModalHades, arrays: dict[str, np.ndarray], train_idx: np.ndarray, cfg: EfficientHadesConfig, device: torch.device) -> dict:
    loader = DataLoader(HadesDataset(arrays, train_idx), batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)
    pred = predict_scores(model, loader, device)
    low = float(pred["score"].quantile(float(cfg.pseudo_low_quantile)))
    high = float(pred["score"].quantile(float(cfg.pseudo_high_quantile)))
    selected = pred.loc[(pred["score"] <= low) | (pred["score"] >= high)].copy()
    selected["pseudo_target"] = (selected["score"] >= high).astype(float)
    target_by_idx = {int(row.row_index): float(row.pseudo_target) for row in selected.itertuples(index=False)}
    weight_by_idx = {int(row.row_index): float(cfg.pseudo_weight) for row in selected.itertuples(index=False)}
    return {
        "target_by_idx": target_by_idx,
        "weight_by_idx": weight_by_idx,
        "count": int(len(selected)),
        "low_threshold": low,
        "high_threshold": high,
    }


def train_model(
    arrays: dict[str, np.ndarray],
    idx: dict[str, np.ndarray],
    model: CrossModalHades,
    cfg: EfficientHadesConfig,
    device: torch.device,
) -> tuple[CrossModalHades, dict]:
    train_y = arrays["target"][idx["train"]]
    pos_weight = torch.tensor(pos_weight_value(train_y), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    best_state = None
    best_monitor = -1e18
    best_epoch = 0
    best_thresholds = {"best_f1": 0.5, "fp_budget": 0.5}
    best_val_snapshot = {
        "val_ap": None,
        "val_best_f1": None,
        "val_fp_budget_f1": None,
        "val_best_f1_precision": None,
        "val_best_f1_recall": None,
        "val_fp_budget_precision": None,
        "val_fp_budget_recall": None,
    }
    stale = 0
    pseudo = None
    pseudo_manifest = {"enabled": bool(cfg.pseudo_labeling), "count": 0}
    history = []

    for epoch in range(1, int(cfg.epochs) + 1):
        if cfg.pseudo_labeling and pseudo is None and epoch >= int(cfg.pseudo_start_epoch):
            pseudo = make_pseudo_labels(model, arrays, idx["train"], cfg, device)
            pseudo_manifest = {key: value for key, value in pseudo.items() if key not in {"target_by_idx", "weight_by_idx"}}
            pseudo_manifest["enabled"] = True
        train_loader = DataLoader(
            HadesDataset(arrays, idx["train"], pseudo=pseudo),
            batch_size=int(cfg.batch_size),
            shuffle=True,
            num_workers=0,
        )
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["log_emb"], batch["log_mask"], batch["metric_seg"], batch["metric_mask"])
            loss = weighted_bce_loss(logits, batch["target"], batch["weight"], pos_weight)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_loader = DataLoader(HadesDataset(arrays, idx["validation"]), batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)
        val_pred = predict_scores(model, val_loader, device)
        val_ap = safe_ap(val_pred["target"], val_pred["score"]) or 0.0
        val_thr_best_f1 = best_f1_threshold(val_pred["target"], val_pred["score"])
        val_metrics_best_f1 = evaluate_binary(val_pred["target"], val_pred["score"], val_thr_best_f1)
        val_thr_fp_budget = fp_budget_threshold(val_pred["target"], val_pred["score"], cfg.fp_budget)
        val_metrics_fp_budget = evaluate_binary(val_pred["target"], val_pred["score"], val_thr_fp_budget)
        monitor_value = float(val_ap)
        if cfg.train_monitor == "best_f1":
            monitor_value = float(val_metrics_best_f1["f1"])
        elif cfg.train_monitor == "fp_budget_f1":
            monitor_value = float(val_metrics_fp_budget["f1"])
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)) if losses else None,
                "val_ap": float(val_ap),
                "val_thr_best_f1": float(val_thr_best_f1),
                "val_f1_best_f1_policy": float(val_metrics_best_f1["f1"]),
                "val_precision_best_f1_policy": float(val_metrics_best_f1["precision"]),
                "val_recall_best_f1_policy": float(val_metrics_best_f1["recall"]),
                "val_thr_fp_budget": float(val_thr_fp_budget),
                "val_f1_fp_budget_policy": float(val_metrics_fp_budget["f1"]),
                "val_precision_fp_budget_policy": float(val_metrics_fp_budget["precision"]),
                "val_recall_fp_budget_policy": float(val_metrics_fp_budget["recall"]),
                "train_monitor": cfg.train_monitor,
                "train_monitor_value": float(monitor_value),
            }
        )
        if monitor_value > best_monitor:
            best_monitor = float(monitor_value)
            best_epoch = int(epoch)
            best_thresholds = {"best_f1": float(val_thr_best_f1), "fp_budget": float(val_thr_fp_budget)}
            best_val_snapshot = {
                "val_ap": float(val_ap),
                "val_best_f1": float(val_metrics_best_f1["f1"]),
                "val_fp_budget_f1": float(val_metrics_fp_budget["f1"]),
                "val_best_f1_precision": float(val_metrics_best_f1["precision"]),
                "val_best_f1_recall": float(val_metrics_best_f1["recall"]),
                "val_fp_budget_precision": float(val_metrics_fp_budget["precision"]),
                "val_fp_budget_recall": float(val_metrics_fp_budget["recall"]),
            }
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_epoch": int(best_epoch),
        "best_monitor": float(best_monitor),
        "best_thresholds": best_thresholds,
        "best_val_snapshot": best_val_snapshot,
        "history": history,
        "pseudo": pseudo_manifest,
        "train_monitor": cfg.train_monitor,
    }


def benchmark_torch_latency(model: CrossModalHades, arrays: dict[str, np.ndarray], indices: np.ndarray, cfg: EfficientHadesConfig, device: torch.device) -> dict:
    if len(indices) == 0:
        return {}
    sample = indices[: min(len(indices), int(cfg.batch_size))]
    batch = {
        "log_emb": torch.from_numpy(arrays["log_emb"][sample]).float().to(device),
        "log_mask": torch.from_numpy(arrays["log_mask"][sample]).float().to(device),
        "metric_seg": torch.from_numpy(arrays["metric_seg"][sample]).float().to(device),
        "metric_mask": torch.from_numpy(arrays["metric_mask"][sample]).float().to(device),
    }
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            _ = model(**batch)
        start = time.perf_counter()
        repeats = max(1, int(cfg.benchmark_batches))
        for _ in range(repeats):
            _ = model(**batch)
        elapsed = time.perf_counter() - start
    return {
        "backend": "torch_fp32",
        "batch_size": int(len(sample)),
        "batches": int(repeats),
        "latency_ms_per_batch": float(elapsed * 1000.0 / repeats),
        "latency_ms_per_sample": float(elapsed * 1000.0 / (repeats * len(sample))),
    }


def export_onnx(model: CrossModalHades, arrays: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.cpu().eval()
    log_emb = torch.from_numpy(arrays["log_emb"][:1]).float()
    log_mask = torch.from_numpy(arrays["log_mask"][:1]).float()
    metric_seg = torch.from_numpy(arrays["metric_seg"][:1]).float()
    metric_mask = torch.from_numpy(arrays["metric_mask"][:1]).float()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Constant folding - Only steps=1 can be constant folded.*",
            category=UserWarning,
        )
        torch.onnx.export(
            model,
            (log_emb, log_mask, metric_seg, metric_mask),
            str(path),
            input_names=["log_emb", "log_mask", "metric_seg", "metric_mask"],
            output_names=["logit"],
            dynamic_axes={
                "log_emb": {0: "batch"},
                "log_mask": {0: "batch"},
                "metric_seg": {0: "batch"},
                "metric_mask": {0: "batch"},
                "logit": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=False,
        )


class CalibrationReader:
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray, batch_size: int = 32):
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.pos = 0

    def get_next(self):
        if self.pos >= len(self.indices):
            return None
        batch_idx = self.indices[self.pos : self.pos + self.batch_size]
        self.pos += self.batch_size
        return {
            "log_emb": self.arrays["log_emb"][batch_idx].astype(np.float32),
            "log_mask": self.arrays["log_mask"][batch_idx].astype(np.float32),
            "metric_seg": self.arrays["metric_seg"][batch_idx].astype(np.float32),
            "metric_mask": self.arrays["metric_mask"][batch_idx].astype(np.float32),
        }


def quantize_onnx(onnx_path: Path, int8_path: Path, arrays: dict[str, np.ndarray], train_idx: np.ndarray, cfg: EfficientHadesConfig) -> dict:
    if cfg.quantization == "none":
        return {"status": "skipped", "mode": "none"}
    quant_logger = logging.getLogger()
    prev_level = quant_logger.level
    quant_logger.setLevel(logging.ERROR)
    try:
        if cfg.quantization == "dynamic":
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantize_dynamic(
                str(onnx_path),
                str(int8_path),
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["MatMul", "Gemm", "Attention"],
            )
        elif cfg.quantization == "static":
            from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

            cal_idx = train_idx[: min(len(train_idx), int(cfg.calibration_size))]
            reader = CalibrationReader(arrays, cal_idx, batch_size=32)
            quantize_static(
                str(onnx_path),
                str(int8_path),
                reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["MatMul", "Gemm", "Attention", "Conv"],
            )
        else:
            raise ValueError(f"Unsupported quantization mode: {cfg.quantization}")
    except Exception as exc:
        quant_logger.setLevel(prev_level)
        return {"status": "failed", "mode": cfg.quantization, "error": f"{type(exc).__name__}: {exc}"}
    quant_logger.setLevel(prev_level)
    return {"status": "ok", "mode": cfg.quantization, "path": str(int8_path), "bytes": int(int8_path.stat().st_size)}


def benchmark_onnx(onnx_path: Path, arrays: dict[str, np.ndarray], indices: np.ndarray, cfg: EfficientHadesConfig) -> dict:
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        sample = indices[: min(len(indices), int(cfg.batch_size))]
        inputs = {
            "log_emb": arrays["log_emb"][sample].astype(np.float32),
            "log_mask": arrays["log_mask"][sample].astype(np.float32),
            "metric_seg": arrays["metric_seg"][sample].astype(np.float32),
            "metric_mask": arrays["metric_mask"][sample].astype(np.float32),
        }
        for _ in range(5):
            session.run(None, inputs)
        repeats = max(1, int(cfg.benchmark_batches))
        start = time.perf_counter()
        for _ in range(repeats):
            session.run(None, inputs)
        elapsed = time.perf_counter() - start
        return {
            "backend": "onnx",
            "path": str(onnx_path),
            "batch_size": int(len(sample)),
            "batches": int(repeats),
            "latency_ms_per_batch": float(elapsed * 1000.0 / repeats),
            "latency_ms_per_sample": float(elapsed * 1000.0 / (repeats * len(sample))),
        }
    except Exception as exc:
        return {"backend": "onnx", "path": str(onnx_path), "status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def onnx_predict(onnx_path: Path, arrays: dict[str, np.ndarray], indices: np.ndarray, batch_size: int) -> np.ndarray | None:
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        scores = []
        for start in range(0, len(indices), int(batch_size)):
            batch_idx = indices[start : start + int(batch_size)]
            inputs = {
                "log_emb": arrays["log_emb"][batch_idx].astype(np.float32),
                "log_mask": arrays["log_mask"][batch_idx].astype(np.float32),
                "metric_seg": arrays["metric_seg"][batch_idx].astype(np.float32),
                "metric_mask": arrays["metric_mask"][batch_idx].astype(np.float32),
            }
            logits = session.run(None, inputs)[0].reshape(-1)
            scores.extend((1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))).tolist())
        return np.asarray(scores, dtype=np.float32)
    except Exception:
        return None


def model_file_size(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    return int(path.stat().st_size)


def train_variant_seed(
    arrays_by_embedding: dict[str, dict[str, np.ndarray]],
    metric_arrays: dict[str, np.ndarray],
    idx: dict[str, np.ndarray],
    cfg: EfficientHadesConfig,
    variant: str,
    seed: int,
    output_dir: Path,
) -> dict:
    set_seed(seed)
    spec = VARIANT_SPECS[variant]
    embedding_key = cfg.modern_log_backend if spec["log_embedding"] == "modern" else "fasttext_hash"
    arrays = {
        **arrays_by_embedding[embedding_key],
        **metric_arrays,
    }
    arrays["target"] = metric_arrays["target"]
    device = torch.device(cfg.device)
    model = CrossModalHades(
        log_input_dim=arrays["log_emb"].shape[-1],
        metric_input_dim=arrays["metric_seg"].shape[-1],
        d_model=cfg.d_model,
        heads=cfg.heads,
        log_layers=cfg.log_layers,
        metric_layers=cfg.metric_layers,
        metric_encoder=spec["metric_encoder"],
    ).to(device)
    model, train_info = train_model(arrays, idx, model, cfg, device)

    val_loader = DataLoader(HadesDataset(arrays, idx["validation"]), batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)
    test_loader = DataLoader(HadesDataset(arrays, idx["test"]), batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)
    val_pred = predict_scores(model, val_loader, device)
    test_pred = predict_scores(model, test_loader, device)
    train_thresholds = train_info.get("best_thresholds", {}) if isinstance(train_info, dict) else {}
    f1_thr = float(train_thresholds.get("best_f1", best_f1_threshold(val_pred["target"], val_pred["score"])))
    fp_thr = float(train_thresholds.get("fp_budget", fp_budget_threshold(val_pred["target"], val_pred["score"], cfg.fp_budget)))

    state_path = output_dir / "models" / f"efficient_hades_{variant}_seed{seed}.pt"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "variant": variant,
            "seed": int(seed),
            "spec": spec,
            "thresholds": {"best_f1": f1_thr, "fp_budget": fp_thr},
            "config": asdict(cfg),
        },
        state_path,
    )
    torch_bench = benchmark_torch_latency(model, arrays, idx["test"], cfg, device)

    export_report = {"status": "skipped", "reason": "export disabled"}
    runtime_metrics = []
    if cfg.export_onnx and spec["runtime"] == "onnx_int8":
        export_dir = output_dir / "exports" / f"{variant}_seed{seed}"
        onnx_path = export_dir / "model_fp32.onnx"
        int8_path = export_dir / "model_int8.onnx"
        try:
            export_onnx(model, arrays, onnx_path)
            q_report = quantize_onnx(onnx_path, int8_path, arrays, idx["train"], cfg)
            onnx_fp32_bench = benchmark_onnx(onnx_path, arrays, idx["test"], cfg)
            export_report = {
                "status": "ok",
                "onnx_fp32": str(onnx_path),
                "onnx_fp32_bytes": model_file_size(onnx_path),
                "quantization": q_report,
                "benchmarks": [onnx_fp32_bench],
            }
            if int8_path.exists():
                onnx_int8_bench = benchmark_onnx(int8_path, arrays, idx["test"], cfg)
                export_report["benchmarks"].append(onnx_int8_bench)
                int8_scores = onnx_predict(int8_path, arrays, idx["test"], int(cfg.batch_size))
                if int8_scores is not None:
                    y_test = arrays["target"][idx["test"]]
                    runtime_metrics.append(
                        {
                            "variant": variant,
                            "seed": int(seed),
                            "runtime": "onnx_int8",
                            "threshold_policy": "validation_best_f1",
                            **evaluate_binary(y_test, int8_scores, f1_thr),
                            "torch_latency_ms_per_sample": onnx_int8_bench.get("latency_ms_per_sample"),
                            "onnx_latency_ms_per_sample": onnx_int8_bench.get("latency_ms_per_sample"),
                            "onnx_model_bytes": model_file_size(int8_path),
                            "state_path": str(state_path),
                            "train_monitor": train_info.get("train_monitor"),
                            "train_monitor_value": train_info.get("best_monitor"),
                            "train_selected_epoch": train_info.get("best_epoch"),
                        }
                    )
                    runtime_metrics.append(
                        {
                            "variant": variant,
                            "seed": int(seed),
                            "runtime": "onnx_int8",
                            "threshold_policy": f"validation_fp_budget_{cfg.fp_budget}",
                            **evaluate_binary(y_test, int8_scores, fp_thr),
                            "torch_latency_ms_per_sample": onnx_int8_bench.get("latency_ms_per_sample"),
                            "onnx_latency_ms_per_sample": onnx_int8_bench.get("latency_ms_per_sample"),
                            "onnx_model_bytes": model_file_size(int8_path),
                            "state_path": str(state_path),
                            "train_monitor": train_info.get("train_monitor"),
                            "train_monitor_value": train_info.get("best_monitor"),
                            "train_selected_epoch": train_info.get("best_epoch"),
                        }
                    )
        except Exception as exc:
            export_report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    metric_rows = []
    for split, pred in [("validation", val_pred), ("test", test_pred)]:
        for policy, threshold in [("validation_best_f1", f1_thr), (f"validation_fp_budget_{cfg.fp_budget}", fp_thr)]:
            metric_rows.append(
                {
                    "variant": variant,
                    "seed": int(seed),
                    "runtime": "torch_fp32",
                    "split": split,
                    "target": cfg.target,
                    "threshold_policy": policy,
                    **evaluate_binary(pred["target"], pred["score"], threshold),
                    "state_path": str(state_path),
                    "model_bytes": model_file_size(state_path),
                    "torch_latency_ms_per_sample": torch_bench.get("latency_ms_per_sample"),
                    "train_monitor": train_info.get("train_monitor"),
                    "train_monitor_value": train_info.get("best_monitor"),
                    "train_selected_epoch": train_info.get("best_epoch"),
                }
            )
    return {
        "metrics": pd.DataFrame(metric_rows),
        "runtime_metrics": pd.DataFrame(runtime_metrics),
        "val_pred": val_pred.assign(variant=variant, seed=int(seed), split="validation"),
        "test_pred": test_pred.assign(variant=variant, seed=int(seed), split="test"),
        "export_report": {"variant": variant, "seed": int(seed), **export_report},
        "train_info": {"variant": variant, "seed": int(seed), **train_info},
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
        target,
    ]
    cols = [col for col in cols if col in frame.columns]
    meta = frame[cols].rename(columns={"row_pos": "row_index", target: "label"})
    return pred.merge(meta, on="row_index", how="left")


def markdown_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "_empty_"
    view = df.head(max_rows).copy()
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


def write_experiment_card(output_dir: Path, cfg: EfficientHadesConfig, metrics: pd.DataFrame) -> None:
    test = metrics.loc[metrics["split"].astype(str).eq("test")].copy() if not metrics.empty else pd.DataFrame()
    cols = [
        "variant",
        "runtime",
        "seed",
        "threshold_policy",
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "false_positives",
        "missed_anomalies",
        "torch_latency_ms_per_sample",
    ]
    cols = [col for col in cols if col in test.columns]
    lines = [
        f"# Efficient HADES Hypothesis: `{cfg.run_id}`",
        "",
        "This experiment preserves the HADES-like architecture: log sequence modeling, metric segment modeling, bidirectional cross-modal attention, pseudo-labeling, and binary normal/abnormal detection.",
        "",
        "It uses raw KPI files from `raw_data`, not only pre-aggregated KPI columns from the contract.",
        "",
        "## Test Metrics",
        "",
        markdown_table(test[cols]) if not test.empty else "_No test metrics produced._",
        "",
        "## Variants",
        "",
    ]
    for name in cfg.variants:
        lines.append(f"- `{name}`: {VARIANT_SPECS[name]['description']}")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `comparison_metrics.csv`: FP32 metrics by variant, seed and threshold policy.",
            "- `runtime_metrics.csv`: ONNX INT8 metrics when export and runtime are available.",
            "- `latency_and_export_report.json`: ONNX/INT8 export status, size and latency details.",
            "- `feature_manifest.json`: log embedding and raw metric segment manifest.",
            "- `train_manifest.json`: pseudo-labeling and validation monitor details.",
        ]
    )
    (output_dir / "EXPERIMENT_CARD.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"efficient_hades_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--root", default=".")
    parser.add_argument("--contract-path", default="dashboard/artifacts/contract/strict_intensity_contract.csv.gz")
    parser.add_argument("--output-dir", default="hypothesis_results/efficient_hades")
    parser.add_argument("--target", choices=["y_anomaly", "y_warning", "y_risk"], default="y_anomaly")
    parser.add_argument("--variants", default="original,modern_log,modern_metric,modern_runtime,efficient_hades")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-log-events", type=int, default=32)
    parser.add_argument("--log-context-sec", type=int, default=120)
    parser.add_argument("--log-embedding-dim", type=int, default=128)
    parser.add_argument("--modern-log-backend", choices=["static_hash", "embeddinggemma"], default="static_hash")
    parser.add_argument("--embeddinggemma-model", default="google/embeddinggemma-300m")
    parser.add_argument("--metric-segment-sec", type=int, default=120)
    parser.add_argument("--metric-segment-len", type=int, default=64)
    parser.add_argument("--max-metric-channels", type=int, default=64)
    parser.add_argument("--metric-nodes", default=",".join(DEFAULT_NODES))
    parser.add_argument("--metric-instruments", default=",".join(DEFAULT_INSTRUMENTS))
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--log-layers", type=int, default=1)
    parser.add_argument("--metric-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--disable-pseudo-labeling", action="store_true")
    parser.add_argument("--pseudo-start-epoch", type=int, default=6)
    parser.add_argument("--pseudo-low-quantile", type=float, default=0.10)
    parser.add_argument("--pseudo-high-quantile", type=float, default=0.90)
    parser.add_argument("--pseudo-weight", type=float, default=0.35)
    parser.add_argument("--fp-budget", type=float, default=0.10)
    parser.add_argument("--train-monitor", choices=["ap", "best_f1", "fp_budget_f1"], default="fp_budget_f1")
    parser.add_argument("--validation-strategy", choices=["auto", "existing", "run_id_holdout", "workload_holdout"], default="auto")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--quantization", choices=["dynamic", "static", "none"], default="dynamic")
    parser.add_argument("--calibration-size", type=int, default=256)
    parser.add_argument("--benchmark-batches", type=int, default=20)
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

    cfg = EfficientHadesConfig(
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
        log_embedding_dim=args.log_embedding_dim,
        modern_log_backend=args.modern_log_backend,
        embeddinggemma_model=args.embeddinggemma_model,
        metric_segment_sec=args.metric_segment_sec,
        metric_segment_len=args.metric_segment_len,
        max_metric_channels=args.max_metric_channels,
        metric_nodes=parse_csv_list(args.metric_nodes, str),
        metric_instruments=parse_csv_list(args.metric_instruments, str),
        d_model=args.d_model,
        heads=args.heads,
        log_layers=args.log_layers,
        metric_layers=args.metric_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pseudo_labeling=not bool(args.disable_pseudo_labeling),
        pseudo_start_epoch=args.pseudo_start_epoch,
        pseudo_low_quantile=args.pseudo_low_quantile,
        pseudo_high_quantile=args.pseudo_high_quantile,
        pseudo_weight=args.pseudo_weight,
        fp_budget=args.fp_budget,
        train_monitor=args.train_monitor,
        validation_strategy=args.validation_strategy,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
        device=args.device,
        export_onnx=not bool(args.skip_onnx),
        quantization=args.quantization,
        calibration_size=args.calibration_size,
        benchmark_batches=args.benchmark_batches,
    )
    unknown = [variant for variant in cfg.variants if variant not in VARIANT_SPECS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Known: {sorted(VARIANT_SPECS)}")
    write_json(output_dir / "run_config.json", asdict(cfg))

    print(f"[1/6] Loading strict contract: {contract_path}")
    frame = load_contract(contract_path, cfg.max_rows, cfg.target)
    frame = apply_validation_split(frame, cfg.validation_strategy, cfg.validation_fraction, cfg.validation_seed)
    idx = split_indices(frame)
    split_summary(frame).to_csv(output_dir / "split_summary.csv", index=False)
    print(split_summary(frame).to_string(index=False))

    print("[2/6] Building raw metric time-series segments")
    metric_seg, metric_mask, metric_manifest = build_metric_segments(
        frame,
        raw_root=root / "raw_data",
        train_idx=idx["train"],
        nodes=cfg.metric_nodes,
        instruments=cfg.metric_instruments,
        max_channels=cfg.max_metric_channels,
        segment_sec=cfg.metric_segment_sec,
        segment_len=cfg.metric_segment_len,
        cache_dir=output_root / "_cache" / "raw_metrics",
    )
    metric_arrays = {
        "metric_seg": metric_seg.astype(np.float32),
        "metric_mask": metric_mask.astype(np.float32),
        "target": frame[cfg.target].astype(np.float32).to_numpy(),
    }

    print("[3/6] Building cached log template embeddings")
    needed_embeddings = {"fasttext_hash"}
    if any(VARIANT_SPECS[variant]["log_embedding"] == "modern" for variant in cfg.variants):
        needed_embeddings.add(cfg.modern_log_backend)
    arrays_by_embedding: dict[str, dict[str, np.ndarray]] = {}
    log_manifests = {}
    for backend in sorted(needed_embeddings):
        log_emb, log_mask, manifest = build_log_arrays(
            frame,
            raw_root=root / "raw_data",
            train_idx=idx["train"],
            max_events=cfg.max_log_events,
            context_sec=cfg.log_context_sec,
            backend=backend,
            dim=cfg.log_embedding_dim,
            output_dir=output_dir,
            embeddinggemma_model=cfg.embeddinggemma_model,
            cache_dir=output_root / "_cache" / "raw_logs",
        )
        arrays_by_embedding[backend] = {"log_emb": log_emb.astype(np.float32), "log_mask": log_mask.astype(np.float32)}
        log_manifests[backend] = manifest

    feature_manifest = {
        "log": log_manifests,
        "metric": metric_manifest,
        "arrays": {
            "metric_seg": list(metric_seg.shape),
            "metric_mask": list(metric_mask.shape),
            "target": list(metric_arrays["target"].shape),
            "log_embeddings": {key: list(value["log_emb"].shape) for key, value in arrays_by_embedding.items()},
        },
        "architecture": "logs->templates->cached_embeddings->Transformer; metrics->raw_segments->TS_encoder; cross_modal_attention->classifier",
    }
    write_json(output_dir / "feature_manifest.json", feature_manifest)

    print("[4/6] Training Efficient HADES ablations")
    results = []
    for variant in cfg.variants:
        for seed in cfg.seeds:
            print(f"  - variant={variant} seed={seed}")
            results.append(train_variant_seed(arrays_by_embedding, metric_arrays, idx, cfg, variant, seed, output_dir))

    print("[5/6] Writing metrics and predictions")
    metrics = pd.concat([item["metrics"] for item in results], ignore_index=True)
    runtime_metric_parts = [item["runtime_metrics"] for item in results if not item["runtime_metrics"].empty]
    runtime_metrics = pd.concat(runtime_metric_parts, ignore_index=True) if runtime_metric_parts else pd.DataFrame()
    if not runtime_metrics.empty:
        runtime_metrics["split"] = "test"
        runtime_metrics["target"] = cfg.target
    all_metrics = pd.concat([metrics, runtime_metrics], ignore_index=True) if not runtime_metrics.empty else metrics
    all_metrics.to_csv(output_dir / "comparison_metrics.csv", index=False)
    runtime_metrics.to_csv(output_dir / "runtime_metrics.csv", index=False)
    val_pred = pd.concat([item["val_pred"] for item in results], ignore_index=True)
    test_pred = pd.concat([item["test_pred"] for item in results], ignore_index=True)
    add_contract_columns(val_pred, frame, cfg.target).to_csv(output_dir / "validation_predictions.csv.gz", index=False, compression="gzip")
    add_contract_columns(test_pred, frame, cfg.target).to_csv(output_dir / "test_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame([item["train_info"] for item in results]).to_json(output_dir / "train_manifest.json", orient="records", indent=2)
    write_json(output_dir / "latency_and_export_report.json", {"exports": [item["export_report"] for item in results]})

    print("[6/6] Writing experiment summary")
    write_json(
        output_dir / "metrics_summary.json",
        {
            "run_id": cfg.run_id,
            "split_summary": split_summary(frame).to_dict("records"),
            "feature_manifest": feature_manifest,
            "metrics": all_metrics.to_dict("records"),
            "exports": [item["export_report"] for item in results],
        },
    )
    write_experiment_card(output_dir, cfg, all_metrics)
    print(f"Done. Efficient HADES hypothesis outputs written to: {output_dir}")
    test_view = all_metrics.loc[all_metrics["split"].astype(str).eq("test")]
    print(test_view.to_string(index=False))


if __name__ == "__main__":
    main()
