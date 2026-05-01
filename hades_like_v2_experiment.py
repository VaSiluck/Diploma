"""
Standalone HADES-like v2 experiment.

Purpose:
    Build a sequence-level multimodal experiment without rewriting notebooks 02/03.
    The script uses the notebook 02 aligned window contract only for windows,
    splits and labels, then rebuilds sequence-style inputs:

    - raw log event snippets from raw_data/<workload>/mapred_logs/<scenario>.log;
    - temporal KPI windows from selected KPI columns over previous windows;
    - cross-modal Transformer/attention model with auxiliary anomaly heads.

Typical GPU run:
    python hades_like_v2_experiment.py --run-id hades_v2_gpu --seeds 42,43,44,45,46

Fast smoke run:
    python hades_like_v2_experiment.py --run-id smoke --text-backend hashing --max-rows 900 --epochs 2 --seeds 42
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

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


@dataclass
class RunConfig:
    run_id: str
    root: str
    contract_path: str
    baseline_predictions_path: str | None
    output_dir: str
    text_backend: str
    embedding_model: str
    embedding_dim: int
    top_log_events: int
    log_context_sec: int
    kpi_seq_len: int
    max_kpi_features: int
    hidden_dim: int
    heads: int
    batch_size: int
    epochs: int
    patience: int
    lr: float
    weight_decay: float
    risk_lambda: float
    aux_lambda: float
    anomaly_lambda: float
    consistency_lambda: float
    intensity_lambda: float
    intensity_mode: str
    intensity_risk_horizon_sec: float
    intensity_warning_horizon_sec: float
    risk_recall_target: float
    validation_strategy: str
    validation_fraction: float
    validation_seed: int
    monitor_mode: str
    top_fracs: list[float]
    seeds: list[int]
    max_rows: int | None
    device: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_float(value):
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


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


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def normalize_log_message(message: str) -> str:
    text = str(message).lower()
    text = re.sub(r"application_\d+_\d+", "<application_id>", text)
    text = re.sub(r"container_\w+", "<container_id>", text)
    text = re.sub(r"[0-9a-f]{8}-[0-9a-f-]{27,}", "<uuid>", text)
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<ip>", text)
    text = re.sub(r"hdfs://\S+|file:/\S+|/\S+", "<path>", text)
    text = re.sub(r"\b\d+\b", "<num>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_raw_log_file(path: Path) -> pd.DataFrame:
    rows = []
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "level", "component", "message_norm", "event_score", "line_id"])
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_id, line in enumerate(f, start=1):
            match = RAW_LOG_RE.match(line.strip())
            if not match:
                continue
            ts = pd.to_datetime(
                f"{match.group('date')} {match.group('time')}",
                format="%y/%m/%d %H:%M:%S",
                utc=True,
                errors="coerce",
            )
            if pd.isna(ts):
                continue
            level = match.group("level")
            message = match.group("message")
            message_norm = normalize_log_message(message)
            lower = message_norm.lower()
            keyword_score = sum((2.0 if kw in {"error", "exception", "failed", "failure", "fatal", "critical"} else 1.0) for kw in KEYWORDS if kw in lower)
            rows.append(
                {
                    "timestamp": ts,
                    "level": level,
                    "component": match.group("component"),
                    "message_norm": message_norm,
                    "event_score": float(SEVERITY_WEIGHT.get(level, 0.0) + keyword_score),
                    "line_id": int(line_id),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["timestamp", "level", "component", "message_norm", "event_score", "line_id"])
    return pd.DataFrame(rows).sort_values(["timestamp", "line_id"]).reset_index(drop=True)


def compact_text(text: str, max_chars: int = 420) -> str:
    value = re.sub(r"\s+", " ", str(text)).strip()
    return value[:max_chars] if value else "__NO_LOG__"


def fallback_snippets_from_window_text(text: str, top_k: int) -> tuple[list[str], np.ndarray]:
    value = str(text).strip()
    if not value or value.lower() == "nan":
        return ["__PAD_LOG__"] * top_k, np.zeros(top_k, dtype=np.float32)
    tokens = value.split()
    chunk_size = 72
    step = max(1, int(chunk_size * 0.75))
    chunks = []
    for start in range(0, len(tokens), step):
        chunks.append(compact_text(" ".join(tokens[start : start + chunk_size])))
        if len(chunks) >= top_k:
            break
    mask = np.ones(len(chunks), dtype=np.float32)
    while len(chunks) < top_k:
        chunks.append("__PAD_LOG__")
    if len(mask) < top_k:
        mask = np.concatenate([mask, np.zeros(top_k - len(mask), dtype=np.float32)])
    return chunks[:top_k], mask[:top_k]


def select_events_for_window(
    events: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    top_k: int,
    context_sec: int,
    fallback_text: str,
) -> tuple[list[str], np.ndarray]:
    if events.empty or pd.isna(window_start) or pd.isna(window_end):
        return fallback_snippets_from_window_text(fallback_text, top_k)
    start = window_start - pd.Timedelta(seconds=int(context_sec))
    part = events.loc[(events["timestamp"] >= start) & (events["timestamp"] < window_end)].copy()
    if part.empty:
        return fallback_snippets_from_window_text(fallback_text, top_k)
    n = len(part)
    part["_pos_bonus"] = 0.0
    part.loc[part.index[:2], "_pos_bonus"] += 0.35
    part.loc[part.index[max(0, n - 2) :], "_pos_bonus"] += 0.45
    part["_rank_score"] = part["event_score"].astype(float) + part["_pos_bonus"]
    keep = set(part.head(min(2, n)).index.tolist())
    keep.update(part.tail(min(2, n)).index.tolist())
    for idx in part.sort_values(["_rank_score", "timestamp"], ascending=[False, True]).index:
        keep.add(idx)
        if len(keep) >= top_k:
            break
    selected = part.loc[sorted(keep)].sort_values(["timestamp", "line_id"]).head(top_k)
    snippets = [
        compact_text(f"{row.level} {row.component}: {row.message_norm}")
        for row in selected.itertuples(index=False)
    ]
    mask = np.ones(len(snippets), dtype=np.float32)
    while len(snippets) < top_k:
        snippets.append("__PAD_LOG__")
    if len(mask) < top_k:
        mask = np.concatenate([mask, np.zeros(top_k - len(mask), dtype=np.float32)])
    return snippets, mask


def build_log_event_matrix(frame: pd.DataFrame, raw_root: Path, top_k: int, context_sec: int) -> tuple[list[list[str]], np.ndarray, dict]:
    event_cache: dict[str, pd.DataFrame] = {}
    rows, masks = [], []
    stats = {"runs_parsed": 0, "events_parsed": 0, "windows": int(len(frame))}
    for run_id, group in frame.groupby("run_id", sort=False):
        workload = str(group["workload"].iloc[0])
        scenario = str(group["scenario"].iloc[0])
        log_path = raw_root / workload / "mapred_logs" / f"{scenario}.log"
        events = parse_raw_log_file(log_path)
        event_cache[run_id] = events
        stats["runs_parsed"] += 1
        stats["events_parsed"] += int(len(events))
        for row in group.itertuples():
            snippets, mask = select_events_for_window(
                events,
                row.window_ts,
                row.window_end,
                top_k=top_k,
                context_sec=context_sec,
                fallback_text=getattr(row, "log_text", ""),
            )
            rows.append((int(row.row_pos), snippets, mask))
    rows = sorted(rows, key=lambda x: x[0])
    text_rows = [item[1] for item in rows]
    mask = np.vstack([item[2] for item in rows]).astype(np.float32)
    stats["mean_events_per_window"] = float(mask.sum(axis=1).mean()) if len(mask) else 0.0
    stats["empty_log_event_share"] = float((mask.sum(axis=1) <= 0).mean()) if len(mask) else 0.0
    return text_rows, mask, stats


def stable_hash(payload: dict, n: int = 16) -> str:
    text = json.dumps(json_safe(payload), ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def encode_log_snippets(
    text_rows: list[list[str]],
    mask: np.ndarray,
    fit_idx: np.ndarray,
    output_dir: Path,
    backend: str,
    model_name: str,
    embedding_dim: int,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    flat_valid = []
    for snippets, mask_row in zip(text_rows, mask):
        flat_valid.extend([text for text, valid in zip(snippets, mask_row) if valid > 0.5])
    unique_texts = []
    seen = set()
    for text in flat_valid:
        if text not in seen:
            unique_texts.append(text)
            seen.add(text)

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = stable_hash(
        {
            "backend": backend,
            "model": model_name,
            "embedding_dim": embedding_dim,
            "unique_sha1": hashlib.sha1("\n".join(unique_texts).encode("utf-8")).hexdigest(),
        },
        n=20,
    )
    cache_file = cache_dir / f"log_snippet_embeddings_{backend}_{cache_key}.npz"

    if cache_file.exists():
        payload = np.load(cache_file, allow_pickle=True)
        cache_texts = payload["texts"].astype(str).tolist()
        cache_embeddings = payload["embeddings"].astype(np.float32)
        if cache_texts == unique_texts and cache_embeddings.shape[1] == embedding_dim:
            embedding_map = {text: emb for text, emb in zip(cache_texts, cache_embeddings)}
            cache_loaded = True
        else:
            embedding_map = {}
            cache_loaded = False
    else:
        embedding_map = {}
        cache_loaded = False

    if not embedding_map:
        if backend == "embeddinggemma":
            try:
                from sentence_transformers import SentenceTransformer

                device_name = "cuda" if torch.cuda.is_available() else "cpu"
                encoder = SentenceTransformer(model_name, device=device_name)
                encode_kwargs = {
                    "batch_size": max(8, int(batch_size)),
                    "show_progress_bar": True,
                    "normalize_embeddings": True,
                    "convert_to_numpy": True,
                }
                try:
                    encoded = encoder.encode(unique_texts, truncate_dim=int(embedding_dim), **encode_kwargs).astype(np.float32)
                except TypeError:
                    encoded = encoder.encode(unique_texts, **encode_kwargs).astype(np.float32)
                    encoded = encoded[:, :embedding_dim]
                if encoded.shape[1] > embedding_dim:
                    encoded = encoded[:, :embedding_dim]
            except Exception as exc:
                print(f"[warn] EmbeddingGemma unavailable ({exc}); falling back to hashing backend.")
                backend = "hashing"
        if backend == "tfidf":
            vectorizer = TfidfVectorizer(min_df=2, max_features=30000, ngram_range=(1, 2))
            sparse = vectorizer.fit_transform(unique_texts if unique_texts else ["__NO_LOG__"])
            n_components = min(int(embedding_dim), max(2, min(sparse.shape) - 1))
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            encoded = svd.fit_transform(sparse).astype(np.float32)
            if encoded.shape[1] < embedding_dim:
                encoded = np.pad(encoded, ((0, 0), (0, embedding_dim - encoded.shape[1]))).astype(np.float32)
        elif backend == "hashing":
            vectorizer = HashingVectorizer(n_features=int(embedding_dim), alternate_sign=True, norm="l2", ngram_range=(1, 2))
            encoded = vectorizer.transform(unique_texts if unique_texts else ["__NO_LOG__"]).toarray().astype(np.float32)
        embedding_map = {text: emb for text, emb in zip(unique_texts, encoded)}
        np.savez_compressed(cache_file, texts=np.array(unique_texts, dtype=object), embeddings=encoded.astype(np.float32))

    n, k = len(text_rows), mask.shape[1]
    arr = np.zeros((n, k, int(embedding_dim)), dtype=np.float32)
    for i, (snippets, mask_row) in enumerate(zip(text_rows, mask)):
        for j, (text, valid) in enumerate(zip(snippets, mask_row)):
            if valid > 0.5 and text in embedding_map:
                arr[i, j] = embedding_map[text]

    scaler = StandardScaler()
    fit_idx = np.asarray(fit_idx, dtype=np.int64)
    train_mask = mask[fit_idx].reshape(-1) > 0.5
    train_values = arr[fit_idx].reshape(-1, arr.shape[-1])[train_mask]
    if len(train_values) == 0:
        train_values = np.zeros((1, arr.shape[-1]), dtype=np.float32)
    scaler.fit(train_values)
    arr = scaler.transform(arr.reshape(-1, arr.shape[-1])).reshape(arr.shape).astype(np.float32)
    arr[mask <= 0.5] = 0.0

    manifest = {
        "backend": backend,
        "model_name": model_name,
        "embedding_dim": int(embedding_dim),
        "top_k": int(k),
        "unique_snippets": int(len(unique_texts)),
        "cache_file": str(cache_file),
        "cache_loaded": bool(cache_loaded),
        "shape": list(arr.shape),
    }
    return arr, manifest


def numeric_kpi_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("kpi_") and c not in {"kpi_available", "kpi_avail_ratio", "kpi_feature_count", "kpi_recency_sec", "kpi_series_available"}]


def select_kpi_features(frame: pd.DataFrame, train_idx: np.ndarray, max_features: int) -> list[str]:
    cols = numeric_kpi_columns(frame)
    if not cols:
        raise RuntimeError("No KPI numeric columns found in contract.")
    x = frame[cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    train = x.iloc[train_idx]
    variances = train.var(axis=0, skipna=True).fillna(0.0)
    y_risk = frame["y_risk"].iloc[train_idx].astype(float).to_numpy()
    y_warning = frame["y_warning"].iloc[train_idx].astype(float).to_numpy()
    scores = []
    for col in cols:
        values = train[col].to_numpy(dtype=np.float32)
        mask = np.isfinite(values)
        corr_score = 0.0
        if mask.sum() > 5 and np.nanstd(values[mask]) > 1e-8:
            corr_r = np.corrcoef(values[mask], y_risk[mask])[0, 1]
            corr_w = np.corrcoef(values[mask], y_warning[mask])[0, 1]
            corr_score = max(abs(finite_float(corr_r) or 0.0), abs(finite_float(corr_w) or 0.0))
        scores.append((col, corr_score, float(variances.get(col, 0.0))))
    scores = sorted(scores, key=lambda x: (x[1], x[2]), reverse=True)
    return [col for col, _, _ in scores[: int(max_features)]]


def build_kpi_sequence(frame: pd.DataFrame, train_idx: np.ndarray, selected_cols: list[str], seq_len: int) -> tuple[np.ndarray, np.ndarray, dict]:
    raw = frame[selected_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32).to_numpy(copy=True)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(raw[train_idx])
    x = imputer.transform(raw).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(x[train_idx])
    x = scaler.transform(x).astype(np.float32)
    x[~np.isfinite(x)] = 0.0

    n, f = x.shape
    seq = np.zeros((n, int(seq_len), f), dtype=np.float32)
    mask = np.zeros((n, int(seq_len)), dtype=np.float32)
    order = frame[["run_id", "window_ts"]].copy()
    order["_row"] = np.arange(n)
    for _, group in order.sort_values(["run_id", "window_ts", "_row"]).groupby("run_id", sort=False):
        rows = group["_row"].to_numpy(dtype=np.int64)
        for pos, row_idx in enumerate(rows):
            start = max(0, pos - int(seq_len) + 1)
            hist = rows[start : pos + 1]
            offset = int(seq_len) - len(hist)
            seq[row_idx, offset:] = x[hist]
            mask[row_idx, offset:] = 1.0
    manifest = {"selected_kpi_features": selected_cols, "shape": list(seq.shape), "seq_len": int(seq_len)}
    return seq, mask, manifest


def _numeric_column_or_default(frame: pd.DataFrame, candidates: list[str], default: float) -> np.ndarray:
    for col in candidates:
        if col in frame.columns:
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float32)
            values = np.where(np.isfinite(values), values, float(default)).astype(np.float32)
            return values
    return np.full(len(frame), float(default), dtype=np.float32)


def build_intensity_targets(
    frame: pd.DataFrame,
    mode: str,
    risk_horizon_sec: float,
    warning_horizon_sec: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    mode = str(mode).strip().lower()
    if mode not in {"auto", "lead_time", "binary"}:
        raise ValueError(f"Unsupported intensity mode: {mode!r}")

    y_warning = frame["y_warning"].astype(np.float32).to_numpy()
    y_risk = frame["y_risk"].astype(np.float32).to_numpy()
    if mode == "binary":
        return y_warning, y_risk, {
            "requested_mode": mode,
            "resolved_mode": "binary",
            "reason": "explicit_binary_mode",
            "warning_mean": float(np.mean(y_warning)),
            "risk_mean": float(np.mean(y_risk)),
        }
    if "strict_lead_sec" not in frame.columns:
        if mode == "lead_time":
            raise RuntimeError("intensity_mode=lead_time requires strict_lead_sec in the contract.")
        return y_warning, y_risk, {
            "requested_mode": mode,
            "resolved_mode": "binary",
            "reason": "strict_lead_sec_missing",
            "warning_mean": float(np.mean(y_warning)),
            "risk_mean": float(np.mean(y_risk)),
        }

    lead = pd.to_numeric(frame["strict_lead_sec"], errors="coerce").to_numpy(dtype=np.float32)
    gap = _numeric_column_or_default(frame, ["strict_lead_gap_sec"], 0.0)
    risk_horizon = _numeric_column_or_default(frame, ["strict_risk_horizon_sec", "risk_horizon_sec"], risk_horizon_sec)
    warning_horizon = _numeric_column_or_default(frame, ["strict_warning_horizon_sec", "warning_horizon_sec"], warning_horizon_sec)

    def curve(label: np.ndarray, horizon: np.ndarray) -> np.ndarray:
        denom = np.maximum(horizon - gap, 1.0)
        valid = np.isfinite(lead) & np.isfinite(gap) & np.isfinite(horizon) & (lead >= gap) & (lead <= horizon)
        value = 1.0 - ((lead - gap) / denom)
        value = np.where(valid, value, 0.0)
        value = np.clip(value, 0.0, 1.0)
        # Keep boundary positives learnable when lead == horizon.
        value = np.maximum(value, 0.05 * label)
        return value.astype(np.float32)

    warning_intensity = curve(y_warning, warning_horizon)
    risk_intensity = curve(y_risk, risk_horizon)
    manifest = {
        "requested_mode": mode,
        "resolved_mode": "lead_time",
        "risk_horizon_sec_default": float(risk_horizon_sec),
        "warning_horizon_sec_default": float(warning_horizon_sec),
        "lead_gap_sec_median": float(np.nanmedian(gap)) if len(gap) else None,
        "risk_horizon_sec_median": float(np.nanmedian(risk_horizon)) if len(risk_horizon) else None,
        "warning_horizon_sec_median": float(np.nanmedian(warning_horizon)) if len(warning_horizon) else None,
        "warning_mean": float(np.mean(warning_intensity)),
        "risk_mean": float(np.mean(risk_intensity)),
        "warning_positive_mean": float(np.mean(warning_intensity[y_warning > 0.5])) if np.any(y_warning > 0.5) else None,
        "risk_positive_mean": float(np.mean(risk_intensity[y_risk > 0.5])) if np.any(y_risk > 0.5) else None,
    }
    return warning_intensity, risk_intensity, manifest


class HadesDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: Iterable[int]):
        self.arrays = arrays
        self.indices = np.asarray(list(indices), dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = self.indices[i]
        return {
            "log_seq": torch.from_numpy(self.arrays["log_seq"][idx]),
            "log_mask": torch.from_numpy(self.arrays["log_mask"][idx]),
            "kpi_seq": torch.from_numpy(self.arrays["kpi_seq"][idx]),
            "kpi_mask": torch.from_numpy(self.arrays["kpi_mask"][idx]),
            "avail": torch.from_numpy(self.arrays["avail"][idx]),
            "y_warning": torch.tensor(self.arrays["y_warning"][idx], dtype=torch.float32),
            "y_risk": torch.tensor(self.arrays["y_risk"][idx], dtype=torch.float32),
            "y_anomaly": torch.tensor(self.arrays["y_anomaly"][idx], dtype=torch.float32),
            "warning_intensity": torch.tensor(self.arrays["warning_intensity"][idx], dtype=torch.float32),
            "risk_intensity": torch.tensor(self.arrays["risk_intensity"][idx], dtype=torch.float32),
            "row_index": torch.tensor(idx, dtype=torch.long),
        }


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = mask.float()
        no_token = mask.sum(dim=1) <= 0.5
        logits = self.score(tokens).squeeze(-1).masked_fill(mask <= 0.5, -1e9)
        weights = torch.softmax(logits, dim=1)
        weights = torch.where(no_token.unsqueeze(1), torch.zeros_like(weights), weights)
        pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights


class HadesLikeV2(nn.Module):
    def __init__(self, log_dim: int, kpi_dim: int, hidden: int = 128, heads: int = 4, log_len: int = 10, kpi_len: int = 12):
        super().__init__()
        heads = max(1, int(heads))
        while hidden % heads != 0 and heads > 1:
            heads -= 1
        self.log_proj = nn.Sequential(nn.Linear(log_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.kpi_proj = nn.Sequential(nn.Linear(kpi_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.log_pos = nn.Parameter(torch.zeros(1, log_len, hidden))
        self.kpi_pos = nn.Parameter(torch.zeros(1, kpi_len, hidden))
        log_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 3, dropout=0.12, batch_first=True, activation="gelu")
        kpi_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 3, dropout=0.12, batch_first=True, activation="gelu")
        self.log_encoder = nn.TransformerEncoder(log_layer, num_layers=2)
        self.kpi_encoder = nn.TransformerEncoder(kpi_layer, num_layers=2)
        self.log_pool = AttentionPool(hidden)
        self.kpi_pool = AttentionPool(hidden)
        self.log_to_kpi = nn.MultiheadAttention(hidden, heads, dropout=0.10, batch_first=True)
        self.kpi_to_log = nn.MultiheadAttention(hidden, heads, dropout=0.10, batch_first=True)
        self.fuser = nn.Sequential(nn.Linear(hidden * 6 + 4, hidden * 2), nn.ReLU(), nn.Dropout(0.12), nn.Linear(hidden * 2, hidden), nn.ReLU())
        self.warning_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.risk_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.warning_intensity_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.risk_intensity_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.anomaly_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.log_warning_head = nn.Linear(hidden, 1)
        self.log_risk_head = nn.Linear(hidden, 1)
        self.kpi_warning_head = nn.Linear(hidden, 1)
        self.kpi_risk_head = nn.Linear(hidden, 1)
        self.consistency_head = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    @staticmethod
    def safe_padding(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = mask > 0.5
        padding = ~valid
        no_key = ~valid.any(dim=1)
        safe = padding.clone()
        if bool(no_key.any()):
            safe[no_key, 0] = False
        return safe, no_key

    @staticmethod
    def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.float().unsqueeze(-1)
        return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        log_mask = batch["log_mask"].float()
        kpi_mask = batch["kpi_mask"].float()
        log_pad, no_log = self.safe_padding(log_mask)
        kpi_pad, no_kpi = self.safe_padding(kpi_mask)
        log_tokens = self.log_proj(batch["log_seq"]) + self.log_pos[:, : batch["log_seq"].shape[1], :]
        kpi_tokens = self.kpi_proj(batch["kpi_seq"]) + self.kpi_pos[:, : batch["kpi_seq"].shape[1], :]
        log_tokens = self.log_encoder(log_tokens, src_key_padding_mask=log_pad) * log_mask.unsqueeze(-1)
        kpi_tokens = self.kpi_encoder(kpi_tokens, src_key_padding_mask=kpi_pad) * kpi_mask.unsqueeze(-1)
        h_log, log_attn = self.log_pool(log_tokens, log_mask)
        h_kpi, kpi_attn = self.kpi_pool(kpi_tokens, kpi_mask)
        log_cross, log_to_kpi_w = self.log_to_kpi(log_tokens, kpi_tokens, kpi_tokens, key_padding_mask=kpi_pad, need_weights=True)
        kpi_cross, kpi_to_log_w = self.kpi_to_log(kpi_tokens, log_tokens, log_tokens, key_padding_mask=log_pad, need_weights=True)
        log_cross = torch.where(no_kpi[:, None, None], torch.zeros_like(log_cross), log_cross)
        kpi_cross = torch.where(no_log[:, None, None], torch.zeros_like(kpi_cross), kpi_cross)
        h_log_cross = self.masked_mean(log_cross, log_mask)
        h_kpi_cross = self.masked_mean(kpi_cross, kpi_mask)
        avail = batch["avail"].float()
        fused = self.fuser(torch.cat([h_log, h_kpi, h_log_cross, h_kpi_cross, torch.abs(h_log - h_kpi), h_log * h_kpi, avail], dim=1))
        return {
            "warning_logit": self.warning_head(fused).squeeze(1),
            "risk_logit": self.risk_head(fused).squeeze(1),
            "warning_intensity_logit": self.warning_intensity_head(fused).squeeze(1),
            "risk_intensity_logit": self.risk_intensity_head(fused).squeeze(1),
            "anomaly_logit": self.anomaly_head(fused).squeeze(1),
            "log_warning_logit": self.log_warning_head(h_log).squeeze(1),
            "log_risk_logit": self.log_risk_head(h_log).squeeze(1),
            "kpi_warning_logit": self.kpi_warning_head(h_kpi).squeeze(1),
            "kpi_risk_logit": self.kpi_risk_head(h_kpi).squeeze(1),
            "consistency_logit": self.consistency_head(torch.cat([h_log, h_kpi, torch.abs(h_log - h_kpi), h_log * h_kpi], dim=1)).squeeze(1),
            "log_attention": log_attn,
            "kpi_attention": kpi_attn,
            "log_to_kpi_attention_mean": log_to_kpi_w.mean(dim=(1, 2)),
            "kpi_to_log_attention_mean": kpi_to_log_w.mean(dim=(1, 2)),
        }


def pos_weight(y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    return float(neg / pos)


def multitask_loss(outputs, batch, weights, cfg: RunConfig):
    bce = nn.functional.binary_cross_entropy_with_logits
    y_w = batch["y_warning"]
    y_r = batch["y_risk"]
    y_a = batch["y_anomaly"]
    warning_loss = bce(outputs["warning_logit"], y_w, pos_weight=weights["warning"])
    risk_loss = bce(outputs["risk_logit"], y_r, pos_weight=weights["risk"])
    aux = torch.stack(
        [
            bce(outputs["log_warning_logit"], y_w, pos_weight=weights["warning"]),
            bce(outputs["kpi_warning_logit"], y_w, pos_weight=weights["warning"]),
            bce(outputs["log_risk_logit"], y_r, pos_weight=weights["risk"]),
            bce(outputs["kpi_risk_logit"], y_r, pos_weight=weights["risk"]),
        ]
    ).mean()
    anomaly = bce(outputs["anomaly_logit"], y_a, pos_weight=weights["anomaly"])
    both = (batch["avail"][:, 0] * batch["avail"][:, 1]).float()
    consistency_raw = bce(outputs["consistency_logit"], y_a, reduction="none")
    consistency = (consistency_raw * both).sum() / both.sum().clamp_min(1.0)
    warning_intensity = nn.functional.smooth_l1_loss(
        torch.sigmoid(outputs["warning_intensity_logit"]),
        batch["warning_intensity"],
    )
    risk_intensity = nn.functional.smooth_l1_loss(
        torch.sigmoid(outputs["risk_intensity_logit"]),
        batch["risk_intensity"],
    )
    intensity = 0.40 * warning_intensity + 0.60 * risk_intensity
    total = (
        warning_loss
        + cfg.risk_lambda * risk_loss
        + cfg.aux_lambda * aux
        + cfg.anomaly_lambda * anomaly
        + cfg.consistency_lambda * consistency
        + cfg.intensity_lambda * intensity
    )
    return total, {
        "warning": float(warning_loss.detach().cpu()),
        "risk": float(risk_loss.detach().cpu()),
        "aux": float(aux.detach().cpu()),
        "anomaly": float(anomaly.detach().cpu()),
        "consistency": float(consistency.detach().cpu()),
        "warning_intensity": float(warning_intensity.detach().cpu()),
        "risk_intensity": float(risk_intensity.detach().cpu()),
        "intensity": float(intensity.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def safe_ap(y, score):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, score))


def safe_roc(y, score):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def fmt_metric(value) -> str:
    if value is None:
        return "nan"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def safe_corr(y, score, method: str = "pearson"):
    data = pd.DataFrame({"target": np.asarray(y, dtype=float), "score": np.asarray(score, dtype=float)}).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    data = data.dropna()
    if len(data) < 2 or data["target"].nunique() < 2 or data["score"].nunique() < 2:
        return None
    return float(data["target"].corr(data["score"], method=method))


def intensity_diagnostics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("y_warning", "warning_intensity", "warning_intensity_score"),
        ("y_risk", "risk_intensity", "risk_intensity_score"),
    ]
    for binary_target, target_col, score_col in specs:
        if target_col not in pred.columns or score_col not in pred.columns:
            continue
        target = pred[target_col].astype(float).to_numpy()
        score = pred[score_col].astype(float).to_numpy()
        finite = np.isfinite(target) & np.isfinite(score)
        if not finite.any():
            continue
        error = score[finite] - target[finite]
        row = {
            "target": target_col,
            "score": score_col,
            "n": int(finite.sum()),
            "target_mean": float(np.mean(target[finite])),
            "score_mean": float(np.mean(score[finite])),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "pearson": safe_corr(target[finite], score[finite], "pearson"),
            "spearman": safe_corr(target[finite], score[finite], "spearman"),
        }
        if binary_target in pred.columns:
            row["binary_average_precision"] = safe_ap(pred[binary_target], pred[score_col])
            row["binary_roc_auc"] = safe_roc(pred[binary_target], pred[score_col])
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_at_threshold(y, score, threshold):
    pred = (np.asarray(score) >= float(threshold)).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(np.asarray(y).astype(int), pred, average="binary", zero_division=0)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "threshold": float(threshold)}


def precision_at_fraction(y, score, fraction: float) -> dict:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    n = len(y)
    if n == 0:
        return {"top_fraction": float(fraction), "top_n": 0, "precision_at_top": None, "recall_at_top": None, "lift_vs_prevalence": None}
    k = max(1, int(math.ceil(n * float(fraction))))
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf))[:k]
    positives = int(y.sum())
    top_pos = int(y[order].sum())
    prevalence = float(y.mean()) if n else 0.0
    precision = top_pos / k
    recall = top_pos / positives if positives else None
    lift = precision / prevalence if prevalence > 0 else None
    return {
        "top_fraction": float(fraction),
        "top_n": int(k),
        "top_positives": int(top_pos),
        "precision_at_top": float(precision),
        "recall_at_top": float(recall) if recall is not None else None,
        "lift_vs_prevalence": float(lift) if lift is not None else None,
    }


def topk_diagnostics(pred: pd.DataFrame, top_fracs: list[float]) -> pd.DataFrame:
    rows = []
    target_score_pairs = [("y_warning", "warning_score"), ("y_risk", "risk_score")]
    for target, score_col in [
        ("y_warning", "warning_intensity_score"),
        ("y_risk", "risk_intensity_score"),
        ("y_warning", "state_score"),
        ("y_risk", "state_score"),
    ]:
        if score_col in pred.columns:
            target_score_pairs.append((target, score_col))
    for target, score_col in target_score_pairs:
        y = pred[target].astype(int).to_numpy()
        score = pred[score_col].astype(float).to_numpy()
        total_positive_runs = None
        if "run_id" in pred.columns:
            positive_runs = set(pred.loc[pred[target].astype(int).eq(1), "run_id"].astype(str))
            total_positive_runs = len(positive_runs)
            no_positive_runs = set(pred.groupby("run_id")[target].sum().loc[lambda s: s <= 0].index.astype(str))
        else:
            positive_runs = set()
            no_positive_runs = set()
        order_all = np.argsort(-np.nan_to_num(score, nan=-np.inf))
        for frac in top_fracs:
            row = precision_at_fraction(y, score, frac)
            top_index = pred.index.to_numpy()[order_all[: row["top_n"]]]
            row.update(
                {
                    "target": target,
                    "score": score_col,
                    "n": int(len(pred)),
                    "positives": int(y.sum()),
                    "prevalence": float(y.mean()) if len(y) else None,
                }
            )
            if "run_id" in pred.columns:
                top_runs = set(pred.loc[top_index, "run_id"].astype(str))
                detected_positive_runs = top_runs & positive_runs
                false_alarm_runs = top_runs & no_positive_runs
                row.update(
                    {
                        "top_runs": int(len(top_runs)),
                        "detected_positive_runs": int(len(detected_positive_runs)),
                        "total_positive_runs": int(total_positive_runs),
                        "detected_positive_run_share": (
                            float(len(detected_positive_runs) / total_positive_runs)
                            if total_positive_runs
                            else None
                        ),
                        "false_alarm_runs_without_target": int(len(false_alarm_runs)),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def select_threshold(y, score, objective: str, min_recall: float = 0.8):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5, {"reason": "one_class"}
    thresholds = np.unique(np.concatenate([np.linspace(0.005, 0.995, 199), np.unique(score[np.isfinite(score)])]))
    rows = [{"threshold": float(t), **metrics_at_threshold(y, score, t)} for t in thresholds]
    table = pd.DataFrame(rows)
    if objective == "risk_recall":
        feasible = table.loc[table["recall"] >= float(min_recall)].copy()
        if feasible.empty:
            best = table.sort_values(["recall", "precision", "f1"], ascending=[False, False, False]).iloc[0]
            reason = "no_feasible_threshold"
        else:
            best = feasible.sort_values(["precision", "f1", "threshold"], ascending=[False, False, False]).iloc[0]
            reason = "max_precision_under_recall"
    else:
        if objective == "precision_top":
            best = table.sort_values(["precision", "recall", "f1"], ascending=[False, False, False]).iloc[0]
            reason = "max_precision"
        else:
            best = table.sort_values(["f1", "precision"], ascending=[False, False]).iloc[0]
            reason = "max_f1"
    return float(best["threshold"]), {"reason": reason, "val_precision": float(best["precision"]), "val_recall": float(best["recall"]), "val_f1": float(best["f1"])}


@torch.no_grad()
def predict(model, dataset, device, batch_size: int) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    rows = []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch)
        warning_intensity_score = torch.sigmoid(out["warning_intensity_logit"]).detach().cpu().numpy()
        risk_intensity_score = torch.sigmoid(out["risk_intensity_logit"]).detach().cpu().numpy()
        state_score = np.maximum(0.65 * warning_intensity_score, risk_intensity_score)
        rows.append(
            pd.DataFrame(
                {
                    "row_index": batch["row_index"].detach().cpu().numpy(),
                    "warning_score": torch.sigmoid(out["warning_logit"]).detach().cpu().numpy(),
                    "risk_score": torch.sigmoid(out["risk_logit"]).detach().cpu().numpy(),
                    "warning_intensity_score": warning_intensity_score,
                    "risk_intensity_score": risk_intensity_score,
                    "state_score": state_score,
                    "anomaly_score": torch.sigmoid(out["anomaly_logit"]).detach().cpu().numpy(),
                    "log_warning_score": torch.sigmoid(out["log_warning_logit"]).detach().cpu().numpy(),
                    "log_risk_score": torch.sigmoid(out["log_risk_logit"]).detach().cpu().numpy(),
                    "kpi_warning_score": torch.sigmoid(out["kpi_warning_logit"]).detach().cpu().numpy(),
                    "kpi_risk_score": torch.sigmoid(out["kpi_risk_logit"]).detach().cpu().numpy(),
                    "consistency_score": torch.sigmoid(out["consistency_logit"]).detach().cpu().numpy(),
                    "log_to_kpi_attention_mean": out["log_to_kpi_attention_mean"].detach().cpu().numpy(),
                    "kpi_to_log_attention_mean": out["kpi_to_log_attention_mean"].detach().cpu().numpy(),
                    "y_warning": batch["y_warning"].detach().cpu().numpy(),
                    "y_risk": batch["y_risk"].detach().cpu().numpy(),
                    "warning_intensity": batch["warning_intensity"].detach().cpu().numpy(),
                    "risk_intensity": batch["risk_intensity"].detach().cpu().numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True).sort_values("row_index").reset_index(drop=True)


def monitor_score(pred: pd.DataFrame, cfg: RunConfig) -> float:
    ap_w = safe_ap(pred["y_warning"], pred["warning_score"]) or 0.0
    ap_r = safe_ap(pred["y_risk"], pred["risk_score"]) or 0.0
    risk_rank_score = pred["risk_intensity_score"] if "risk_intensity_score" in pred.columns else pred["risk_score"]
    warning_rank_score = pred["warning_intensity_score"] if "warning_intensity_score" in pred.columns else pred["warning_score"]
    ap_r_intensity = safe_ap(pred["y_risk"], risk_rank_score) or 0.0
    risk_p5 = precision_at_fraction(pred["y_risk"], risk_rank_score, 0.05)["precision_at_top"] or 0.0
    warning_p10 = precision_at_fraction(pred["y_warning"], warning_rank_score, 0.10)["precision_at_top"] or 0.0
    if cfg.monitor_mode == "strict":
        # Strict forecasting has weak hard-risk labels; choose checkpoints that
        # preserve warning ranking while rewarding lead-time risk concentration.
        return 0.38 * ap_w + 0.15 * ap_r + 0.20 * ap_r_intensity + 0.17 * risk_p5 + 0.10 * warning_p10
    if cfg.monitor_mode == "warning":
        return 0.70 * ap_w + 0.15 * ap_r + 0.15 * warning_p10
    thr, _ = select_threshold(pred["y_risk"], pred["risk_score"], "risk_recall", min_recall=0.70)
    m = metrics_at_threshold(pred["y_risk"], pred["risk_score"], thr)
    return 0.45 * ap_w + 0.45 * ap_r + 0.10 * m["precision"]


def train_one_seed(arrays, idx, cfg: RunConfig, seed: int, output_dir: Path) -> dict:
    set_seed(seed)
    device = torch.device(cfg.device)
    train_ds = HadesDataset(arrays, idx["train"])
    val_ds = HadesDataset(arrays, idx["validation"])
    test_ds = HadesDataset(arrays, idx["test"])
    model = HadesLikeV2(
        log_dim=arrays["log_seq"].shape[-1],
        kpi_dim=arrays["kpi_seq"].shape[-1],
        hidden=cfg.hidden_dim,
        heads=cfg.heads,
        log_len=arrays["log_seq"].shape[1],
        kpi_len=arrays["kpi_seq"].shape[1],
    ).to(device)
    weights = {
        "warning": torch.tensor(pos_weight(arrays["y_warning"][idx["train"]]), dtype=torch.float32, device=device),
        "risk": torch.tensor(pos_weight(arrays["y_risk"][idx["train"]]), dtype=torch.float32, device=device),
        "anomaly": torch.tensor(pos_weight(arrays["y_anomaly"][idx["train"]]), dtype=torch.float32, device=device),
    }
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=max(3, cfg.patience // 4), min_lr=1e-5)
    best_state = None
    best_score = -1e9
    best_epoch = -1
    patience_left = cfg.patience
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            out = model(batch)
            loss, parts = multitask_loss(out, batch, weights, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(parts["total"])
        val_pred = predict(model, val_ds, device, cfg.batch_size)
        score = monitor_score(val_pred, cfg)
        scheduler.step(score)
        ap_w = safe_ap(val_pred["y_warning"], val_pred["warning_score"])
        ap_r = safe_ap(val_pred["y_risk"], val_pred["risk_score"])
        ap_r_intensity = safe_ap(val_pred["y_risk"], val_pred["risk_intensity_score"])
        val_risk_p5 = precision_at_fraction(val_pred["y_risk"], val_pred["risk_intensity_score"], 0.05)["precision_at_top"]
        val_warning_p10 = precision_at_fraction(val_pred["y_warning"], val_pred["warning_intensity_score"], 0.10)["precision_at_top"]
        val_risk_intensity_mae = float(np.mean(np.abs(val_pred["risk_intensity_score"] - val_pred["risk_intensity"])))
        val_warning_intensity_mae = float(np.mean(np.abs(val_pred["warning_intensity_score"] - val_pred["warning_intensity"])))
        improved = score > best_score + 1e-5
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
        history.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else None,
                "val_ap_warning": ap_w,
                "val_ap_risk": ap_r,
                "val_ap_risk_intensity_score": ap_r_intensity,
                "val_warning_precision_top10": val_warning_p10,
                "val_risk_precision_top5": val_risk_p5,
                "val_warning_intensity_mae": val_warning_intensity_mae,
                "val_risk_intensity_mae": val_risk_intensity_mae,
                "monitor_score": float(score),
                "best_score": float(best_score),
                "patience_left": int(patience_left),
                "lr": float(opt.param_groups[0]["lr"]),
            }
        )
        print(
            f"seed={seed} epoch={epoch:03d} loss={np.mean(losses):.4f} "
            f"val_ap_w={fmt_metric(ap_w)} val_ap_r={fmt_metric(ap_r)} "
            f"val_ap_r_int={fmt_metric(ap_r_intensity)} val_p10_w={fmt_metric(val_warning_p10)} "
            f"val_p5_r={fmt_metric(val_risk_p5)} val_mae_r_int={fmt_metric(val_risk_intensity_mae)} "
            f"monitor={score:.4f} best={best_score:.4f} patience={patience_left}"
        )
        if patience_left <= 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_pred = predict(model, val_ds, device, cfg.batch_size)
    test_pred = predict(model, test_ds, device, cfg.batch_size)
    thr_w, info_w = select_threshold(val_pred["y_warning"], val_pred["warning_score"], "f1")
    thr_r, info_r = select_threshold(val_pred["y_risk"], val_pred["risk_score"], "risk_recall", cfg.risk_recall_target + 0.05)
    thr_w_intensity, info_w_intensity = select_threshold(val_pred["y_warning"], val_pred["warning_intensity_score"], "f1")
    thr_r_intensity, info_r_intensity = select_threshold(val_pred["y_risk"], val_pred["risk_intensity_score"], "risk_recall", cfg.risk_recall_target + 0.05)
    test_pred["seed"] = seed
    val_pred["seed"] = seed
    test_pred["threshold_warning"] = thr_w
    test_pred["threshold_risk"] = thr_r
    test_pred["threshold_warning_intensity"] = thr_w_intensity
    test_pred["threshold_risk_intensity"] = thr_r_intensity
    val_pred["threshold_warning"] = thr_w
    val_pred["threshold_risk"] = thr_r
    val_pred["threshold_warning_intensity"] = thr_w_intensity
    val_pred["threshold_risk_intensity"] = thr_r_intensity
    metrics = []
    for target, score_col, thr, score_family in [
        ("y_warning", "warning_score", thr_w, "binary_head"),
        ("y_risk", "risk_score", thr_r, "binary_head"),
        ("y_warning", "warning_intensity_score", thr_w_intensity, "intensity_head"),
        ("y_risk", "risk_intensity_score", thr_r_intensity, "intensity_head"),
    ]:
        m = metrics_at_threshold(test_pred[target], test_pred[score_col], thr)
        p5 = precision_at_fraction(test_pred[target], test_pred[score_col], 0.05)
        p10 = precision_at_fraction(test_pred[target], test_pred[score_col], 0.10)
        metrics.append(
            {
                "seed": seed,
                "target": target,
                "score": score_col,
                "score_family": score_family,
                "average_precision": safe_ap(test_pred[target], test_pred[score_col]),
                "roc_auc": safe_roc(test_pred[target], test_pred[score_col]),
                **m,
                "precision_top5": p5["precision_at_top"],
                "recall_top5": p5["recall_at_top"],
                "precision_top10": p10["precision_at_top"],
                "recall_top10": p10["recall_at_top"],
                "best_epoch": int(best_epoch),
                "best_monitor_score": float(best_score),
            }
        )
    pd.DataFrame(history).to_csv(output_dir / f"history_seed{seed}.csv", index=False)
    torch.save({"state_dict": best_state, "config": asdict(cfg), "seed": seed, "best_epoch": best_epoch}, output_dir / f"model_seed{seed}.pt")
    return {
        "seed": seed,
        "model": model,
        "metrics": pd.DataFrame(metrics),
        "val_pred": val_pred,
        "test_pred": test_pred,
        "thresholds": {
            "warning": thr_w,
            "risk": thr_r,
            "warning_intensity": thr_w_intensity,
            "risk_intensity": thr_r_intensity,
            "warning_info": info_w,
            "risk_info": info_r,
            "warning_intensity_info": info_w_intensity,
            "risk_intensity_info": info_r_intensity,
        },
        "best_epoch": best_epoch,
        "best_score": best_score,
    }


def add_contract_columns(pred: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    candidate_cols = [
        "run_id",
        "workload",
        "scenario",
        "scenario_type",
        "failure_family",
        "anchor_policy",
        "strict_variant",
        "strict_lead_gap_sec",
        "strict_lead_sec",
        "strict_risk_horizon_sec",
        "strict_warning_horizon_sec",
        "risk_horizon_sec",
        "warning_horizon_sec",
        "target_policy",
        "window_ts",
        "split",
        "log_available",
        "kpi_available",
        "has_both_modalities",
    ]
    meta_cols = [c for c in candidate_cols if c in frame.columns]
    meta = frame[meta_cols].copy()
    meta["row_index"] = np.arange(len(meta))
    return pred.merge(meta, on="row_index", how="left")


def evaluate_slices(pred: pd.DataFrame, threshold_warning: float, threshold_risk: float) -> pd.DataFrame:
    rows = []
    slices = {
        "all": np.ones(len(pred), dtype=bool),
        "both": pred["has_both_modalities"].astype(float).to_numpy() > 0.5,
        "kpi_only": (pred["kpi_available"].astype(float).to_numpy() > 0.5) & (pred["log_available"].astype(float).to_numpy() <= 0.5),
        "workload=svd": pred["workload"].astype(str).eq("svd").to_numpy(),
    }
    for slice_name, mask in slices.items():
        part = pred.loc[mask]
        if part.empty:
            continue
        for target, score, thr in [("y_warning", "warning_score", threshold_warning), ("y_risk", "risk_score", threshold_risk)]:
            m = metrics_at_threshold(part[target], part[score], thr)
            rows.append(
                {
                    "slice": slice_name,
                    "target": target,
                    "n": int(len(part)),
                    "positives": int(part[target].sum()),
                    "prevalence": float(part[target].mean()),
                    "average_precision": safe_ap(part[target], part[score]),
                    "roc_auc": safe_roc(part[target], part[score]),
                    **m,
                }
            )
    return pd.DataFrame(rows)


def load_contract(path: Path, max_rows: int | None) -> pd.DataFrame:
    parse_dates = ["window_ts", "window_end"]
    frame = pd.read_csv(path, parse_dates=parse_dates)
    if "split" not in frame.columns:
        raise RuntimeError("Contract must contain split column from notebook 02.")
    split_map = {"valid": "validation", "val": "validation", "dev": "validation"}
    frame["split"] = frame["split"].astype(str).str.strip().str.lower().replace(split_map)
    frame = frame.sort_values(["run_id", "window_ts", "window_index"]).reset_index(drop=True)
    if max_rows:
        # Keep split/target balance for smoke runs.
        parts = []
        rng = np.random.default_rng(42)
        per_split = max(1, int(max_rows) // max(frame["split"].nunique(), 1))
        for split, part in frame.groupby("split", sort=False):
            keep = []
            for target in ["y_warning", "y_risk"]:
                for value in [0, 1]:
                    cand = part.index[part[target].astype(int).eq(value)].to_numpy()
                    if len(cand):
                        keep.extend(rng.choice(cand, size=min(30, len(cand)), replace=False).tolist())
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
            raise RuntimeError("validation_strategy=existing requested, but contract has no validation split.")
        return frame
    train_mask = frame["split"].astype(str).eq("train")
    train_frame = frame.loc[train_mask].copy()
    if train_frame.empty:
        raise RuntimeError("Cannot derive validation split: train split is empty.")
    group_col = "workload" if strategy == "workload_holdout" else "run_id"
    if group_col not in train_frame.columns:
        raise RuntimeError(f"Cannot derive validation split: missing group column {group_col!r}.")
    if train_frame[group_col].astype(str).nunique() < 2:
        if strategy == "workload_holdout":
            group_col = "run_id"
        if train_frame[group_col].astype(str).nunique() < 2:
            raise RuntimeError(f"Cannot derive validation split: {group_col} has fewer than two groups.")

    splitter = GroupShuffleSplit(n_splits=64, test_size=float(fraction), random_state=int(seed))
    best = None
    best_score = -1e9
    train_index = train_frame.index.to_numpy()
    groups = train_frame[group_col].astype(str)
    for tr_rel, val_rel in splitter.split(train_frame, groups=groups):
        tr = train_frame.iloc[tr_rel]
        val = train_frame.iloc[val_rel]
        score = 0.0
        for target in ["y_warning", "y_risk"]:
            tr_pos = int(tr[target].sum())
            val_pos = int(val[target].sum())
            if tr[target].nunique() == 2:
                score += 1.0
            if val[target].nunique() == 2:
                score += 2.0
            score += min(val_pos, 20) / 20.0
            score += min(tr_pos, 20) / 40.0
        size_penalty = abs((len(val) / max(len(train_frame), 1)) - float(fraction))
        score -= size_penalty
        if score > best_score:
            best_score = score
            best = val_rel
    if best is None:
        raise RuntimeError("Cannot derive validation split: GroupShuffleSplit produced no split.")
    val_index = train_index[best]
    frame.loc[val_index, "split"] = "validation"
    frame["validation_group_column"] = group_col
    frame["validation_strategy"] = strategy
    return frame


def split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, part in frame.groupby("split", sort=False):
        rows.append(
            {
                "split": split,
                "rows": int(len(part)),
                "runs": int(part["run_id"].nunique()) if "run_id" in part else None,
                "workloads": int(part["workload"].nunique()) if "workload" in part else None,
                "warning_positives": int(part["y_warning"].sum()),
                "risk_positives": int(part["y_risk"].sum()),
                "warning_prevalence": float(part["y_warning"].mean()),
                "risk_prevalence": float(part["y_risk"].mean()),
                "log_available": float(part["log_available"].mean()) if "log_available" in part else None,
                "both_modalities": float(part["has_both_modalities"].mean()) if "has_both_modalities" in part else None,
            }
        )
    return pd.DataFrame(rows)


def split_indices(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    result = {}
    for split in ["train", "validation", "test"]:
        result[split] = frame.index[frame["split"].astype(str).eq(split)].to_numpy(dtype=np.int64)
    if not all(len(v) for v in result.values()):
        raise RuntimeError(f"Empty split detected: { {k: len(v) for k, v in result.items()} }")
    return result


def baseline_comparison(frame: pd.DataFrame, baseline_path: Path | None, test_idx: np.ndarray) -> pd.DataFrame:
    if baseline_path is None or not baseline_path.exists():
        return pd.DataFrame()
    pred = pd.read_csv(baseline_path)
    required = {"run_id", "window_ts", "target", "baseline", "score"}
    if not required.issubset(pred.columns):
        return pd.DataFrame()
    key = frame.iloc[test_idx][["run_id", "window_ts", "y_warning", "y_risk"]].copy()
    key["window_ts"] = pd.to_datetime(key["window_ts"], utc=True, errors="coerce")
    rows = []
    for target in ["y_warning", "y_risk"]:
        for baseline in ["log-only", "kpi-only", "fused", "availability-only"]:
            part = pred.loc[pred["target"].astype(str).eq(target) & pred["baseline"].astype(str).eq(baseline), ["run_id", "window_ts", "score"]].copy()
            if part.empty:
                continue
            part["window_ts"] = pd.to_datetime(part["window_ts"], utc=True, errors="coerce")
            merged = key.merge(part.drop_duplicates(["run_id", "window_ts"]), on=["run_id", "window_ts"], how="left")
            if merged["score"].notna().sum() == 0:
                continue
            rows.append(
                {
                    "target": target,
                    "baseline": baseline,
                    "average_precision": safe_ap(merged[target], merged["score"].fillna(0.0)),
                    "roc_auc": safe_roc(merged[target], merged["score"].fillna(0.0)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"hades_v2_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--root", default=".")
    parser.add_argument("--contract-path", default=None)
    parser.add_argument("--baseline-predictions-path", default=None)
    parser.add_argument("--output-dir", default="outputs/hades_like_v2")
    parser.add_argument("--text-backend", choices=["embeddinggemma", "tfidf", "hashing"], default="embeddinggemma")
    parser.add_argument("--embedding-model", default="google/embeddinggemma-300m")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--top-log-events", type=int, default=10)
    parser.add_argument("--log-context-sec", type=int, default=60)
    parser.add_argument("--kpi-seq-len", type=int, default=12)
    parser.add_argument("--max-kpi-features", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=22)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--risk-lambda", type=float, default=0.85)
    parser.add_argument("--aux-lambda", type=float, default=0.10)
    parser.add_argument("--anomaly-lambda", type=float, default=0.15)
    parser.add_argument("--consistency-lambda", type=float, default=0.10)
    parser.add_argument("--intensity-lambda", type=float, default=0.25)
    parser.add_argument("--intensity-mode", choices=["auto", "lead_time", "binary"], default="auto")
    parser.add_argument("--intensity-risk-horizon-sec", type=float, default=30.0)
    parser.add_argument("--intensity-warning-horizon-sec", type=float, default=120.0)
    parser.add_argument("--risk-recall-target", type=float, default=0.80)
    parser.add_argument("--validation-strategy", choices=["auto", "existing", "run_id_holdout", "workload_holdout"], default="auto")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--monitor-mode", choices=["balanced", "strict", "warning"], default="balanced")
    parser.add_argument("--top-fracs", default="0.01,0.05,0.10")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    contract_path = Path(args.contract_path) if args.contract_path else first_existing(
        root / "outputs/raw_multirate/aligned_multirate_windows.csv.gz",
        root / "outputs/raw_multirate/aligned_multirate_windows.csv",
    )
    baseline_path = Path(args.baseline_predictions_path) if args.baseline_predictions_path else first_existing(
        root / "outputs/raw_multirate/baseline_predictions.csv.gz",
        root / "outputs/raw_multirate/model_predictions_smoke.csv",
    )
    if not baseline_path.exists():
        baseline_path = None
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = RunConfig(
        run_id=args.run_id,
        root=str(root),
        contract_path=str(contract_path),
        baseline_predictions_path=str(baseline_path) if baseline_path else None,
        output_dir=str(output_dir),
        text_backend=args.text_backend,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        top_log_events=args.top_log_events,
        log_context_sec=args.log_context_sec,
        kpi_seq_len=args.kpi_seq_len,
        max_kpi_features=args.max_kpi_features,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        risk_lambda=args.risk_lambda,
        aux_lambda=args.aux_lambda,
        anomaly_lambda=args.anomaly_lambda,
        consistency_lambda=args.consistency_lambda,
        intensity_lambda=args.intensity_lambda,
        intensity_mode=args.intensity_mode,
        intensity_risk_horizon_sec=args.intensity_risk_horizon_sec,
        intensity_warning_horizon_sec=args.intensity_warning_horizon_sec,
        risk_recall_target=args.risk_recall_target,
        validation_strategy=args.validation_strategy,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
        monitor_mode=args.monitor_mode,
        top_fracs=[float(s.strip()) for s in args.top_fracs.split(",") if s.strip()],
        seeds=[int(s.strip()) for s in args.seeds.split(",") if s.strip()],
        max_rows=args.max_rows,
        device=args.device,
    )
    write_json(output_dir / "run_config.json", asdict(cfg))
    print(f"[1/7] Loading contract: {contract_path}")
    frame = load_contract(contract_path, max_rows=cfg.max_rows)
    frame = apply_validation_split(frame, cfg.validation_strategy, cfg.validation_fraction, cfg.validation_seed)
    idx = split_indices(frame)
    print(f"rows={len(frame)} train={len(idx['train'])} val={len(idx['validation'])} test={len(idx['test'])}")
    split_report = split_summary(frame)
    split_report.to_csv(output_dir / "split_summary.csv", index=False)
    print(split_report.to_string(index=False))

    print("[2/7] Building raw log event sequence")
    log_text_rows, log_mask, log_manifest = build_log_event_matrix(frame, root / "raw_data", cfg.top_log_events, cfg.log_context_sec)
    print(log_manifest)

    print("[3/7] Encoding log snippets")
    log_seq, embedding_manifest = encode_log_snippets(
        log_text_rows,
        log_mask,
        idx["train"],
        output_dir,
        cfg.text_backend,
        cfg.embedding_model,
        cfg.embedding_dim,
        cfg.batch_size,
    )
    print(embedding_manifest)

    print("[4/7] Building KPI temporal sequence")
    selected_kpi = select_kpi_features(frame, idx["train"], cfg.max_kpi_features)
    kpi_seq, kpi_mask, kpi_manifest = build_kpi_sequence(frame, idx["train"], selected_kpi, cfg.kpi_seq_len)
    print({"kpi_shape": kpi_seq.shape, "features": len(selected_kpi)})

    print("[4b/7] Building lead-time intensity targets")
    warning_intensity, risk_intensity, intensity_manifest = build_intensity_targets(
        frame,
        cfg.intensity_mode,
        cfg.intensity_risk_horizon_sec,
        cfg.intensity_warning_horizon_sec,
    )
    print(intensity_manifest)

    arrays = {
        "log_seq": log_seq.astype(np.float32),
        "log_mask": log_mask.astype(np.float32),
        "kpi_seq": kpi_seq.astype(np.float32),
        "kpi_mask": kpi_mask.astype(np.float32),
        "avail": frame[["log_available", "kpi_available", "has_both_modalities", "kpi_avail_ratio"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        "y_warning": frame["y_warning"].astype(np.float32).to_numpy(),
        "y_risk": frame["y_risk"].astype(np.float32).to_numpy(),
        "y_anomaly": np.maximum(frame["y_warning"].astype(np.float32).to_numpy(), frame["y_risk"].astype(np.float32).to_numpy()),
        "warning_intensity": warning_intensity.astype(np.float32),
        "risk_intensity": risk_intensity.astype(np.float32),
    }
    write_json(output_dir / "feature_manifest.json", {"log": log_manifest, "embedding": embedding_manifest, "kpi": kpi_manifest, "intensity": intensity_manifest, "arrays": {k: list(v.shape) for k, v in arrays.items()}})

    print("[5/7] Training HADES-like v2 seeds")
    results = []
    for seed in cfg.seeds:
        results.append(train_one_seed(arrays, idx, cfg, seed, output_dir))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[6/7] Aggregating predictions and metrics")
    seed_metrics = pd.concat([r["metrics"] for r in results], ignore_index=True)
    seed_metrics.to_csv(output_dir / "seed_metrics.csv", index=False)
    all_test_pred = pd.concat([r["test_pred"] for r in results], ignore_index=True)
    all_test_pred.to_csv(output_dir / "seed_predictions.csv", index=False)
    all_val_pred = pd.concat([r["val_pred"] for r in results], ignore_index=True)
    all_val_pred.to_csv(output_dir / "validation_seed_predictions.csv", index=False)

    ensemble = all_test_pred.groupby("row_index", as_index=False).mean(numeric_only=True)
    ensemble = add_contract_columns(ensemble, frame)
    val_ensemble = all_val_pred.groupby("row_index", as_index=False).mean(numeric_only=True)
    val_ensemble = add_contract_columns(val_ensemble, frame)
    # Thresholds from ensemble test are not leakage-free; final AP is threshold-free.
    # For F1/recall, use median seed thresholds to avoid tuning on test.
    thr_w = float(np.median([r["thresholds"]["warning"] for r in results]))
    thr_r = float(np.median([r["thresholds"]["risk"] for r in results]))
    thr_w_intensity = float(np.median([r["thresholds"]["warning_intensity"] for r in results]))
    thr_r_intensity = float(np.median([r["thresholds"]["risk_intensity"] for r in results]))
    ensemble_metrics = []
    for target, score, thr, model_name in [
        ("y_warning", "warning_score", thr_w, "seed_ensemble_mean"),
        ("y_risk", "risk_score", thr_r, "seed_ensemble_mean"),
        ("y_warning", "warning_intensity_score", thr_w_intensity, "seed_ensemble_mean_intensity"),
        ("y_risk", "risk_intensity_score", thr_r_intensity, "seed_ensemble_mean_intensity"),
        ("y_warning", "state_score", 0.5, "seed_ensemble_state_score"),
        ("y_risk", "state_score", 0.5, "seed_ensemble_state_score"),
    ]:
        m = metrics_at_threshold(ensemble[target], ensemble[score], thr)
        ensemble_metrics.append({"target": target, "model": model_name, "score": score, "average_precision": safe_ap(ensemble[target], ensemble[score]), "roc_auc": safe_roc(ensemble[target], ensemble[score]), **m})
    ensemble_metrics = pd.DataFrame(ensemble_metrics)
    intensity_metrics = intensity_diagnostics(ensemble)
    intensity_metrics["split"] = "test"
    val_intensity_metrics = intensity_diagnostics(val_ensemble)
    val_intensity_metrics["split"] = "validation"
    intensity_metrics_all = pd.concat([val_intensity_metrics, intensity_metrics], ignore_index=True)
    slices = evaluate_slices(ensemble, thr_w, thr_r)
    baseline_df = baseline_comparison(frame, baseline_path, idx["test"])
    ranking = topk_diagnostics(ensemble, cfg.top_fracs)
    ranking["split"] = "test"
    val_ranking = topk_diagnostics(val_ensemble, cfg.top_fracs)
    val_ranking["split"] = "validation"
    ranking_all = pd.concat([val_ranking, ranking], ignore_index=True)

    print("[7/7] Writing outputs")
    ensemble.to_csv(output_dir / "ensemble_predictions.csv", index=False)
    val_ensemble.to_csv(output_dir / "validation_ensemble_predictions.csv", index=False)
    ensemble_metrics.to_csv(output_dir / "ensemble_metrics.csv", index=False)
    intensity_metrics_all.to_csv(output_dir / "intensity_metrics.csv", index=False)
    slices.to_csv(output_dir / "slice_metrics.csv", index=False)
    ranking_all.to_csv(output_dir / "ranking_diagnostics.csv", index=False)
    baseline_df.to_csv(output_dir / "baseline_comparison.csv", index=False)
    summary = {
        "run_id": cfg.run_id,
        "seed_metrics": seed_metrics.to_dict("records"),
        "ensemble_metrics": ensemble_metrics.to_dict("records"),
        "intensity_metrics": intensity_metrics_all.to_dict("records"),
        "slice_metrics": slices.to_dict("records"),
        "ranking_diagnostics": ranking_all.to_dict("records"),
        "baseline_comparison": baseline_df.to_dict("records"),
        "thresholds": {
            "warning_median_seed": thr_w,
            "risk_median_seed": thr_r,
            "warning_intensity_median_seed": thr_w_intensity,
            "risk_intensity_median_seed": thr_r_intensity,
        },
        "split_summary": split_report.to_dict("records"),
        "feature_manifest": {"log": log_manifest, "embedding": embedding_manifest, "kpi_shape": list(kpi_seq.shape), "kpi_feature_count": len(selected_kpi), "intensity": intensity_manifest},
    }
    write_json(output_dir / "metrics_summary.json", summary)
    print(f"Done. Outputs written to: {output_dir}")
    print(ensemble_metrics.to_string(index=False))
    if not intensity_metrics_all.empty:
        print(intensity_metrics_all.to_string(index=False))
    print(ranking_all.to_string(index=False))
    if not baseline_df.empty:
        print(baseline_df.to_string(index=False))


if __name__ == "__main__":
    main()
