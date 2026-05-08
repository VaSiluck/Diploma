"""
Training core for the lightweight multimodal warning contour.

This module contains the production training logic behind notebook 03:
stratified workload split, workload-aware metric normalization, temporal
deviation channels, separate target heads, HADES-like cross-modal fusion,
and alert-level evaluation.

Typical CUDA run is exposed through:
    python -m dashboard.training.final_model --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dashboard.modeling import hades_blocks as eh


DEFAULT_TARGETS = ["y_warning", "y_risk", "y_anomaly", "horizon_30s", "horizon_60s", "horizon_120s", "horizon_300s"]


@dataclass
class ProtocolConfig:
    run_id: str
    root: str
    contract_path: str
    output_dir: str
    split_protocols: list[str]
    leave_one_workloads: list[str]
    variants: list[str]
    seeds: list[int]
    targets: list[str]
    alert_target: str
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
    normalization: str
    metric_feature_mode: str
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
    monitor_target: str
    validation_fraction: float
    validation_seed: int
    device: str
    export_onnx: bool
    quantization: str
    calibration_size: int
    benchmark_batches: int


class ProtocolDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: Iterable[int], pseudo: dict | None = None):
        self.arrays = arrays
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.pseudo = pseudo

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        target = self.arrays["target_matrix"][idx].astype(np.float32).copy()
        weight = np.ones_like(target, dtype=np.float32)
        if self.pseudo is not None:
            pseudo_value = self.pseudo["values"][idx]
            pseudo_weight = self.pseudo["weights"][idx]
            use_pseudo = np.isfinite(pseudo_value) & (pseudo_weight > 0)
            target[use_pseudo] = pseudo_value[use_pseudo]
            weight[use_pseudo] = pseudo_weight[use_pseudo]
        return {
            "log_emb": torch.from_numpy(self.arrays["log_emb"][idx]).float(),
            "log_mask": torch.from_numpy(self.arrays["log_mask"][idx]).float(),
            "metric_seg": torch.from_numpy(self.arrays["metric_seg"][idx]).float(),
            "metric_mask": torch.from_numpy(self.arrays["metric_mask"][idx]).float(),
            "target": torch.from_numpy(target).float(),
            "weight": torch.from_numpy(weight).float(),
            "row_index": torch.tensor(idx, dtype=torch.long),
        }


class MultiHeadCrossModalHades(nn.Module):
    def __init__(
        self,
        log_input_dim: int,
        metric_input_dim: int,
        d_model: int,
        heads: int,
        log_layers: int,
        metric_layers: int,
        metric_encoder: str,
        target_names: list[str],
    ):
        super().__init__()
        self.target_names = list(target_names)
        self.log_encoder = eh.LogTransformerEncoder(log_input_dim, d_model, heads, log_layers)
        if metric_encoder == "tiny_time_mixer":
            self.metric_encoder = eh.TinyTimeMixerMetricEncoder(metric_input_dim, d_model, metric_layers)
        elif metric_encoder == "causal_conv":
            self.metric_encoder = eh.CausalConvMetricEncoder(metric_input_dim, d_model, metric_layers)
        else:
            raise ValueError(f"Unsupported metric encoder: {metric_encoder}")
        self.log_to_metric = nn.MultiheadAttention(d_model, heads, dropout=0.05, batch_first=True)
        self.metric_to_log = nn.MultiheadAttention(d_model, heads, dropout=0.05, batch_first=True)
        self.shared = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict({name: nn.Linear(d_model, 1) for name in self.target_names})

    @staticmethod
    def safe_pad(mask: torch.Tensor) -> torch.Tensor:
        return eh.CrossModalHades.safe_pad(mask)

    @staticmethod
    def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return eh.CrossModalHades.masked_mean(tokens, mask)

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
        shared = self.shared(fused)
        return torch.cat([self.heads[name](shared) for name in self.target_names], dim=1)


def parse_csv_list(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_target_names(value: str) -> list[str]:
    if str(value).strip().lower() == "default":
        return list(DEFAULT_TARGETS)
    return parse_csv_list(value, str)


def resolve_leave_one_workloads(frame: pd.DataFrame, value: str) -> list[str]:
    value = str(value).strip()
    if not value or value.lower() == "none":
        return []
    if value.lower() == "all":
        return sorted(frame["workload"].astype(str).unique())
    if value.lower() == "test":
        test = frame.loc[frame["split"].astype(str).eq("test"), "workload"].astype(str).unique()
        return sorted(test)
    return parse_csv_list(value, str)


def split_indices(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    result = {}
    for split in ["train", "validation", "test"]:
        result[split] = frame.index[frame["split"].astype(str).eq(split)].to_numpy(dtype=np.int64)
    if not all(len(value) for value in result.values()):
        raise RuntimeError(f"Empty split detected: { {k: len(v) for k, v in result.items()} }")
    return result


def stratified_by_workload_split(frame: pd.DataFrame, seed: int, validation_fraction: float) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    out = frame.copy()
    out["split"] = "train"
    for _, group in out.groupby("workload", sort=True):
        runs = np.array(sorted(group["run_id"].astype(str).unique()))
        rng.shuffle(runs)
        n = len(runs)
        if n >= 5:
            test_n = max(1, int(round(n * 0.20)))
            val_n = max(1, int(round(n * float(validation_fraction))))
            test_runs = set(runs[:test_n])
            val_runs = set(runs[test_n : test_n + val_n])
        elif n >= 3:
            test_runs = {runs[0]}
            val_runs = {runs[1]}
        else:
            test_runs = {runs[0]}
            val_runs = set()
        out.loc[out["run_id"].astype(str).isin(test_runs), "split"] = "test"
        out.loc[out["run_id"].astype(str).isin(val_runs), "split"] = "validation"
    return out


def leave_one_workload_out_split(frame: pd.DataFrame, workload: str, seed: int, validation_fraction: float) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    out = frame.copy()
    out["split"] = "train"
    out.loc[out["workload"].astype(str).eq(str(workload)), "split"] = "test"
    for other, group in out.loc[~out["workload"].astype(str).eq(str(workload))].groupby("workload", sort=True):
        runs = np.array(sorted(group["run_id"].astype(str).unique()))
        rng.shuffle(runs)
        val_n = max(1, int(round(len(runs) * float(validation_fraction)))) if len(runs) > 1 else 0
        if val_n:
            out.loc[out["run_id"].astype(str).isin(set(runs[:val_n])), "split"] = "validation"
    return out


def make_protocol_frame(base: pd.DataFrame, protocol: str, cfg: ProtocolConfig) -> pd.DataFrame:
    frame = base.copy()
    frame["original_split"] = frame["split"].astype(str)
    if protocol == "current":
        return eh.apply_validation_split(frame, "auto", cfg.validation_fraction, cfg.validation_seed)
    if protocol == "stratified_by_workload":
        return stratified_by_workload_split(frame, cfg.validation_seed, cfg.validation_fraction)
    if protocol.startswith("leave_one_workload_out:"):
        workload = protocol.split(":", 1)[1]
        return leave_one_workload_out_split(frame, workload, cfg.validation_seed, cfg.validation_fraction)
    raise ValueError(f"Unsupported split protocol: {protocol}")


def target_for_horizon(frame: pd.DataFrame, horizon: int) -> pd.Series:
    scenario_failure = frame["scenario_type"].astype(str).str.lower().eq("failure")
    lead = pd.to_numeric(frame.get("strict_lead_sec"), errors="coerce")
    gap = pd.to_numeric(frame.get("strict_lead_gap_sec", pd.Series(10, index=frame.index)), errors="coerce").fillna(10)
    return (scenario_failure & lead.notna() & lead.ge(gap) & lead.le(float(horizon))).astype(np.float32)


def make_target_matrix(frame: pd.DataFrame, target_names: list[str]) -> tuple[np.ndarray, dict]:
    cols = []
    manifest = {}
    for name in target_names:
        if name.startswith("horizon_") and name.endswith("s"):
            horizon = int(name.replace("horizon_", "").replace("s", ""))
            values = target_for_horizon(frame, horizon)
            manifest[name] = {"type": "synthetic_horizon", "horizon_sec": horizon, "positives": int(values.sum())}
        else:
            if name not in frame.columns:
                raise RuntimeError(f"Missing target column: {name}")
            values = frame[name].astype(np.float32)
            manifest[name] = {"type": "contract_column", "positives": int(values.sum())}
        cols.append(values.to_numpy(dtype=np.float32))
    matrix = np.stack(cols, axis=1).astype(np.float32)
    if {"y_anomaly", "y_warning"}.issubset(set(target_names)):
        i = target_names.index("y_anomaly")
        j = target_names.index("y_warning")
        manifest["y_anomaly_equals_y_warning"] = bool(np.array_equal(matrix[:, i], matrix[:, j]))
    return matrix, manifest


def load_metric_frames(frame: pd.DataFrame, raw_root: Path, nodes: list[str], instruments: list[str], cache_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for run_id, group in frame.groupby("run_id", sort=False):
        workload = str(group["workload"].iloc[0])
        scenario = str(group["scenario"].iloc[0])
        run_root = raw_root / workload / "kpis" / scenario
        frames[str(run_id)] = eh.load_run_metrics(run_root, nodes=nodes, instruments=instruments, cache_dir=cache_dir)
    return frames


def build_raw_metric_segments(
    frame: pd.DataFrame,
    metric_frames: dict[str, pd.DataFrame],
    train_idx: np.ndarray,
    max_channels: int,
    segment_sec: int,
    segment_len: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    train_runs = set(frame.iloc[train_idx]["run_id"].astype(str))
    selected_cols = eh.select_metric_channels(metric_frames, train_runs, max_channels=max_channels)
    if not selected_cols:
        raise RuntimeError("No raw KPI metric channels were loaded from raw_data.")
    segments = np.full((len(frame), int(segment_len), len(selected_cols)), np.nan, dtype=np.float32)
    masks = np.zeros((len(frame), int(segment_len)), dtype=np.float32)
    for row in frame.itertuples(index=False):
        run_frame = metric_frames.get(str(row.run_id), pd.DataFrame())
        segment, mask = eh.segment_from_run_frame(run_frame, row.window_end, selected_cols, segment_sec, segment_len)
        segments[int(row.row_pos)] = segment
        masks[int(row.row_pos)] = mask
    manifest = {
        "source": "raw_data_kpi_files_unscaled",
        "segment_sec": int(segment_sec),
        "segment_len": int(segment_len),
        "channels": selected_cols,
        "channel_count": int(len(selected_cols)),
        "runs_loaded": int(sum(not frame.empty for frame in metric_frames.values())),
        "mean_valid_steps": float(masks.sum(axis=1).mean()) if len(masks) else 0.0,
    }
    return segments, masks, selected_cols, manifest


def valid_train_values(segments: np.ndarray, masks: np.ndarray, indices: np.ndarray) -> np.ndarray:
    part = segments[indices]
    part_mask = masks[indices] > 0.5
    if part.size == 0:
        return np.empty((0, segments.shape[-1]), dtype=np.float32)
    return part[part_mask]


def fill_segments(segments: np.ndarray, masks: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_values = valid_train_values(segments, masks, train_idx)
    if len(train_values):
        median = np.nanmedian(train_values, axis=0)
    else:
        median = np.zeros(segments.shape[-1], dtype=np.float32)
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
    filled = np.where(np.isfinite(segments), segments, median[None, None, :]).astype(np.float32)
    filled[masks <= 0.5] = 0.0
    return filled, median


def stats_from_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(values):
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
    else:
        mean = np.zeros(values.shape[-1] if values.ndim == 2 else 1, dtype=np.float32)
        std = np.ones_like(mean)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    return mean, std


def normalize_metric_segments(
    frame: pd.DataFrame,
    raw_segments: np.ndarray,
    masks: np.ndarray,
    train_idx: np.ndarray,
    normalization: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    filled, median = fill_segments(raw_segments, masks, train_idx)
    global_mean, global_std = stats_from_values(valid_train_values(filled, masks, train_idx))
    if normalization == "raw":
        base = filled.copy()
        z = ((filled - global_mean[None, None, :]) / global_std[None, None, :]).astype(np.float32)
        manifest = {"normalization": "raw", "fallback_workloads": []}
    elif normalization == "global_norm":
        base = ((filled - global_mean[None, None, :]) / global_std[None, None, :]).astype(np.float32)
        z = base.copy()
        manifest = {"normalization": "global_norm", "fallback_workloads": []}
    elif normalization in {"workload_norm", "workload_node_norm"}:
        base = np.zeros_like(filled, dtype=np.float32)
        fallback = []
        train_workloads = frame.iloc[train_idx]["workload"].astype(str)
        for workload, group in frame.groupby("workload", sort=False):
            row_idx = group.index.to_numpy(dtype=np.int64)
            group_train_idx = frame.iloc[train_idx].index[train_workloads.eq(str(workload)).to_numpy()].to_numpy(dtype=np.int64)
            if len(group_train_idx):
                mean, std = stats_from_values(valid_train_values(filled, masks, group_train_idx))
            else:
                mean, std = global_mean, global_std
                fallback.append(str(workload))
            base[row_idx] = ((filled[row_idx] - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
        z = base.copy()
        manifest = {"normalization": normalization, "fallback_workloads": sorted(set(fallback))}
    else:
        raise ValueError(f"Unsupported normalization: {normalization}")
    base[masks <= 0.5] = 0.0
    z[masks <= 0.5] = 0.0
    manifest["median"] = median.tolist()
    return base.astype(np.float32), z.astype(np.float32), manifest


def temporal_deviation_channels(z: np.ndarray, masks: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(z, dtype=np.float32)
    delta[:, 1:, :] = z[:, 1:, :] - z[:, :-1, :]
    mean = (z * masks[:, :, None]).sum(axis=1, keepdims=True) / masks.sum(axis=1, keepdims=True).clip(min=1.0)[:, :, None]
    residual = z - mean
    spike = (np.abs(z) > 3.0).astype(np.float32)
    abs_z = np.abs(z)
    p95 = np.quantile(abs_z, 0.95, axis=2, keepdims=True).astype(np.float32)
    p99 = np.quantile(abs_z, 0.99, axis=2, keepdims=True).astype(np.float32)
    spike_count = spike.mean(axis=2, keepdims=True).astype(np.float32)
    dev = np.concatenate([z, delta, residual, spike, p95, p99, spike_count], axis=2).astype(np.float32)
    dev[masks <= 0.5] = 0.0
    return dev


def transform_metric_segments(
    frame: pd.DataFrame,
    raw_segments: np.ndarray,
    masks: np.ndarray,
    train_idx: np.ndarray,
    normalization: str,
    feature_mode: str,
) -> tuple[np.ndarray, dict]:
    base, z, norm_manifest = normalize_metric_segments(frame, raw_segments, masks, train_idx, normalization)
    dev = temporal_deviation_channels(z, masks)
    if feature_mode == "raw":
        out = base
    elif feature_mode == "deviation":
        out = dev
    elif feature_mode == "raw_plus_deviation":
        out = np.concatenate([base, dev], axis=2).astype(np.float32)
    else:
        raise ValueError(f"Unsupported metric feature mode: {feature_mode}")
    out[masks <= 0.5] = 0.0
    return out.astype(np.float32), {
        **norm_manifest,
        "metric_feature_mode": feature_mode,
        "input_dim": int(out.shape[-1]),
        "deviation_channels": int(dev.shape[-1]),
    }


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def pos_weight_vector(target_matrix: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    y = target_matrix[train_idx].astype(int)
    pos = np.maximum((y == 1).sum(axis=0), 1)
    neg = np.maximum((y == 0).sum(axis=0), 1)
    return (neg / pos).astype(np.float32)


def multi_bce_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    return (raw * weight).sum() / weight.sum().clamp_min(1.0)


def predict_matrix(model: MultiHeadCrossModalHades, arrays: dict[str, np.ndarray], indices: np.ndarray, cfg: ProtocolConfig, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(ProtocolDataset(arrays, indices), batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)
    model.eval()
    scores = []
    row_indices = []
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            logits = model(batch["log_emb"], batch["log_mask"], batch["metric_seg"], batch["metric_mask"])
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
            row_indices.append(batch["row_index"].detach().cpu().numpy())
    if not scores:
        return np.empty((0, len(cfg.targets)), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.concatenate(scores, axis=0).astype(np.float32), np.concatenate(row_indices).astype(np.int64)


def make_pseudo_labels(model: MultiHeadCrossModalHades, arrays: dict[str, np.ndarray], train_idx: np.ndarray, cfg: ProtocolConfig, device: torch.device) -> dict:
    scores, rows = predict_matrix(model, arrays, train_idx, cfg, device)
    values = np.full_like(arrays["target_matrix"], np.nan, dtype=np.float32)
    weights = np.zeros_like(arrays["target_matrix"], dtype=np.float32)
    counts = {}
    for j, target in enumerate(cfg.targets):
        score = scores[:, j]
        low = float(np.quantile(score, float(cfg.pseudo_low_quantile)))
        high = float(np.quantile(score, float(cfg.pseudo_high_quantile)))
        selected = (score <= low) | (score >= high)
        values[rows[selected], j] = (score[selected] >= high).astype(np.float32)
        weights[rows[selected], j] = float(cfg.pseudo_weight)
        counts[target] = int(selected.sum())
    return {"values": values, "weights": weights, "counts": counts}


def evaluate_target(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    return eh.evaluate_binary(y, score, threshold)


def thresholds_from_validation(target_matrix: np.ndarray, val_scores: np.ndarray, val_rows: np.ndarray, cfg: ProtocolConfig) -> dict:
    thresholds = {}
    for j, target in enumerate(cfg.targets):
        y = target_matrix[val_rows, j]
        score = val_scores[:, j]
        thresholds[target] = {
            "best_f1": float(eh.best_f1_threshold(y, score)),
            "fp_budget": float(eh.fp_budget_threshold(y, score, cfg.fp_budget)),
        }
    return thresholds


def train_model(arrays: dict[str, np.ndarray], idx: dict[str, np.ndarray], model: MultiHeadCrossModalHades, cfg: ProtocolConfig, device: torch.device) -> tuple[MultiHeadCrossModalHades, dict]:
    pos_weight = torch.from_numpy(pos_weight_vector(arrays["target_matrix"], idx["train"])).float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    monitor_idx = cfg.targets.index(cfg.monitor_target) if cfg.monitor_target in cfg.targets else 0
    best_state = None
    best_monitor = -1e18
    best_epoch = 0
    stale = 0
    pseudo = None
    pseudo_manifest = {"enabled": bool(cfg.pseudo_labeling), "counts": {}}
    history = []
    best_thresholds = {}

    for epoch in range(1, int(cfg.epochs) + 1):
        if cfg.pseudo_labeling and pseudo is None and epoch >= int(cfg.pseudo_start_epoch):
            pseudo = make_pseudo_labels(model, arrays, idx["train"], cfg, device)
            pseudo_manifest = {"enabled": True, "counts": pseudo["counts"]}
        loader = DataLoader(ProtocolDataset(arrays, idx["train"], pseudo=pseudo), batch_size=int(cfg.batch_size), shuffle=True, num_workers=0)
        model.train()
        losses = []
        for batch in loader:
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["log_emb"], batch["log_mask"], batch["metric_seg"], batch["metric_mask"])
            loss = multi_bce_loss(logits, batch["target"], batch["weight"], pos_weight)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_scores, val_rows = predict_matrix(model, arrays, idx["validation"], cfg, device)
        thresholds = thresholds_from_validation(arrays["target_matrix"], val_scores, val_rows, cfg)
        y_monitor = arrays["target_matrix"][val_rows, monitor_idx]
        s_monitor = val_scores[:, monitor_idx]
        val_ap = eh.safe_ap(y_monitor, s_monitor) or 0.0
        best_f1_metrics = eh.evaluate_binary(y_monitor, s_monitor, thresholds[cfg.targets[monitor_idx]]["best_f1"])
        fp_metrics = eh.evaluate_binary(y_monitor, s_monitor, thresholds[cfg.targets[monitor_idx]]["fp_budget"])
        monitor_value = float(val_ap)
        if cfg.train_monitor == "best_f1":
            monitor_value = float(best_f1_metrics["f1"])
        elif cfg.train_monitor == "fp_budget_f1":
            monitor_value = float(fp_metrics["f1"])
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)) if losses else None,
                "monitor_target": cfg.monitor_target,
                "val_ap": float(val_ap),
                "val_best_f1": float(best_f1_metrics["f1"]),
                "val_fp_budget_f1": float(fp_metrics["f1"]),
                "monitor_value": float(monitor_value),
            }
        )
        if monitor_value > best_monitor:
            best_monitor = float(monitor_value)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            best_thresholds = thresholds
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
        "history": history,
        "thresholds": best_thresholds,
        "pseudo": pseudo_manifest,
        "train_monitor": cfg.train_monitor,
        "monitor_target": cfg.monitor_target,
    }


def metrics_from_scores(
    target_matrix: np.ndarray,
    scores: np.ndarray,
    rows: np.ndarray,
    thresholds: dict,
    cfg: ProtocolConfig,
    split_name: str,
    variant: str,
    seed: int,
    split_protocol: str,
) -> pd.DataFrame:
    out = []
    for j, target in enumerate(cfg.targets):
        y = target_matrix[rows, j]
        score = scores[:, j]
        for policy, threshold in [
            ("validation_best_f1", thresholds[target]["best_f1"]),
            (f"validation_fp_budget_{cfg.fp_budget}", thresholds[target]["fp_budget"]),
        ]:
            out.append(
                {
                    "split_protocol": split_protocol,
                    "variant": variant,
                    "seed": int(seed),
                    "runtime": "torch_fp32",
                    "split": split_name,
                    "target": target,
                    "threshold_policy": policy,
                    **evaluate_target(y, score, threshold),
                }
            )
    return pd.DataFrame(out)


def prediction_frame(
    frame: pd.DataFrame,
    target_matrix: np.ndarray,
    scores: np.ndarray,
    rows: np.ndarray,
    cfg: ProtocolConfig,
    split_protocol: str,
    variant: str,
    seed: int,
    split_name: str,
) -> pd.DataFrame:
    parts = []
    meta_cols = [
        "row_pos",
        "run_id",
        "workload",
        "scenario",
        "scenario_type",
        "failure_family",
        "anchor_policy",
        "window_ts",
        "window_end",
        "split",
        "original_split",
    ]
    meta_cols = [col for col in meta_cols if col in frame.columns]
    meta = frame.iloc[rows][meta_cols].rename(columns={"row_pos": "row_index"}).reset_index(drop=True)
    for j, target in enumerate(cfg.targets):
        part = meta.copy()
        part["target_name"] = target
        part["target"] = target_matrix[rows, j].astype(int)
        part["score"] = scores[:, j].astype(float)
        part["split_protocol"] = split_protocol
        part["variant"] = variant
        part["seed"] = int(seed)
        part["pred_split"] = split_name
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def workload_metrics(pred: pd.DataFrame, metrics: pd.DataFrame, cfg: ProtocolConfig) -> pd.DataFrame:
    rows = []
    thresholds = metrics.loc[
        metrics["split"].astype(str).eq("validation")
        & metrics["threshold_policy"].astype(str).eq("validation_best_f1")
    ][["split_protocol", "variant", "seed", "target", "threshold"]]
    for key, group in pred.groupby(["split_protocol", "variant", "seed", "target_name", "workload"], dropna=False):
        split_protocol, variant, seed, target, workload = key
        threshold_row = thresholds.loc[
            thresholds["split_protocol"].astype(str).eq(str(split_protocol))
            & thresholds["variant"].astype(str).eq(str(variant))
            & thresholds["seed"].astype(int).eq(int(seed))
            & thresholds["target"].astype(str).eq(str(target))
        ]
        threshold = float(threshold_row.iloc[0]["threshold"]) if not threshold_row.empty else 0.5
        rows.append(
            {
                "split_protocol": split_protocol,
                "variant": variant,
                "seed": int(seed),
                "target": target,
                "workload": workload,
                **evaluate_target(group["target"].to_numpy(), group["score"].to_numpy(), threshold),
            }
        )
    return pd.DataFrame(rows)


def alert_level_metrics(pred: pd.DataFrame, metrics: pd.DataFrame, cfg: ProtocolConfig) -> pd.DataFrame:
    rows = []
    target = cfg.alert_target if cfg.alert_target in cfg.targets else cfg.monitor_target
    threshold_table = metrics.loc[
        metrics["split"].astype(str).eq("validation")
        & metrics["target"].astype(str).eq(target)
        & metrics["threshold_policy"].astype(str).eq("validation_best_f1")
    ]
    test = pred.loc[pred["target_name"].astype(str).eq(target)].copy()
    if test.empty:
        return pd.DataFrame()
    test["window_ts"] = pd.to_datetime(test["window_ts"], utc=True, errors="coerce")
    for key, group in test.groupby(["split_protocol", "variant", "seed"], dropna=False):
        split_protocol, variant, seed = key
        thr_row = threshold_table.loc[
            threshold_table["split_protocol"].astype(str).eq(str(split_protocol))
            & threshold_table["variant"].astype(str).eq(str(variant))
            & threshold_table["seed"].astype(int).eq(int(seed))
        ]
        threshold = float(thr_row.iloc[0]["threshold"]) if not thr_row.empty else 0.5
        part = group.copy()
        part["base_alert"] = part["score"].ge(threshold)
        for consecutive in [1, 2, 3]:
            for cooldown_windows in [0, 3, 6]:
                false_alerts = 0
                detected_incidents = 0
                missed_incidents = 0
                alert_events = 0
                for _, run in part.sort_values(["run_id", "window_ts"]).groupby("run_id", sort=False):
                    raw = run["base_alert"].to_numpy(dtype=bool)
                    confirmed = np.zeros(len(run), dtype=bool)
                    streak = 0
                    cooldown = 0
                    for i, value in enumerate(raw):
                        if cooldown > 0:
                            cooldown -= 1
                            if not value:
                                streak = 0
                            continue
                        streak = streak + 1 if value else 0
                        if streak >= consecutive:
                            confirmed[i] = True
                            cooldown = int(cooldown_windows)
                            streak = 0
                    y = run["target"].astype(int).to_numpy()
                    has_positive = bool(y.sum() > 0)
                    has_alert_on_positive = bool((confirmed & (y == 1)).any())
                    alert_events += int(confirmed.sum())
                    if has_positive:
                        detected_incidents += int(has_alert_on_positive)
                        missed_incidents += int(not has_alert_on_positive)
                    elif confirmed.any():
                        false_alerts += int(confirmed.sum())
                rows.append(
                    {
                        "split_protocol": split_protocol,
                        "variant": variant,
                        "seed": int(seed),
                        "target": target,
                        "threshold": threshold,
                        "consecutive_windows": int(consecutive),
                        "cooldown_windows": int(cooldown_windows),
                        "alert_events": int(alert_events),
                        "false_alerts": int(false_alerts),
                        "detected_incidents": int(detected_incidents),
                        "missed_incidents": int(missed_incidents),
                    }
                )
    return pd.DataFrame(rows)


def benchmark_torch_latency(model: nn.Module, arrays: dict[str, np.ndarray], indices: np.ndarray, cfg: ProtocolConfig, device: torch.device) -> dict:
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
        for _ in range(3):
            _ = model(**batch)
        repeats = max(1, int(cfg.benchmark_batches))
        start = time.perf_counter()
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


def export_onnx(model: nn.Module, arrays: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.cpu().eval()
    log_emb = torch.from_numpy(arrays["log_emb"][:1]).float()
    log_mask = torch.from_numpy(arrays["log_mask"][:1]).float()
    metric_seg = torch.from_numpy(arrays["metric_seg"][:1]).float()
    metric_mask = torch.from_numpy(arrays["metric_mask"][:1]).float()
    torch.onnx.export(
        model,
        (log_emb, log_mask, metric_seg, metric_mask),
        str(path),
        input_names=["log_emb", "log_mask", "metric_seg", "metric_mask"],
        output_names=["logits"],
        dynamic_axes={
            "log_emb": {0: "batch"},
            "log_mask": {0: "batch"},
            "metric_seg": {0: "batch"},
            "metric_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=False,
    )


def train_variant_protocol(
    arrays_by_embedding: dict[str, dict[str, np.ndarray]],
    metric_arrays: dict[str, np.ndarray],
    target_matrix: np.ndarray,
    idx: dict[str, np.ndarray],
    cfg: ProtocolConfig,
    split_protocol: str,
    variant: str,
    seed: int,
    output_dir: Path,
) -> dict:
    eh.set_seed(seed)
    spec = eh.VARIANT_SPECS[variant]
    embedding_key = eh.resolve_log_embedding_backend(cfg.modern_log_backend, spec)
    arrays = {**arrays_by_embedding[embedding_key], **metric_arrays, "target_matrix": target_matrix.astype(np.float32)}
    device = torch.device(cfg.device)
    model = MultiHeadCrossModalHades(
        log_input_dim=arrays["log_emb"].shape[-1],
        metric_input_dim=arrays["metric_seg"].shape[-1],
        d_model=cfg.d_model,
        heads=cfg.heads,
        log_layers=cfg.log_layers,
        metric_layers=cfg.metric_layers,
        metric_encoder=spec["metric_encoder"],
        target_names=cfg.targets,
    ).to(device)
    model, train_info = train_model(arrays, idx, model, cfg, device)
    val_scores, val_rows = predict_matrix(model, arrays, idx["validation"], cfg, device)
    test_scores, test_rows = predict_matrix(model, arrays, idx["test"], cfg, device)
    thresholds = train_info.get("thresholds") or thresholds_from_validation(target_matrix, val_scores, val_rows, cfg)
    val_metrics = metrics_from_scores(target_matrix, val_scores, val_rows, thresholds, cfg, "validation", variant, seed, split_protocol)
    test_metrics = metrics_from_scores(target_matrix, test_scores, test_rows, thresholds, cfg, "test", variant, seed, split_protocol)
    torch_bench = benchmark_torch_latency(model, arrays, idx["test"], cfg, device)
    for df in [val_metrics, test_metrics]:
        df["train_monitor"] = train_info.get("train_monitor")
        df["train_monitor_target"] = train_info.get("monitor_target")
        df["train_monitor_value"] = train_info.get("best_monitor")
        df["train_selected_epoch"] = train_info.get("best_epoch")
        df["torch_latency_ms_per_sample"] = torch_bench.get("latency_ms_per_sample")

    state_path = output_dir / "models" / split_protocol.replace(":", "_") / f"{variant}_seed{seed}.pt"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "variant": variant,
            "seed": int(seed),
            "split_protocol": split_protocol,
            "target_names": cfg.targets,
            "thresholds": thresholds,
            "spec": spec,
            "config": asdict(cfg),
        },
        state_path,
    )
    for df in [val_metrics, test_metrics]:
        df["state_path"] = str(state_path)
        df["model_bytes"] = int(state_path.stat().st_size)

    export_report = {"status": "skipped", "reason": "export disabled"}
    if cfg.export_onnx:
        export_dir = output_dir / "exports" / split_protocol.replace(":", "_") / f"{variant}_seed{seed}"
        onnx_path = export_dir / "model_fp32.onnx"
        int8_path = export_dir / "model_int8.onnx"
        try:
            export_onnx(model, arrays, onnx_path)
            qcfg = SimpleNamespace(
                quantization=cfg.quantization,
                calibration_size=cfg.calibration_size,
                batch_size=cfg.batch_size,
                benchmark_batches=cfg.benchmark_batches,
            )
            q_report = eh.quantize_onnx(onnx_path, int8_path, arrays, idx["train"], qcfg)
            export_report = {
                "status": "ok",
                "split_protocol": split_protocol,
                "variant": variant,
                "seed": int(seed),
                "source_runtime": str(spec.get("runtime", "unknown")),
                "onnx_fp32": str(onnx_path),
                "onnx_fp32_bytes": eh.model_file_size(onnx_path),
                "quantization": q_report,
                "benchmarks": [eh.benchmark_onnx(onnx_path, arrays, idx["test"], qcfg)],
            }
            if int8_path.exists():
                export_report["benchmarks"].append(eh.benchmark_onnx(int8_path, arrays, idx["test"], qcfg))
        except Exception as exc:
            export_report = {"status": "failed", "split_protocol": split_protocol, "variant": variant, "seed": int(seed), "error": f"{type(exc).__name__}: {exc}"}

    return {
        "metrics": pd.concat([val_metrics, test_metrics], ignore_index=True),
        "val_pred": prediction_frame(metric_arrays["frame"], target_matrix, val_scores, val_rows, cfg, split_protocol, variant, seed, "validation"),
        "test_pred": prediction_frame(metric_arrays["frame"], target_matrix, test_scores, test_rows, cfg, split_protocol, variant, seed, "test"),
        "train_info": {"split_protocol": split_protocol, "variant": variant, "seed": int(seed), **train_info},
        "export_report": export_report,
    }


def split_summary(frame: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    rows = []
    target_matrix, _ = make_target_matrix(frame, targets)
    for split, group in frame.groupby("split", sort=False):
        idx = group.index.to_numpy(dtype=np.int64)
        row = {"split": split, "rows": int(len(group)), "runs": int(group["run_id"].nunique()), "workloads": int(group["workload"].nunique())}
        for j, target in enumerate(targets):
            y = target_matrix[idx, j]
            row[f"{target}_positives"] = int(y.sum())
            row[f"{target}_prevalence"] = float(y.mean()) if len(y) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def write_card(output_dir: Path, cfg: ProtocolConfig, metrics: pd.DataFrame, alert_metrics: pd.DataFrame) -> None:
    test = metrics.loc[metrics["split"].astype(str).eq("test")].copy()
    best = test.sort_values(["target", "f1", "average_precision"], ascending=[True, False, False]).groupby("target", as_index=False).head(5)
    cols = [
        "split_protocol",
        "variant",
        "seed",
        "target",
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
    cols = [col for col in cols if col in best.columns]
    lines = [
        f"# Облегченный мультимодальный контур предупреждения: `{cfg.run_id}`",
        "",
        "This experiment keeps the HADES-like architecture and changes the data protocol first:",
        "",
        "- split comparison: current, stratified-by-workload, leave-one-workload-out;",
        f"- metric normalization: `{cfg.normalization}`;",
        f"- metric features: `{cfg.metric_feature_mode}`;",
        "- separate heads for warning/risk/anomaly and horizons 30s/60s/120s/300s;",
        "- alert-level evaluation with consecutive windows and cooldown.",
        "",
        "## Best Test Rows By Target",
        "",
        eh.markdown_table(best[cols]) if not best.empty else "_No metrics._",
        "",
        "## Alert-Level Metrics",
        "",
        eh.markdown_table(alert_metrics.head(80)) if not alert_metrics.empty else "_No alert metrics._",
        "",
        "## Files",
        "",
        "- `comparison_metrics.csv`",
        "- `test_predictions.csv.gz`",
        "- `validation_predictions.csv.gz`",
        "- `workload_metrics.csv`",
        "- `alert_level_metrics.csv`",
        "- `feature_manifest.json`",
        "- `train_manifest.json`",
        "- `latency_and_export_report.json`",
    ]
    output_dir.joinpath("EXPERIMENT_CARD.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="final_protocol_fixed_hades")
    parser.add_argument("--root", default=".")
    parser.add_argument("--contract-path", default="dashboard/artifacts/contract/strict_intensity_contract.csv.gz")
    parser.add_argument("--output-dir", default="dashboard/artifacts/model_runs/protocol_fixed_hades")
    parser.add_argument("--split-protocols", default="current,stratified_by_workload,leave_one_workload_out")
    parser.add_argument("--leave-one-workloads", default="test")
    parser.add_argument("--variants", default="efficient_hades")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--targets", default="default")
    parser.add_argument("--alert-target", default="horizon_120s")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-log-events", type=int, default=32)
    parser.add_argument("--log-context-sec", type=int, default=120)
    parser.add_argument("--log-embedding-dim", type=int, default=128)
    parser.add_argument("--modern-log-backend", choices=["embeddinggemma"], default="embeddinggemma")
    parser.add_argument("--embeddinggemma-model", default="google/embeddinggemma-300m")
    parser.add_argument("--metric-segment-sec", type=int, default=120)
    parser.add_argument("--metric-segment-len", type=int, default=64)
    parser.add_argument("--max-metric-channels", type=int, default=64)
    parser.add_argument("--metric-nodes", default=",".join(eh.DEFAULT_NODES))
    parser.add_argument("--metric-instruments", default=",".join(eh.DEFAULT_INSTRUMENTS))
    parser.add_argument("--normalization", choices=["raw", "global_norm", "workload_norm", "workload_node_norm"], default="workload_norm")
    parser.add_argument("--metric-feature-mode", choices=["raw", "deviation", "raw_plus_deviation"], default="raw_plus_deviation")
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
    parser.add_argument("--monitor-target", default="horizon_120s")
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

    base_frame = eh.load_contract(contract_path, args.max_rows, "y_anomaly")
    leave_workloads = resolve_leave_one_workloads(base_frame, args.leave_one_workloads)
    split_protocols = []
    for item in parse_csv_list(args.split_protocols, str):
        if item == "leave_one_workload_out":
            split_protocols.extend([f"leave_one_workload_out:{workload}" for workload in leave_workloads])
        else:
            split_protocols.append(item)
    cfg = ProtocolConfig(
        run_id=args.run_id,
        root=str(root),
        contract_path=str(contract_path),
        output_dir=str(output_dir),
        split_protocols=split_protocols,
        leave_one_workloads=leave_workloads,
        variants=parse_csv_list(args.variants, str),
        seeds=parse_csv_list(args.seeds, int),
        targets=parse_target_names(args.targets),
        alert_target=args.alert_target,
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
        normalization=args.normalization,
        metric_feature_mode=args.metric_feature_mode,
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
        monitor_target=args.monitor_target,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
        device=args.device,
        export_onnx=not bool(args.skip_onnx),
        quantization=args.quantization,
        calibration_size=args.calibration_size,
        benchmark_batches=args.benchmark_batches,
    )
    unknown = [variant for variant in cfg.variants if variant not in eh.VARIANT_SPECS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Known: {sorted(eh.VARIANT_SPECS)}")
    eh.write_json(output_dir / "run_config.json", asdict(cfg))

    print(f"[1/7] Loading strict contract: {contract_path}")
    print(f"Rows: {len(base_frame)}; split protocols: {cfg.split_protocols}; targets: {cfg.targets}")

    print("[2/7] Loading raw KPI frames")
    metric_frames = load_metric_frames(
        base_frame,
        raw_root=root / "raw_data",
        nodes=cfg.metric_nodes,
        instruments=cfg.metric_instruments,
        cache_dir=output_root / "_cache" / "raw_metrics",
    )

    all_results = []
    all_val_pred = []
    all_test_pred = []
    all_exports = []
    all_train_info = []
    all_split_summaries = []
    feature_manifests = {}

    for split_protocol in cfg.split_protocols:
        print(f"[3/7] Preparing protocol: {split_protocol}")
        frame = make_protocol_frame(base_frame, split_protocol, cfg)
        frame = frame.sort_values(["run_id", "window_ts", "window_index"]).reset_index(drop=True)
        frame["row_pos"] = np.arange(len(frame), dtype=np.int64)
        idx = split_indices(frame)
        summary = split_summary(frame, cfg.targets)
        summary["split_protocol"] = split_protocol
        all_split_summaries.append(summary)
        print(summary.to_string(index=False))

        target_matrix, target_manifest = make_target_matrix(frame, cfg.targets)
        raw_metric_seg, metric_mask, selected_cols, metric_manifest = build_raw_metric_segments(
            frame,
            metric_frames,
            train_idx=idx["train"],
            max_channels=cfg.max_metric_channels,
            segment_sec=cfg.metric_segment_sec,
            segment_len=cfg.metric_segment_len,
        )
        metric_seg, metric_transform_manifest = transform_metric_segments(
            frame,
            raw_metric_seg,
            metric_mask,
            train_idx=idx["train"],
            normalization=cfg.normalization,
            feature_mode=cfg.metric_feature_mode,
        )
        metric_arrays = {
            "metric_seg": metric_seg.astype(np.float32),
            "metric_mask": metric_mask.astype(np.float32),
            "frame": frame,
        }

        print("[4/7] Building cached EmbeddingGemma log template embeddings")
        needed_embeddings = {
            eh.resolve_log_embedding_backend(cfg.modern_log_backend, eh.VARIANT_SPECS[variant])
            for variant in cfg.variants
        }
        arrays_by_embedding = {}
        log_manifests = {}
        for backend in sorted(needed_embeddings):
            log_emb, log_mask, manifest = eh.build_log_arrays(
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

        feature_manifests[split_protocol] = {
            "targets": target_manifest,
            "log": log_manifests,
            "metric": {**metric_manifest, **metric_transform_manifest, "selected_channels": selected_cols},
            "arrays": {
                "metric_seg": list(metric_seg.shape),
                "metric_mask": list(metric_mask.shape),
                "target_matrix": list(target_matrix.shape),
                "log_embeddings": {key: list(value["log_emb"].shape) for key, value in arrays_by_embedding.items()},
            },
            "architecture": "shared HADES backbone: logs->templates->cached_embeddings->Transformer; metrics->raw_segments+deviation->TS_encoder; cross_modal_attention->multi_head_classifier",
        }

        print("[5/7] Training protocol variants")
        for variant in cfg.variants:
            for seed in cfg.seeds:
                print(f"  - split_protocol={split_protocol} variant={variant} seed={seed}")
                result = train_variant_protocol(
                    arrays_by_embedding,
                    metric_arrays,
                    target_matrix,
                    idx,
                    cfg,
                    split_protocol,
                    variant,
                    seed,
                    output_dir,
                )
                all_results.append(result["metrics"])
                all_val_pred.append(result["val_pred"])
                all_test_pred.append(result["test_pred"])
                all_exports.append(result["export_report"])
                all_train_info.append(result["train_info"])

    print("[6/7] Writing reports")
    metrics = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    val_pred = pd.concat(all_val_pred, ignore_index=True) if all_val_pred else pd.DataFrame()
    test_pred = pd.concat(all_test_pred, ignore_index=True) if all_test_pred else pd.DataFrame()
    split_summary_df = pd.concat(all_split_summaries, ignore_index=True) if all_split_summaries else pd.DataFrame()
    workload_df = workload_metrics(test_pred, metrics, cfg) if not test_pred.empty and not metrics.empty else pd.DataFrame()
    alert_df = alert_level_metrics(test_pred, metrics, cfg) if not test_pred.empty and not metrics.empty else pd.DataFrame()

    metrics.to_csv(output_dir / "comparison_metrics.csv", index=False)
    val_pred.to_csv(output_dir / "validation_predictions.csv.gz", index=False, compression="gzip")
    test_pred.to_csv(output_dir / "test_predictions.csv.gz", index=False, compression="gzip")
    split_summary_df.to_csv(output_dir / "split_summary.csv", index=False)
    workload_df.to_csv(output_dir / "workload_metrics.csv", index=False)
    alert_df.to_csv(output_dir / "alert_level_metrics.csv", index=False)
    pd.DataFrame(all_train_info).to_json(output_dir / "train_manifest.json", orient="records", indent=2)
    eh.write_json(output_dir / "feature_manifest.json", feature_manifests)
    eh.write_json(output_dir / "latency_and_export_report.json", {"exports": all_exports})
    eh.write_json(
        output_dir / "metrics_summary.json",
        {
            "config": asdict(cfg),
            "metrics": metrics.to_dict("records"),
            "alert_level": alert_df.to_dict("records"),
        },
    )

    print("[7/7] Writing experiment card")
    write_card(output_dir, cfg, metrics, alert_df)
    print(f"Done. Lightweight multimodal warning contour outputs written to: {output_dir}")
    test_view = metrics.loc[metrics["split"].astype(str).eq("test")].copy()
    if not test_view.empty:
        cols = ["split_protocol", "variant", "seed", "target", "threshold_policy", "average_precision", "roc_auc", "precision", "recall", "f1", "false_positives", "missed_anomalies"]
        print(test_view[cols].sort_values(["target", "f1"], ascending=[True, False]).head(40).to_string(index=False))


if __name__ == "__main__":
    main()
