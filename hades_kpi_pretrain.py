"""
KPI-only pretraining on HADES labeled CSV data.

Purpose:
    Use hades_data/data/labeled_csv/kpi_csv as a large KPI teacher before
    fine-tuning the multimodal HADES-like model on the raw_data contract.

The script deliberately does not join drain_labeled.csv logs with KPI. The log
file has no workload/scenario/run_id columns, so a blind timestamp join would
create many missing or ambiguous rows.

Typical run:
    python hades_kpi_pretrain.py --run-id kpi_pretrain_01 --device cuda

Then fine-tune:
    python hades_like_v2_experiment.py ... --pretrained-kpi-path outputs/hades_kpi_pretrain/kpi_pretrain_01/kpi_encoder_pretrained.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


KPI_FILE_MAP = {
    "cpustat": "cpu",
    "cpu": "cpu",
    "iostat": "iostat",
    "memstat": "memstat",
    "netstat": "netstat",
    "vmstat": "vmstat",
    "jvms": "jvms",
}


@dataclass
class PretrainConfig:
    run_id: str
    root: str
    hades_kpi_root: str
    reference_contract_path: str | None
    selected_kpi_features_path: str | None
    output_dir: str
    window_sec: int
    stride_sec: int
    seq_len: int
    max_kpi_features: int
    hidden_dim: int
    heads: int
    batch_size: int
    epochs: int
    patience: int
    lr: float
    weight_decay: float
    validation_fraction: float
    validation_seed: int
    seeds: list[int]
    max_scenarios: int | None
    exclude_workloads: list[str]
    exclude_reference_test_workloads: bool
    device: str


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv_list(value: str | None, cast=str) -> list:
    if not value:
        return []
    return [cast(part.strip()) for part in str(value).split(",") if part.strip()]


def numeric_kpi_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"kpi_available", "kpi_avail_ratio", "kpi_feature_count", "kpi_recency_sec", "kpi_series_available"}
    return [c for c in frame.columns if c.startswith("kpi_") and c not in excluded]


def select_reference_kpi_features(reference_contract_path: Path | None, max_features: int) -> list[str]:
    if reference_contract_path is None or not reference_contract_path.exists():
        return []
    frame = pd.read_csv(reference_contract_path)
    cols = numeric_kpi_columns(frame)
    if not cols:
        return []
    if "split" in frame.columns:
        train_idx = frame.index[frame["split"].astype(str).str.lower().eq("train")].to_numpy()
        if len(train_idx) == 0:
            train_idx = frame.index.to_numpy()
    else:
        train_idx = frame.index.to_numpy()
    x = frame[cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    train = x.iloc[train_idx]
    variances = train.var(axis=0, skipna=True).fillna(0.0)
    if "y_risk" in frame.columns:
        y_risk = frame["y_risk"].iloc[train_idx].astype(float).to_numpy()
    else:
        y_risk = np.zeros(len(train_idx), dtype=float)
    if "y_warning" in frame.columns:
        y_warning = frame["y_warning"].iloc[train_idx].astype(float).to_numpy()
    else:
        y_warning = np.zeros(len(train_idx), dtype=float)
    scores = []
    for col in cols:
        values = train[col].to_numpy(dtype=np.float32)
        mask = np.isfinite(values)
        corr_score = 0.0
        if mask.sum() > 5 and np.nanstd(values[mask]) > 1e-8:
            corr_r = np.corrcoef(values[mask], y_risk[mask])[0, 1] if y_risk.std() > 0 else 0.0
            corr_w = np.corrcoef(values[mask], y_warning[mask])[0, 1] if y_warning.std() > 0 else 0.0
            corr_score = max(abs(float(np.nan_to_num(corr_r))), abs(float(np.nan_to_num(corr_w))))
        scores.append((col, corr_score, float(variances.get(col, 0.0))))
    scores = sorted(scores, key=lambda x: (x[1], x[2]), reverse=True)
    return [col for col, _, _ in scores[: int(max_features)]]


def load_selected_kpi_features(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidates = [
            payload.get("selected_kpi_features"),
            payload.get("kpi", {}).get("selected_kpi_features") if isinstance(payload.get("kpi"), dict) else None,
            payload.get("feature_manifest", {}).get("kpi", {}).get("selected_kpi_features")
            if isinstance(payload.get("feature_manifest"), dict)
            else None,
        ]
        for value in candidates:
            if isinstance(value, list) and value:
                return [str(item) for item in value]
        return []
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        frame = pd.read_csv(path, sep=sep)
        for col in ["selected_kpi_features", "feature", "name", "kpi_feature"]:
            if col in frame.columns:
                return frame[col].dropna().astype(str).tolist()
        return frame.iloc[:, 0].dropna().astype(str).tolist() if len(frame.columns) else []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_name(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def parse_raw_kpi_feature(name: str) -> dict:
    parts = str(name).split("__")
    if len(parts) < 5 or not parts[0].startswith("kpi_"):
        return {"feature": name, "supported": False, "reason": "unexpected_feature_name"}
    metric = parts[1]
    series = parts[2]
    stat = parts[-1]
    source_file = KPI_FILE_MAP.get(metric.lower())
    if source_file is None:
        return {"feature": name, "supported": False, "reason": f"unsupported_metric:{metric}"}
    return {
        "feature": name,
        "host": parts[0].replace("kpi_", "", 1),
        "metric": metric,
        "source_file": source_file,
        "series": series,
        "series_norm": normalize_name(series),
        "stat": stat.lower(),
        "supported": source_file in {"cpu", "iostat", "memstat", "netstat"},
    }


def default_labeled_features() -> list[str]:
    features = []
    for metric, series_list in {
        "cpu": ["user", "system", "iowait", "idle"],
        "iostat": ["rkB_s", "wkB_s", "util"],
        "memstat": ["memused", "commit"],
        "netstat": ["rxkB_s", "txkB_s"],
    }.items():
        for series in series_list:
            for stat in ["mean", "last", "std", "delta"]:
                features.append(f"kpi_labeled__{metric}__{series}__value__{stat}")
    return features


def parse_labeled_feature(name: str) -> dict:
    parts = str(name).split("__")
    if len(parts) < 5 or not parts[0].startswith("kpi_"):
        return {"feature": name, "supported": False, "reason": "unexpected_feature_name"}
    return {
        "feature": name,
        "host": parts[0].replace("kpi_", "", 1),
        "metric": parts[1],
        "source_file": parts[1],
        "series": parts[2],
        "series_norm": normalize_name(parts[2]),
        "stat": parts[-1].lower(),
        "supported": parts[1] in {"cpu", "iostat", "memstat", "netstat"},
    }


def feature_specs(selected_features: list[str]) -> list[dict]:
    specs = []
    for feature in selected_features:
        spec = parse_raw_kpi_feature(feature)
        if not spec.get("supported"):
            spec = parse_labeled_feature(feature)
        specs.append(spec)
    return specs


def scenario_dirs(kpi_root: Path, exclude_workloads: set[str], max_scenarios: int | None) -> list[Path]:
    dirs = []
    for workload_dir in sorted(kpi_root.iterdir()):
        if not workload_dir.is_dir() or workload_dir.name.startswith("."):
            continue
        if workload_dir.name in exclude_workloads:
            continue
        for scenario_dir in sorted(workload_dir.iterdir()):
            if scenario_dir.is_dir() and not scenario_dir.name.startswith("."):
                dirs.append(scenario_dir)
    if max_scenarios:
        dirs = dirs[: int(max_scenarios)]
    return dirs


def read_metric_files(scenario_dir: Path) -> tuple[dict[tuple[str, str], pd.DataFrame], dict]:
    result = {}
    summary = {"files": 0, "rows": 0, "positive_rows": 0, "series": set()}
    files = sorted(scenario_dir.glob("*-labeled.csv"))
    if not files:
        files = [
            p
            for p in sorted(scenario_dir.glob("*.csv"))
            if not p.name.endswith("-intersect.csv") and not p.name.endswith("-labeled.csv")
        ]
    for file_path in files:
        metric = file_path.name.replace("-labeled.csv", "").replace(".csv", "")
        try:
            frame = pd.read_csv(file_path, usecols=["series", "timestamp", "value", "label"])
        except Exception:
            continue
        if frame.empty:
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame["label"] = pd.to_numeric(frame["label"], errors="coerce").fillna(0).astype(int)
        frame = frame.dropna(subset=["timestamp"])
        if frame.empty:
            continue
        summary["files"] += 1
        summary["rows"] += int(len(frame))
        summary["positive_rows"] += int(frame["label"].sum())
        for series, part in frame.groupby(frame["series"].astype(str), sort=False):
            series_norm = normalize_name(series)
            result[(metric, series_norm)] = part.sort_values("timestamp").reset_index(drop=True)
            summary["series"].add(f"{metric}/{series}")
    summary["series"] = sorted(summary["series"])
    return result, summary


def aggregate_values(values: np.ndarray, stat: str) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    if stat == "mean":
        return float(np.mean(values))
    if stat == "last":
        return float(values[-1])
    if stat == "std":
        return float(np.std(values))
    if stat == "delta":
        return float(values[-1] - values[0])
    if stat == "slope":
        return float(values[-1] - values[0])
    if stat == "max":
        return float(np.max(values))
    if stat == "min":
        return float(np.min(values))
    return float(np.mean(values))


def build_windows(kpi_root: Path, specs: list[dict], cfg: PretrainConfig) -> tuple[pd.DataFrame, dict]:
    rows = []
    coverage = {spec["feature"]: {"matched_windows": 0, "total_windows": 0} for spec in specs}
    scenario_summaries = []
    dirs = scenario_dirs(kpi_root, set(cfg.exclude_workloads), cfg.max_scenarios)
    for scenario_idx, scen_dir in enumerate(dirs, start=1):
        workload = scen_dir.parent.name
        scenario = scen_dir.name
        metric_data, metric_summary = read_metric_files(scen_dir)
        if not metric_data:
            continue
        all_times = pd.concat([part["timestamp"] for part in metric_data.values()], ignore_index=True)
        start = all_times.min().floor(f"{int(cfg.stride_sec)}s")
        end = all_times.max().ceil(f"{int(cfg.stride_sec)}s")
        if pd.isna(start) or pd.isna(end) or start >= end:
            continue
        scenario_rows = 0
        for window_start in pd.date_range(start=start, end=end, freq=f"{int(cfg.stride_sec)}s", inclusive="left"):
            window_end = window_start + pd.Timedelta(seconds=int(cfg.window_sec))
            row = {
                "run_id": f"{workload}/{scenario}",
                "workload": workload,
                "scenario": scenario,
                "window_ts": window_start,
                "window_end": window_end,
                "window_index": scenario_rows,
            }
            label_rows = 0
            positive_rows = 0
            for part in metric_data.values():
                mask = part["timestamp"].ge(window_start) & part["timestamp"].lt(window_end)
                if mask.any():
                    labels = part.loc[mask, "label"].to_numpy(dtype=np.float32)
                    label_rows += int(len(labels))
                    positive_rows += int(labels.sum())
            row["kpi_label_rows"] = int(label_rows)
            row["kpi_positive_rows"] = int(positive_rows)
            row["kpi_label_density"] = float(positive_rows / label_rows) if label_rows else 0.0
            row["y_anomaly"] = int(positive_rows > 0)
            row["y_warning"] = int(positive_rows > 0)
            row["y_risk"] = int(row["kpi_label_density"] >= 0.20 or positive_rows >= 4)
            for spec in specs:
                value = np.nan
                if spec.get("supported"):
                    part = metric_data.get((spec["source_file"], spec["series_norm"]))
                    if part is not None:
                        mask = part["timestamp"].ge(window_start) & part["timestamp"].lt(window_end)
                        if mask.any():
                            value = aggregate_values(part.loc[mask, "value"].to_numpy(dtype=np.float32), spec["stat"])
                row[spec["feature"]] = value
                coverage[spec["feature"]]["total_windows"] += 1
                if np.isfinite(value):
                    coverage[spec["feature"]]["matched_windows"] += 1
            rows.append(row)
            scenario_rows += 1
        scenario_summaries.append(
            {
                "workload": workload,
                "scenario": scenario,
                "windows": int(scenario_rows),
                "files": metric_summary["files"],
                "rows": metric_summary["rows"],
                "positive_rows": metric_summary["positive_rows"],
                "series": len(metric_summary["series"]),
            }
        )
        if scenario_idx % 50 == 0:
            print(f"processed_scenarios={scenario_idx}/{len(dirs)} windows={len(rows)}")
    contract = pd.DataFrame(rows)
    coverage_rows = []
    for feature, stats in coverage.items():
        total = stats["total_windows"]
        matched = stats["matched_windows"]
        coverage_rows.append(
            {
                "feature": feature,
                "matched_windows": int(matched),
                "total_windows": int(total),
                "coverage": float(matched / total) if total else 0.0,
            }
        )
    manifest = {
        "scenarios_requested": int(len(dirs)),
        "scenarios_used": int(len(scenario_summaries)),
        "scenario_summary": scenario_summaries,
        "feature_coverage": coverage_rows,
    }
    return contract, manifest


def split_contract(frame: pd.DataFrame, validation_fraction: float, seed: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["split"] = "train"
    groups = frame["workload"].astype(str)
    if groups.nunique() < 2:
        frame.loc[frame.sample(frac=validation_fraction, random_state=seed).index, "split"] = "validation"
        return frame
    splitter = GroupShuffleSplit(n_splits=32, test_size=float(validation_fraction), random_state=int(seed))
    best_val = None
    best_score = -1.0
    for tr_idx, val_idx in splitter.split(frame, groups=groups):
        tr = frame.iloc[tr_idx]
        val = frame.iloc[val_idx]
        score = 0.0
        score += 2.0 if tr["y_anomaly"].nunique() == 2 else 0.0
        score += 3.0 if val["y_anomaly"].nunique() == 2 else 0.0
        score += min(int(val["y_anomaly"].sum()), 100) / 100.0
        score -= abs(len(val) / len(frame) - float(validation_fraction))
        if score > best_score:
            best_score = score
            best_val = val_idx
    if best_val is None:
        raise RuntimeError("Could not build validation split.")
    frame.iloc[best_val, frame.columns.get_loc("split")] = "validation"
    return frame


def build_sequences(frame: pd.DataFrame, selected_features: list[str], train_idx: np.ndarray, seq_len: int) -> tuple[dict[str, np.ndarray], dict]:
    raw = frame[selected_features].apply(pd.to_numeric, errors="coerce").astype(np.float32).to_numpy(copy=True)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
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
    arrays = {
        "kpi_seq": seq.astype(np.float32),
        "kpi_mask": mask.astype(np.float32),
        "y_anomaly": frame["y_anomaly"].astype(np.float32).to_numpy(),
        "y_warning": frame["y_warning"].astype(np.float32).to_numpy(),
        "y_risk": frame["y_risk"].astype(np.float32).to_numpy(),
        "label_density": frame["kpi_label_density"].astype(np.float32).to_numpy(),
    }
    manifest = {"shape": list(seq.shape), "seq_len": int(seq_len), "feature_count": int(f)}
    return arrays, manifest


class KpiDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: Iterable[int]):
        self.arrays = arrays
        self.indices = np.asarray(list(indices), dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = self.indices[i]
        return {
            "kpi_seq": torch.from_numpy(self.arrays["kpi_seq"][idx]),
            "kpi_mask": torch.from_numpy(self.arrays["kpi_mask"][idx]),
            "y_anomaly": torch.tensor(self.arrays["y_anomaly"][idx], dtype=torch.float32),
            "y_warning": torch.tensor(self.arrays["y_warning"][idx], dtype=torch.float32),
            "y_risk": torch.tensor(self.arrays["y_risk"][idx], dtype=torch.float32),
            "label_density": torch.tensor(self.arrays["label_density"][idx], dtype=torch.float32),
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


class KpiPretrainModel(nn.Module):
    def __init__(self, kpi_dim: int, hidden: int, heads: int, seq_len: int):
        super().__init__()
        heads = max(1, int(heads))
        while hidden % heads != 0 and heads > 1:
            heads -= 1
        self.kpi_proj = nn.Sequential(nn.Linear(kpi_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.kpi_pos = nn.Parameter(torch.zeros(1, seq_len, hidden))
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 3, dropout=0.12, batch_first=True, activation="gelu")
        self.kpi_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.kpi_pool = AttentionPool(hidden)
        self.kpi_pretrain_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.kpi_density_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    @staticmethod
    def safe_padding(mask: torch.Tensor) -> torch.Tensor:
        valid = mask > 0.5
        padding = ~valid
        no_key = ~valid.any(dim=1)
        if bool(no_key.any()):
            padding = padding.clone()
            padding[no_key, 0] = False
        return padding

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mask = batch["kpi_mask"].float()
        padding = self.safe_padding(mask)
        tokens = self.kpi_proj(batch["kpi_seq"]) + self.kpi_pos[:, : batch["kpi_seq"].shape[1], :]
        tokens = self.kpi_encoder(tokens, src_key_padding_mask=padding) * mask.unsqueeze(-1)
        pooled, attn = self.kpi_pool(tokens, mask)
        return {
            "anomaly_logit": self.kpi_pretrain_head(pooled).squeeze(1),
            "density_logit": self.kpi_density_head(pooled).squeeze(1),
            "kpi_attention": attn,
        }


def pos_weight(y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    return float(neg / pos)


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


def precision_at_fraction(y, score, fraction: float) -> float | None:
    y = np.asarray(y).astype(int)
    if len(y) == 0:
        return None
    k = max(1, int(math.ceil(len(y) * float(fraction))))
    order = np.argsort(-np.asarray(score, dtype=float))[:k]
    return float(y[order].mean())


@torch.no_grad()
def predict(model: KpiPretrainModel, dataset: KpiDataset, device: torch.device, batch_size: int) -> pd.DataFrame:
    model.eval()
    rows = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch)
        rows.append(
            pd.DataFrame(
                {
                    "row_index": batch["row_index"].detach().cpu().numpy(),
                    "anomaly_score": torch.sigmoid(out["anomaly_logit"]).detach().cpu().numpy(),
                    "density_score": torch.sigmoid(out["density_logit"]).detach().cpu().numpy(),
                    "y_anomaly": batch["y_anomaly"].detach().cpu().numpy(),
                    "y_warning": batch["y_warning"].detach().cpu().numpy(),
                    "y_risk": batch["y_risk"].detach().cpu().numpy(),
                    "label_density": batch["label_density"].detach().cpu().numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True).sort_values("row_index").reset_index(drop=True)


def train_seed(arrays: dict[str, np.ndarray], idx: dict[str, np.ndarray], cfg: PretrainConfig, seed: int, output_dir: Path) -> dict:
    set_seed(seed)
    device = torch.device(cfg.device)
    train_ds = KpiDataset(arrays, idx["train"])
    val_ds = KpiDataset(arrays, idx["validation"])
    model = KpiPretrainModel(
        kpi_dim=arrays["kpi_seq"].shape[-1],
        hidden=cfg.hidden_dim,
        heads=cfg.heads,
        seq_len=arrays["kpi_seq"].shape[1],
    ).to(device)
    weight = torch.tensor(pos_weight(arrays["y_anomaly"][idx["train"]]), dtype=torch.float32, device=device)
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
            bce = nn.functional.binary_cross_entropy_with_logits(out["anomaly_logit"], batch["y_anomaly"], pos_weight=weight)
            density = nn.functional.smooth_l1_loss(torch.sigmoid(out["density_logit"]), batch["label_density"])
            loss = bce + 0.30 * density
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_pred = predict(model, val_ds, device, cfg.batch_size)
        ap = safe_ap(val_pred["y_anomaly"], val_pred["anomaly_score"]) or 0.0
        p5 = precision_at_fraction(val_pred["y_anomaly"], val_pred["anomaly_score"], 0.05) or 0.0
        score = 0.80 * ap + 0.20 * p5
        scheduler.step(score)
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
                "val_ap_anomaly": ap,
                "val_precision_top5": p5,
                "monitor_score": score,
                "best_score": best_score,
                "patience_left": patience_left,
                "lr": float(opt.param_groups[0]["lr"]),
            }
        )
        print(f"seed={seed} epoch={epoch:03d} loss={np.mean(losses):.4f} val_ap={ap:.4f} val_p5={p5:.4f} monitor={score:.4f} best={best_score:.4f} patience={patience_left}")
        if patience_left <= 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_pred = predict(model, val_ds, device, cfg.batch_size)
    pd.DataFrame(history).to_csv(output_dir / f"history_seed{seed}.csv", index=False)
    return {
        "seed": seed,
        "model": model,
        "state_dict": best_state,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "val_pred": val_pred,
        "history": pd.DataFrame(history),
        "metrics": {
            "seed": seed,
            "best_epoch": int(best_epoch),
            "best_score": float(best_score),
            "val_ap_anomaly": safe_ap(val_pred["y_anomaly"], val_pred["anomaly_score"]),
            "val_roc_anomaly": safe_roc(val_pred["y_anomaly"], val_pred["anomaly_score"]),
            "val_precision_top5": precision_at_fraction(val_pred["y_anomaly"], val_pred["anomaly_score"], 0.05),
            "val_density_mae": float(np.mean(np.abs(val_pred["density_score"] - val_pred["label_density"]))),
        },
    }


def kpi_branch_state_dict(model_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("kpi_proj.", "kpi_encoder.", "kpi_pool.", "kpi_pos")
    return {k: v for k, v in model_state.items() if k.startswith(prefixes)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"kpi_pretrain_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--root", default=".")
    parser.add_argument("--hades-kpi-root", default="hades_data/data/labeled_csv/kpi_csv")
    parser.add_argument("--reference-contract-path", default=None)
    parser.add_argument("--selected-kpi-features-path", default=None)
    parser.add_argument("--output-dir", default="outputs/hades_kpi_pretrain")
    parser.add_argument("--window-sec", type=int, default=10)
    parser.add_argument("--stride-sec", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--max-kpi-features", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--exclude-workloads", default="")
    parser.add_argument("--exclude-reference-test-workloads", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    hades_kpi_root = (root / args.hades_kpi_root).resolve()
    reference_contract = Path(args.reference_contract_path) if args.reference_contract_path else None
    selected_kpi_features_path = Path(args.selected_kpi_features_path) if args.selected_kpi_features_path else None
    if reference_contract is None:
        for candidate in [
            root / "outputs/strict_forecast/strict_forecast_01/contracts/strict_explicit_only_gap10.csv.gz",
            root / "outputs/raw_multirate/aligned_multirate_windows.csv.gz",
            root / "outputs/raw_multirate/aligned_multirate_windows.csv",
        ]:
            if candidate.exists():
                reference_contract = candidate
                break
    exclude_workloads = parse_csv_list(args.exclude_workloads, str)
    if args.exclude_reference_test_workloads and reference_contract is not None and reference_contract.exists():
        ref = pd.read_csv(reference_contract, usecols=lambda c: c in {"workload", "split"})
        if {"workload", "split"}.issubset(ref.columns):
            test_workloads = sorted(ref.loc[ref["split"].astype(str).str.lower().eq("test"), "workload"].astype(str).unique())
            exclude_workloads = sorted(set(exclude_workloads) | set(test_workloads))
            print(f"excluding reference test workloads from pretrain: {test_workloads}")
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = PretrainConfig(
        run_id=args.run_id,
        root=str(root),
        hades_kpi_root=str(hades_kpi_root),
        reference_contract_path=str(reference_contract) if reference_contract else None,
        selected_kpi_features_path=str(selected_kpi_features_path) if selected_kpi_features_path else None,
        output_dir=str(output_dir),
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        seq_len=args.seq_len,
        max_kpi_features=args.max_kpi_features,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
        seeds=parse_csv_list(args.seeds, int),
        max_scenarios=args.max_scenarios,
        exclude_workloads=exclude_workloads,
        exclude_reference_test_workloads=bool(args.exclude_reference_test_workloads),
        device=args.device,
    )
    write_json(output_dir / "run_config.json", asdict(cfg))

    if not hades_kpi_root.exists():
        raise RuntimeError(f"HADES KPI root not found: {hades_kpi_root}")

    selected_features = load_selected_kpi_features(selected_kpi_features_path)
    feature_source = "selected_kpi_features_path" if selected_features else "reference_contract"
    if selected_features:
        selected_features = selected_features[: cfg.max_kpi_features]
    else:
        selected_features = select_reference_kpi_features(reference_contract, cfg.max_kpi_features)
    if not selected_features:
        selected_features = default_labeled_features()[: cfg.max_kpi_features]
        feature_source = "default_labeled_features"
    specs = feature_specs(selected_features)
    print(f"[1/5] Building KPI-labeled windows from {hades_kpi_root}")
    print(f"features={len(selected_features)} feature_source={feature_source}")
    contract, build_manifest = build_windows(hades_kpi_root, specs, cfg)
    if contract.empty:
        raise RuntimeError("No windows built from HADES labeled KPI data.")
    contract = split_contract(contract, cfg.validation_fraction, cfg.validation_seed)
    train_idx = contract.index[contract["split"].eq("train")].to_numpy(dtype=np.int64)
    val_idx = contract.index[contract["split"].eq("validation")].to_numpy(dtype=np.int64)
    idx = {"train": train_idx, "validation": val_idx}
    print(f"windows={len(contract)} train={len(train_idx)} validation={len(val_idx)} positives={int(contract['y_anomaly'].sum())}")

    split_summary = (
        contract.groupby("split")
        .agg(rows=("run_id", "size"), workloads=("workload", "nunique"), runs=("run_id", "nunique"), positives=("y_anomaly", "sum"), prevalence=("y_anomaly", "mean"))
        .reset_index()
    )
    workload_summary = (
        contract.groupby("workload")
        .agg(rows=("run_id", "size"), scenarios=("scenario", "nunique"), positives=("y_anomaly", "sum"), prevalence=("y_anomaly", "mean"))
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    contract.to_csv(output_dir / "hades_kpi_window_contract.csv.gz", index=False)
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    workload_summary.to_csv(output_dir / "workload_summary.csv", index=False)

    print("[2/5] Building temporal KPI sequences")
    arrays, sequence_manifest = build_sequences(contract, selected_features, train_idx, cfg.seq_len)
    feature_coverage = pd.DataFrame(build_manifest["feature_coverage"]).sort_values("coverage")
    feature_coverage.to_csv(output_dir / "feature_coverage.csv", index=False)
    write_json(
        output_dir / "feature_manifest.json",
        {
            "feature_source": feature_source,
            "selected_kpi_features": selected_features,
            "feature_specs": specs,
            "sequence": sequence_manifest,
            "build": build_manifest,
        },
    )

    print("[3/5] Training KPI encoder")
    results = []
    for seed in cfg.seeds:
        results.append(train_seed(arrays, idx, cfg, seed, output_dir))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[4/5] Aggregating validation predictions")
    metrics = pd.DataFrame([r["metrics"] for r in results])
    metrics.to_csv(output_dir / "seed_metrics.csv", index=False)
    val_pred = pd.concat([r["val_pred"].assign(seed=r["seed"]) for r in results], ignore_index=True)
    val_pred.to_csv(output_dir / "validation_seed_predictions.csv", index=False)
    ensemble = val_pred.groupby("row_index", as_index=False).mean(numeric_only=True)
    ensemble_metrics = {
        "val_ap_anomaly": safe_ap(ensemble["y_anomaly"], ensemble["anomaly_score"]),
        "val_roc_anomaly": safe_roc(ensemble["y_anomaly"], ensemble["anomaly_score"]),
        "val_precision_top5": precision_at_fraction(ensemble["y_anomaly"], ensemble["anomaly_score"], 0.05),
        "val_density_mae": float(np.mean(np.abs(ensemble["density_score"] - ensemble["label_density"]))),
    }
    ensemble.to_csv(output_dir / "validation_ensemble_predictions.csv", index=False)

    best = max(results, key=lambda r: float(r["best_score"]))
    checkpoint = {
        "state_dict": best["state_dict"],
        "kpi_branch_state_dict": kpi_branch_state_dict(best["state_dict"]),
        "config": asdict(cfg),
        "selected_kpi_features": selected_features,
        "feature_source": feature_source,
        "feature_specs": specs,
        "sequence_manifest": sequence_manifest,
        "best_seed": int(best["seed"]),
        "best_epoch": int(best["best_epoch"]),
        "best_score": float(best["best_score"]),
        "seed_metrics": metrics.to_dict("records"),
        "ensemble_metrics": ensemble_metrics,
    }
    torch.save(checkpoint, output_dir / "kpi_encoder_pretrained.pt")

    print("[5/5] Writing summary")
    summary = {
        "run_id": cfg.run_id,
        "rows": int(len(contract)),
        "workloads": int(contract["workload"].nunique()),
        "runs": int(contract["run_id"].nunique()),
        "positives": int(contract["y_anomaly"].sum()),
        "prevalence": float(contract["y_anomaly"].mean()),
        "split_summary": split_summary.to_dict("records"),
        "workload_summary": workload_summary.to_dict("records"),
        "seed_metrics": metrics.to_dict("records"),
        "ensemble_metrics": ensemble_metrics,
        "feature_source": feature_source,
        "feature_count": int(len(selected_features)),
        "feature_coverage_min": float(feature_coverage["coverage"].min()) if not feature_coverage.empty else None,
        "feature_coverage_mean": float(feature_coverage["coverage"].mean()) if not feature_coverage.empty else None,
        "checkpoint": str(output_dir / "kpi_encoder_pretrained.pt"),
    }
    write_json(output_dir / "pretrain_summary.json", summary)
    print(f"Done. Outputs written to: {output_dir}")
    print(metrics.to_string(index=False))
    print(json.dumps(ensemble_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
