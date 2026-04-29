from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.contract_pipeline import APP_ROOT, RAW_CENTRAL_MODEL_DIR, ArtifactCheck


COMPACT_BUNDLE_FILES = [
    "central_model.pt",
    "preprocessing_artifacts.joblib",
    "central_model_metrics.json",
    "baseline_comparison.csv",
    "model_card_central_fusion.md",
    "dashboard_model_report.json",
    "dashboard_replay_store.csv",
    "seed_metrics.csv",
    "calibration_bins.csv",
]

DEBUG_ARTIFACT_FILES = [
    "feature_manifest.json",
    "preprocessing_manifest.json",
    "model_config.json",
    "dry_run_report.json",
    "inference_manifest.json",
    "inference_store.csv",
    "central_model_predictions.csv",
]

PREDICTION_STORE_FILES = [
    "dashboard_replay_store.csv",
    "inference_store.csv",
    "central_model_predictions.csv",
]


@dataclass
class InferenceBundleStatus:
    model_dir: Path
    ready: bool
    mode: str
    missing_required: list[str]
    artifact_checks: list[ArtifactCheck]
    metrics: dict[str, Any]
    inference_manifest: dict[str, Any]
    model_card: str
    baseline_comparison: pd.DataFrame
    prediction_store: pd.DataFrame
    prediction_store_path: Path | None
    dashboard_report: dict[str, Any]
    dashboard_replay_store: pd.DataFrame
    seed_metrics: pd.DataFrame
    calibration_bins: pd.DataFrame
    debug_metadata: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_prediction_store(model_dir: Path) -> tuple[pd.DataFrame, Path | None]:
    for name in PREDICTION_STORE_FILES:
        path = model_dir / name
        frame = _read_csv(path)
        if not frame.empty:
            return frame, path
    return pd.DataFrame(), None


def inspect_inference_bundle(model_dir: Path = RAW_CENTRAL_MODEL_DIR) -> InferenceBundleStatus:
    checks: list[ArtifactCheck] = []
    for name in COMPACT_BUNDLE_FILES:
        path = model_dir / name
        checks.append(
            ArtifactCheck(
                name=name,
                path=path,
                required=True,
                found=path.exists(),
                note="Compact inference bundle" if path.exists() else "Нужен train-run notebook 03.",
            )
        )

    for name in DEBUG_ARTIFACT_FILES:
        path = model_dir / name
        checks.append(
            ArtifactCheck(
                name=name,
                path=path,
                required=False,
                found=path.exists(),
                note="Dry-run/preprocessing readiness artifact.",
            )
        )

    missing_required = [
        check.name
        for check in checks
        if check.required and not check.found
    ]
    ready = not missing_required
    mode = "model_scoring" if ready else "preview_without_model"

    debug_metadata = {
        "feature_manifest": _read_json(model_dir / "feature_manifest.json"),
        "preprocessing_manifest": _read_json(model_dir / "preprocessing_manifest.json"),
        "model_config": _read_json(model_dir / "model_config.json"),
        "dry_run_report": _read_json(model_dir / "dry_run_report.json"),
    }
    dashboard_report = _read_json(model_dir / "dashboard_model_report.json")
    dashboard_replay_store = _read_csv(model_dir / "dashboard_replay_store.csv")
    seed_metrics = _read_csv(model_dir / "seed_metrics.csv")
    calibration_bins = _read_csv(model_dir / "calibration_bins.csv")
    prediction_store, prediction_store_path = _read_prediction_store(model_dir)

    return InferenceBundleStatus(
        model_dir=model_dir,
        ready=ready,
        mode=mode,
        missing_required=missing_required,
        artifact_checks=checks,
        metrics=_read_json(model_dir / "central_model_metrics.json"),
        inference_manifest=_read_json(model_dir / "inference_manifest.json"),
        model_card=_read_text(model_dir / "model_card_central_fusion.md"),
        baseline_comparison=_read_csv(model_dir / "baseline_comparison.csv"),
        prediction_store=prediction_store,
        prediction_store_path=prediction_store_path,
        dashboard_report=dashboard_report,
        dashboard_replay_store=dashboard_replay_store,
        seed_metrics=seed_metrics,
        calibration_bins=calibration_bins,
        debug_metadata=debug_metadata,
    )


def artifact_checks_frame(checks: list[ArtifactCheck]) -> pd.DataFrame:
    rows = []
    for check in checks:
        try:
            path_text = str(check.path.relative_to(APP_ROOT))
        except ValueError:
            path_text = str(check.path)
        rows.append(
            {
                "Артефакт": check.name,
                "Путь": path_text,
                "Обязателен для scoring": "да" if check.required else "нет",
                "Найден": "да" if check.found else "нет",
                "Комментарий": check.note,
            }
        )
    return pd.DataFrame(rows)
