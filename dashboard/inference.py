from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.contract_pipeline import APP_ROOT, ALIGNED_CONTRACT_PATH, RAW_CENTRAL_MODEL_DIR, RAW_MULTIRATE_DIR, ArtifactCheck


COMPACT_BUNDLE_FILES = [
    "central_model.pt",
    "preprocessing_artifacts.joblib",
    "central_model_metrics.json",
    "model_report.json",
    "dashboard_model_report.json",
    "baseline_comparison.csv",
    "model_card_central_fusion.md",
    "dashboard_replay_store.csv.gz",
    "central_model_predictions.csv",
    "inference_store.csv",
]

DEBUG_ARTIFACT_FILES = [
    "feature_manifest.json",
    "preprocessing_manifest.json",
    "model_config.json",
    "dry_run_report.json",
    "inference_manifest.json",
    "dashboard_replay_store.csv",
    "seed_metrics.csv",
    "calibration_bins.csv",
    "hypothesis_registry.csv",
    "hypothesis_summary.csv",
    "baseline_sanity_diagnostics.csv",
    "artifact_consistency_report.csv",
    "experiment_run_manifest.json",
]

PREDICTION_STORE_FILES = [
    "dashboard_replay_store.csv.gz",
    "dashboard_replay_store.csv",
    "inference_store.csv",
    "central_model_predictions.csv",
]

ARTIFACT_ALTERNATIVES = {
    "model_report.json": ["model_report.json", "central_model_metrics.json", "dashboard_model_report.json"],
    "dashboard_replay_store.csv.gz": ["dashboard_replay_store.csv.gz", "dashboard_replay_store.csv"],
}


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


def _first_existing(model_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        path = model_dir / name
        if path.exists():
            return path
    return None


def _read_first_json(model_dir: Path, names: list[str]) -> dict[str, Any]:
    path = _first_existing(model_dir, names)
    return _read_json(path) if path is not None else {}


def _read_first_csv(model_dir: Path, names: list[str]) -> tuple[pd.DataFrame, Path | None]:
    for name in names:
        path = model_dir / name
        frame = _read_csv(path)
        if not frame.empty:
            return frame, path
    return pd.DataFrame(), None


def _read_prediction_store(model_dir: Path) -> tuple[pd.DataFrame, Path | None]:
    return _read_first_csv(model_dir, PREDICTION_STORE_FILES)


def _ensure_cached_replay_store(model_dir: Path, model_report: dict[str, Any]) -> Path | None:
    existing = _first_existing(model_dir, ["dashboard_replay_store.csv.gz", "dashboard_replay_store.csv"])
    if existing is not None:
        return existing

    cache_dir = APP_ROOT / ".cache" / "dashboard"
    cached = _first_existing(cache_dir, ["dashboard_replay_store.csv.gz", "dashboard_replay_store.csv"])
    if cached is not None:
        return cached

    prediction_store, _ = _read_first_csv(model_dir, ["inference_store.csv", "central_model_predictions.csv"])
    baseline_comparison = _read_csv(model_dir / "baseline_comparison.csv")
    baseline_predictions, _ = _read_first_csv(RAW_MULTIRATE_DIR, ["baseline_predictions.csv.gz", "model_predictions_smoke.csv"])
    metrics_payload = model_report.get("metrics_payload", model_report) if isinstance(model_report, dict) else {}
    if prediction_store.empty or baseline_comparison.empty or not ALIGNED_CONTRACT_PATH.exists():
        return None

    try:
        from dashboard.model_report import write_dashboard_diagnostics

        timeline = pd.read_csv(ALIGNED_CONTRACT_PATH, low_memory=False)
        artifacts = write_dashboard_diagnostics(
            timeline=timeline,
            prediction_store=prediction_store,
            baseline_predictions=baseline_predictions,
            baseline_comparison=baseline_comparison,
            metrics_payload=metrics_payload if isinstance(metrics_payload, dict) else {},
            output_dir=cache_dir,
        )
        replay_path = Path(artifacts["dashboard_replay_store"])
        return replay_path if replay_path.exists() else None
    except Exception:
        return None


def inspect_inference_bundle(model_dir: Path = RAW_CENTRAL_MODEL_DIR) -> InferenceBundleStatus:
    model_report = _read_first_json(model_dir, ["model_report.json", "dashboard_model_report.json", "central_model_metrics.json"])
    cached_replay_store_path = _ensure_cached_replay_store(model_dir, model_report)
    checks: list[ArtifactCheck] = []
    for name in COMPACT_BUNDLE_FILES:
        alternatives = ARTIFACT_ALTERNATIVES.get(name, [name])
        found_path = _first_existing(model_dir, alternatives)
        if found_path is None and name == "dashboard_replay_store.csv.gz":
            found_path = cached_replay_store_path
        path = found_path or (model_dir / name)
        checks.append(
            ArtifactCheck(
                name=name,
                path=path,
                required=True,
                found=found_path is not None,
                note="Compact inference bundle" if found_path is not None else "Нужен train-run notebook 03.",
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
    dashboard_report = model_report
    dashboard_replay_store, _ = _read_first_csv(model_dir, ["dashboard_replay_store.csv.gz", "dashboard_replay_store.csv"])
    if dashboard_replay_store.empty and cached_replay_store_path is not None:
        dashboard_replay_store = _read_csv(cached_replay_store_path)
    seed_metrics = pd.DataFrame(model_report.get("seed_metrics", [])) if isinstance(model_report, dict) else pd.DataFrame()
    if seed_metrics.empty:
        seed_metrics = _read_csv(model_dir / "seed_metrics.csv")
    calibration_bins = pd.DataFrame(model_report.get("calibration_bins", [])) if isinstance(model_report, dict) else pd.DataFrame()
    if calibration_bins.empty:
        calibration_bins = _read_csv(model_dir / "calibration_bins.csv")
    prediction_store, prediction_store_path = _read_prediction_store(model_dir)
    if cached_replay_store_path is not None and (prediction_store_path is None or prediction_store_path.name != "dashboard_replay_store.csv.gz"):
        cached_store = _read_csv(cached_replay_store_path)
        if not cached_store.empty:
            prediction_store = cached_store
            prediction_store_path = cached_replay_store_path
    metrics = model_report.get("metrics_payload", model_report) if isinstance(model_report, dict) else {}

    return InferenceBundleStatus(
        model_dir=model_dir,
        ready=ready,
        mode=mode,
        missing_required=missing_required,
        artifact_checks=checks,
        metrics=metrics if isinstance(metrics, dict) else {},
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
