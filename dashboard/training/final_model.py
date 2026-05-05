from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from dashboard.export.bundle import DEFAULT_MODEL_DIR, export_protocol_fixed_bundle


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = ROOT / "dashboard" / "artifacts"
DEFAULT_CONTRACT_PATH = DEFAULT_ARTIFACT_DIR / "contract" / "strict_intensity_contract.csv.gz"
DEFAULT_RUN_ROOT = DEFAULT_ARTIFACT_DIR / "model_runs" / "protocol_fixed_hades"
DEFAULT_BUNDLE_DIR = DEFAULT_ARTIFACT_DIR / "bundle"


def _resolve(path: str | Path, default_base: Path = ROOT) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = default_base / value
    return value.resolve()


def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_train_command(args: argparse.Namespace, run_root: Path, contract_path: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "dashboard.modeling.protocol_fixed_hades",
        "--run-id",
        args.run_id,
        "--root",
        str(ROOT),
        "--contract-path",
        str(contract_path),
        "--output-dir",
        str(run_root),
        "--device",
        args.device,
        "--variants",
        args.variant,
        "--seeds",
        str(args.seed),
        "--split-protocols",
        args.split_protocol,
        "--leave-one-workloads",
        "none",
        "--targets",
        "default",
        "--monitor-target",
        args.monitor_target,
        "--alert-target",
        args.alert_target,
        "--normalization",
        args.normalization,
        "--metric-feature-mode",
        args.metric_feature_mode,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--patience",
        str(args.patience),
        "--fp-budget",
        str(args.fp_budget),
        "--validation-seed",
        str(args.validation_seed),
        "--train-monitor",
        args.train_monitor,
        "--quantization",
        args.quantization,
    ]
    if args.max_rows is not None:
        command.extend(["--max-rows", str(args.max_rows)])
    if args.skip_onnx:
        command.append("--skip-onnx")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the single production Protocol-Fixed Efficient HADES model.")
    parser.add_argument("--run-id", default="final_protocol_fixed_hades")
    parser.add_argument("--contract-path", default=str(DEFAULT_CONTRACT_PATH.relative_to(ROOT)))
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT.relative_to(ROOT)))
    parser.add_argument("--model-out-dir", default=str(DEFAULT_MODEL_DIR.relative_to(ROOT)))
    parser.add_argument("--bundle-out-dir", default=str(DEFAULT_BUNDLE_DIR.relative_to(ROOT)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-protocol", default="stratified_by_workload")
    parser.add_argument("--variant", default="modern_metric")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--normalization", choices=["raw", "global_norm", "workload_norm", "workload_node_norm"], default="workload_norm")
    parser.add_argument("--metric-feature-mode", choices=["raw", "deviation", "raw_plus_deviation"], default="raw_plus_deviation")
    parser.add_argument("--monitor-target", default="horizon_60s")
    parser.add_argument("--alert-target", default="horizon_60s")
    parser.add_argument("--train-monitor", choices=["ap", "best_f1", "fp_budget_f1"], default="fp_budget_f1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke/debug row cap passed to the core trainer.")
    parser.add_argument("--fp-budget", type=float, default=0.10)
    parser.add_argument("--validation-seed", type=int, default=43)
    parser.add_argument("--quantization", choices=["dynamic", "static", "none"], default="dynamic")
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="Only promote/export an already trained run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract_path = _resolve(args.contract_path)
    run_root = _resolve(args.run_root)
    model_out_dir = _resolve(args.model_out_dir)
    bundle_out_dir = _resolve(args.bundle_out_dir)
    protocol_run_dir = run_root / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)

    command = build_train_command(args, run_root, contract_path)
    manifest = {
        "source": "dashboard.training.final_model",
        "run_id": args.run_id,
        "contract_path": _safe_rel(contract_path),
        "run_root": _safe_rel(run_root),
        "protocol_run_dir": _safe_rel(protocol_run_dir),
        "model_out_dir": _safe_rel(model_out_dir),
        "bundle_out_dir": _safe_rel(bundle_out_dir),
        "train_command": command,
        "single_model_policy": {
            "split_protocol": args.split_protocol,
            "variant": args.variant,
            "seed": int(args.seed),
            "normalization": args.normalization,
            "metric_feature_mode": args.metric_feature_mode,
            "monitor_target": args.monitor_target,
            "alert_target": args.alert_target,
        },
    }
    _write_json(model_out_dir / "final_train_command.json", manifest)

    if not args.skip_train:
        subprocess.run(command, check=True)
    elif not protocol_run_dir.exists():
        raise FileNotFoundError(f"--skip-train requested but run dir is missing: {protocol_run_dir}")

    report = export_protocol_fixed_bundle(
        protocol_run_dir=protocol_run_dir,
        contract_path=contract_path,
        out_dir=bundle_out_dir,
        model_out_dir=model_out_dir,
        split_protocol=args.split_protocol,
        variant=args.variant,
        seed=int(args.seed),
    )
    _write_json(
        model_out_dir / "final_training_manifest.json",
        {
            **manifest,
            "export_report": {
                "model_name": report.get("model_name"),
                "model_family": report.get("model_family"),
                "runtime": report.get("runtime"),
                "primary_dashboard_target": report.get("primary_dashboard_target"),
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "protocol_run_dir": _safe_rel(protocol_run_dir),
                "model_out_dir": _safe_rel(model_out_dir),
                "bundle_out_dir": _safe_rel(bundle_out_dir),
                "runtime": report.get("runtime"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
