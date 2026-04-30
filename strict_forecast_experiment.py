"""
Strict pre-anchor forecast experiment.

This script builds stricter target variants on top of notebook 02's aligned
contract. It is meant to answer a specific methodological question:

    What remains of log/KPI/fusion quality when the model must predict before
    the failure anchor, instead of seeing anchor-adjacent or post-anchor windows?

It does not replace notebooks 02/03. It writes derived contracts that can be
passed to hades_like_v2_experiment.py via --contract-path.

Typical full run:
    python strict_forecast_experiment.py --run-id strict_forecast_01 --lead-gaps 0,10,20 --random-seeds 42,43,44

Then a strict HADES-like run can use, for example:
    python hades_like_v2_experiment.py --run-id hades_v2_strict_explicit_gap10 --contract-path outputs/strict_forecast/strict_forecast_01/contracts/strict_explicit_only_gap10.csv.gz --text-backend embeddinggemma --seeds 42,43,44 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from deep_multimodal_audit import (
    evaluate_model,
    feature_groups,
    first_existing,
    json_safe,
    load_contract,
    metric_row,
    split_indices,
    write_json,
)


RANDOM_SEED = 42

MODEL_NAMES = [
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

ANCHOR_VARIANTS = {
    "explicit_only": {"explicit_raw_log_keyword_or_severity"},
    "all_anchors": {
        "explicit_raw_log_keyword_or_severity",
        "run_end_proxy",
        "run_end_proxy_no_log_events",
    },
}


@dataclass
class StrictConfig:
    run_id: str
    root: str
    output_dir: str
    contract_path: str
    lead_gaps: list[int]
    anchor_variants: list[str]
    risk_horizon_sec: int
    warning_horizon_sec: int
    max_tfidf_features: int
    random_seeds: list[int]
    models: list[str]
    protocols: list[str]


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
        rows.append(f"\n_Показаны первые {max_rows} строк из {len(df)}._")
    return "\n".join(rows)


def prepare_time_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["window_ts", "window_end", "event_anchor", "fail_start", "run_start", "run_end"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    return out


def build_strict_variant(
    frame: pd.DataFrame,
    variant_name: str,
    allowed_anchor_policies: set[str],
    lead_gap_sec: int,
    risk_horizon_sec: int,
    warning_horizon_sec: int,
) -> tuple[pd.DataFrame, dict]:
    df = prepare_time_columns(frame)
    if "event_anchor" not in df.columns:
        raise RuntimeError("Contract has no event_anchor column; cannot build strict pre-anchor target.")

    scenario_type = df["scenario_type"].astype(str).str.lower()
    anchor_policy = df["anchor_policy"].astype(str)
    is_normal = scenario_type.eq("normal") | anchor_policy.eq("normal_no_failure")
    is_failure_with_allowed_anchor = scenario_type.eq("failure") & anchor_policy.isin(allowed_anchor_policies) & df["event_anchor"].notna()

    df["strict_lead_sec"] = (df["event_anchor"] - df["window_end"]).dt.total_seconds()
    df["strict_anchor_policy_group"] = np.where(is_normal, "normal", anchor_policy)
    too_close_or_after_anchor = is_failure_with_allowed_anchor & df["strict_lead_sec"].lt(float(lead_gap_sec))
    too_far = is_failure_with_allowed_anchor & df["strict_lead_sec"].gt(float(warning_horizon_sec))

    keep = is_normal | (is_failure_with_allowed_anchor & ~too_close_or_after_anchor)
    strict = df.loc[keep].copy()

    failure_kept = strict["scenario_type"].astype(str).str.lower().eq("failure")
    lead = strict["strict_lead_sec"].astype(float)
    strict["y_risk_original"] = strict["y_risk"].astype(int)
    strict["y_warning_original"] = strict["y_warning"].astype(int)
    strict["y_risk"] = (
        failure_kept
        & lead.ge(float(lead_gap_sec))
        & lead.le(float(risk_horizon_sec))
    ).astype(int)
    strict["y_warning"] = (
        failure_kept
        & lead.ge(float(lead_gap_sec))
        & lead.le(float(warning_horizon_sec))
    ).astype(int)
    strict["y_anomaly"] = np.maximum(strict["y_warning"].to_numpy(dtype=int), strict["y_risk"].to_numpy(dtype=int))
    strict["target_policy"] = (
        f"strict_pre_anchor_{variant_name}_gap{lead_gap_sec}"
        f"_risk{risk_horizon_sec}_warning{warning_horizon_sec}"
    )
    strict["target_available"] = 1
    strict["strict_variant"] = variant_name
    strict["strict_lead_gap_sec"] = int(lead_gap_sec)
    strict["strict_risk_horizon_sec"] = int(risk_horizon_sec)
    strict["strict_warning_horizon_sec"] = int(warning_horizon_sec)

    summary = {
        "variant": variant_name,
        "lead_gap_sec": int(lead_gap_sec),
        "allowed_anchor_policies": sorted(allowed_anchor_policies),
        "input_rows": int(len(df)),
        "output_rows": int(len(strict)),
        "dropped_rows": int(len(df) - len(strict)),
        "dropped_too_close_or_after_anchor": int(too_close_or_after_anchor.sum()),
        "dropped_disallowed_or_missing_failure_anchor": int((scenario_type.eq("failure") & ~is_failure_with_allowed_anchor).sum()),
        "far_before_warning_horizon_kept_as_negative": int(too_far.sum()),
        "runs": int(strict["run_id"].nunique()),
        "workloads": int(strict["workload"].nunique()),
        "risk_positives": int(strict["y_risk"].sum()),
        "warning_positives": int(strict["y_warning"].sum()),
        "risk_prevalence": float(strict["y_risk"].mean()) if len(strict) else None,
        "warning_prevalence": float(strict["y_warning"].mean()) if len(strict) else None,
    }
    return strict.reset_index(drop=True), summary


def dataset_summary(frame: pd.DataFrame, variant: str, lead_gap: int) -> pd.DataFrame:
    rows = []
    for cols in [["split"], ["split", "anchor_policy"], ["split", "workload"], ["split", "failure_family"]]:
        for key, part in frame.groupby(cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            row = {col: value for col, value in zip(cols, key)}
            row.update(
                {
                    "variant": variant,
                    "lead_gap_sec": int(lead_gap),
                    "section": "+".join(cols),
                    "rows": int(len(part)),
                    "runs": int(part["run_id"].nunique()),
                    "workloads": int(part["workload"].nunique()),
                    "warning_positives": int(part["y_warning"].sum()),
                    "risk_positives": int(part["y_risk"].sum()),
                    "warning_prevalence": float(part["y_warning"].mean()) if len(part) else None,
                    "risk_prevalence": float(part["y_risk"].mean()) if len(part) else None,
                    "log_available": float(part["log_available"].mean()) if "log_available" in part else None,
                    "both_modalities": float(part["has_both_modalities"].mean()) if "has_both_modalities" in part else None,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_variant(
    frame: pd.DataFrame,
    groups: dict[str, list[str]],
    cfg: StrictConfig,
    variant: str,
    lead_gap: int,
) -> pd.DataFrame:
    rows = []
    protocol_seed_pairs: list[tuple[str, int | None]] = []
    if "official_workload_holdout" in cfg.protocols:
        protocol_seed_pairs.append(("official_workload_holdout", None))
    for protocol in ["random_run_holdout", "train_workload_validation"]:
        if protocol in cfg.protocols:
            protocol_seed_pairs.extend((protocol, seed) for seed in cfg.random_seeds)

    for protocol, seed in protocol_seed_pairs:
        try:
            train_idx, test_idx = split_indices(frame, protocol, seed or RANDOM_SEED)
        except Exception as exc:
            rows.append(
                {
                    "variant": variant,
                    "lead_gap_sec": int(lead_gap),
                    "protocol": protocol,
                    "seed": seed,
                    "status": f"split_error: {type(exc).__name__}: {exc}",
                }
            )
            continue
        for target in ["y_warning", "y_risk"]:
            for model_name in cfg.models:
                print(f"[strict] {variant} gap={lead_gap} protocol={protocol} seed={seed} target={target} model={model_name}")
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
                row.update({"variant": variant, "lead_gap_sec": int(lead_gap)})
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    ok = metrics.loc[metrics["status"].eq("ok")].copy()
    if ok.empty:
        return ok
    return (
        ok.groupby(["variant", "lead_gap_sec", "protocol", "target", "model"], dropna=False)
        .agg(
            ap_mean=("average_precision", "mean"),
            ap_std=("average_precision", "std"),
            roc_mean=("roc_auc", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            n_mean=("n", "mean"),
            positives_mean=("positives", "mean"),
            prevalence_mean=("prevalence", "mean"),
        )
        .reset_index()
        .sort_values(["variant", "lead_gap_sec", "target", "protocol", "ap_mean"], ascending=[True, True, True, True, False])
    )


def write_report(output_dir: Path, cfg: StrictConfig, variant_summaries: list[dict], dataset_df: pd.DataFrame, metric_summary: pd.DataFrame) -> None:
    lines = []
    lines.append(f"# Strict Forecast Experiment: `{cfg.run_id}`")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Проверяется качество до `event_anchor`: окна после anchor или ближе заданного lead gap исключаются, а targets пересчитываются заново.")
    lines.append("")
    lines.append("## Variants")
    lines.append("")
    lines.append(markdown_table(pd.DataFrame(variant_summaries)))
    lines.append("")
    if not dataset_df.empty:
        lines.append("## Dataset Summary")
        lines.append("")
        compact = dataset_df.loc[dataset_df["section"].eq("split")].copy()
        lines.append(markdown_table(compact))
        lines.append("")
    if not metric_summary.empty:
        lines.append("## Primary Metrics")
        lines.append("")
        primary = metric_summary.loc[
            metric_summary["protocol"].eq("official_workload_holdout")
            & metric_summary["target"].eq("y_risk")
        ].copy()
        lines.append(markdown_table(primary[["variant", "lead_gap_sec", "model", "ap_mean", "roc_mean", "recall_mean", "prevalence_mean"]]))
        lines.append("")
        lines.append("## Warning Metrics")
        lines.append("")
        warning = metric_summary.loc[
            metric_summary["protocol"].eq("official_workload_holdout")
            & metric_summary["target"].eq("y_warning")
        ].copy()
        lines.append(markdown_table(warning[["variant", "lead_gap_sec", "model", "ap_mean", "roc_mean", "recall_mean", "prevalence_mean"]]))
        lines.append("")
    lines.append("## Files")
    lines.append("")
    for path in sorted(output_dir.glob("*.csv")):
        lines.append(f"- `{path.name}`")
    lines.append("- `strict_config.json`")
    lines.append("- `strict_contract_paths.json`")
    lines.append("")
    (output_dir / "strict_forecast_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_csv_list(value: str, cast=str) -> list:
    return [cast(v.strip()) for v in str(value).split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"strict_forecast_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--root", default=".")
    parser.add_argument("--contract-path", default=None)
    parser.add_argument("--output-dir", default="outputs/strict_forecast")
    parser.add_argument("--lead-gaps", default="0,10,20")
    parser.add_argument("--anchor-variants", default="explicit_only,all_anchors")
    parser.add_argument("--risk-horizon-sec", type=int, default=30)
    parser.add_argument("--warning-horizon-sec", type=int, default=120)
    parser.add_argument("--max-tfidf-features", type=int, default=2048)
    parser.add_argument("--random-seeds", default="42,43,44")
    parser.add_argument("--models", default=",".join(MODEL_NAMES))
    parser.add_argument("--protocols", default="official_workload_holdout,random_run_holdout,train_workload_validation")
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
    output_dir = Path(args.output_dir) / args.run_id
    contracts_dir = output_dir / "contracts"
    output_dir.mkdir(parents=True, exist_ok=True)
    contracts_dir.mkdir(parents=True, exist_ok=True)

    anchor_variants = parse_csv_list(args.anchor_variants, str)
    unknown_variants = [v for v in anchor_variants if v not in ANCHOR_VARIANTS]
    if unknown_variants:
        raise ValueError(f"Unknown anchor variants: {unknown_variants}. Known: {sorted(ANCHOR_VARIANTS)}")
    models = parse_csv_list(args.models, str)
    unknown_models = [m for m in models if m not in MODEL_NAMES]
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}. Known: {MODEL_NAMES}")

    cfg = StrictConfig(
        run_id=args.run_id,
        root=str(root),
        output_dir=str(output_dir),
        contract_path=str(contract_path),
        lead_gaps=parse_csv_list(args.lead_gaps, int),
        anchor_variants=anchor_variants,
        risk_horizon_sec=int(args.risk_horizon_sec),
        warning_horizon_sec=int(args.warning_horizon_sec),
        max_tfidf_features=int(args.max_tfidf_features),
        random_seeds=parse_csv_list(args.random_seeds, int),
        models=models,
        protocols=parse_csv_list(args.protocols, str),
    )
    write_json(output_dir / "strict_config.json", asdict(cfg))

    print(f"[1/5] Loading base contract: {contract_path}")
    base = load_contract(contract_path)

    variant_summaries: list[dict] = []
    dataset_tables = []
    metrics_tables = []
    contract_paths = {}

    print("[2/5] Building strict contracts")
    for variant in cfg.anchor_variants:
        for gap in cfg.lead_gaps:
            strict, summary = build_strict_variant(
                base,
                variant,
                ANCHOR_VARIANTS[variant],
                gap,
                cfg.risk_horizon_sec,
                cfg.warning_horizon_sec,
            )
            variant_name = f"strict_{variant}_gap{gap}"
            path = contracts_dir / f"{variant_name}.csv.gz"
            strict.to_csv(path, index=False, compression="gzip")
            contract_paths[variant_name] = str(path)
            summary["contract_path"] = str(path)
            variant_summaries.append(summary)
            dataset_tables.append(dataset_summary(strict, variant, gap))

    write_json(output_dir / "strict_contract_paths.json", contract_paths)
    pd.DataFrame(variant_summaries).to_csv(output_dir / "strict_variant_summary.csv", index=False)
    dataset_df = pd.concat(dataset_tables, ignore_index=True) if dataset_tables else pd.DataFrame()
    dataset_df.to_csv(output_dir / "strict_dataset_summary.csv", index=False)

    print("[3/5] Evaluating strict baselines")
    for variant in cfg.anchor_variants:
        for gap in cfg.lead_gaps:
            path = Path(contract_paths[f"strict_{variant}_gap{gap}"])
            strict = load_contract(path)
            groups = feature_groups(strict)
            metrics_tables.append(evaluate_variant(strict, groups, cfg, variant, gap))

    print("[4/5] Aggregating")
    metrics = pd.concat(metrics_tables, ignore_index=True) if metrics_tables else pd.DataFrame()
    metrics.to_csv(output_dir / "strict_model_metrics.csv", index=False)
    metric_summary = summarize_metrics(metrics)
    metric_summary.to_csv(output_dir / "strict_model_summary.csv", index=False)

    print("[5/5] Writing report")
    write_report(output_dir, cfg, variant_summaries, dataset_df, metric_summary)
    print(f"Done. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
