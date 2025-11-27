#!/usr/bin/env python3
"""Compute wind-resource metrics from uncensored wind-speed datasets.

This helper treats the input GeoParquet as buoy-like observations (no
range censoring), but allows callers to inject an RMSE to propagate
measurement uncertainty through the bootstrap pipeline. All parameters are
controlled via a JSON configuration file to remain consistent with the
rest of the toolkit.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hf_wind_resource.io import resolve_catalog_asset
from hf_wind_resource.stats import (
    GlobalRmseProvider,
    GlobalRmseRecord,
    HeightCorrection,
    NodeBootstrapInput,
    PowerCurve,
    StratifiedBootstrapConfig,
    compute_stratified_bootstrap_uncertainty,
    load_kaplan_meier_selection_criteria,
)
from hf_wind_resource.preprocessing.censoring import load_range_thresholds


DEFAULT_CONFIG = Path("config/uncensored_resource.json")
DEFAULT_STAC_CONFIG = Path("config/stac_catalogs.json")
DEFAULT_ASSET_KEY = "data"
DEFAULT_SPEED_COLUMN = "wind_speed"
DEFAULT_NODE_COLUMN = "node_id"
DEFAULT_TIMESTAMP_COLUMN = "timestamp"
DEFAULT_OUTPUT_DIR = Path("artifacts/uncensored_resource")
DEFAULT_POWER_CURVE_CONFIG = Path("config/power_curves.json")
DEFAULT_POWER_CURVE_KEY = "reference_offshore_6mw"


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload: {path}") from exc


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("Configuration must be a JSON object.")
    return payload


def _resolve_dataset(args: argparse.Namespace, config: Mapping[str, object], repo_root: Path) -> Path:
    if args.dataset is not None:
        return (repo_root / args.dataset).resolve() if not args.dataset.is_absolute() else args.dataset.resolve()

    dataset_key = args.stac_dataset or config.get("stac_dataset")
    if dataset_key is None:
        raise ValueError("Either --dataset or stac_dataset in the config must be provided.")

    stac_config = args.stac_config or Path(config.get("stac_config", DEFAULT_STAC_CONFIG))
    asset_key = args.asset_key or config.get("asset_key") or DEFAULT_ASSET_KEY
    resolved = resolve_catalog_asset(
        str(dataset_key),
        config_path=stac_config,
        root=repo_root,
        asset_key=str(asset_key),
    )
    return resolved.require_local_path()


def _load_power_curve(path: Path | None, key: str) -> PowerCurve:
    target = path or DEFAULT_POWER_CURVE_CONFIG
    target = target if target.is_absolute() else target.resolve()
    payload = _load_json(target)
    if key not in payload:
        raise KeyError(f"Power-curve key '{key}' not present in {target}")
    entry = payload[key]
    return PowerCurve(
        name=str(entry.get("name", key)),
        speeds=tuple(float(x) for x in entry["speeds"]),
        power_kw=tuple(float(x) for x in entry["power_kw"]),
        reference_air_density=float(entry.get("reference_air_density", 1.225)),
        hub_height_m=float(entry["hub_height_m"]) if entry.get("hub_height_m") is not None else None,
        notes=tuple(str(note) for note in entry.get("notes", ())),
    )


def _build_height_correction(params: Mapping[str, object] | None, power_curve: PowerCurve) -> HeightCorrection:
    params = dict(params or {})
    method = str(params.get("method", "none")).lower()
    source = float(params.get("source_height_m", params.get("measurement_height_m", power_curve.hub_height_m or 10.0)))
    target_default = power_curve.hub_height_m if power_curve.hub_height_m is not None else source
    target = float(params.get("target_height_m", target_default))

    if method == "none" or math.isclose(source, target, rel_tol=1e-9, abs_tol=1e-9):
        speed_scale = float(params.get("speed_scale", 1.0))
        return HeightCorrection(
            method="none",
            source_height_m=source,
            target_height_m=target,
            speed_scale=speed_scale,
        )

    if method == "power":
        alpha = float(params.get("power_law_alpha", 0.11))
        if alpha <= 0.0:
            raise ValueError("power_law_alpha must be positive for power-law height correction.")
        if source <= 0.0 or target <= 0.0:
            raise ValueError("source_height_m and target_height_m must be positive for power-law height correction.")
        speed_scale = (target / source) ** alpha
        return HeightCorrection(
            method="power",
            source_height_m=source,
            target_height_m=target,
            speed_scale=speed_scale,
            power_law_alpha=alpha,
        )

    if method == "log":
        roughness = float(params.get("roughness_length_m", 0.0002))
        if roughness <= 0.0:
            raise ValueError("roughness_length_m must be positive for log-law height correction.")
        if source <= roughness or target <= roughness:
            raise ValueError("Heights must exceed roughness_length_m for log-law correction.")
        numerator = math.log(target / roughness)
        denominator = math.log(source / roughness)
        if denominator == 0.0:
            raise ValueError("Invalid log-law configuration: denominator equals zero.")
        speed_scale = numerator / denominator
        return HeightCorrection(
            method="log",
            source_height_m=source,
            target_height_m=target,
            speed_scale=speed_scale,
            roughness_length_m=roughness,
        )

    raise ValueError(f"Unsupported height-correction method: {method}")


def _build_rmse_provider(value: float, source: str | None) -> GlobalRmseProvider:
    note = "User-provided RMSE for uncensored resource estimation."

    def loader() -> Sequence[GlobalRmseRecord]:
        now = datetime.now(timezone.utc)
        record = GlobalRmseRecord(
            version="custom",
            value=float(value),
            unit="m/s",
            effective_from=now,
            effective_until=None,
            source=source or "user_provided",
            computed_at=now,
            notes=(note,),
        )
        return (record,)

    return GlobalRmseProvider(loader=loader)


def _build_records(
    frame: pd.DataFrame,
    *,
    node_column: str,
    speed_column: str,
    timestamp_column: str,
) -> Mapping[str, list[Mapping[str, object]]]:
    frame = frame[[node_column, speed_column, timestamp_column]].copy()
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    frame[speed_column] = pd.to_numeric(frame[speed_column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=[node_column, speed_column, timestamp_column])
    frame = frame[frame[speed_column] > 0.0]

    grouped = frame.groupby(node_column)
    records: dict[str, list[Mapping[str, object]]] = {}
    for node_id, node_frame in grouped:
        node_records: list[Mapping[str, object]] = []
        for _, row in node_frame.iterrows():
            timestamp = row[timestamp_column].to_pydatetime()
            node_records.append(
                {
                    "timestamp": timestamp,
                    "pred_wind_speed": float(row[speed_column]),
                    "prob_range_below": 0.0,
                    "prob_range_in": 1.0,
                    "prob_range_above": 0.0,
                    "range_flag": "in",
                    "range_flag_confident": True,
                }
            )
        records[str(node_id)] = node_records
    return records


def _serialise_result(node_id: str, result) -> Mapping[str, object]:
    row: dict[str, object] = {
        "node_id": node_id,
        "samples": result.total_samples,
        "rmse_value": result.rmse_record.value,
        "rmse_unit": result.rmse_record.unit,
        "rmse_version": result.rmse_record.version,
    }
    for metric, ci in result.metrics.items():
        row[f"{metric}_estimate"] = ci.estimate
        row[f"{metric}_lower"] = ci.lower
        row[f"{metric}_upper"] = ci.upper
    return row


def _default_json(value: object) -> object:
    if isinstance(value, (datetime, )):
        return value.isoformat()
    return str(value)


def _write_outputs(
    output_dir: Path,
    *,
    summaries: Sequence[Mapping[str, object]],
    detailed: Mapping[str, Mapping[str, object]],
    config_path: Path,
    dataset_path: Path,
    config_payload: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "uncensored_resource_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    detail_path = output_dir / "uncensored_resource_details.json"
    detail_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "dataset_path": str(dataset_path),
        "config": config_payload,
        "per_node": detailed,
    }
    detail_path.write_text(json.dumps(detail_payload, indent=2, default=_default_json), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to the JSON configuration file.")
    parser.add_argument("--dataset", type=Path, help="Optional direct path to the GeoParquet dataset.")
    parser.add_argument("--stac-config", type=Path, default=DEFAULT_STAC_CONFIG, help="STAC index to resolve the dataset.")
    parser.add_argument("--stac-dataset", help="Dataset key in the STAC index (overrides config file).")
    parser.add_argument("--asset-key", help="Asset key inside the STAC item (default: config value or 'data').")
    parser.add_argument("--output-dir", type=Path, help="Destination directory for outputs (overrides config).")
    parser.add_argument("--rmse", type=float, help="User-provided RMSE (m/s) to inject in the bootstrap.")
    parser.add_argument("--rmse-source", help="Optional source string for the RMSE record.")
    parser.add_argument("--speed-column", default=DEFAULT_SPEED_COLUMN, help="Name of the wind-speed column.")
    parser.add_argument("--node-column", default=DEFAULT_NODE_COLUMN, help="Name of the node identifier column.")
    parser.add_argument("--timestamp-column", default=DEFAULT_TIMESTAMP_COLUMN, help="Name of the timestamp column.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    config = _load_config(args.config)

    dataset_path = _resolve_dataset(args, config, repo_root)
    output_dir = args.output_dir or Path(config.get("output_dir", DEFAULT_OUTPUT_DIR))
    output_dir = output_dir if output_dir.is_absolute() else (repo_root / output_dir)

    rmse_value = args.rmse if args.rmse is not None else config.get("rmse_m_s")
    if rmse_value is None:
        raise ValueError("RMSE value must be provided either via --rmse or in the configuration (rmse_m_s).")

    speed_column = args.speed_column or config.get("speed_column") or DEFAULT_SPEED_COLUMN
    node_column = args.node_column or config.get("node_column") or DEFAULT_NODE_COLUMN
    timestamp_column = args.timestamp_column or config.get("timestamp_column") or DEFAULT_TIMESTAMP_COLUMN

    power_curve_key = config.get("power_curve_key", DEFAULT_POWER_CURVE_KEY)
    power_curve_config = Path(config.get("power_curve_config", DEFAULT_POWER_CURVE_CONFIG))
    power_curve = _load_power_curve(power_curve_config, key=str(power_curve_key))

    height_params = config.get("height") or {}
    height = _build_height_correction(height_params, power_curve)

    thresholds = load_range_thresholds(config.get("range_thresholds"))
    km_criteria_config = config.get("km_criteria_config")
    km_criteria = load_kaplan_meier_selection_criteria(Path(km_criteria_config)) if km_criteria_config else load_kaplan_meier_selection_criteria()

    bootstrap_cfg_raw = config.get("bootstrap", {})
    bootstrap_cfg = StratifiedBootstrapConfig(
        replicas=int(bootstrap_cfg_raw.get("replicas", 200)),
        confidence_level=float(bootstrap_cfg_raw.get("confidence_level", 0.95)),
        random_seed=(int(bootstrap_cfg_raw["random_seed"]) if bootstrap_cfg_raw.get("random_seed") is not None else None),
        apply_rmse_noise=bool(bootstrap_cfg_raw.get("apply_rmse_noise", True)),
        rmse_mode=str(bootstrap_cfg_raw.get("rmse_mode", "velocity")),
        ci_method=str(bootstrap_cfg_raw.get("ci_method", "percentile")),
        jackknife_max_samples=int(bootstrap_cfg_raw.get("jackknife_max_samples", 200)),
        label_strategy=str(bootstrap_cfg_raw.get("label_strategy", "fixed")),
        resampling_mode=str(bootstrap_cfg_raw.get("resampling_mode", "iid")),
        block_length=int(bootstrap_cfg_raw.get("block_length", 1)),
        air_density=float(config.get("air_density", bootstrap_cfg_raw.get("air_density", 1.225))),
        lower_threshold=float(bootstrap_cfg_raw.get("lower_threshold", thresholds.lower)),
        upper_threshold=float(bootstrap_cfg_raw.get("upper_threshold", thresholds.upper)),
        min_confidence=float(bootstrap_cfg_raw.get("min_confidence", 0.5)),
        min_in_range_weight=float(bootstrap_cfg_raw.get("min_in_range_weight", config.get("min_in_range_weight", 500.0))),
        tail_surrogate=bootstrap_cfg_raw.get("tail_surrogate", config.get("tail_surrogate", thresholds.upper)),
        noise_truncation_multiplier=float(bootstrap_cfg_raw.get("noise_truncation_multiplier", 4.0)),
        power_curve=power_curve,
        km_criteria=km_criteria,
    )

    rmse_provider = _build_rmse_provider(float(rmse_value), args.rmse_source or config.get("rmse_source"))

    frame = pd.read_parquet(dataset_path)
    node_records = _build_records(
        frame,
        node_column=node_column,
        speed_column=speed_column,
        timestamp_column=timestamp_column,
    )

    summaries = []
    detailed: dict[str, Mapping[str, object]] = {}
    for node_id, records in node_records.items():
        data = NodeBootstrapInput(node_id=node_id, records=records, height=height)
        result = compute_stratified_bootstrap_uncertainty(
            data,
            config=bootstrap_cfg,
            rmse_provider=rmse_provider,
        )
        summaries.append(_serialise_result(node_id, result))
        detailed[node_id] = {
            "metrics": {metric: asdict(ci) for metric, ci in result.metrics.items()},
            "bootstrap_means": result.bootstrap_means,
            "label_counts": result.label_counts,
            "label_proportions": result.label_proportions,
            "rmse_record": asdict(result.rmse_record),
            "notes": result.notes,
            "power_diagnostics": asdict(result.power_diagnostics) if result.power_diagnostics is not None else None,
        }

    _write_outputs(
        output_dir,
        summaries=summaries,
        detailed=detailed,
        config_path=args.config,
        dataset_path=dataset_path,
        config_payload=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
