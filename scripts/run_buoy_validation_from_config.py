#!/usr/bin/env python3
"""Load buoy-validation options from JSON and invoke validate_buoy accordingly."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Final, Iterable

from hf_wind_resource.cli import main as cli_main


FLAG_MAP: Final[dict[str, str]] = {
    "ann_dataset": "--ann-dataset",
    "stac_config": "--stac-config",
    "stac_dataset": "--stac-dataset",
    "python_backend": "--python-backend",
    "buoy_dataset": "--buoy-dataset",
    "node_id": "--node-id",
    "output_parquet": "--output-parquet",
    "output_summary": "--output-summary",
    "direction_output": "--direction-output",
    "tolerance_minutes": "--tolerance-minutes",
    "buoy_height_method": "--buoy-height-method",
    "buoy_measurement_height": "--buoy-measurement-height",
    "buoy_target_height": "--buoy-target-height",
    "buoy_power_law_alpha": "--buoy-power-law-alpha",
    "buoy_roughness_length": "--buoy-roughness-length",
    "scatter_sample_limit": "--scatter-sample-limit",
    "resource_output_dir": "--resource-output-dir",
    "resource_power_curve_config": "--resource-power-curve-config",
    "resource_power_curve_key": "--resource-power-curve-key",
    "resource_height_config": "--resource-height-config",
    "resource_range_thresholds": "--resource-range-thresholds",
    "resource_bootstrap_summary": "--resource-bootstrap-summary",
    "resource_right_tail_surrogate": "--resource-right-tail-surrogate",
    "resource_air_density": "--resource-air-density",
    "resource_min_confidence": "--resource-min-confidence",
    "resource_min_in_range": "--resource-min-in-range",
    "resource_km_criteria_config": "--resource-km-criteria-config",
    "resource_buoy_bootstrap_replicates": "--resource-buoy-bootstrap-replicates",
    "resource_buoy_bootstrap_confidence": "--resource-buoy-bootstrap-confidence",
    "resource_buoy_bootstrap_seed": "--resource-buoy-bootstrap-seed",
    "ann_paired_bootstrap_replicates": "--ann-paired-bootstrap-replicates",
    "ann_paired_bootstrap_confidence": "--ann-paired-bootstrap-confidence",
    "ann_paired_bootstrap_seed": "--ann-paired-bootstrap-seed",
}


BOOL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "nearest_matching",
        "disable_height_correction",
        "resource_overwrite",
        "verbose",
    }
)


REQUIRED_KEYS: Final[frozenset[str]] = frozenset({"buoy_dataset", "node_id"})


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        raise TypeError("Boolean values must use the dedicated boolean keys.")
    return str(value)


def _build_cli_args(config: dict[str, Any]) -> list[str]:
    missing = REQUIRED_KEYS - config.keys()
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Missing required config key(s): {joined}")

    args: list[str] = []
    for key, flag in FLAG_MAP.items():
        if key not in config:
            continue
        value = config[key]
        if value is None:
            continue
        args.extend([flag, _stringify(value)])

    for key in BOOL_KEYS:
        value = config.get(key)
        if value:
            flag = f"--{key.replace('_', '-')}"
            args.append(flag)

    extra: Iterable[str] | None = config.get("extra_args")
    if extra:
        args.extend(extra)

    known_keys = set(FLAG_MAP) | set(BOOL_KEYS) | set(REQUIRED_KEYS) | {"extra_args"}
    unknown = set(config) - known_keys
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown config key(s): {joined}")

    return args


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run validate_buoy with arguments sourced from a JSON config file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the JSON file describing validate_buoy overrides.",
    )
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_file():
        parser.error(f"Config file not found: {config_path}")

    config = _load_config(config_path)
    cli_args = ["validate_buoy", *_build_cli_args(config)]

    return cli_main.main(cli_args)


if __name__ == "__main__":
    raise SystemExit(main())
