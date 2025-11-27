"""CLI helper to prepare buoy reference datasets for ANN validation."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from hf_wind_resource.io import resolve_catalog_asset
from hf_wind_resource.preprocessing import (
    BuoySentinelConfig,
    HeightCorrectionConfig,
    load_height_correction_from_config,
    SynchronisationConfig,
    build_geoparquet_table,
    prepare_buoy_timeseries,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAC_CONFIG = Path("config/stac_catalogs.json")
DEFAULT_STAC_DATASET = "sar_range_final_pivots_joined"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a buoy GeoParquet time series, filter sentinel values, and "
            "align it with the ANN predictions for a given node."
        ),
    )
    parser.add_argument(
        "--buoy-dataset",
        type=Path,
        required=True,
        help="Path to the buoy GeoParquet asset (must expose timestamp/wind_speed/wind_dir).",
    )
    parser.add_argument(
        "--ann-dataset",
        type=Path,
        default=None,
        help=(
            "Override the ANN GeoParquet snapshot path. When omitted the asset is "
            "resolved via the STAC catalog configuration."
        ),
    )
    parser.add_argument(
        "--ann-kind",
        choices=("ann", "uncensored"),
        default="ann",
        help=(
            "Interpretation of the ANN dataset: use 'ann' for standard inference snapshots "
            "or 'uncensored' for generic wind series (e.g., interpolated models) that lack "
            "pred_* columns. Defaults to ann."
        ),
    )
    parser.add_argument(
        "--stac-config",
        type=Path,
        default=DEFAULT_STAC_CONFIG,
        help=(
            "Path to the STAC catalog index JSON. Ignored when --ann-dataset is provided. "
            "Defaults to config/stac_catalogs.json."
        ),
    )
    parser.add_argument(
        "--stac-dataset",
        default=DEFAULT_STAC_DATASET,
        help=(
            "Dataset key within the STAC catalog used to resolve the ANN snapshot. "
            "Ignored when --ann-dataset is provided."
        ),
    )
    parser.add_argument(
        "--node-id",
        required=True,
        help="ANN node identifier to extract from the GeoParquet dataset.",
    )
    parser.add_argument(
        "--tolerance-minutes",
        type=float,
        default=30.0,
        help="Maximum allowed time difference (in minutes) between ANN and buoy records.",
    )
    parser.add_argument(
        "--nearest-matching",
        action="store_true",
        help="Enable nearest-neighbour matching when exact timestamps are unavailable.",
    )
    parser.add_argument(
        "--height-method",
        choices=("power_law", "log_profile"),
        help=(
            "Vertical adjustment law applied to the buoy wind speeds before comparison. "
            "Defaults to the value declared in config/buoy_height.json (currently log_profile)."
        ),
        default=None,
    )
    parser.add_argument(
        "--measurement-height-m",
        type=float,
        help=(
            "Physical height of the buoy anemometer in metres. "
            "Defaults to the value declared in config/buoy_height.json."
        ),
        default=None,
    )
    parser.add_argument(
        "--target-height-m",
        type=float,
        help=(
            "Reference height in metres to which the buoy winds are extrapolated. "
            "Defaults to the value declared in config/buoy_height.json."
        ),
        default=None,
    )
    parser.add_argument(
        "--power-law-alpha",
        type=float,
        help=(
            "Exponent used by the power-law profile when --height-method=power_law. "
            "Defaults to the value declared in config/buoy_height.json."
        ),
        default=None,
    )
    parser.add_argument(
        "--roughness-length-m",
        type=float,
        help=(
            "Surface roughness length in metres for --height-method=log_profile. "
            "Defaults to the value declared in config/buoy_height.json."
        ),
        default=None,
    )
    parser.add_argument(
        "--disable-height-correction",
        action="store_true",
        help="Skip the vertical correction and keep the buoy wind speeds at the sensor height.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        help="Optional path to store the matched dataset in Parquet format.",
    )
    parser.add_argument(
        "--geometry-column",
        default="geometry",
        help="Name of the column containing WKB geometries (default: geometry).",
    )
    parser.add_argument(
        "--geometry-crs",
        default="EPSG:4326",
        help="CRS identifier to embed in the GeoParquet metadata (default: EPSG:4326).",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        help="Optional path to store a JSON summary with ingestion diagnostics.",
    )
    return parser.parse_args()


def build_summary_payload(result) -> dict[str, Any]:
    buoy = result.buoy
    sync = result.synchronisation
    height_summary: dict[str, Any] | None = None
    if buoy.height_correction is not None:
        height_summary = {
            "method": buoy.height_correction.method,
            "measurement_height_m": buoy.height_correction.measurement_height_m,
            "target_height_m": buoy.height_correction.target_height_m,
            "scale_factor": buoy.height_correction.scale_factor,
            "parameters": dict(buoy.height_correction.parameters),
            "hypotheses": list(buoy.height_correction.hypotheses),
        }
    return {
        "buoy": {
            "total_records": buoy.total_records,
            "records_after_speed_filter": len(buoy.dataframe),
            "dropped_speed_records": buoy.dropped_speed_records,
            "direction_sentinel_records": buoy.direction_sentinel_records,
            "coverage_start": buoy.coverage_start.isoformat() if buoy.coverage_start else None,
            "coverage_end": buoy.coverage_end.isoformat() if buoy.coverage_end else None,
            "cadence_seconds": buoy.cadence.nominal.total_seconds() if buoy.cadence.nominal else None,
            "unique_cadence_seconds": [
                interval.total_seconds() for interval in buoy.cadence.unique_intervals
            ],
            "height_correction": height_summary,
        },
        "synchronisation": {
            "matched_rows": sync.matched_rows,
            "unmatched_ann_rows": sync.unmatched_ann_rows,
            "unmatched_buoy_rows": sync.unmatched_buoy_rows,
            "exact_matches": sync.exact_matches,
            "nearest_matches": sync.nearest_matches,
            "match_ratio_ann": sync.match_ratio_ann,
            "match_ratio_buoy": sync.match_ratio_buoy,
        },
    }


def main() -> None:
    args = parse_args()

    if args.tolerance_minutes <= 0:
        raise SystemExit("--tolerance-minutes must be positive")

    sync_config = SynchronisationConfig(
        tolerance=timedelta(minutes=args.tolerance_minutes),
        prefer_nearest=args.nearest_matching,
    )

    height_config: HeightCorrectionConfig | None = None
    if not args.disable_height_correction:
        base_height = load_height_correction_from_config()
        height_config = HeightCorrectionConfig(
            method=args.height_method or base_height.method,
            measurement_height_m=(
                args.measurement_height_m
                if args.measurement_height_m is not None
                else base_height.measurement_height_m
            ),
            target_height_m=(
                args.target_height_m
                if args.target_height_m is not None
                else base_height.target_height_m
            ),
            power_law_alpha=(
                args.power_law_alpha
                if args.power_law_alpha is not None
                else base_height.power_law_alpha
            ),
            roughness_length_m=(
                args.roughness_length_m
                if args.roughness_length_m is not None
                else base_height.roughness_length_m
            ),
        )

    if args.ann_dataset is not None:
        ann_dataset_path = args.ann_dataset
    else:
        stac_config_path = args.stac_config
        if not stac_config_path.is_absolute():
            stac_config_path = (REPO_ROOT / stac_config_path).resolve()
        resolved = resolve_catalog_asset(
            args.stac_dataset,
            config_path=stac_config_path,
            root=REPO_ROOT,
        )
        ann_dataset_path = resolved.require_local_path()

    result = prepare_buoy_timeseries(
        buoy_dataset=args.buoy_dataset,
        ann_dataset=ann_dataset_path,
        node_id=args.node_id,
        height_correction_config=height_config,
        sentinel_config=BuoySentinelConfig(),
        synchronisation_config=sync_config,
        ann_dataset_kind=args.ann_kind,
    )

    print("Buoy preparation summary")
    print("-----------------------")
    print(f"Total buoy records: {result.buoy.total_records}")
    print(f"Records after speed filtering: {len(result.buoy.dataframe)}")
    print(f"Dropped speed sentinels: {result.buoy.dropped_speed_records}")
    print(f"Direction sentinels converted to NA: {result.buoy.direction_sentinel_records}")
    print(f"Coverage start: {result.buoy.coverage_start}")
    print(f"Coverage end: {result.buoy.coverage_end}")
    if result.buoy.cadence.nominal:
        print(f"Nominal cadence: {result.buoy.cadence.nominal}")
    if result.buoy.cadence.unique_intervals:
        uniq = ", ".join(f"{interval}" for interval in result.buoy.cadence.unique_intervals)
        print(f"Observed cadence intervals: {uniq}")
    if result.buoy.height_correction:
        correction = result.buoy.height_correction
        print()
        print("Height correction")
        print(f"  Method: {correction.method}")
        print(
            f"  Measurement height: {correction.measurement_height_m:.2f} m -> Target: "
            f"{correction.target_height_m:.2f} m"
        )
        print(f"  Scale factor applied: {correction.scale_factor:.6f}")
        if correction.parameters:
            params = ", ".join(f"{key}={value}" for key, value in correction.parameters)
            print(f"  Parameters: {params}")
        if correction.hypotheses:
            hyp = "; ".join(correction.hypotheses)
            print(f"  Hypotheses: {hyp}")
    print()
    print(f"Matched rows: {result.synchronisation.matched_rows}")
    print(f"Unmatched ANN rows: {result.synchronisation.unmatched_ann_rows}")
    print(f"Unmatched buoy rows (post-filter): {result.synchronisation.unmatched_buoy_rows}")
    print(f"Exact matches: {result.synchronisation.exact_matches}")
    print(f"Nearest matches: {result.synchronisation.nearest_matches}")
    if result.synchronisation.match_ratio_ann is not None:
        print(f"Match ratio (ANN): {result.synchronisation.match_ratio_ann:.3f}")
    if result.synchronisation.match_ratio_buoy is not None:
        print(f"Match ratio (buoy): {result.synchronisation.match_ratio_buoy:.3f}")

    if args.output_parquet:
        args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
        table = build_geoparquet_table(
            result.matched_dataframe,
            geometry_column=args.geometry_column,
            crs=args.geometry_crs,
        )
        pq.write_table(table, args.output_parquet)
        print(f"\nSaved GeoParquet dataset to {args.output_parquet}")

    if args.output_summary:
        payload = build_summary_payload(result)
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(json.dumps(payload, indent=2))
        print(f"Summary JSON written to {args.output_summary}")


if __name__ == "__main__":
    main()
