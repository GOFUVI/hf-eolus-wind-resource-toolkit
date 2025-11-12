#!/usr/bin/env python3
"""Compute wind power density and reference turbine outputs per node.

This script queries the HF radar ANN inference GeoParquet snapshot using
containerised DuckDB, builds censored Weibull datasets per node, and derives
theoretical wind-resource indicators. When the censored Weibull fit is deemed
unreliable or the Kaplan–Meier fallback triggers due to heavy censoring, the
Kaplan–Meier estimator is used to obtain conservative (lower-bound) power
estimates. In both cases the routine records the air-density assumption and
applies a configurable reference power curve to obtain expected turbine
generation and capacity factors. Outputs land in
``artifacts/power_estimates/`` as CSV, JSON, and per-node diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple, MutableMapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hf_wind_resource.io import resolve_catalog_asset
from hf_wind_resource.preprocessing.censoring import RangeThresholds, load_range_thresholds
from hf_wind_resource.stats import (
    HeightCorrection,
    KaplanMeierResult,
    KaplanMeierSelectionCriteria,
    ParametricComparison,
    ParametricComparisonConfig,
    PowerCurve,
    PowerCurveEstimate,
    PowerDensityEstimate,
    build_censored_data_from_records,
    compute_power_distribution,
    format_height_note,
    load_kaplan_meier_selection_criteria,
    summarise_records_for_selection,
    evaluate_parametric_models,
)
from hf_wind_resource.stats.weibull import CensoredWeibullData, WeibullFitResult


DEFAULT_IMAGE = "duckdb/duckdb:latest"
DEFAULT_STAC_CONFIG = Path("config/stac_catalogs.json")
DEFAULT_STAC_DATASET = "sar_range_final_pivots_joined"
DEFAULT_OUTPUT_DIR = Path("artifacts/power_estimates")
DEFAULT_POWER_CURVE_CONFIG = Path("config/power_curves.json")
DEFAULT_HEIGHT_CONFIG = Path("config/power_height.json")
DEFAULT_POWER_CURVE_KEY = "reference_offshore_6mw"
DEFAULT_ENGINE = "docker"
DEFAULT_SOURCE_HEIGHT_M = 10.0
DEFAULT_ROUGHNESS_LENGTH_M = 0.0002
DEFAULT_POWER_LAW_ALPHA = 0.11

REPO_ROOT = Path(__file__).resolve().parents[1]
NODES_DIRNAME = "nodes"
SUMMARY_FILENAME = "power_estimates_summary.csv"
SUMMARY_JSON = "power_estimates_summary.json"
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class NodeReport:
    """Bundle storing row-level summary data and detailed payload."""

    summary: Mapping[str, object]
    payload: Mapping[str, object]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None, help="Direct path to the GeoParquet dataset.")
    parser.add_argument(
        "--stac-config",
        type=Path,
        default=DEFAULT_STAC_CONFIG,
        help="STAC catalog index JSON used when --dataset is omitted.",
    )
    parser.add_argument(
        "--stac-dataset",
        default=DEFAULT_STAC_DATASET,
        help="Dataset key within the STAC index (ignored when --dataset is provided).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT / "artifacts" / "empirical_metrics" / "per_node_summary.csv",
        help="CSV summary produced by generate_empirical_metrics.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for CSV/JSON artefacts.",
    )
    parser.add_argument(
        "--power-curve-config",
        type=Path,
        default=DEFAULT_POWER_CURVE_CONFIG,
        help="JSON file containing reference power curves.",
    )
    parser.add_argument(
        "--power-curve-key",
        default=DEFAULT_POWER_CURVE_KEY,
        help="Key selecting the power curve inside --power-curve-config.",
    )
    parser.add_argument(
        "--air-density",
        type=float,
        default=1.225,
        help="Air density in kg/m³ used to scale the power curve (defaults to sea-level 1.225 kg/m³).",
    )
    parser.add_argument(
        "--right-tail-surrogate",
        type=float,
        default=None,
        help="Optional surrogate wind speed for right-censored probability mass (defaults to ANN upper threshold).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum ANN range-flag confidence to treat labels as deterministic.",
    )
    parser.add_argument(
        "--min-in-range",
        type=float,
        default=500.0,
        help="Minimum in-range weight required to accept the Weibull fit as reliable.",
    )
    parser.add_argument(
        "--parametric-min-in-weight",
        type=float,
        default=200.0,
        help="Minimum in-range weight required to evaluate alternative parametric fits (log-normal, gamma).",
    )
    parser.add_argument(
        "--parametric-ks-min-weight",
        type=float,
        default=75.0,
        help="Minimum in-range weight required to compute KS diagnostics for parametric models.",
    )
    parser.add_argument(
        "--parametric-selection-metric",
        choices=("aic", "bic"),
        default="bic",
        help="Information criterion (AIC or BIC) used to select the preferred parametric model.",
    )
    parser.add_argument(
        "--disable-gamma-fit",
        action="store_true",
        help="Skip the gamma candidate when comparing parametric fits (useful if SciPy is unavailable).",
    )
    parser.add_argument(
        "--km-criteria-config",
        type=Path,
        default=None,
        help="Optional JSON overriding the Kaplan–Meier activation thresholds.",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="Docker image hosting DuckDB.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Limit the number of nodes processed (debug helper).",
    )
    parser.add_argument(
        "--source-height-m",
        type=float,
        default=None,
        help="Reference height (m) of the ANN wind speeds. Defaults to the value declared in --height-config.",
    )
    parser.add_argument(
        "--target-height-m",
        type=float,
        default=None,
        help="Target height (m). Defaults to the hub height from the power curve or the configuration file.",
    )
    parser.add_argument(
        "--height-method",
        choices=("log", "power", "none"),
        default=None,
        help="Vertical extrapolation method (log/power/none). Defaults to the configuration file.",
    )
    parser.add_argument(
        "--power-law-alpha",
        type=float,
        default=None,
        help="Exponent for the power-law profile (used when --height-method=power).",
    )
    parser.add_argument(
        "--roughness-length-m",
        type=float,
        default=None,
        help="Surface roughness length in metres for the log-law profile.",
    )
    parser.add_argument(
        "--height-config",
        type=Path,
        default=DEFAULT_HEIGHT_CONFIG,
        help="JSON file with default height-correction parameters (method, source/target heights, etc.).",
    )
    parser.add_argument(
        "--engine",
        choices=("docker", "python"),
        default=DEFAULT_ENGINE,
        help=(
            "Execution engine for DuckDB queries. "
            "`docker` invokes duckdb/duckdb:latest (default); "
            "`python` runs queries via the local DuckDB module."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = _resolve_dataset(args.dataset, catalog=args.stac_config, dataset=args.stac_dataset)
    summary_rows = _read_summary_csv(args.summary)
    if args.max_nodes is not None:
        summary_rows = summary_rows[: args.max_nodes]

    power_curve = _load_power_curve(args.power_curve_config, args.power_curve_key)
    height_defaults, height_config_path = _load_height_config(args.height_config)
    height_correction = _resolve_height_correction(args, power_curve, height_defaults)
    thresholds = load_range_thresholds()
    tail_surrogate = args.right_tail_surrogate if args.right_tail_surrogate is not None else thresholds.upper
    km_criteria = load_kaplan_meier_selection_criteria(args.km_criteria_config)
    parametric_config = ParametricComparisonConfig(
        min_in_weight=args.parametric_min_in_weight,
        ks_min_weight=args.parametric_ks_min_weight,
        selection_metric=args.parametric_selection_metric,
        enable_gamma=not args.disable_gamma_fit,
    )

    output_dir = _prepare_output_directory(args.output)
    nodes_dir = output_dir / NODES_DIRNAME
    nodes_dir.mkdir(parents=True, exist_ok=True)

    reports: List[NodeReport] = []
    seasonal_rows_all: List[Mapping[str, object]] = []
    monthly_rows_all: List[Mapping[str, object]] = []

    for row in summary_rows:
        node_id = str(row["node_id"])
        records = fetch_node_records(
            dataset,
            node_id,
            image=args.image,
            engine=args.engine,
        )
        data = build_censored_data_from_records(
            records,
            lower_threshold=thresholds.lower,
            upper_threshold=thresholds.upper,
            min_confidence=args.min_confidence,
        )

        report = _evaluate_node(
            node_id=node_id,
            summary_row=row,
            data=data,
            thresholds=thresholds,
            power_curve=power_curve,
            air_density=args.air_density,
            tail_surrogate=tail_surrogate,
            min_in_range=args.min_in_range,
            km_criteria=km_criteria,
            height=height_correction,
            parametric_config=parametric_config,
        )
        reports.append(report)

        node_path = nodes_dir / f"{node_id}.json"
        node_path.write_text(json.dumps(report.payload, indent=2, sort_keys=True), encoding="utf-8")

        seasonal_rows, monthly_rows = _aggregate_temporal_power(
            node_id=node_id,
            records=records,
            thresholds=thresholds,
            min_confidence=args.min_confidence,
            power_curve=power_curve,
            air_density=args.air_density,
            tail_surrogate=tail_surrogate,
            min_in_range=args.min_in_range,
            km_criteria=km_criteria,
            height=height_correction,
        )
        seasonal_rows_all.extend(seasonal_rows)
        monthly_rows_all.extend(monthly_rows)

    summary_path = output_dir / SUMMARY_FILENAME
    _write_summary_csv([report.summary for report in reports], summary_path)

    json_path = output_dir / SUMMARY_JSON
    json_path.write_text(
        json.dumps([report.summary for report in reports], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    metadata_path = output_dir / METADATA_FILENAME
    metadata = _build_metadata(
        dataset=dataset,
        power_curve=power_curve,
        args=args,
        thresholds=thresholds,
        tail_surrogate=tail_surrogate,
        km_criteria=km_criteria,
        height=height_correction,
        height_config_path=height_config_path,
        seasonal_path=(output_dir / "seasonal_power_summary.csv") if seasonal_rows_all else None,
        monthly_path=(output_dir / "monthly_power_timeseries.csv") if monthly_rows_all else None,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    if seasonal_rows_all:
        seasonal_path = output_dir / "seasonal_power_summary.csv"
        _write_summary_csv(seasonal_rows_all, seasonal_path)
    if monthly_rows_all:
        monthly_path = output_dir / "monthly_power_timeseries.csv"
        _write_summary_csv(monthly_rows_all, monthly_path)

    print(f"Wrote {len(reports)} node estimates to {output_dir}")


def _resolve_dataset(path: Path | None, *, catalog: Path, dataset: str) -> Path:
    if path is not None:
        return _resolve_with_root(path)

    catalog_path = _resolve_with_root(catalog)
    asset = resolve_catalog_asset(
        dataset,
        config_path=catalog_path,
        root=REPO_ROOT,
    )
    return asset.path.resolve()


def _resolve_with_root(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _prepare_output_directory(path: Path) -> Path:
    target = _resolve_with_root(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _read_summary_csv(path: Path) -> List[Mapping[str, object]]:
    resolved = _resolve_with_root(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Summary CSV not found: {resolved}")

    rows: List[Mapping[str, object]] = []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            parsed: dict[str, object] = {}
            for key, value in raw_row.items():
                parsed[key] = _parse_csv_value(value)
            rows.append(parsed)
    return rows


def _parse_csv_value(value: str | None) -> object:
    if value is None or value == "":
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_power_curve(path: Path, key: str) -> PowerCurve:
    resolved = _resolve_with_root(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Power-curve configuration not found: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in power-curve configuration: {resolved}") from exc

    if key not in payload:
        raise KeyError(f"Power-curve key '{key}' not present in {resolved}")

    entry = payload[key]
    return PowerCurve(
        name=str(entry.get("name", key)),
        speeds=tuple(float(x) for x in entry["speeds"]),
        power_kw=tuple(float(x) for x in entry["power_kw"]),
        reference_air_density=float(entry.get("reference_air_density", 1.225)),
        hub_height_m=float(entry["hub_height_m"]) if entry.get("hub_height_m") is not None else None,
        notes=tuple(str(note) for note in entry.get("notes", ())),
    )


def _load_height_config(path: Path | None) -> tuple[dict[str, object], Path | None]:
    if path is None:
        return {}, None
    resolved = _resolve_with_root(path)
    if not resolved.exists():
        return {}, resolved
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in height configuration: {resolved}") from exc
    return payload or {}, resolved


def _resolve_height_correction(
    args: argparse.Namespace,
    power_curve: PowerCurve,
    defaults: Mapping[str, object],
) -> HeightCorrection:
    defaults = dict(defaults or {})

    method_value = args.height_method if args.height_method is not None else defaults.get("method")
    method = (method_value or "log").lower()

    source_value = args.source_height_m if args.source_height_m is not None else defaults.get("source_height_m", DEFAULT_SOURCE_HEIGHT_M)
    source = float(source_value)
    if source <= 0.0:
        raise ValueError("Source height must be positive.")

    target_default = defaults.get("target_height_m")
    if target_default is None:
        target_default = power_curve.hub_height_m if power_curve.hub_height_m is not None else source
    target_value = args.target_height_m if args.target_height_m is not None else target_default
    target = float(target_value)
    if target <= 0.0:
        raise ValueError("Target height must be positive when provided.")

    if method == "none" or math.isclose(target, source, rel_tol=1e-9, abs_tol=1e-9):
        return HeightCorrection(method="none", source_height_m=source, target_height_m=target, speed_scale=1.0)

    if method == "log":
        roughness_value = args.roughness_length_m if args.roughness_length_m is not None else defaults.get("roughness_length_m", DEFAULT_ROUGHNESS_LENGTH_M)
        roughness = float(roughness_value)
        if roughness <= 0.0:
            raise ValueError("Surface roughness length must be positive for log-law corrections.")
        if source <= roughness or target <= roughness:
            raise ValueError("Source and target heights must exceed the roughness length for log-law corrections.")
        numerator = math.log(target / roughness)
        denominator = math.log(source / roughness)
        if denominator == 0.0:
            raise ValueError("Source height leads to zero denominator in log-law correction; adjust roughness or heights.")
        speed_scale = numerator / denominator
        if speed_scale <= 0.0 or not math.isfinite(speed_scale):
            raise ValueError("Computed speed scale is invalid for log-law correction.")
        return HeightCorrection(
            method="log",
            source_height_m=source,
            target_height_m=target,
            speed_scale=speed_scale,
            power_law_alpha=None,
            roughness_length_m=roughness,
        )

    if method == "power":
        alpha_value = args.power_law_alpha if args.power_law_alpha is not None else defaults.get("power_law_alpha", DEFAULT_POWER_LAW_ALPHA)
        alpha = float(alpha_value)
        speed_scale = (target / source) ** alpha
        if speed_scale <= 0.0 or not math.isfinite(speed_scale):
            raise ValueError("Computed speed scale is invalid for power-law correction.")
        return HeightCorrection(
            method="power",
            source_height_m=source,
            target_height_m=target,
            speed_scale=speed_scale,
            power_law_alpha=alpha,
            roughness_length_m=None,
        )

    raise ValueError(f"Unsupported height correction method: {method}")


def fetch_node_records(
    dataset: Path,
    node_id: str,
    *,
    image: str,
    engine: str,
) -> List[Mapping[str, object]]:
    node_literal = node_id.replace("'", "''")

    if engine == "docker":
        if shutil.which("docker") is None:
            raise RuntimeError(
                "docker CLI is not available in this environment; rerun with --engine python."
            )
        container_dataset = _dataset_inside_container(dataset)
        sql = _build_node_query(container_dataset, node_literal)
        output = _run_duckdb_docker(sql, image=image)
    else:
        dataset_literal = _escape_path_for_sql(dataset)
        sql = _build_node_query(dataset_literal, node_literal)
        output = _run_duckdb_python(sql)

    if not output:
        return []

    return _parse_records(output)


def _dataset_inside_container(dataset: Path) -> str:
    resolved = dataset.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Dataset {resolved} lies outside the repository root; mount logic expects in-repo paths."
        ) from exc
    return (Path("/workspace") / relative).as_posix()


def _resolve_host_workspace(workdir: Path) -> Path:
    host_root = os.environ.get("HOST_WORKSPACE_PATH")
    if host_root:
        return Path(host_root).resolve()
    return workdir.resolve()


def _parse_records(payload: str) -> List[Mapping[str, object]]:
    rows: List[Mapping[str, object]] = []
    reader = csv.DictReader(payload.splitlines())
    for raw in reader:
        parsed: dict[str, object] = {}
        for key, value in raw.items():
            parsed[key] = _parse_csv_value(value)
        rows.append(parsed)
    return rows


def _parse_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None


_SEASON_BY_MONTH = {
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
    12: "DJF",
}


def _season_label_and_year(ts: datetime) -> tuple[str, int]:
    season = _SEASON_BY_MONTH.get(ts.month, "UNKNOWN")
    year = ts.year
    if season == "DJF" and ts.month == 12:
        year += 1
    return season, year


def _month_period_start(year: int, month: int) -> datetime:
    return datetime(year=year, month=month, day=1, tzinfo=timezone.utc)


def _season_period_start(season: str, year: int) -> datetime:
    if season == "DJF":
        return datetime(year=year - 1, month=12, day=1, tzinfo=timezone.utc)
    if season == "MAM":
        return datetime(year=year, month=3, day=1, tzinfo=timezone.utc)
    if season == "JJA":
        return datetime(year=year, month=6, day=1, tzinfo=timezone.utc)
    if season == "SON":
        return datetime(year=year, month=9, day=1, tzinfo=timezone.utc)
    return datetime(year=year, month=1, day=1, tzinfo=timezone.utc)


def _build_node_query(dataset_literal: str, node_literal: str) -> str:
    return textwrap.dedent(
        f"""
        SELECT
            timestamp,
            pred_wind_speed,
            prob_range_below,
            prob_range_in,
            prob_range_above,
            range_flag,
            range_flag_confident
        FROM read_parquet('{dataset_literal}')
        WHERE node_id = '{node_literal}'
        """
    )


def _run_duckdb_docker(sql: str, *, image: str) -> str:
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{_resolve_host_workspace(REPO_ROOT)}:/workspace",
        "-w",
        "/workspace",
        image,
        "duckdb",
        "-csv",
        "-header",
        "-cmd",
        sql,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout


def _run_duckdb_python(sql: str) -> str:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:  # pragma: no cover - defensive fallback
        raise RuntimeError(
            "DuckDB Python module not available. Install `duckdb` or use --engine docker."
        ) from exc

    with duckdb.connect(database=":memory:") as conn:
        conn.execute("SET threads TO 4")
        result = conn.execute(sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()

    if not columns:
        return ""

    lines = [",".join(columns)]
    for row in rows:
        fields = []
        for value in row:
            fields.append("" if value is None else str(value))
        lines.append(",".join(fields))
    return "\n".join(lines)


def _escape_path_for_sql(path: Path) -> str:
    return path.as_posix().replace("'", "''")
def _evaluate_node(
    *,
    node_id: str,
    summary_row: Mapping[str, object],
    data: CensoredWeibullData,
    thresholds: RangeThresholds,
    power_curve: PowerCurve,
    air_density: float,
    tail_surrogate: float,
    min_in_range: float,
    km_criteria: KaplanMeierSelectionCriteria,
    height: HeightCorrection,
    parametric_config: ParametricComparisonConfig,
) -> NodeReport:
    notes: list[str] = []
    height_note = format_height_note(height)
    if height_note:
        notes.append(height_note)
    if data.is_empty():
        payload = {
            "node_id": node_id,
            "method": "none",
            "notes": ["No data available for this node."],
            "power_density": None,
            "power_curve_estimate": None,
            "weibull": None,
            "kaplan_meier": None,
            "parametric_models": None,
        }
        summary = {
            "node_id": node_id,
            "method": "none",
            "power_density_w_m2": None,
            "power_density_method": None,
            "power_density_notes": "No data available.",
            "air_density": air_density,
            "turbine_mean_power_kw": None,
            "capacity_factor": None,
            "power_curve_name": power_curve.name,
            "power_curve_notes": " | ".join(power_curve.notes),
            "weibull_success": False,
            "weibull_reliable": False,
            "weibull_shape": None,
            "weibull_scale": None,
            "weibull_message": "No data available.",
            "kaplan_meier_tail_probability": None,
            "kaplan_meier_reasons": "",
        }
        summary.update(_parametric_summary_fields(None))
        return NodeReport(summary=summary, payload=payload)

    (
        method,
        power_density,
        power_curve_estimate,
        weibull,
        km_result,
        selection_reasons,
        method_notes,
    ) = compute_power_distribution(
        data=data,
        summary_row=summary_row,
        power_curve=power_curve,
        air_density=air_density,
        tail_surrogate=tail_surrogate,
        min_in_range=min_in_range,
        km_criteria=km_criteria,
        height=height,
    )
    notes.extend(method_notes)

    parametric_result = evaluate_parametric_models(
        data=data,
        weibull=weibull,
        config=parametric_config,
    )

    summary = _build_summary_row(
        node_id=node_id,
        method=method,
        air_density=air_density,
        power_density=power_density,
        power_curve_estimate=power_curve_estimate,
        power_curve=power_curve,
        weibull=weibull,
        km_result=km_result,
        selection_reasons=selection_reasons,
        extra_notes=notes,
        height=height,
    )
    summary.update(_parametric_summary_fields(parametric_result))

    payload = {
        "node_id": node_id,
        "method": method,
        "air_density": air_density,
        "power_density": _power_density_to_mapping(power_density),
        "power_curve_estimate": _power_curve_estimate_to_mapping(power_curve_estimate),
        "power_curve": power_curve.to_mapping(),
        "weibull": _weibull_to_mapping(weibull),
        "kaplan_meier": _kaplan_meier_to_mapping(km_result, tail_surrogate, selection_reasons),
        "height_correction": height.to_mapping(),
        "parametric_models": _parametric_result_to_mapping(parametric_result),
        "notes": notes + list(power_density.notes) + list(power_curve_estimate.notes),
    }

    return NodeReport(summary=summary, payload=payload)


def _build_summary_row(
    *,
    node_id: str,
    method: str,
    air_density: float,
    power_density: PowerDensityEstimate,
    power_curve_estimate: PowerCurveEstimate,
    power_curve: PowerCurve,
    weibull: WeibullFitResult,
    km_result: KaplanMeierResult | None,
    selection_reasons: Tuple[str, ...],
    extra_notes: Sequence[str],
    height: HeightCorrection,
) -> Mapping[str, object]:
    notes = list(extra_notes) + list(power_density.notes) + list(power_curve_estimate.notes)
    row: dict[str, object] = {
        "node_id": node_id,
        "method": method,
        "air_density": air_density,
        "power_density_w_m2": power_density.estimate_w_per_m2,
        "power_density_method": power_density.method,
        "turbine_mean_power_kw": power_curve_estimate.mean_power_kw,
        "capacity_factor": power_curve_estimate.capacity_factor,
        "power_curve_name": power_curve.name,
        "power_curve_notes": " | ".join(power_curve.notes),
        "power_curve_reference_density": power_curve.reference_air_density,
        "power_curve_hub_height_m": power_curve.hub_height_m,
        "weibull_success": weibull.success,
        "weibull_reliable": weibull.reliable,
        "weibull_shape": weibull.shape,
        "weibull_scale": weibull.scale,
        "weibull_log_likelihood": weibull.log_likelihood,
        "weibull_in_weight": weibull.in_count,
        "weibull_left_weight": weibull.left_count,
        "weibull_right_weight": weibull.right_count,
        "weibull_iterations": weibull.diagnostics.iterations,
        "weibull_gradient_norm": weibull.diagnostics.gradient_norm,
        "weibull_last_step_size": weibull.diagnostics.last_step_size,
        "weibull_message": weibull.diagnostics.message,
        "kaplan_meier_tail_probability": (
            km_result.right_tail_probability if km_result is not None else None
        ),
        "kaplan_meier_selection_reasons": " | ".join(selection_reasons),
        "height_method": height.method,
        "height_source_m": height.source_height_m,
        "height_target_m": height.target_height_m,
        "height_speed_scale": height.speed_scale,
        "height_power_law_alpha": height.power_law_alpha,
        "height_roughness_length_m": height.roughness_length_m,
    }
    row["power_density_notes"] = " | ".join(notes)
    return row


def _power_density_to_mapping(estimate: PowerDensityEstimate) -> Mapping[str, object]:
    return {
        "method": estimate.method,
        "estimate_w_per_m2": estimate.estimate_w_per_m2,
        "air_density": estimate.air_density,
        "notes": list(estimate.notes),
    }


def _power_curve_estimate_to_mapping(estimate: PowerCurveEstimate) -> Mapping[str, object]:
    return {
        "curve_name": estimate.curve.name,
        "mean_power_kw": estimate.mean_power_kw,
        "capacity_factor": estimate.capacity_factor,
        "air_density": estimate.air_density,
        "notes": list(estimate.notes),
    }


def _weibull_to_mapping(result: WeibullFitResult) -> Mapping[str, object]:
    diagnostics = result.diagnostics
    diag_mapping = {
        "iterations": diagnostics.iterations,
        "gradient_norm": diagnostics.gradient_norm,
        "last_step_size": diagnostics.last_step_size,
        "message": diagnostics.message,
    }
    return {
        "success": result.success,
        "reliable": result.reliable,
        "shape": result.shape,
        "scale": result.scale,
        "log_likelihood": result.log_likelihood,
        "used_gradients": result.used_gradients,
        "diagnostics": diag_mapping,
        "in_weight": result.in_count,
        "left_weight": result.left_count,
        "right_weight": result.right_count,
    }


def _kaplan_meier_to_mapping(
    result: KaplanMeierResult | None,
    tail_surrogate: float,
    reasons: Tuple[str, ...],
) -> Mapping[str, object] | None:
    if result is None:
        return None
    return {
        "support": list(result.support),
        "cdf": list(result.cdf),
        "survival": list(result.survival),
        "total_weight": result.total_weight,
        "left_censored_weight": result.left_censored_weight,
        "right_censored_weight": result.right_censored_weight,
        "right_tail_probability": result.right_tail_probability,
        "tail_surrogate": tail_surrogate,
        "selection_reasons": list(reasons),
    }


_PARAMETRIC_CANDIDATES = ("weibull", "lognormal", "gamma")
_PARAMETRIC_PARAMETER_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "lognormal": ("mu", "sigma"),
    "gamma": ("shape", "scale"),
}


def _parametric_summary_fields(result: ParametricComparison | None) -> Mapping[str, object]:
    fields: dict[str, object] = {
        "parametric_selection_metric": result.selection_metric if result else None,
        "parametric_preferred_model": result.preferred_model if result else None,
        "parametric_preferred_metric_value": result.preferred_metric_value if result else None,
        "parametric_notes": " | ".join(result.notes) if result and result.notes else "",
    }

    lookup = {candidate.name.lower(): candidate for candidate in (result.candidates if result else ())}
    for name in _PARAMETRIC_CANDIDATES:
        candidate = lookup.get(name)
        fields[f"{name}_log_likelihood"] = candidate.log_likelihood if candidate else None
        fields[f"{name}_aic"] = candidate.aic if candidate else None
        fields[f"{name}_bic"] = candidate.bic if candidate else None
        fields[f"{name}_ks_statistic"] = candidate.ks_statistic if candidate else None
        fields[f"{name}_ks_pvalue"] = candidate.ks_pvalue if candidate else None
        fields[f"{name}_parametric_success"] = candidate.success if candidate else False
        fields[f"{name}_parametric_notes"] = " | ".join(candidate.notes) if candidate and candidate.notes else ""

        if name != "weibull":
            parameter_fields = _PARAMETRIC_PARAMETER_FIELDS.get(name, ())
            for param in parameter_fields:
                key = f"{name}_{param}"
                fields[key] = candidate.parameters.get(param) if candidate else None
            # Include any unexpected parameter names for completeness.
            if candidate:
                for param_name, param_value in candidate.parameters.items():
                    key = f"{name}_{param_name}"
                    if key not in fields:
                        fields[key] = param_value
    return fields


def _parametric_result_to_mapping(result: ParametricComparison | None) -> Mapping[str, object] | None:
    if result is None:
        return None
    return {
        "selection_metric": result.selection_metric,
        "preferred_model": result.preferred_model,
        "preferred_metric_value": result.preferred_metric_value,
        "notes": list(result.notes),
        "candidates": [
            {
                "name": candidate.name,
                "parameters": dict(candidate.parameters),
                "log_likelihood": candidate.log_likelihood,
                "aic": candidate.aic,
                "bic": candidate.bic,
                "ks_statistic": candidate.ks_statistic,
                "ks_pvalue": candidate.ks_pvalue,
                "success": candidate.success,
                "notes": list(candidate.notes),
            }
            for candidate in result.candidates
        ],
    }


def _aggregate_temporal_power(
    *,
    node_id: str,
    records: Sequence[Mapping[str, object]],
    thresholds: RangeThresholds,
    min_confidence: float,
    power_curve: PowerCurve,
    air_density: float,
    tail_surrogate: float,
    min_in_range: float,
    km_criteria: KaplanMeierSelectionCriteria,
    height: HeightCorrection,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    timestamps: list[tuple[datetime, Mapping[str, object]]] = []
    for record in records:
        ts = _parse_timestamp(record.get("timestamp"))
        if ts is None:
            continue
        timestamps.append((ts, record))

    if not timestamps:
        return [], []

    monthly_groups: MutableMapping[tuple[int, int], list[Mapping[str, object]]] = defaultdict(list)
    seasonal_groups: MutableMapping[tuple[int, str], list[Mapping[str, object]]] = defaultdict(list)

    for ts, record in timestamps:
        monthly_groups[(ts.year, ts.month)].append(record)
        season_label, season_year = _season_label_and_year(ts)
        seasonal_groups[(season_year, season_label)].append(record)

    monthly_rows: list[Mapping[str, object]] = []
    for (year, month) in sorted(monthly_groups.keys()):
        subset = monthly_groups[(year, month)]
        data_subset = build_censored_data_from_records(
            subset,
            lower_threshold=thresholds.lower,
            upper_threshold=thresholds.upper,
            min_confidence=min_confidence,
        )
        summary_values = summarise_records_for_selection(subset, min_confidence=min_confidence)
        summary_values["valid_count"] = data_subset.in_count
        if summary_values["total_observations"] <= 0.0 or data_subset.is_empty():
            continue

        (
            method,
            power_density,
            power_curve_estimate,
            weibull,
            km_result,
            selection_reasons,
            method_notes,
        ) = compute_power_distribution(
            data=data_subset,
            summary_row=summary_values,
            power_curve=power_curve,
            air_density=air_density,
            tail_surrogate=tail_surrogate,
            min_in_range=min_in_range,
            km_criteria=km_criteria,
            height=height,
        )

        extra_notes = []
        height_note = format_height_note(height)
        if height_note:
            extra_notes.append(height_note)
        extra_notes.extend(method_notes)
        extra_notes.append(f"Temporal aggregate: {year:04d}-{month:02d}.")

        summary_row = _build_summary_row(
            node_id=node_id,
            method=method,
            air_density=air_density,
            power_density=power_density,
            power_curve_estimate=power_curve_estimate,
            power_curve=power_curve,
            weibull=weibull,
            km_result=km_result,
            selection_reasons=selection_reasons,
            extra_notes=extra_notes,
            height=height,
        )
        period_start = _month_period_start(year, month)
        summary_row.update(
            {
                "period_type": "monthly",
                "year": year,
                "month": month,
                "period_start": period_start.isoformat(),
                "total_observations": summary_values["total_observations"],
                "censored_ratio": summary_values["censored_ratio"],
                "below_ratio": summary_values["below_ratio"],
                "in_ratio": summary_values["in_ratio"],
                "uncensored_weight": data_subset.in_count,
                "left_censored_weight": data_subset.left_count,
                "right_censored_weight": data_subset.right_count,
                "sample_count": len(subset),
            }
        )
        monthly_rows.append(summary_row)

    seasonal_rows: list[Mapping[str, object]] = []
    for (season_year, season_label) in sorted(seasonal_groups.keys()):
        subset = seasonal_groups[(season_year, season_label)]
        data_subset = build_censored_data_from_records(
            subset,
            lower_threshold=thresholds.lower,
            upper_threshold=thresholds.upper,
            min_confidence=min_confidence,
        )
        summary_values = summarise_records_for_selection(subset, min_confidence=min_confidence)
        summary_values["valid_count"] = data_subset.in_count
        if summary_values["total_observations"] <= 0.0 or data_subset.is_empty():
            continue

        (
            method,
            power_density,
            power_curve_estimate,
            weibull,
            km_result,
            selection_reasons,
            method_notes,
        ) = compute_power_distribution(
            data=data_subset,
            summary_row=summary_values,
            power_curve=power_curve,
            air_density=air_density,
            tail_surrogate=tail_surrogate,
            min_in_range=min_in_range,
            km_criteria=km_criteria,
            height=height,
        )

        extra_notes = []
        height_note = format_height_note(height)
        if height_note:
            extra_notes.append(height_note)
        extra_notes.extend(method_notes)
        extra_notes.append(f"Temporal aggregate: season {season_label} {season_year}.")

        summary_row = _build_summary_row(
            node_id=node_id,
            method=method,
            air_density=air_density,
            power_density=power_density,
            power_curve_estimate=power_curve_estimate,
            power_curve=power_curve,
            weibull=weibull,
            km_result=km_result,
            selection_reasons=selection_reasons,
            extra_notes=extra_notes,
            height=height,
        )
        period_start = _season_period_start(season_label, season_year)
        summary_row.update(
            {
                "period_type": "seasonal",
                "season": season_label,
                "season_year": season_year,
                "period_start": period_start.isoformat(),
                "total_observations": summary_values["total_observations"],
                "censored_ratio": summary_values["censored_ratio"],
                "below_ratio": summary_values["below_ratio"],
                "in_ratio": summary_values["in_ratio"],
                "uncensored_weight": data_subset.in_count,
                "left_censored_weight": data_subset.left_count,
                "right_censored_weight": data_subset.right_count,
                "sample_count": len(subset),
            }
        )
        seasonal_rows.append(summary_row)

    return seasonal_rows, monthly_rows


def _write_summary_csv(rows: Iterable[Mapping[str, object]], path: Path) -> None:
    entries = list(rows)
    if not entries:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(entries[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)


def _build_metadata(
    *,
    dataset: Path,
    power_curve: PowerCurve,
    args: argparse.Namespace,
    thresholds: RangeThresholds,
    tail_surrogate: float,
    km_criteria: KaplanMeierSelectionCriteria,
    height: HeightCorrection,
    height_config_path: Path | None,
    seasonal_path: Path | None,
    monthly_path: Path | None,
) -> Mapping[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "generated_at": now,
        "dataset": dataset.as_posix(),
        "air_density": args.air_density,
        "power_curve_key": args.power_curve_key,
        "power_curve": power_curve.to_mapping(),
        "range_thresholds": {"lower": thresholds.lower, "upper": thresholds.upper},
        "right_tail_surrogate": tail_surrogate,
        "min_confidence": args.min_confidence,
        "min_in_range": args.min_in_range,
        "height_correction": height.to_mapping(),
        "height_config_path": height_config_path.as_posix() if height_config_path is not None else None,
        "engine": args.engine,
        "km_criteria": {
            "min_total_observations": km_criteria.min_total_observations,
            "min_total_censored_ratio": km_criteria.min_total_censored_ratio,
            "min_below_ratio": km_criteria.min_below_ratio,
            "max_valid_share": km_criteria.max_valid_share,
            "min_uncensored_weight": km_criteria.min_uncensored_weight,
        },
        "parametric_evaluation": {
            "min_in_weight": args.parametric_min_in_weight,
            "ks_min_weight": args.parametric_ks_min_weight,
            "selection_metric": args.parametric_selection_metric,
            "gamma_enabled": not args.disable_gamma_fit,
        },
        "docker_image": args.image,
        "summary_source": _resolve_with_root(args.summary).as_posix(),
        "seasonal_power_summary": seasonal_path.as_posix() if seasonal_path is not None else None,
        "monthly_power_timeseries": monthly_path.as_posix() if monthly_path is not None else None,
    }


if __name__ == "__main__":
    main()
