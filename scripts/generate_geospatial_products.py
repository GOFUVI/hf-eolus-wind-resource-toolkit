#!/usr/bin/env python3
"""Generate geospatial artefacts and lightweight visualisations for wind resource outputs.

This helper ingests the per-node summary table produced by
``scripts/generate_node_summary_table.py``, resolves the ANN GeoParquet snapshot via STAC,
and emits:

* ``node_resource_map.csv`` – tabular export with lon/lat and the core indicators.
* ``node_resource_map.parquet`` – GeoParquet counterpart created through DuckDB.
* ``node_resource_map.geojson`` – Feature collection for GIS/web tooling.
* ``power_density_map.svg`` – Single-panel map highlighting power density.
* ``uncertainty_map.svg`` – Single-panel map highlighting the bootstrap confidence width of power density.
* ``wind_rose_panels.svg`` – Representative wind roses.
* ``wind_rose_histogram.csv`` – Directional histogram backing the roses.
* ``metadata.json`` – Provenance record linking inputs and generated outputs.

All heavy queries execute inside ``duckdb/duckdb:latest`` so we never read the GeoParquet
directly from Python, keeping the workflow aligned with project constraints.
"""

from __future__ import annotations

import argparse
import codecs
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from hf_wind_resource.io import resolve_ann_asset, resolve_catalog_asset
    from hf_wind_resource.stats.empirical import load_taxonomy_records
except ImportError as exc:  # pragma: no cover - defensive guard for missing package
    raise RuntimeError("Unable to import hf_wind_resource helpers.") from exc

DEFAULT_NODE_SUMMARY = (
    REPO_ROOT / "artifacts" / "power_estimates" / "node_summary" / "node_summary.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "power_estimates" / "geospatial"
DEFAULT_IMAGE = "duckdb/duckdb:latest"
DEFAULT_ENGINE = "docker"
DEFAULT_STAC_CONFIG = REPO_ROOT / "config" / "stac_catalogs.json"
DEFAULT_STAC_DATASET = "sar_range_final_pivots_joined"
DEFAULT_MAX_WIND_ROSES = 4
DEFAULT_BUOY_DATASET = REPO_ROOT / "catalogs" / "pde_vilano_buoy" / "assets" / "Vilano.parquet"
DEFAULT_BUOY_NODE_ID = "Vilano_buoy"
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "geospatial_products.json"
DEFAULT_TAXONOMY_PATH = REPO_ROOT / "config" / "node_taxonomy.json"
WIND_ROSE_SECTORS = 12

BASE_FIGURE_WIDTH = 680
BASE_FIGURE_HEIGHT = 620
BASE_LEFT_MARGIN = 90
BASE_RIGHT_MARGIN = 90
BASE_TOP_MARGIN = 110
BASE_BOTTOM_MARGIN = 160
BASE_COLORBAR_WIDTH = 220
DEFAULT_NODE_PADDING = 0.05
DEFAULT_LOW_COVERAGE_STROKE = "#22c55e"
DEFAULT_LOW_COVERAGE_STROKE_WIDTH = 2.5
DEFAULT_MISSING_STYLE = {
    "fill": "#ffffff",
    "outline": "#cbd5f5",
    "stroke_width": 1.3,
    "legend": "Hollow marker: KM spread unavailable (Weibull selected)",
}
DEFAULT_REFERENCE_OUTLINE = "#0ea5e9"
DEFAULT_REFERENCE_OUTLINE_WIDTH = 3.0
DEFAULT_REFERENCE_NOTE = "Blue outline marks buoy comparison node."
DEFAULT_POWER_GRADIENT = ("#1d4ed8", "#f97316")
DEFAULT_UNCERTAINTY_GRADIENT = ("#2563eb", "#facc15")
DEFAULT_LABEL_NOTE = "Named nodes highlight locations discussed in the analysis."
DEFAULT_UNCERTAINTY_NOTE = "Metric shown = KM P90 minus KM P50 (m/s). Hollow markers indicate nodes published via Weibull."
DEFAULT_POWER_UNCERTAINTY_SUMMARY = Path("artifacts/bootstrap_velocity_block/bootstrap_summary.csv")
DEFAULT_POWER_UNCERTAINTY_FIELDS = {
    "estimate": "power_density_estimate",
    "lower": "power_density_lower",
    "upper": "power_density_upper",
    "bootstrap_estimate": "power_density_bootstrap_estimate",
    "replicates": "power_density_replicates",
    "confidence": None,
}


def load_json_config(path: Path | None) -> Mapping[str, object]:
    if path is None:
        return {}
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON configuration: {path}") from exc


def _cfg_get(config: Mapping[str, object], *keys: str, default: object | None = None) -> object | None:
    data: object = config
    for key in keys:
        if not isinstance(data, Mapping) or key not in data:
            return default
        data = data[key]  # type: ignore[index]
    return data


def _to_path(value: Path | str | None, default: Path) -> Path:
    if value is None:
        return default
    if isinstance(value, Path):
        return value
    return Path(value)


def _as_bool(value: object | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"false", "0", "no", "off", ""}:
            return False
        if lowered in {"true", "1", "yes", "on"}:
            return True
    return bool(value)


class DuckDBError(RuntimeError):
    """Raised when DuckDB returns a non-zero exit status."""


@dataclass(frozen=True)
class NodeMetrics:
    """Minimal per-node bundle required for geospatial products."""

    node_id: str
    lon: float
    lat: float
    power_density_w_m2: float | None
    capacity_factor: float | None
    turbine_mean_power_kw: float | None
    mean_speed_m_s: float | None
    speed_p50_m_s: float | None
    speed_p90_m_s: float | None
    km_p50_m_s: float | None
    km_p90_m_s: float | None
    km_interval_width_m_s: float | None
    reliable_estimate: bool | None
    selected_method: str | None
    alternate_method: str | None
    low_coverage: bool | None
    coverage_band: str | None
    continuity_band: str | None
    in_ratio: float | None
    below_ratio: float | None
    above_ratio: float | None
    uncertain_ratio: float | None
    censored_ratio: float | None
    any_bias: bool | None
    bias_notes: str | None

    @property
    def display_name(self) -> str:
        """Friendly label for plots."""

        label = self.node_id
        if self.low_coverage:
            label += " (low coverage)"
        elif self.reliable_estimate is False:
            label += " (uncertain)"
        return label


@dataclass(frozen=True)
class RoseDescriptor:
    """Minimal description required to plot a wind-rose panel."""

    identifier: str
    label: str
    mean_speed_m_s: float | None
    power_density_w_m2: float | None
    capacity_factor: float | None
    low_coverage: bool = False


@dataclass(frozen=True)
class PowerUncertaintyInterval:
    """Confidence-interval bundle extracted from bootstrap summaries."""

    estimate: float | None
    lower: float | None
    upper: float | None
    bootstrap_estimate: float | None
    replicates: Optional[int]
    confidence: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a JSON configuration file controlling inputs and outputs.",
    )
    parser.add_argument(
        "--node-summary",
        type=Path,
        default=None,
        help="CSV generated by scripts/generate_node_summary_table.py.",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=None,
        help="Path to node taxonomy metadata (defaults to config/node_taxonomy.json).",
    )
    parser.add_argument(
        "--stac-config",
        type=Path,
        default=None,
        help="STAC catalog index used to resolve the ANN GeoParquet when --dataset is omitted.",
    )
    parser.add_argument(
        "--stac-dataset",
        default=None,
        help="Dataset key inside the STAC index (ignored when --dataset is provided).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Direct path to the ANN GeoParquet. Overrides STAC resolution when provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for GeoParquet/GeoJSON/visualisations.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Docker image hosting DuckDB (used when --engine docker).",
    )
    parser.add_argument(
        "--engine",
        choices=("docker", "python"),
        default=None,
        help="Execution engine for DuckDB queries.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing artefacts inside --output-dir.",
    )
    parser.add_argument(
        "--max-wind-roses",
        type=int,
        default=None,
        help="Maximum number of representative wind roses to render.",
    )
    parser.add_argument(
        "--buoy-dataset",
        type=Path,
        default=None,
        help="Path to the reference buoy dataset (Vilano) used for comparison.",
    )
    parser.add_argument(
        "--buoy-node-id",
        default=None,
        help="Node identifier in the ANN dataset that corresponds to the reference buoy.",
    )
    parser.add_argument(
        "--disable-buoy-rose",
        action="store_true",
        help="Disable generation of the reference buoy wind-rose panel.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def decode_wkb_point(raw: str) -> Tuple[float, float]:
    """Decode a WKB point stored as an escaped string."""

    data = codecs.decode(raw, "unicode_escape").encode("latin1")
    if len(data) != 21:
        raise ValueError("Unsupported geometry length for WKB point.")
    little_endian = data[0] == 1
    fmt = "<dd" if little_endian else ">dd"
    lon, lat = _struct_unpack(fmt, data[5:21])
    return lon, lat


def _struct_unpack(fmt: str, payload: bytes) -> Tuple[float, float]:
    from struct import unpack

    return unpack(fmt, payload)


def load_node_metrics(
    path: Path,
    *,
    taxonomy: Mapping[str, Mapping[str, object]] | None = None,
) -> List[NodeMetrics]:
    ensure_exists(path, label="node summary CSV")
    nodes: List[NodeMetrics] = []
    taxonomy_lookup = taxonomy or {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            node_id = row["node_id"]
            lon, lat = decode_wkb_point(row["geometry"])
            taxonomy_row = taxonomy_lookup.get(node_id, {})
            low_coverage_value = taxonomy_row.get("low_coverage") if taxonomy_row else None
            coverage_band_value = taxonomy_row.get("coverage_band") if taxonomy_row else None
            continuity_band_value = taxonomy_row.get("continuity_band") if taxonomy_row else None
            low_coverage = _optional_bool(row.get("low_coverage"))
            if low_coverage_value is not None:
                low_coverage = _optional_bool(low_coverage_value)
            coverage_band = _optional_str(row.get("coverage_band"))
            if coverage_band_value not in (None, ""):
                coverage_band = str(coverage_band_value)
            continuity_band = _optional_str(row.get("continuity_band"))
            if continuity_band_value not in (None, ""):
                continuity_band = str(continuity_band_value)
            nodes.append(
                NodeMetrics(
                    node_id=node_id,
                    lon=lon,
                    lat=lat,
                    power_density_w_m2=_optional_float(row.get("power_density_w_m2")),
                    capacity_factor=_optional_float(row.get("capacity_factor")),
                    turbine_mean_power_kw=_optional_float(row.get("turbine_mean_power_kw")),
                    mean_speed_m_s=_optional_float(row.get("mean_speed_m_s")),
                    speed_p50_m_s=_optional_float(row.get("speed_p50_m_s")),
                    speed_p90_m_s=_optional_float(row.get("speed_p90_m_s")),
                    km_p50_m_s=_optional_float(row.get("km_p50_m_s")),
                    km_p90_m_s=_optional_float(row.get("km_p90_m_s")),
                    km_interval_width_m_s=_optional_float(row.get("km_interval_width_m_s")),
                    reliable_estimate=_optional_bool(row.get("reliable_estimate")),
                    selected_method=_optional_str(row.get("selected_method")),
                    alternate_method=_optional_str(row.get("alternate_method")),
                    low_coverage=low_coverage,
                    coverage_band=coverage_band,
                    continuity_band=continuity_band,
                    in_ratio=_optional_float(row.get("in_ratio")),
                    below_ratio=_optional_float(row.get("below_ratio")),
                    above_ratio=_optional_float(row.get("above_ratio")),
                    uncertain_ratio=_optional_float(row.get("uncertain_ratio")),
                    censored_ratio=_optional_float(row.get("censored_ratio")),
                    any_bias=_optional_bool(row.get("any_bias")),
                    bias_notes=_optional_str(row.get("bias_notes")),
                )
            )
    return nodes


def _load_power_uncertainty_summary(
    path: Path,
    *,
    estimate_field: str,
    lower_field: str,
    upper_field: str,
    bootstrap_field: str | None,
    replicates_field: str | None,
    confidence_field: str | None,
) -> Dict[str, PowerUncertaintyInterval]:
    lookup: Dict[str, PowerUncertaintyInterval] = {}
    if not path.exists():
        return lookup

    def _coerce_float(value: object | None) -> float | None:
        if value in (None, "", "null"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _coerce_int(value: object | None) -> Optional[int]:
        if value in (None, "", "null"):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                maybe_float = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
            return int(maybe_float)

    def _follow_path(mapping: Mapping[str, object], path: str) -> object | None:
        current: object = mapping
        for part in path.split('.'):
            if not isinstance(current, Mapping):
                return None
            if part not in current:
                return None
            current = current[part]  # type: ignore[index]
            if current is None:
                return None
        return current

    def _derive_metric_key(*field_names: object | None) -> str | None:
        for name in field_names:
            if not isinstance(name, str):
                continue
            if '.' in name:
                components = name.split('.')
                if len(components) >= 2:
                    return components[-2]
            if '_' in name:
                prefix = name.rsplit('_', 1)[0]
                if prefix:
                    return prefix
        return None

    metric_key = _derive_metric_key(
        estimate_field,
        lower_field,
        upper_field,
        bootstrap_field,
        replicates_field,
        confidence_field,
    )

    def _build_interval(record: Mapping[str, object]) -> tuple[str, PowerUncertaintyInterval] | None:
        node_id_obj = record.get("node_id")
        if not node_id_obj:
            return None
        node_id = str(node_id_obj)

        metrics_payload: Mapping[str, object] | None = None
        metrics_section = record.get("metrics")
        if isinstance(metrics_section, Mapping) and metric_key:
            candidate = metrics_section.get(metric_key)
            if isinstance(candidate, Mapping):
                metrics_payload = candidate

        def _resolve_value(field: object | None, *, default_key: str) -> object | None:
            if isinstance(field, str):
                direct = record.get(field)
                if direct not in (None, ""):
                    return direct
                if '.' in field:
                    dotted = _follow_path(record, field)
                    if dotted not in (None, ""):
                        return dotted
                if metric_key and field.startswith(f"{metric_key}_"):
                    stripped = field[len(metric_key) + 1 :]
                else:
                    stripped = field
                if metrics_payload and isinstance(stripped, str) and stripped in metrics_payload:
                    value = metrics_payload[stripped]
                    if value not in (None, ""):
                        return value
            if metrics_payload is not None:
                candidate = metrics_payload.get(default_key)
                if candidate not in (None, ""):
                    return candidate
            return None

        interval = PowerUncertaintyInterval(
            estimate=_coerce_float(_resolve_value(estimate_field, default_key="estimate")),
            lower=_coerce_float(_resolve_value(lower_field, default_key="lower")),
            upper=_coerce_float(_resolve_value(upper_field, default_key="upper")),
            bootstrap_estimate=_coerce_float(_resolve_value(bootstrap_field, default_key="bootstrap_estimate"))
            if bootstrap_field or metrics_payload is not None
            else None,
            replicates=_coerce_int(_resolve_value(replicates_field, default_key="replicates"))
            if replicates_field or metrics_payload is not None
            else None,
            confidence=_coerce_float(_resolve_value(confidence_field, default_key="confidence"))
            if confidence_field or metrics_payload is not None
            else None,
        )
        return node_id, interval

    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as handle:
            if suffix == ".jsonl":
                for line in handle:
                    payload = line.strip()
                    if not payload:
                        continue
                    record = json.loads(payload)
                    outcome = _build_interval(record)
                    if outcome is not None:
                        node_id, interval = outcome
                        lookup[node_id] = interval
            else:
                payload = json.load(handle)
                if isinstance(payload, list):
                    iterable = payload
                elif isinstance(payload, Mapping):
                    iterable = payload.values()
                else:
                    iterable = []
                for item in iterable:
                    if isinstance(item, Mapping):
                        outcome = _build_interval(item)
                        if outcome is not None:
                            node_id, interval = outcome
                            lookup[node_id] = interval
        return lookup

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            outcome = _build_interval(row)
            if outcome is None:
                continue
            node_id, interval = outcome
            lookup[node_id] = interval
    return lookup


def summarise_power_uncertainty(
    intervals: Mapping[str, PowerUncertaintyInterval],
    *,
    total_nodes: Optional[int] = None,
) -> Dict[str, float | int]:
    stats: Dict[str, float | int] = {}
    provided_nodes = len(intervals)
    nodes_with_ci = sum(
        1 for interval in intervals.values() if interval.lower is not None and interval.upper is not None
    )
    stats["nodes_with_ci"] = nodes_with_ci
    effective_total = total_nodes if total_nodes is not None else provided_nodes
    if effective_total is not None:
        missing_summary = max(effective_total - provided_nodes, 0)
        missing_ci = max(effective_total - nodes_with_ci, 0)
        stats["nodes_without_summary"] = missing_summary
        stats["nodes_without_ci"] = missing_ci

    replicate_values = [
        interval.replicates
        for interval in intervals.values()
        if interval.replicates is not None
    ]
    if replicate_values:
        stats["replicates_min"] = float(min(replicate_values))
        stats["replicates_max"] = float(max(replicate_values))
        stats["replicates_mean"] = float(sum(replicate_values) / len(replicate_values))

    confidence_values = [
        interval.confidence for interval in intervals.values() if interval.confidence is not None
    ]
    if confidence_values:
        stats["confidence_min"] = min(confidence_values)
        stats["confidence_max"] = max(confidence_values)
    return stats


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _optional_bool(value: object | None) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def _optional_str(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


def write_resource_csv(nodes: Sequence[NodeMetrics], path: Path) -> None:
    fieldnames = [
        "node_id",
        "lon",
        "lat",
        "power_density_w_m2",
        "capacity_factor",
        "turbine_mean_power_kw",
        "mean_speed_m_s",
        "speed_p50_m_s",
        "speed_p90_m_s",
        "km_p50_m_s",
        "km_p90_m_s",
        "km_interval_width_m_s",
        "reliable_estimate",
        "selected_method",
        "alternate_method",
        "low_coverage",
        "coverage_band",
        "continuity_band",
        "in_ratio",
        "below_ratio",
        "above_ratio",
        "uncertain_ratio",
        "censored_ratio",
        "any_bias",
        "bias_notes",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for node in nodes:
            writer.writerow(
                {
                    "node_id": node.node_id,
                    "lon": f"{node.lon:.6f}",
                    "lat": f"{node.lat:.6f}",
                    "power_density_w_m2": _format_optional(node.power_density_w_m2),
                    "capacity_factor": _format_optional(node.capacity_factor),
                    "turbine_mean_power_kw": _format_optional(node.turbine_mean_power_kw),
                    "mean_speed_m_s": _format_optional(node.mean_speed_m_s),
                    "speed_p50_m_s": _format_optional(node.speed_p50_m_s),
                    "speed_p90_m_s": _format_optional(node.speed_p90_m_s),
                    "km_p50_m_s": _format_optional(node.km_p50_m_s),
                    "km_p90_m_s": _format_optional(node.km_p90_m_s),
                    "km_interval_width_m_s": _format_optional(node.km_interval_width_m_s),
                    "reliable_estimate": _format_optional_bool(node.reliable_estimate),
                    "selected_method": node.selected_method or "",
                    "alternate_method": node.alternate_method or "",
                    "low_coverage": _format_optional_bool(node.low_coverage),
                    "coverage_band": node.coverage_band or "",
                    "continuity_band": node.continuity_band or "",
                    "in_ratio": _format_optional(node.in_ratio),
                    "below_ratio": _format_optional(node.below_ratio),
                    "above_ratio": _format_optional(node.above_ratio),
                    "uncertain_ratio": _format_optional(node.uncertain_ratio),
                    "censored_ratio": _format_optional(node.censored_ratio),
                    "any_bias": _format_optional_bool(node.any_bias),
                    "bias_notes": node.bias_notes or "",
                }
            )


def _format_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def resolve_output_path(base_dir: Path, value: Path | str | None, default_name: str) -> Path:
    if value is None or value == "":
        return base_dir / default_name
    if isinstance(value, Path):
        path = value
    else:
        path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _to_float(value: object | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def run_duckdb_sql(sql: str, *, engine: str, image: str) -> None:
    normalized = " ".join(sql.strip().split())
    if engine == "docker":
        command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{REPO_ROOT}:/workspace",
            "-w",
            "/workspace",
            image,
            "duckdb",
            "-cmd",
            normalized,
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise DuckDBError(f"DuckDB failed (docker, code {result.returncode}):\n{result.stderr.strip()}")
        return

    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise DuckDBError("DuckDB Python module not available; rerun with --engine docker.") from exc

    with duckdb.connect(database=":memory:") as conn:
        conn.execute("PRAGMA memory_limit='1024MB';")
        conn.execute(normalized)


def build_geoparquet(csv_path: Path, parquet_path: Path, *, engine: str, image: str) -> None:
    sql = f"""
    INSTALL spatial;
    LOAD spatial;
    CREATE OR REPLACE TABLE node_resource_map AS
    SELECT * FROM read_csv_auto('{_repo_relative(csv_path)}', header=True, union_by_name=True);
    COPY (
        SELECT
            node_id,
            ST_Point(lon, lat) AS geometry,
            power_density_w_m2,
            capacity_factor,
            turbine_mean_power_kw,
            mean_speed_m_s,
            speed_p50_m_s,
            speed_p90_m_s,
            km_p50_m_s,
            km_p90_m_s,
            km_interval_width_m_s,
            reliable_estimate,
            selected_method,
            alternate_method,
            CAST(low_coverage AS BOOLEAN) AS low_coverage,
            coverage_band,
            continuity_band,
            in_ratio,
            below_ratio,
            above_ratio,
            uncertain_ratio,
            censored_ratio,
            any_bias,
            bias_notes
        FROM node_resource_map
    ) TO '{_repo_relative(parquet_path)}' WITH (FORMAT PARQUET);
    """
    run_duckdb_sql(sql, engine=engine, image=image)


def compute_wind_direction_histogram(
    dataset_path: Path,
    node_ids: Sequence[str],
    output_csv: Path,
    *,
    engine: str,
    image: str,
) -> None:
    if not node_ids:
        output_csv.write_text("node_id,sector,weight,samples\n", encoding="utf-8")
        return

    quoted_ids = ",".join(f"'{node_id}'" for node_id in node_ids)
    sql = f"""
    COPY (
        WITH base AS (
            SELECT
                node_id,
                ((CAST(FLOOR(((pred_wind_direction % 360) + 360) % 360 / (360.0 / {WIND_ROSE_SECTORS})) AS INTEGER)) % {WIND_ROSE_SECTORS}) AS sector,
                prob_range_in AS weight
            FROM read_parquet('{_repo_relative(dataset_path)}')
            WHERE node_id IN ({quoted_ids})
        )
        SELECT
            node_id,
            sector,
            SUM(COALESCE(weight, 0.0)) AS weight,
            COUNT(*) AS samples
        FROM base
        GROUP BY node_id, sector
        ORDER BY node_id, sector
    )
    TO '{_repo_relative(output_csv)}' WITH (HEADER, DELIMITER ',');
    """
    run_duckdb_sql(sql, engine=engine, image=image)


def load_direction_weights(path: Path) -> Mapping[str, List[Tuple[int, float]]]:
    ensure_exists(path, label="wind rose histogram CSV")
    payload: MutableMapping[str, List[Tuple[int, float]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            node_id = row["node_id"]
            sector = int(row["sector"])
            weight = float(row["weight"])
            payload.setdefault(node_id, []).append((sector, weight))
    for weights in payload.values():
        weights.sort(key=lambda item: item[0])
    return payload


def compute_buoy_direction_histogram(
    dataset_path: Path,
    *,
    output_csv: Path,
    stats_csv: Path,
    engine: str,
    image: str,
) -> None:
    ensure_exists(dataset_path, label="buoy dataset")
    hist_sql = f"""
    COPY (
        SELECT
            'reference_buoy' AS node_id,
            ((CAST(FLOOR(((wind_dir % 360) + 360) % 360 / (360.0 / {WIND_ROSE_SECTORS})) AS INTEGER)) % {WIND_ROSE_SECTORS}) AS sector,
            COUNT(*) AS weight,
            COUNT(*) AS samples
        FROM read_parquet('{_repo_relative(dataset_path)}')
        WHERE wind_dir BETWEEN 0 AND 360
        GROUP BY 2
        ORDER BY 2
    )
    TO '{_repo_relative(output_csv)}' WITH (HEADER, DELIMITER ',');
    """
    run_duckdb_sql(hist_sql, engine=engine, image=image)

    stats_sql = f"""
    COPY (
        SELECT
            AVG(NULLIF(wind_speed, -10000)) FILTER (WHERE wind_speed >= 0) AS mean_speed,
            COUNT(*) AS samples
        FROM read_parquet('{_repo_relative(dataset_path)}')
        WHERE wind_dir BETWEEN 0 AND 360 AND wind_speed IS NOT NULL AND wind_speed >= 0
    )
    TO '{_repo_relative(stats_csv)}' WITH (HEADER, DELIMITER ',');
    """
    run_duckdb_sql(stats_sql, engine=engine, image=image)


def load_single_row_csv(path: Path) -> Mapping[str, float | None]:
    ensure_exists(path, label="stats CSV")
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            result: Dict[str, float | None] = {}
            for key, value in row.items():
                if value in {"", None}:
                    result[key] = None
                else:
                    try:
                        result[key] = float(value)
                    except ValueError:
                        result[key] = None
            return result
    return {}


def write_geojson(nodes: Sequence[NodeMetrics], path: Path) -> None:
    features = []
    for node in nodes:
        properties = {
            "power_density_w_m2": node.power_density_w_m2,
            "capacity_factor": node.capacity_factor,
            "turbine_mean_power_kw": node.turbine_mean_power_kw,
            "mean_speed_m_s": node.mean_speed_m_s,
            "speed_p50_m_s": node.speed_p50_m_s,
            "speed_p90_m_s": node.speed_p90_m_s,
            "km_p50_m_s": node.km_p50_m_s,
            "km_p90_m_s": node.km_p90_m_s,
            "km_interval_width_m_s": node.km_interval_width_m_s,
            "reliable_estimate": node.reliable_estimate,
            "selected_method": node.selected_method,
            "alternate_method": node.alternate_method,
            "low_coverage": node.low_coverage,
            "coverage_band": node.coverage_band,
            "continuity_band": node.continuity_band,
            "in_ratio": node.in_ratio,
            "below_ratio": node.below_ratio,
            "above_ratio": node.above_ratio,
            "uncertain_ratio": node.uncertain_ratio,
            "censored_ratio": node.censored_ratio,
            "any_bias": node.any_bias,
            "bias_notes": node.bias_notes,
        }
        features.append(
            {
                "type": "Feature",
                "id": node.node_id,
                "geometry": {"type": "Point", "coordinates": [node.lon, node.lat]},
                "properties": properties,
            }
        )
    payload = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_metric_map(
    nodes: Sequence[NodeMetrics],
    output_path: Path,
    *,
    title: str,
    subtitle: str,
    value_accessor,
    color_start: str,
    color_end: str,
    gradient_id: str,
    colorbar_title: str,
    colorbar_format: str,
    legend_items: Sequence[Tuple[str, str]],
    dash_when_unreliable: bool,
    width: int,
    height: int,
    margins: Mapping[str, float],
    node_padding: float,
    low_coverage_color: str,
    low_coverage_stroke_width: float,
    note_text: str,
    colorbar_width: float,
    missing_style: Mapping[str, object] | None = None,
    reference_node_id: str | None = None,
    reference_outline_color: str | None = None,
    reference_outline_width: float | None = None,
    reference_note: str | None = None,
    annotation_text: str | None = None,
) -> None:
    values = [value_accessor(node) for node in nodes if value_accessor(node) is not None]
    lons = [node.lon for node in nodes]
    lats = [node.lat for node in nodes]

    min_lon = min(lons)
    max_lon = max(lons)
    min_lat = min(lats)
    max_lat = max(lats)
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat
    padding = max(node_padding, 0.0)
    if math.isclose(lon_span, 0.0):
        lon_span = 1.0
    if math.isclose(lat_span, 0.0):
        lat_span = 1.0
    min_lon -= lon_span * padding
    max_lon += lon_span * padding
    min_lat -= lat_span * padding
    max_lat += lat_span * padding

    left_margin = int(round(margins.get("left", BASE_LEFT_MARGIN)))
    right_margin = int(round(margins.get("right", BASE_RIGHT_MARGIN)))
    top_margin = int(round(margins.get("top", BASE_TOP_MARGIN)))
    bottom_margin = int(round(margins.get("bottom", BASE_BOTTOM_MARGIN)))
    panel_width = max(width - left_margin - right_margin, 1)
    panel_height = max(height - top_margin - bottom_margin, 1)

    def project(lon: float, lat: float) -> Tuple[float, float]:
        x_norm = 0.5 if math.isclose(max_lon, min_lon) else (lon - min_lon) / (max_lon - min_lon)
        y_norm = 0.5 if math.isclose(max_lat, min_lat) else (lat - min_lat) / (max_lat - min_lat)
        x = left_margin + x_norm * panel_width
        y = top_margin + panel_height - y_norm * panel_height
        return x, y

    svg: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
        "<defs>",
        _build_linear_gradient(gradient_id, color_start, color_end),
        "</defs>",
        f'<rect width="{width}" height="{height}" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>',
        f'<text x="{left_margin}" y="58" font-size="20" font-weight="bold">{title}</text>',
        f'<text x="{left_margin}" y="78" font-size="12" fill="#6b7280">{subtitle}</text>',
    ]

    if annotation_text:
        svg.append(
            f'<text x="{left_margin}" y="{top_margin - 30}" font-size="11" fill="#4b5563">{annotation_text}</text>'
        )

    svg.extend(
        _render_panel_background(
            x=left_margin,
            y=top_margin,
            width=panel_width,
            height=panel_height,
            title="",
            subtitle="",
        )
    )

    label_nodes = _select_label_nodes(nodes)

    missing_present = False
    missing_fill = None
    missing_outline = None
    missing_stroke_width = DEFAULT_MISSING_STYLE["stroke_width"]
    missing_legend_text = None
    if missing_style:
        missing_fill = str(missing_style.get("fill", DEFAULT_MISSING_STYLE["fill"]))
        missing_outline = str(missing_style.get("outline", DEFAULT_MISSING_STYLE["outline"]))
        missing_stroke_width = _to_float(missing_style.get("stroke_width"), DEFAULT_MISSING_STYLE["stroke_width"])
        missing_legend_text = missing_style.get("legend") or DEFAULT_MISSING_STYLE["legend"]

    reference_entries: List[Tuple[str, str]] = []
    if reference_node_id and reference_note:
        reference_entries.append(("Blue outline", reference_note))

    for node in nodes:
        value = value_accessor(node)
        x, y = project(node.lon, node.lat)
        radius = 6 + 6 * (node.capacity_factor or 0.0)
        is_reference = reference_node_id is not None and node.node_id == reference_node_id

        outline = "#111827"
        stroke_width = 1.0
        if node.low_coverage:
            outline = low_coverage_color
            stroke_width = low_coverage_stroke_width
        if is_reference:
            outline = reference_outline_color or outline
            stroke_width = reference_outline_width or (stroke_width + 1.5)

        dash_attr = ' stroke-dasharray="4 2"' if dash_when_unreliable and node.reliable_estimate is False else ""

        if value is None:
            if missing_fill is None or missing_outline is None:
                continue
            svg.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{missing_fill}" '
                f'stroke="{missing_outline}" stroke-width="{missing_stroke_width}" stroke-dasharray="3 3"/>'
            )
            missing_present = True
        else:
            color = _interpolate_color(value, values, color_start, color_end)
            svg.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{color}" '
                f'stroke="{outline}" stroke-width="{stroke_width}"{dash_attr}/>'
            )

        if node.node_id in label_nodes:
            suffix = " (buoy)" if is_reference else ""
            svg.append(
                f'<text x="{x + 8:.2f}" y="{y - 8:.2f}" font-size="10">{node.display_name}{suffix}</text>'
            )

    legend_y = height - bottom_margin + 40
    if note_text:
        svg.append(
            f'<text x="{left_margin}" y="{legend_y - 16}" font-size="11" fill="#4b5563">{note_text}</text>'
        )

    legend_entries = list(legend_items) + reference_entries
    if missing_present and missing_legend_text:
        legend_entries.append(("Hollow marker", missing_legend_text))

    svg.extend(
        _render_legend(
            x=left_margin,
            y=legend_y,
            title="Legend",
            items=legend_entries,
        )
    )

    cb_width = int(round(colorbar_width))
    colorbar_x = width - right_margin - cb_width
    colorbar_y = legend_y + 20
    if values:
        min_value = min(values)
        max_value = max(values)
    else:
        min_value = 0.0
        max_value = 0.0
    svg.extend(
        _render_colorbar(
            x=colorbar_x,
            y=colorbar_y,
            width=cb_width,
            title=colorbar_title,
            gradient_id=gradient_id,
            min_value=min_value,
            max_value=max_value,
            format_spec=colorbar_format,
        )
    )

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def _render_panel_background(
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
) -> List[str]:
    return [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="#f9fafb" stroke="#e5e7eb" stroke-width="1"/>',
        *((
            f'<text x="{x + 8}" y="{y + 18}" font-size="16" font-weight="bold">{title}</text>',
            f'<text x="{x + 8}" y="{y + 34}" font-size="11" fill="#6b7280">{subtitle}</text>',
        ) if title or subtitle else ()),
    ]


def _render_legend(*, x: float, y: float, title: str, items: Sequence[Tuple[str, str]]) -> List[str]:
    lines = [f'<text x="{x}" y="{y}" font-size="12" font-weight="bold">{title}</text>']
    line_height = 14
    for idx, (label, desc) in enumerate(items, start=1):
        lines.append(
            f'<text x="{x}" y="{y + idx * line_height}" font-size="11"><tspan font-weight="bold">{label}:</tspan> {desc}</text>'
        )
    return lines


def _build_linear_gradient(gradient_id: str, start_hex: str, end_hex: str) -> str:
    return (
        f'<linearGradient id="{gradient_id}" x1="0%" y1="0%" x2="100%" y2="0%">'
        f'<stop offset="0%" stop-color="{start_hex}"/>'
        f'<stop offset="100%" stop-color="{end_hex}"/>'
        "</linearGradient>"
    )


def _render_colorbar(
    *,
    x: float,
    y: float,
    width: float,
    title: str,
    gradient_id: str,
    min_value: float,
    max_value: float,
    format_spec: str,
) -> List[str]:
    height = 16
    text_offset = 14
    return [
        f'<text x="{x}" y="{y - 10}" font-size="11" font-weight="bold">{title}</text>',
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="url(#{gradient_id})" stroke="#d1d5db" stroke-width="1"/>',
        f'<text x="{x}" y="{y + height + text_offset}" font-size="10">{format(min_value, format_spec)}</text>',
        f'<text x="{x + width}" y="{y + height + text_offset}" font-size="10" text-anchor="end">{format(max_value, format_spec)}</text>',
    ]


def _interpolate_color(value: float, values: Sequence[float], start_hex: str, end_hex: str) -> str:
    if not values:
        return start_hex
    min_val = min(values)
    max_val = max(values)
    if math.isclose(min_val, max_val):
        t = 0.5
    else:
        t = (value - min_val) / (max_val - min_val)
        t = max(0.0, min(1.0, t))
    start_rgb = _hex_to_rgb(start_hex)
    end_rgb = _hex_to_rgb(end_hex)
    interp = tuple(int(round(s + (e - s) * t)) for s, e in zip(start_rgb, end_rgb))
    return f"#{interp[0]:02x}{interp[1]:02x}{interp[2]:02x}"


def _hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    hex_code = hex_code.lstrip("#")
    return tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))


def render_wind_rose_panels(
    descriptors: Sequence[RoseDescriptor],
    weights: Mapping[str, List[Tuple[int, float]]],
    output_path: Path,
) -> None:
    if not descriptors:
        output_path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="200"><text x="20" y="40">No nodes selected.</text></svg>',
            encoding="utf-8",
        )
        return

    cols = min(2, len(descriptors))
    rows = math.ceil(len(descriptors) / cols)
    cell_width = 320
    cell_height = 320
    width = cols * cell_width
    height = rows * cell_height
    svg: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    for index, descriptor in enumerate(descriptors):
        row = index // cols
        col = index % cols
        origin_x = col * cell_width
        origin_y = row * cell_height
        cx = origin_x + cell_width / 2
        cy = origin_y + cell_height / 2 + 20
        radius = 110

        svg.append(
            f'<text x="{origin_x + 16}" y="{origin_y + 26}" font-size="16" font-weight="bold">{descriptor.label}</text>'
        )
        metrics_parts: List[str] = []
        if descriptor.mean_speed_m_s is not None:
            metrics_parts.append(f"Mean speed {descriptor.mean_speed_m_s:.2f} m/s")
        if descriptor.power_density_w_m2 is not None:
            metrics_parts.append(f"Power density {descriptor.power_density_w_m2:.0f} W/m²")
        if descriptor.capacity_factor is not None:
            metrics_parts.append(f"Capacity factor {descriptor.capacity_factor:.2f}")
        if descriptor.low_coverage:
            metrics_parts.append("Taxonomy flag: low coverage")
        metrics_text = " · ".join(metrics_parts) if metrics_parts else "No aggregated stats available"
        svg.append(
            f'<text x="{origin_x + 16}" y="{origin_y + 46}" font-size="11" fill="#6b7280">{metrics_text}</text>'
        )

        svg.extend(_render_rose_grid(cx, cy, radius))

        node_weights = weights.get(descriptor.identifier, [])
        total_weight = sum(weight for _, weight in node_weights)
        if total_weight <= 0:
            continue

        for sector, weight in node_weights:
            fraction = weight / total_weight if total_weight else 0.0
            outer_radius = radius * fraction
            angle_start = sector * (360 / WIND_ROSE_SECTORS)
            angle_end = (sector + 1) * (360 / WIND_ROSE_SECTORS)
            svg.append(_render_sector_path(cx, cy, outer_radius, angle_start, angle_end))

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def _render_rose_grid(cx: float, cy: float, radius: float) -> List[str]:
    circles: List[str] = []
    steps = [0.25, 0.5, 0.75, 1.0]
    for step in steps:
        r = radius * step
        circles.append(f'<circle cx="{cx}" cy="{cy}" r="{r:.2f}" fill="none" stroke="#e5e7eb" stroke-width="1"/>')
        circles.append(
            f'<text x="{cx + r + 4:.2f}" y="{cy - 2:.2f}" font-size="10" fill="#9ca3af">{int(step * 100)}%</text>'
        )

    directions = [("N", 0), ("E", 90), ("S", 180), ("W", 270)]
    for label, angle in directions:
        x, y = _polar_to_cartesian(cx, cy, radius + 12, angle)
        circles.append(f'<text x="{x - 6:.2f}" y="{y + 4:.2f}" font-size="12" font-weight="bold">{label}</text>')

    return circles


def _render_sector_path(
    cx: float,
    cy: float,
    outer_radius: float,
    angle_start_deg: float,
    angle_end_deg: float,
) -> str:
    start_x, start_y = _polar_to_cartesian(cx, cy, outer_radius, angle_start_deg)
    end_x, end_y = _polar_to_cartesian(cx, cy, outer_radius, angle_end_deg)
    large_arc = 1 if (angle_end_deg - angle_start_deg) > 180 else 0
    fill = _interpolate_color(
        (angle_start_deg + angle_end_deg) / 2,
        list(range(0, 360, int(360 / WIND_ROSE_SECTORS))),
        "#2563eb",
        "#10b981",
    )
    return (
        f'<path d="M {cx:.2f} {cy:.2f} L {start_x:.2f} {start_y:.2f} '
        f'A {outer_radius:.2f} {outer_radius:.2f} 0 {large_arc} 1 {end_x:.2f} {end_y:.2f} Z" '
        f'fill="{fill}" fill-opacity="0.85" stroke="#1f2937" stroke-width="0.6"/>'
    )


def _polar_to_cartesian(cx: float, cy: float, radius: float, angle_deg: float) -> Tuple[float, float]:
    angle_rad = math.radians(90 - angle_deg)
    x = cx + radius * math.cos(angle_rad)
    y = cy - radius * math.sin(angle_rad)
    return x, y


def _select_label_nodes(nodes: Sequence[NodeMetrics]) -> set[str]:
    label_nodes: set[str] = set()

    reliable_nodes = [n for n in nodes if n.reliable_estimate is not False]
    sorted_power = sorted(reliable_nodes, key=lambda n: (n.power_density_w_m2 or -math.inf), reverse=True)

    power_quota = min(len(sorted_power), 2)
    for node in sorted_power[:power_quota]:
        label_nodes.add(node.node_id)

    high_uncertainty = sorted(reliable_nodes, key=lambda n: (n.km_interval_width_m_s or 0.0), reverse=True)
    for node in high_uncertainty:
        before = len(label_nodes)
        label_nodes.add(node.node_id)
        if len(label_nodes) > before:
            break

    low_coverage = [n for n in nodes if n.low_coverage]
    for node in low_coverage:
        if node.node_id not in label_nodes:
            label_nodes.add(node.node_id)
            break

    if len(label_nodes) < 4:
        remaining = sorted(nodes, key=lambda n: (n.capacity_factor or -math.inf), reverse=True)
        for node in remaining:
            label_nodes.add(node.node_id)
            if len(label_nodes) >= 4:
                break

    return label_nodes


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def resolve_dataset(dataset: Path | None, *, config_path: Path, dataset_key: str) -> Path:
    if dataset is not None:
        ensure_exists(dataset, label="ANN GeoParquet dataset")
        return dataset.resolve()
    if dataset_key == DEFAULT_STAC_DATASET:
        resolved = resolve_ann_asset(config_path=config_path, item_id="data", asset_key="data")
    else:
        resolved = resolve_catalog_asset(dataset_key, config_path=config_path, item_id="data", asset_key="data")
    return resolved.require_local_path()


def write_metadata(
    path: Path,
    *,
    node_summary: Path,
    dataset: Path,
    geo_parquet: Path,
    geojson: Path,
    power_map: Path,
    uncertainty_map: Path,
    roses: Path,
    histogram: Path,
    selected_nodes: Sequence[str],
    buoy_histogram: Path | None = None,
    buoy_mean_speed: float | None = None,
    buoy_samples: float | None = None,
    config_path: Path | None = None,
    buoy_panels: Path | None = None,
    taxonomy: Path | None = None,
    low_coverage_nodes: Sequence[str] | None = None,
    power_uncertainty: Mapping[str, object] | None = None,
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "node_summary_source": _repo_relative(node_summary),
        "dataset_source": _repo_relative(dataset),
        "geo_parquet": _repo_relative(geo_parquet),
        "geojson": _repo_relative(geojson),
        "power_density_map": _repo_relative(power_map),
        "uncertainty_map": _repo_relative(uncertainty_map),
        "wind_rose_panels": _repo_relative(roses),
        "wind_rose_histogram": _repo_relative(histogram),
        "selected_nodes": list(selected_nodes),
    }
    if buoy_histogram is not None:
        payload["buoy_wind_rose_histogram"] = _repo_relative(buoy_histogram)
    if buoy_mean_speed is not None:
        payload["buoy_mean_speed_m_s"] = buoy_mean_speed
    if buoy_samples is not None:
        payload["buoy_samples"] = buoy_samples
    if config_path is not None:
        payload["config_source"] = _repo_relative(config_path)
    if buoy_panels is not None:
        payload["buoy_wind_rose_panels"] = _repo_relative(buoy_panels)
    if taxonomy is not None:
        payload["taxonomy_source"] = _repo_relative(taxonomy)
    if low_coverage_nodes:
        payload["low_coverage_nodes"] = list(low_coverage_nodes)
    if power_uncertainty:
        payload["power_uncertainty"] = power_uncertainty
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_json_config(args.config)
    config_source = args.config if args.config and args.config.exists() else None

    node_summary_value = args.node_summary if args.node_summary is not None else _cfg_get(config, "node_summary")
    node_summary_path = _to_path(node_summary_value, DEFAULT_NODE_SUMMARY)

    taxonomy_value = args.taxonomy if args.taxonomy is not None else _cfg_get(config, "taxonomy")
    if taxonomy_value in (None, ""):
        taxonomy_path = DEFAULT_TAXONOMY_PATH
    else:
        taxonomy_path = _to_path(taxonomy_value, DEFAULT_TAXONOMY_PATH)
    ensure_exists(taxonomy_path, label="node taxonomy JSON")
    taxonomy_records = load_taxonomy_records(taxonomy_path)

    stac_config_value = args.stac_config if args.stac_config is not None else _cfg_get(config, "stac", "config_path")
    stac_config_path = _to_path(stac_config_value, DEFAULT_STAC_CONFIG)

    stac_dataset_value = args.stac_dataset if args.stac_dataset is not None else _cfg_get(config, "stac", "dataset")
    if stac_dataset_value in (None, ""):
        stac_dataset = DEFAULT_STAC_DATASET
    else:
        stac_dataset = str(stac_dataset_value)

    dataset_override_value = args.dataset if args.dataset is not None else _cfg_get(config, "ann_dataset")
    dataset_override_path: Path | None = None
    if dataset_override_value not in (None, ""):
        dataset_override_path = dataset_override_value if isinstance(dataset_override_value, Path) else Path(dataset_override_value)

    output_dir_value = args.output_dir if args.output_dir is not None else _cfg_get(config, "output_dir")
    output_dir = _to_path(output_dir_value, DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs_cfg = _cfg_get(config, "outputs")
    if not isinstance(outputs_cfg, Mapping):
        outputs_cfg = {}

    csv_path = resolve_output_path(output_dir, outputs_cfg.get("node_resource_map"), "node_resource_map.csv")
    parquet_path = resolve_output_path(output_dir, outputs_cfg.get("node_resource_parquet"), "node_resource_map.parquet")
    geojson_path = resolve_output_path(output_dir, outputs_cfg.get("node_resource_geojson"), "node_resource_map.geojson")
    power_map_path = resolve_output_path(output_dir, outputs_cfg.get("power_density_map"), "power_density_map.svg")
    uncertainty_map_path = resolve_output_path(output_dir, outputs_cfg.get("uncertainty_map"), "uncertainty_map.svg")
    roses_path = resolve_output_path(output_dir, outputs_cfg.get("wind_rose_panels"), "wind_rose_panels.svg")
    buoy_panels_path = resolve_output_path(output_dir, outputs_cfg.get("buoy_wind_rose_panels"), "buoy_wind_rose_panels.svg")
    histogram_csv = resolve_output_path(output_dir, outputs_cfg.get("wind_rose_histogram"), "wind_rose_histogram.csv")
    buoy_histogram_csv = resolve_output_path(output_dir, outputs_cfg.get("buoy_wind_rose_histogram"), "buoy_wind_rose_histogram.csv")
    buoy_stats_csv = resolve_output_path(output_dir, outputs_cfg.get("buoy_wind_rose_stats"), "buoy_wind_rose_stats.csv")
    metadata_path = resolve_output_path(output_dir, outputs_cfg.get("metadata"), "metadata.json")

    for path in {
        csv_path,
        parquet_path,
        geojson_path,
        power_map_path,
        uncertainty_map_path,
        roses_path,
        buoy_panels_path,
        histogram_csv,
        buoy_histogram_csv,
        buoy_stats_csv,
        metadata_path,
    }:
        path.parent.mkdir(parents=True, exist_ok=True)

    image_value = args.image if args.image is not None else _cfg_get(config, "image")
    image = image_value or DEFAULT_IMAGE

    engine_value = args.engine if args.engine is not None else _cfg_get(config, "engine")
    engine = engine_value or DEFAULT_ENGINE
    if engine not in {"docker", "python"}:
        engine = DEFAULT_ENGINE

    max_roses_value = args.max_wind_roses if args.max_wind_roses is not None else _cfg_get(config, "max_wind_roses")
    if max_roses_value in (None, ""):
        max_wind_roses = DEFAULT_MAX_WIND_ROSES
    else:
        max_wind_roses = int(max_roses_value)
        if max_wind_roses < 0:
            max_wind_roses = 0

    style_cfg = _cfg_get(config, "style")
    if not isinstance(style_cfg, Mapping):
        style_cfg = {}

    figure_scale = _to_float(style_cfg.get("figure_scale"), 1.2)
    figure_cfg = style_cfg.get("figure")
    if not isinstance(figure_cfg, Mapping):
        figure_cfg = {}

    figure_width = max(int(round(_to_float(figure_cfg.get("width"), BASE_FIGURE_WIDTH * figure_scale))), 1)
    figure_height = max(int(round(_to_float(figure_cfg.get("height"), BASE_FIGURE_HEIGHT * figure_scale))), 1)
    figure_margins = {
        "left": _to_float(figure_cfg.get("left_margin"), BASE_LEFT_MARGIN * figure_scale),
        "right": _to_float(figure_cfg.get("right_margin"), BASE_RIGHT_MARGIN * figure_scale),
        "top": _to_float(figure_cfg.get("top_margin"), BASE_TOP_MARGIN * figure_scale),
        "bottom": _to_float(figure_cfg.get("bottom_margin"), BASE_BOTTOM_MARGIN * figure_scale),
    }

    node_padding = _to_float(style_cfg.get("node_padding"), DEFAULT_NODE_PADDING)
    low_coverage_stroke = str(style_cfg.get("low_coverage_stroke") or DEFAULT_LOW_COVERAGE_STROKE)
    low_coverage_stroke_width = _to_float(style_cfg.get("low_coverage_stroke_width"), DEFAULT_LOW_COVERAGE_STROKE_WIDTH)

    missing_style_cfg = style_cfg.get("missing_marker")
    if isinstance(missing_style_cfg, Mapping):
        missing_style = dict(missing_style_cfg)
    else:
        missing_style = dict(DEFAULT_MISSING_STYLE)

    colorbar_cfg = style_cfg.get("colorbar")
    if isinstance(colorbar_cfg, Mapping):
        colorbar_width = _to_float(colorbar_cfg.get("width"), BASE_COLORBAR_WIDTH * figure_scale)
    else:
        colorbar_width = BASE_COLORBAR_WIDTH * figure_scale

    power_gradient_cfg = style_cfg.get("power_gradient")
    if isinstance(power_gradient_cfg, Mapping):
        power_color_start = str(power_gradient_cfg.get("start") or DEFAULT_POWER_GRADIENT[0])
        power_color_end = str(power_gradient_cfg.get("end") or DEFAULT_POWER_GRADIENT[1])
    else:
        power_color_start, power_color_end = DEFAULT_POWER_GRADIENT

    uncertainty_gradient_cfg = style_cfg.get("uncertainty_gradient")
    if isinstance(uncertainty_gradient_cfg, Mapping):
        uncertainty_color_start = str(uncertainty_gradient_cfg.get("start") or DEFAULT_UNCERTAINTY_GRADIENT[0])
        uncertainty_color_end = str(uncertainty_gradient_cfg.get("end") or DEFAULT_UNCERTAINTY_GRADIENT[1])
    else:
        uncertainty_color_start, uncertainty_color_end = DEFAULT_UNCERTAINTY_GRADIENT

    legend_note_text = str(style_cfg.get("label_note") or DEFAULT_LABEL_NOTE)
    uncertainty_annotation = str(style_cfg.get("uncertainty_note") or DEFAULT_UNCERTAINTY_NOTE)

    reference_outline_color = str(style_cfg.get("reference_outline", DEFAULT_REFERENCE_OUTLINE))
    reference_outline_width = _to_float(style_cfg.get("reference_outline_width"), DEFAULT_REFERENCE_OUTLINE_WIDTH)
    reference_note = str(style_cfg.get("reference_note") or DEFAULT_REFERENCE_NOTE)

    power_subtitle = str(
        style_cfg.get(
            "power_subtitle",
            "Circle size scales with capacity factor; highlighted outline marks low coverage nodes.",
        )
    )
    uncertainty_subtitle = str(
        style_cfg.get("uncertainty_subtitle", "Dashed outline marks unreliable estimators.")
    )

    power_uncertainty_cfg = _cfg_get(config, "power_uncertainty")
    if not isinstance(power_uncertainty_cfg, Mapping):
        power_uncertainty_cfg = {}
    power_uncertainty_summary = _to_path(
        power_uncertainty_cfg.get("summary"),
        DEFAULT_POWER_UNCERTAINTY_SUMMARY,
    ).resolve()
    power_uncertainty_fields = power_uncertainty_cfg.get("fields")
    if not isinstance(power_uncertainty_fields, Mapping):
        power_uncertainty_fields = DEFAULT_POWER_UNCERTAINTY_FIELDS
    power_uncertainty_lookup = _load_power_uncertainty_summary(
        power_uncertainty_summary,
        estimate_field=str(power_uncertainty_fields.get("estimate", DEFAULT_POWER_UNCERTAINTY_FIELDS["estimate"])),
        lower_field=str(power_uncertainty_fields.get("lower", DEFAULT_POWER_UNCERTAINTY_FIELDS["lower"])),
        upper_field=str(power_uncertainty_fields.get("upper", DEFAULT_POWER_UNCERTAINTY_FIELDS["upper"])),
        bootstrap_field=power_uncertainty_fields.get("bootstrap_estimate"),
        replicates_field=power_uncertainty_fields.get("replicates"),
        confidence_field=power_uncertainty_fields.get("confidence"),
    )
    if not power_uncertainty_lookup:
        print(
            f"[geospatial_products] Warning: power uncertainty summary not found or empty: {power_uncertainty_summary}",
            file=sys.stderr,
        )

    def power_uncertainty_accessor(node: NodeMetrics) -> float | None:
        interval = power_uncertainty_lookup.get(node.node_id)
        if interval is None:
            return None
        if interval.lower is None or interval.upper is None:
            return None
        return max(interval.upper - interval.lower, 0.0)

    buoy_cfg = _cfg_get(config, "buoy")
    if not isinstance(buoy_cfg, Mapping):
        buoy_cfg = {}
    buoy_dataset_value = args.buoy_dataset if args.buoy_dataset is not None else buoy_cfg.get("dataset")
    buoy_dataset_path = _to_path(buoy_dataset_value, DEFAULT_BUOY_DATASET)
    buoy_node_id_value = args.buoy_node_id if args.buoy_node_id is not None else buoy_cfg.get("node_id")
    buoy_node_id = str(buoy_node_id_value or DEFAULT_BUOY_NODE_ID)
    buoy_enable_config = _as_bool(buoy_cfg.get("enable"))
    buoy_enabled = True
    if buoy_enable_config is not None:
        buoy_enabled = buoy_enable_config
    if args.disable_buoy_rose:
        buoy_enabled = False
    if buoy_enabled and not buoy_dataset_path.exists():
        buoy_enabled = False

    nodes = load_node_metrics(node_summary_path, taxonomy=taxonomy_records)
    if not nodes:
        raise RuntimeError("Node summary is empty; nothing to publish.")

    total_nodes = len(nodes)
    low_coverage_nodes = [node.node_id for node in nodes if node.low_coverage]
    power_uncertainty_stats: Dict[str, float | int] = {}
    if power_uncertainty_lookup:
        power_uncertainty_stats = summarise_power_uncertainty(
            power_uncertainty_lookup,
            total_nodes=total_nodes,
        )

    power_uncertainty_metadata_path: Path | None = None
    metadata_override = power_uncertainty_cfg.get("metadata") if isinstance(power_uncertainty_cfg, Mapping) else None
    if metadata_override not in (None, ""):
        candidate = _to_path(metadata_override, DEFAULT_POWER_UNCERTAINTY_SUMMARY).resolve()
        if candidate.exists():
            power_uncertainty_metadata_path = candidate
    if power_uncertainty_metadata_path is None:
        candidate = power_uncertainty_summary.with_name("bootstrap_metadata.json")
        if candidate.exists():
            power_uncertainty_metadata_path = candidate

    power_uncertainty_metadata_payload: Mapping[str, object] | None = None
    if power_uncertainty_metadata_path is not None:
        try:
            raw_text = power_uncertainty_metadata_path.read_text(encoding="utf-8")
            maybe_json = json.loads(raw_text)
            if isinstance(maybe_json, Mapping):
                power_uncertainty_metadata_payload = maybe_json
        except json.JSONDecodeError:
            power_uncertainty_metadata_payload = None

    power_uncertainty_section: Mapping[str, object] | None = None
    if power_uncertainty_lookup or power_uncertainty_summary.exists():
        section: Dict[str, object] = {
            "summary_source": _repo_relative(power_uncertainty_summary),
        }
        if power_uncertainty_metadata_path is not None:
            section["metadata_source"] = _repo_relative(power_uncertainty_metadata_path)
        if power_uncertainty_stats:
            section["stats"] = power_uncertainty_stats
        if isinstance(power_uncertainty_metadata_payload, Mapping):
            for key in ("confidence_level", "replicas", "resampling_mode", "block_length"):
                value = power_uncertainty_metadata_payload.get(key)
                if value is not None:
                    section[key] = value
        power_uncertainty_section = section

    dataset_path = resolve_dataset(dataset_override_path, config_path=stac_config_path, dataset_key=stac_dataset)

    artefacts = [
        csv_path,
        parquet_path,
        geojson_path,
        power_map_path,
        uncertainty_map_path,
        roses_path,
        histogram_csv,
        metadata_path,
    ]
    if buoy_enabled:
        artefacts.extend([buoy_panels_path, buoy_histogram_csv, buoy_stats_csv])
    if not args.overwrite:
        clashes = [path for path in artefacts if path.exists()]
        if clashes:
            joined = ", ".join(str(path) for path in clashes)
            raise FileExistsError(f"Artefacts already exist: {joined}. Use --overwrite to replace them.")

    write_resource_csv(nodes, csv_path)
    build_geoparquet(csv_path, parquet_path, engine=engine, image=image)
    write_geojson(nodes, geojson_path)

    render_metric_map(
        nodes,
        power_map_path,
        title="Power density (W/m²)",
        subtitle=power_subtitle,
        value_accessor=lambda node: node.power_density_w_m2,
        color_start=power_color_start,
        color_end=power_color_end,
        gradient_id="power-gradient",
        colorbar_title="Power density scale (W/m²)",
        colorbar_format=".0f",
        legend_items=[
            ("Circle size", "Capacity factor (scaled)"),
            ("Highlighted outline", "Low coverage flagged in taxonomy"),
        ],
        dash_when_unreliable=False,
        width=figure_width,
        height=figure_height,
        margins=figure_margins,
        node_padding=node_padding,
        low_coverage_color=low_coverage_stroke,
        low_coverage_stroke_width=low_coverage_stroke_width,
        note_text=legend_note_text,
        colorbar_width=colorbar_width,
        missing_style=None,
        reference_node_id=buoy_node_id if buoy_enabled else None,
        reference_outline_color=reference_outline_color,
        reference_outline_width=reference_outline_width,
        reference_note=reference_note,
        annotation_text=None,
    )

    render_metric_map(
        nodes,
        uncertainty_map_path,
        title="Power density uncertainty (bootstrap CI width)",
        subtitle=uncertainty_subtitle,
        value_accessor=power_uncertainty_accessor,
        color_start=uncertainty_color_start,
        color_end=uncertainty_color_end,
        gradient_id="uncertainty-gradient",
        colorbar_title="Power density CI width (W/m²)",
        colorbar_format=".1f",
        legend_items=[
            ("Circle size", "Capacity factor (scaled)"),
            ("Fill colour", "Bootstrap CI width (W/m²)"),
            ("Highlighted outline", "Low coverage flagged in taxonomy"),
            ("Dashed outline", "Estimator failed reliability checks"),
        ],
        dash_when_unreliable=True,
        width=figure_width,
        height=figure_height,
        margins=figure_margins,
        node_padding=node_padding,
        low_coverage_color=low_coverage_stroke,
        low_coverage_stroke_width=low_coverage_stroke_width,
        note_text=legend_note_text,
        colorbar_width=colorbar_width,
        missing_style=missing_style,
        reference_node_id=buoy_node_id if buoy_enabled else None,
        reference_outline_color=reference_outline_color,
        reference_outline_width=reference_outline_width,
        reference_note=reference_note,
        annotation_text=uncertainty_annotation,
    )

    representative_nodes = select_representative_nodes_for_roses(nodes, max_wind_roses)
    buoy_node = next((node for node in nodes if node.node_id == buoy_node_id), None)
    if buoy_enabled and buoy_node is not None and all(n.node_id != buoy_node.node_id for n in representative_nodes):
        if max_wind_roses > 0 and len(representative_nodes) >= max_wind_roses:
            representative_nodes = representative_nodes[:-1]
        representative_nodes = [buoy_node] + representative_nodes

    node_ids_for_hist: List[str] = []
    for node in representative_nodes:
        if node.node_id not in node_ids_for_hist:
            node_ids_for_hist.append(node.node_id)

    compute_wind_direction_histogram(
        dataset_path,
        node_ids_for_hist,
        histogram_csv,
        engine=engine,
        image=image,
    )
    ann_weights = load_direction_weights(histogram_csv)
    combined_weights: Dict[str, List[Tuple[int, float]]] = dict(ann_weights)

    comparison_descriptors: List[RoseDescriptor] = []
    ann_descriptors: List[RoseDescriptor] = []
    selected_nodes: List[str] = []
    buoy_hist_available = False
    buoy_mean_speed: float | None = None
    buoy_sample_count: float | None = None

    if buoy_enabled:
        compute_buoy_direction_histogram(
            buoy_dataset_path,
            output_csv=buoy_histogram_csv,
            stats_csv=buoy_stats_csv,
            engine=engine,
            image=image,
        )
        buoy_weight_map = load_direction_weights(buoy_histogram_csv)
        combined_weights.update(buoy_weight_map)
        stats_row = load_single_row_csv(buoy_stats_csv)
        buoy_mean_speed = stats_row.get("mean_speed")
        buoy_sample_count = stats_row.get("samples")
        buoy_hist_available = True
        comparison_descriptors.append(
            RoseDescriptor(
                identifier="reference_buoy",
                label="Vilano buoy (observed)",
                mean_speed_m_s=buoy_mean_speed,
                power_density_w_m2=None,
                capacity_factor=None,
            )
        )

    seen_ann_nodes: set[str] = set()
    for node in representative_nodes:
        if node.node_id in seen_ann_nodes:
            continue
        seen_ann_nodes.add(node.node_id)
        if node.node_id not in selected_nodes:
            selected_nodes.append(node.node_id)
        descriptor = RoseDescriptor(
            identifier=node.node_id,
            label=node.display_name,
            mean_speed_m_s=node.mean_speed_m_s,
            power_density_w_m2=node.power_density_w_m2,
            capacity_factor=node.capacity_factor,
            low_coverage=bool(node.low_coverage),
        )
        if buoy_enabled and node.node_id == buoy_node_id:
            comparison_descriptors.append(
                RoseDescriptor(
                    identifier=node.node_id,
                    label=f"{node.display_name} (ANN)",
                    mean_speed_m_s=node.mean_speed_m_s,
                    power_density_w_m2=node.power_density_w_m2,
                    capacity_factor=node.capacity_factor,
                    low_coverage=bool(node.low_coverage),
                )
            )
            continue
        ann_descriptors.append(descriptor)

    render_wind_rose_panels(ann_descriptors, combined_weights, roses_path)
    if comparison_descriptors:
        render_wind_rose_panels(comparison_descriptors, combined_weights, buoy_panels_path)

    write_metadata(
        metadata_path,
        node_summary=node_summary_path,
        dataset=dataset_path,
        geo_parquet=parquet_path,
        geojson=geojson_path,
        power_map=power_map_path,
        uncertainty_map=uncertainty_map_path,
        roses=roses_path,
        histogram=histogram_csv,
        selected_nodes=selected_nodes,
        buoy_histogram=buoy_histogram_csv if buoy_hist_available else None,
        buoy_mean_speed=buoy_mean_speed,
        buoy_samples=buoy_sample_count,
        config_path=config_source,
        buoy_panels=buoy_panels_path if comparison_descriptors else None,
        taxonomy=taxonomy_path,
        low_coverage_nodes=low_coverage_nodes,
        power_uncertainty=power_uncertainty_section,
    )


def select_representative_nodes_for_roses(
    nodes: Sequence[NodeMetrics],
    max_items: int,
) -> List[NodeMetrics]:
    if max_items <= 0:
        return []
    ordered: List[NodeMetrics] = []

    def consider(candidate: NodeMetrics) -> None:
        if len(ordered) >= max_items:
            return
        if candidate not in ordered:
            ordered.append(candidate)

    reliable_nodes = [n for n in nodes if n.reliable_estimate is not False]
    sorted_power = sorted(reliable_nodes, key=lambda n: (n.power_density_w_m2 or -math.inf), reverse=True)
    for node in sorted_power:
        consider(node)
        if len(ordered) >= max_items:
            break

    high_uncertainty = sorted(reliable_nodes, key=lambda n: (n.km_interval_width_m_s or 0.0), reverse=True)
    for node in high_uncertainty:
        consider(node)
        if len(ordered) >= max_items:
            break

    low_coverage = [n for n in nodes if n.low_coverage]
    for node in low_coverage:
        consider(node)
        if len(ordered) >= max_items:
            break

    if len(ordered) < max_items:
        remaining = sorted(nodes, key=lambda n: (n.capacity_factor or -math.inf), reverse=True)
        for node in remaining:
            consider(node)
            if len(ordered) >= max_items:
                break

    return ordered[:max_items]


if __name__ == "__main__":
    main()
