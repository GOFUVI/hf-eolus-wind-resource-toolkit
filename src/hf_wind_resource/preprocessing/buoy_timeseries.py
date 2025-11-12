"""Ingestion and synchronisation helpers for buoy reference datasets.

These utilities operate on GeoParquet time series delivered by agencies such
as Puertos del Estado and align them with ANN predictions produced at the
corresponding mesh nodes. The code assumes the buoy dataset exposes at least
the columns ``timestamp``, ``wind_speed`` and ``wind_dir`` (with sentinel
values for missing measurements), while the ANN GeoParquet follows the schema
documented in ``docs/sar_range_final_schema.md``.

Typical workflow:

* load the raw buoy time series, converting sentinel values to missing
  observations and quantifying their occurrence;
* extract the ANN inference subset for a target ``node_id``;
* align both series on a common temporal grid, optionally allowing
  nearest-neighbour matches within a configurable tolerance.

Downstream statistics can use the returned artefacts to compute skill scores,
bootstrap uncertainty estimates, or derive wind-resource metrics with explicit
traceability back to the raw measurements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Literal, Sequence

import math

import numpy as np
import pandas as pd
import pyarrow as pa

__all__ = [
    "BuoySentinelConfig",
    "BuoySeries",
    "HeightCorrectionConfig",
    "HeightCorrectionResult",
    "load_height_correction_from_config",
    "CadenceSummary",
    "SynchronisationConfig",
    "SynchronisationSummary",
    "BuoyPreparation",
    "build_geoparquet_table",
    "load_buoy_timeseries",
    "load_ann_node_timeseries",
    "synchronise_buoy_and_ann",
    "prepare_buoy_timeseries",
]


@dataclass(frozen=True)
class BuoySentinelConfig:
    """Sentinel filtering thresholds for buoy time series."""

    speed_threshold: float = -9000.0
    direction_threshold: int = -9000


@dataclass(frozen=True)
class CadenceSummary:
    """Cadence diagnostics for a time series."""

    nominal: timedelta | None
    unique_intervals: tuple[timedelta, ...]
    sample_count: int


@dataclass(frozen=True)
class BuoySeries:
    """Container for the cleaned buoy data and QC diagnostics."""

    dataframe: pd.DataFrame
    total_records: int
    dropped_speed_records: int
    direction_sentinel_records: int
    coverage_start: datetime | None
    coverage_end: datetime | None
    cadence: CadenceSummary
    height_correction: "HeightCorrectionResult | None"


@dataclass(frozen=True)
class HeightCorrectionConfig:
    """Parameters controlling the vertical adjustment of buoy wind speeds."""

    method: Literal["power_law", "log_profile"] = "log_profile"
    measurement_height_m: float = 3.0
    target_height_m: float = 10.0
    power_law_alpha: float = 0.11
    roughness_length_m: float = 0.0002

    def __post_init__(self) -> None:
        if self.method not in {"power_law", "log_profile"}:
            raise ValueError(
                "method must be either 'power_law' or 'log_profile'",
            )
        if self.measurement_height_m <= 0.0:
            raise ValueError("measurement_height_m must be positive")
        if self.target_height_m <= 0.0:
            raise ValueError("target_height_m must be positive")
        if math.isclose(self.measurement_height_m, self.target_height_m):
            raise ValueError("measurement_height_m must differ from target_height_m")
        if self.method == "power_law" and self.power_law_alpha <= 0.0:
            raise ValueError("power_law_alpha must be positive")
        if self.method == "log_profile":
            if self.roughness_length_m <= 0.0:
                raise ValueError("roughness_length_m must be positive")
            if self.roughness_length_m >= min(
                self.measurement_height_m, self.target_height_m
            ):
                raise ValueError(
                    "roughness_length_m must be lower than both measurement and target heights"
                )


@dataclass(frozen=True)
class HeightCorrectionResult:
    """Diagnostics describing the vertical wind-speed adjustment."""

    method: str
    measurement_height_m: float
    target_height_m: float
    parameters: tuple[tuple[str, float], ...]
    scale_factor: float
    hypotheses: tuple[str, ...]


_DEFAULT_HEIGHT_CONFIG_PATH = Path("config") / "buoy_height.json"


def load_height_correction_from_config(
    path: Path | str | None = None,
) -> HeightCorrectionConfig:
    """Load the default height-correction parameters from JSON configuration."""

    target_path = Path(path) if path is not None else _DEFAULT_HEIGHT_CONFIG_PATH
    base = HeightCorrectionConfig()

    if not target_path.exists():
        return base

    try:
        payload = json.loads(target_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in height-correction configuration: {target_path}",
        ) from exc

    method_raw = str(payload.get("method", base.method)).lower()
    if method_raw not in {"power_law", "log_profile"}:
        raise ValueError(
            "Height-correction configuration must declare method 'power_law' or 'log_profile'."
        )
    method: Literal["power_law", "log_profile"] = (
        "power_law" if method_raw == "power_law" else "log_profile"
    )
    measurement = float(payload.get("measurement_height_m", base.measurement_height_m))
    target_height = float(payload.get("target_height_m", base.target_height_m))
    power_alpha = float(payload.get("power_law_alpha", base.power_law_alpha))
    roughness = float(payload.get("roughness_length_m", base.roughness_length_m))

    return HeightCorrectionConfig(
        method=method,
        measurement_height_m=measurement,
        target_height_m=target_height,
        power_law_alpha=power_alpha,
        roughness_length_m=roughness,
    )


def _calculate_height_scaling(
    config: HeightCorrectionConfig,
) -> tuple[float, tuple[tuple[str, float], ...], tuple[str, ...]]:
    """Return the multiplicative scale factor and metadata for a configuration."""

    ratio: float
    parameters: tuple[tuple[str, float], ...]
    hypotheses: tuple[str, ...]

    if config.method == "power_law":
        ratio = (config.target_height_m / config.measurement_height_m) ** (
            config.power_law_alpha
        )
        parameters = (("power_law_alpha", config.power_law_alpha),)
        hypotheses = (
            "Neutral-stability power law with marine boundary-layer exponent.",
        )
    else:
        numerator = math.log(config.target_height_m / config.roughness_length_m)
        denominator = math.log(config.measurement_height_m / config.roughness_length_m)
        ratio = numerator / denominator
        parameters = (("roughness_length_m", config.roughness_length_m),)
        hypotheses = (
            "Neutral logarithmic profile over homogeneous sea surface roughness.",
        )

    return ratio, parameters, hypotheses


def _apply_height_correction(
    frame: pd.DataFrame,
    config: HeightCorrectionConfig,
) -> HeightCorrectionResult:
    """Apply an in-place vertical wind-speed correction."""

    if "wind_speed" not in frame.columns:
        raise KeyError("wind_speed column missing from buoy dataframe")

    scale_factor, parameters, hypotheses = _calculate_height_scaling(config)

    frame["wind_speed_original_height"] = frame["wind_speed"].astype(float)
    frame["wind_speed"] = frame["wind_speed_original_height"] * scale_factor

    return HeightCorrectionResult(
        method=config.method,
        measurement_height_m=config.measurement_height_m,
        target_height_m=config.target_height_m,
        parameters=parameters,
        scale_factor=scale_factor,
        hypotheses=hypotheses,
    )


@dataclass(frozen=True)
class SynchronisationConfig:
    """Behavioural knobs for temporal synchronisation."""

    tolerance: timedelta = timedelta(minutes=30)
    prefer_nearest: bool = False

    def __post_init__(self) -> None:
        if self.tolerance <= timedelta(0):
            raise ValueError("tolerance must be positive")


@dataclass(frozen=True)
class SynchronisationSummary:
    """Diagnostics describing the synchronisation outcome."""

    matched_rows: int
    unmatched_ann_rows: int
    unmatched_buoy_rows: int
    exact_matches: int
    nearest_matches: int
    match_ratio_ann: float | None
    match_ratio_buoy: float | None


@dataclass(frozen=True)
class BuoyPreparation:
    """Aggregate artefacts produced when preparing a buoy dataset."""

    buoy: BuoySeries
    ann_dataframe: pd.DataFrame
    matched_dataframe: pd.DataFrame
    synchronisation: SynchronisationSummary


def _ensure_datetime_utc(series: pd.Series) -> pd.Series:
    converted = pd.to_datetime(series, utc=True)
    # ``to_datetime`` already returns tz-aware values when utc=True. We call
    # ``tz_convert`` defensively in case the input was a tz-aware dtype.
    return converted.dt.tz_convert("UTC")


def _compute_cadence(timestamps: pd.Series) -> CadenceSummary:
    if timestamps.empty:
        return CadenceSummary(nominal=None, unique_intervals=tuple(), sample_count=0)

    deltas = timestamps.sort_values().diff().dropna()
    if deltas.empty:
        return CadenceSummary(nominal=None, unique_intervals=tuple(), sample_count=0)

    seconds = deltas.dt.total_seconds().astype(int)
    value_counts = seconds.value_counts()
    max_count = int(value_counts.max())
    dominant_candidates: Iterable[int] = (
        value for value, count in value_counts.items() if int(count) == max_count
    )
    nominal_seconds = min(int(value) for value in dominant_candidates)
    nominal = timedelta(seconds=nominal_seconds)
    unique_intervals = tuple(
        timedelta(seconds=int(value)) for value in sorted(set(int(v) for v in seconds))
    )
    return CadenceSummary(
        nominal=nominal,
        unique_intervals=unique_intervals,
        sample_count=len(seconds),
    )


def load_buoy_timeseries(
    dataset_path: Path | str,
    sentinel_config: BuoySentinelConfig | None = None,
    height_correction: HeightCorrectionConfig | Literal["auto"] | None = "auto",
) -> BuoySeries:
    """Load and clean a buoy GeoParquet time series.

    Parameters
    ----------
    dataset_path:
        Path to the buoy GeoParquet asset.
    sentinel_config:
        Thresholds that define the sentinel values used to encode missing wind
        speed and direction observations.
    height_correction:
        Configuration for the vertical wind-speed correction. When set to
        ``"auto"``, the parameters declared in ``config/buoy_height.json`` are
        used. When ``None``, the raw buoy wind speeds are returned at their
        original measurement height.

    Returns
    -------
    BuoySeries
        Cleaned DataFrame alongside QC diagnostics such as sentinel counts,
        coverage, and cadence estimates.
    """

    if sentinel_config is None:
        sentinel_config = BuoySentinelConfig()

    frame = pd.read_parquet(dataset_path)
    required_columns = {"timestamp", "wind_speed", "wind_dir"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise KeyError(f"Buoy dataset is missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["timestamp"] = _ensure_datetime_utc(frame["timestamp"])
    frame.sort_values("timestamp", inplace=True)
    frame.reset_index(drop=True, inplace=True)

    total_records = len(frame)
    speed_mask = frame["wind_speed"] <= sentinel_config.speed_threshold
    dropped_speed_records = int(speed_mask.sum())
    if dropped_speed_records:
        frame = frame.loc[~speed_mask].copy()

    # Convert explicit sentinel markers in the wind direction to missing values
    # but keep the records so that the speed measurements remain available.
    frame["wind_dir"] = frame["wind_dir"].astype("Int64")
    direction_mask = frame["wind_dir"] <= sentinel_config.direction_threshold
    direction_sentinel_records = int(direction_mask.sum())
    if direction_sentinel_records:
        frame.loc[direction_mask, "wind_dir"] = pd.NA

    frame["timestamp"] = _ensure_datetime_utc(frame["timestamp"])
    frame.sort_values("timestamp", inplace=True)
    frame.reset_index(drop=True, inplace=True)

    height_result: HeightCorrectionResult | None = None
    if height_correction == "auto":
        height_correction = load_height_correction_from_config()

    if height_correction is not None:
        height_result = _apply_height_correction(frame, height_correction)

    coverage_start: datetime | None
    coverage_end: datetime | None
    if frame.empty:
        coverage_start = None
        coverage_end = None
    else:
        coverage_start = frame.iloc[0]["timestamp"].to_pydatetime()
        coverage_end = frame.iloc[-1]["timestamp"].to_pydatetime()

    cadence = _compute_cadence(frame["timestamp"])

    return BuoySeries(
        dataframe=frame,
        total_records=total_records,
        dropped_speed_records=dropped_speed_records,
        direction_sentinel_records=direction_sentinel_records,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        cadence=cadence,
        height_correction=height_result,
    )


def load_ann_node_timeseries(
    dataset_path: Path | str,
    node_id: str,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load the subset of the ANN GeoParquet corresponding to a single node."""

    if columns is None:
        columns = (
            "timestamp",
            "node_id",
            "pred_wind_speed",
            "pred_wind_direction",
            "pred_range_label",
            "prob_range_below",
            "prob_range_in",
            "prob_range_above",
            "range_flag",
            "range_flag_confident",
        )

    frame = pd.read_parquet(dataset_path, columns=list(columns))
    if "node_id" not in frame.columns:
        raise KeyError("ANN dataset must expose a node_id column")

    frame = frame.loc[frame["node_id"] == node_id].copy()
    if frame.empty:
        raise ValueError(f"No ANN records found for node_id={node_id!r}")

    frame["timestamp"] = _ensure_datetime_utc(frame["timestamp"])
    frame.sort_values("timestamp", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def synchronise_buoy_and_ann(
    buoy: pd.DataFrame,
    ann: pd.DataFrame,
    config: SynchronisationConfig | None = None,
) -> tuple[pd.DataFrame, SynchronisationSummary]:
    """Synchronise buoy measurements with ANN predictions on a time axis."""

    if config is None:
        config = SynchronisationConfig()

    if "timestamp" not in buoy.columns:
        raise KeyError("Buoy dataframe requires a timestamp column")
    if "timestamp" not in ann.columns:
        raise KeyError("ANN dataframe requires a timestamp column")

    ann_sorted = ann.copy()
    ann_sorted.rename(columns={"timestamp": "timestamp_ann"}, inplace=True)
    ann_sorted.sort_values("timestamp_ann", inplace=True)
    ann_sorted.reset_index(drop=True, inplace=True)

    buoy_sorted = buoy.copy()
    buoy_sorted.rename(columns={"timestamp": "timestamp_buoy"}, inplace=True)
    buoy_sorted.sort_values("timestamp_buoy", inplace=True)
    buoy_sorted.reset_index(drop=True, inplace=True)

    direction = "nearest" if config.prefer_nearest else "backward"
    matched = pd.merge_asof(
        ann_sorted,
        buoy_sorted,
        left_on="timestamp_ann",
        right_on="timestamp_buoy",
        direction=direction,
        tolerance=pd.Timedelta(config.tolerance),
    )

    # Rows outside the tolerance remain unmatched (NaNs on buoy columns).
    matched = matched.dropna(subset=["wind_speed"]).copy()
    if matched.empty:
        summary = SynchronisationSummary(
            matched_rows=0,
            unmatched_ann_rows=len(ann_sorted),
            unmatched_buoy_rows=len(buoy_sorted),
            exact_matches=0,
            nearest_matches=0,
            match_ratio_ann=0.0 if len(ann_sorted) else None,
            match_ratio_buoy=0.0 if len(buoy_sorted) else None,
        )
        return matched, summary

    matched["timestamp_ann"] = _ensure_datetime_utc(matched["timestamp_ann"])
    matched["timestamp_buoy"] = _ensure_datetime_utc(matched["timestamp_buoy"])
    matched["time_offset_seconds"] = (
        matched["timestamp_ann"] - matched["timestamp_buoy"]
    ).dt.total_seconds()
    matched["is_exact_match"] = np.isclose(matched["time_offset_seconds"], 0.0)

    if not config.prefer_nearest:
        matched = matched.loc[matched["is_exact_match"]].copy()

    exact_matches = int(matched["is_exact_match"].sum())
    nearest_matches = int(len(matched) - exact_matches)

    matched_rows = len(matched)
    ann_total = len(ann_sorted)
    buoy_total = len(buoy_sorted)
    unmatched_ann = ann_total - matched_rows
    unmatched_buoy = max(buoy_total - matched_rows, 0)

    match_ratio_ann = matched_rows / ann_total if ann_total else None
    match_ratio_buoy = matched_rows / buoy_total if buoy_total else None

    summary = SynchronisationSummary(
        matched_rows=matched_rows,
        unmatched_ann_rows=unmatched_ann,
        unmatched_buoy_rows=unmatched_buoy,
        exact_matches=exact_matches,
        nearest_matches=nearest_matches,
        match_ratio_ann=match_ratio_ann,
        match_ratio_buoy=match_ratio_buoy,
    )

    return matched, summary


def prepare_buoy_timeseries(
    buoy_dataset: Path | str,
    ann_dataset: Path | str,
    node_id: str,
    height_correction_config: HeightCorrectionConfig | Literal["auto"] | None = "auto",
    sentinel_config: BuoySentinelConfig | None = None,
    synchronisation_config: SynchronisationConfig | None = None,
    ann_columns: Sequence[str] | None = None,
) -> BuoyPreparation:
    """End-to-end helper that loads, cleans, and synchronises a buoy dataset.

    The buoy wind speeds are vertically adjusted before the temporal alignment
    so that downstream comparisons operate at the reference height.
    """

    buoy_series = load_buoy_timeseries(
        dataset_path=buoy_dataset,
        sentinel_config=sentinel_config,
        height_correction=height_correction_config,
    )
    ann_frame = load_ann_node_timeseries(
        dataset_path=ann_dataset,
        node_id=node_id,
        columns=ann_columns,
    )
    matched, summary = synchronise_buoy_and_ann(
        buoy=buoy_series.dataframe,
        ann=ann_frame,
        config=synchronisation_config,
    )
    return BuoyPreparation(
        buoy=buoy_series,
        ann_dataframe=ann_frame,
        matched_dataframe=matched,
        synchronisation=summary,
    )


def build_geoparquet_table(
    frame: pd.DataFrame,
    geometry_column: str = "geometry",
    crs: str = "EPSG:4326",
    geometry_types: Sequence[str] | None = None,
) -> pa.Table:
    """Convert *frame* into a GeoParquet-ready ``pyarrow.Table``."""

    if geometry_column not in frame.columns:
        raise KeyError(f"geometry column {geometry_column!r} not found in frame")

    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    column_meta: dict[str, object] = {"encoding": "WKB", "crs": crs}
    if geometry_types:
        column_meta["geometry_types"] = list(geometry_types)

    geo_meta = {
        "version": "1.0.0",
        "primary_column": geometry_column,
        "columns": {geometry_column: column_meta},
    }
    metadata[b"geo"] = json.dumps(geo_meta).encode("utf-8")
    return table.replace_schema_metadata(metadata)
