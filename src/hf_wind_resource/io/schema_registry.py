"""Central registry for the SAR range-aware GeoParquet schema.

The ANN-derived dataset powering the wind-resource pipeline is expected to
remain stable throughout the project lifecycle. Keeping its column layout,
logical types, and documentation in a single module ensures all
data-access layers share the same view of the schema and that automated
tests can detect accidental drift (e.g. columns added/removed upstream).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Iterator, Mapping, Tuple

try:  # Optional dependency: only required when callers request PyArrow objects.
    import pyarrow as pa
except ImportError:  # pragma: no cover - exercised when PyArrow is unavailable.
    pa = None

__all__ = ["SchemaField", "SarRangeFinalSchema"]


@dataclass(frozen=True, slots=True)
class SchemaField:
    """Represents a column in the SAR range-aware GeoParquet dataset."""

    name: str
    stac_type: str
    description: str

    def stac_entry(self) -> Mapping[str, str]:
        """Return a mapping compatible with STAC `table:columns` entries."""

        return {"name": self.name, "type": self.stac_type, "description": self.description}


class SarRangeFinalSchema:
    """Canonical schema definition for ``sar_range_final_pivots_joined``.

    The field catalogue mirrors the STAC metadata tracked under
    ``use_case/catalogs/sar_range_final_pivots_joined/items/data.json`` and the
    narrative documentation in ``docs/sar_range_final_schema.md``. Columns
    are declared in the exact order stored in the GeoParquet snapshot. Tests
    should rely on :class:`SarRangeFinalSchema` rather than redefining the
    schema ad-hoc so that column changes are caught in a single place.
    """

    _ARROW_TYPE_FACTORIES: ClassVar[Mapping[str, "pa.DataType"]] = {}
    _DUCKDB_TYPE_MAP: ClassVar[Mapping[str, str]] = {
        "datetime": "TIMESTAMP",
        "string": "VARCHAR",
        "binary": "BLOB",
        "integer": "BIGINT",
        "number": "DOUBLE",
        "boolean": "BOOLEAN",
    }

    _FIELDS: ClassVar[Tuple[SchemaField, ...]] = (
        SchemaField("timestamp", "datetime", "Observation timestamp (UTC)."),
        SchemaField("node_id", "string", "Spatial grid node identifier (unitless)."),
        SchemaField("geometry", "binary", "Cell geometry encoded as WKB in CRS84 (lon/lat)."),
        SchemaField(
            "vila_aggregated__n_0",
            "integer",
            "VILA: count of contributing observations (negative Bragg peak, unitless).",
        ),
        SchemaField(
            "vila_aggregated__n_1",
            "integer",
            "VILA: count of contributing observations (positive Bragg peak, unitless).",
        ),
        SchemaField(
            "vila_aggregated__pwr_mean_0",
            "number",
            "VILA: mean signal power (negative Bragg peak), in dB.",
        ),
        SchemaField(
            "vila_aggregated__pwr_mean_1",
            "number",
            "VILA: mean signal power (positive Bragg peak), in dB.",
        ),
        SchemaField(
            "vila_aggregated__pwr_median_0",
            "number",
            "VILA: median signal power (negative Bragg peak), in dB.",
        ),
        SchemaField(
            "vila_aggregated__pwr_median_1",
            "number",
            "VILA: median signal power (positive Bragg peak), in dB.",
        ),
        SchemaField(
            "vila_aggregated__pwr_stddev_0",
            "number",
            "VILA: standard deviation of signal power (negative Bragg peak), in dB.",
        ),
        SchemaField(
            "vila_aggregated__pwr_stddev_1",
            "number",
            "VILA: standard deviation of signal power (positive Bragg peak), in dB.",
        ),
        SchemaField(
            "vila_aggregated__pwr_min_0",
            "number",
            "VILA: minimum signal power (negative Bragg peak), in dB.",
        ),
        SchemaField(
            "vila_aggregated__pwr_min_1",
            "number",
            "VILA: minimum signal power (positive Bragg peak), in dB.",
        ),
        SchemaField(
            "vila_aggregated__pwr_max_0",
            "number",
            "VILA: maximum signal power (negative Bragg peak), in dB.",
        ),
        SchemaField(
            "vila_aggregated__pwr_max_1",
            "number",
            "VILA: maximum signal power (positive Bragg peak), in dB.",
        ),
        SchemaField(
            "vila_aggregated__pwr_n_0",
            "integer",
            "VILA: sample count used for power statistics (negative Bragg peak, unitless).",
        ),
        SchemaField(
            "vila_aggregated__pwr_n_1",
            "integer",
            "VILA: sample count used for power statistics (positive Bragg peak, unitless).",
        ),
        SchemaField(
            "vila_aggregated__pwr_mad_0",
            "number",
            "VILA: median absolute deviation of power (negative Bragg peak), in dB.",
        ),
        SchemaField(
            "vila_aggregated__pwr_mad_1",
            "number",
            "VILA: median absolute deviation of power (positive Bragg peak), in dB.",
        ),
        SchemaField(
            "vila_aggregated__velo_mean_0",
            "number",
            "VILA: mean radial velocity (negative Bragg peak), in cm/s.",
        ),
        SchemaField(
            "vila_aggregated__velo_mean_1",
            "number",
            "VILA: mean radial velocity (positive Bragg peak), in cm/s.",
        ),
        SchemaField(
            "vila_aggregated__velo_median_0",
            "number",
            "VILA: median radial velocity (negative Bragg peak), in cm/s.",
        ),
        SchemaField(
            "vila_aggregated__velo_median_1",
            "number",
            "VILA: median radial velocity (positive Bragg peak), in cm/s.",
        ),
        SchemaField(
            "vila_aggregated__velo_stddev_0",
            "number",
            "VILA: standard deviation of radial velocity (negative Bragg peak), in cm/s.",
        ),
        SchemaField(
            "vila_aggregated__velo_stddev_1",
            "number",
            "VILA: standard deviation of radial velocity (positive Bragg peak), in cm/s.",
        ),
        SchemaField(
            "vila_aggregated__velo_min_0",
            "number",
            "VILA: minimum radial velocity (negative Bragg peak), in cm/s.",
        ),
        SchemaField(
            "vila_aggregated__velo_min_1",
            "number",
            "VILA: minimum radial velocity (positive Bragg peak), in cm/s.",
        ),
        SchemaField(
            "vila_aggregated__velo_max_0",
            "number",
            "VILA: maximum radial velocity (negative Bragg peak), in cm/s.",
        ),
        SchemaField(
            "vila_aggregated__velo_max_1",
            "number",
            "VILA: maximum radial velocity (positive Bragg peak), in cm/s.",
        ),
        SchemaField(
            "vila_aggregated__velo_n_0",
            "integer",
            "VILA: sample count used for radial velocity statistics (negative Bragg peak, unitless).",
        ),
        SchemaField(
            "vila_aggregated__velo_n_1",
            "integer",
            "VILA: sample count used for radial velocity statistics (positive Bragg peak, unitless).",
        ),
        SchemaField(
            "vila_aggregated__velo_mad_0",
            "number",
            "VILA: MAD of radial velocity (negative Bragg peak), in cm/s.",
        ),
        SchemaField(
            "vila_aggregated__velo_mad_1",
            "number",
            "VILA: MAD of radial velocity (positive Bragg peak), in cm/s.",
        ),
        SchemaField(
            "vila_bearing",
            "number",
            "VILA: bearing from station to cell centroid, degrees clockwise from geographic north.",
        ),
        SchemaField(
            "vila_dist_km",
            "number",
            "VILA: great-circle distance from station to cell centroid, in kilometres.",
        ),
        SchemaField(
            "vila_maintenance_interval_id",
            "string",
            "VILA: identifier of the maintenance interval covering the observation (unitless).",
        ),
        SchemaField(
            "vila_maintenance_type",
            "string",
            "VILA: maintenance category reported by station logs (e.g., calibration, outage).",
        ),
        SchemaField(
            "vila_maintenance_start",
            "string",
            "VILA: ISO8601 timestamp marking the start of the active maintenance interval.",
        ),
        SchemaField(
            "vila_hours_since_last_calibration",
            "number",
            "VILA: elapsed hours since the station's last calibration event.",
        ),
        SchemaField(
            "prio_aggregated__n_0",
            "integer",
            "PRIO: count of contributing observations (negative Bragg peak, unitless).",
        ),
        SchemaField(
            "prio_aggregated__n_1",
            "integer",
            "PRIO: count of contributing observations (positive Bragg peak, unitless).",
        ),
        SchemaField(
            "prio_aggregated__pwr_mean_0",
            "number",
            "PRIO: mean signal power (negative Bragg peak), in dB.",
        ),
        SchemaField(
            "prio_aggregated__pwr_mean_1",
            "number",
            "PRIO: mean signal power (positive Bragg peak), in dB.",
        ),
        SchemaField(
            "prio_aggregated__pwr_median_0",
            "number",
            "PRIO: median signal power (negative Bragg peak), in dB.",
        ),
        SchemaField(
            "prio_aggregated__pwr_median_1",
            "number",
            "PRIO: median signal power (positive Bragg peak), in dB.",
        ),
        SchemaField(
            "prio_aggregated__pwr_stddev_0",
            "number",
            "PRIO: standard deviation of signal power (negative Bragg peak), in dB.",
        ),
        SchemaField(
            "prio_aggregated__pwr_stddev_1",
            "number",
            "PRIO: standard deviation of signal power (positive Bragg peak), in dB.",
        ),
        SchemaField(
            "prio_aggregated__pwr_min_0",
            "number",
            "PRIO: minimum signal power (negative Bragg peak), in dB.",
        ),
        SchemaField(
            "prio_aggregated__pwr_min_1",
            "number",
            "PRIO: minimum signal power (positive Bragg peak), in dB.",
        ),
        SchemaField(
            "prio_aggregated__pwr_max_0",
            "number",
            "PRIO: maximum signal power (negative Bragg peak), in dB.",
        ),
        SchemaField(
            "prio_aggregated__pwr_max_1",
            "number",
            "PRIO: maximum signal power (positive Bragg peak), in dB.",
        ),
        SchemaField(
            "prio_aggregated__pwr_n_0",
            "integer",
            "PRIO: sample count used for power statistics (negative Bragg peak, unitless).",
        ),
        SchemaField(
            "prio_aggregated__pwr_n_1",
            "integer",
            "PRIO: sample count used for power statistics (positive Bragg peak, unitless).",
        ),
        SchemaField(
            "prio_aggregated__pwr_mad_0",
            "number",
            "PRIO: median absolute deviation of power (negative Bragg peak), in dB.",
        ),
        SchemaField(
            "prio_aggregated__pwr_mad_1",
            "number",
            "PRIO: median absolute deviation of power (positive Bragg peak), in dB.",
        ),
        SchemaField(
            "prio_aggregated__velo_mean_0",
            "number",
            "PRIO: mean radial velocity (negative Bragg peak), in cm/s.",
        ),
        SchemaField(
            "prio_aggregated__velo_mean_1",
            "number",
            "PRIO: mean radial velocity (positive Bragg peak), in cm/s.",
        ),
        SchemaField(
            "prio_aggregated__velo_median_0",
            "number",
            "PRIO: median radial velocity (negative Bragg peak), in cm/s.",
        ),
        SchemaField(
            "prio_aggregated__velo_median_1",
            "number",
            "PRIO: median radial velocity (positive Bragg peak), in cm/s.",
        ),
        SchemaField(
            "prio_aggregated__velo_stddev_0",
            "number",
            "PRIO: standard deviation of radial velocity (negative Bragg peak), in cm/s.",
        ),
        SchemaField(
            "prio_aggregated__velo_stddev_1",
            "number",
            "PRIO: standard deviation of radial velocity (positive Bragg peak), in cm/s.",
        ),
        SchemaField(
            "prio_aggregated__velo_min_0",
            "number",
            "PRIO: minimum radial velocity (negative Bragg peak), in cm/s.",
        ),
        SchemaField(
            "prio_aggregated__velo_min_1",
            "number",
            "PRIO: minimum radial velocity (positive Bragg peak), in cm/s.",
        ),
        SchemaField(
            "prio_aggregated__velo_max_0",
            "number",
            "PRIO: maximum radial velocity (negative Bragg peak), in cm/s.",
        ),
        SchemaField(
            "prio_aggregated__velo_max_1",
            "number",
            "PRIO: maximum radial velocity (positive Bragg peak), in cm/s.",
        ),
        SchemaField(
            "prio_aggregated__velo_n_0",
            "integer",
            "PRIO: sample count used for radial velocity statistics (negative Bragg peak, unitless).",
        ),
        SchemaField(
            "prio_aggregated__velo_n_1",
            "integer",
            "PRIO: sample count used for radial velocity statistics (positive Bragg peak, unitless).",
        ),
        SchemaField(
            "prio_aggregated__velo_mad_0",
            "number",
            "PRIO: MAD of radial velocity (negative Bragg peak), in cm/s.",
        ),
        SchemaField(
            "prio_aggregated__velo_mad_1",
            "number",
            "PRIO: MAD of radial velocity (positive Bragg peak), in cm/s.",
        ),
        SchemaField(
            "prio_bearing",
            "number",
            "PRIO: bearing from station to cell centroid, degrees clockwise from geographic north.",
        ),
        SchemaField(
            "prio_dist_km",
            "number",
            "PRIO: great-circle distance from station to cell centroid, in kilometres.",
        ),
        SchemaField(
            "prio_maintenance_interval_id",
            "string",
            "PRIO: identifier of the maintenance interval covering the observation (unitless).",
        ),
        SchemaField(
            "prio_maintenance_type",
            "string",
            "PRIO: maintenance category reported by station logs (e.g., calibration, outage).",
        ),
        SchemaField(
            "prio_maintenance_start",
            "string",
            "PRIO: ISO8601 timestamp marking the start of the active maintenance interval.",
        ),
        SchemaField(
            "prio_hours_since_last_calibration",
            "number",
            "PRIO: elapsed hours since the station's last calibration event.",
        ),
        SchemaField("pred_wind_speed", "number", "Model prediction for wind speed (m/s)."),
        SchemaField(
            "pred_cos_wind_dir",
            "number",
            "Cosine component of the predicted wind direction unit vector (unitless).",
        ),
        SchemaField(
            "pred_sin_wind_dir",
            "number",
            "Sine component of the predicted wind direction unit vector (unitless).",
        ),
        SchemaField(
            "pred_wind_direction",
            "number",
            "Model prediction for wind direction (degrees clockwise from geographic north).",
        ),
        SchemaField(
            "prob_range_below",
            "number",
            "Posterior probability that wind speed lies below the calibrated operating range (0-1).",
        ),
        SchemaField(
            "prob_range_in",
            "number",
            "Posterior probability that wind speed lies within the calibrated operating range (0-1).",
        ),
        SchemaField(
            "prob_range_above",
            "number",
            "Posterior probability that wind speed lies above the calibrated operating range (0-1).",
        ),
        SchemaField(
            "pred_range_label",
            "string",
            "Discrete range class selected by the classifier (below/in/above).",
        ),
        SchemaField(
            "pred_range_confidence",
            "number",
            "Confidence associated with the predicted range label (0-1).",
        ),
        SchemaField(
            "pred_speed_range_label",
            "string",
            "Range class inferred deterministically from the predicted wind speed (below/in/above).",
        ),
        SchemaField(
            "range_near_lower_margin",
            "boolean",
            "Flag indicating the predicted wind speed lies near the lower reliability margin.",
        ),
        SchemaField(
            "range_near_upper_margin",
            "boolean",
            "Flag indicating the predicted wind speed lies near the upper reliability margin.",
        ),
        SchemaField(
            "range_near_any_margin",
            "boolean",
            "Flag indicating the predicted wind speed lies near any reliability margin.",
        ),
        SchemaField(
            "range_flag",
            "string",
            "Consolidated range-awareness flag (below/in/above/uncertain).",
        ),
        SchemaField(
            "range_flag_confident",
            "boolean",
            "True when the range-awareness flag exceeds the configured confidence threshold.",
        ),
        SchemaField(
            "range_prediction_consistent",
            "boolean",
            "Indicates whether the range-awareness flag matches the class derived from predicted wind speed.",
        ),
    )

    def __iter__(self) -> Iterator[SchemaField]:
        """Iterate over declared schema fields in storage order."""

        return iter(self._FIELDS)

    def field_names(self) -> Tuple[str, ...]:
        """Return the ordered list of column names."""

        return tuple(field.name for field in self._FIELDS)

    def stac_types(self) -> Tuple[str, ...]:
        """Return the logical (STAC) types for each column."""

        return tuple(field.stac_type for field in self._FIELDS)

    def to_pyarrow_schema(self) -> "pa.Schema":
        """Materialise the expected PyArrow schema."""

        if pa is None:
            raise RuntimeError("PyArrow is required to build the schema representation.")

        fields = []
        for field in self._FIELDS:
            data_type = self._pyarrow_type_for(field.stac_type)
            fields.append(pa.field(field.name, data_type, metadata=None))
        return pa.schema(fields)

    def duckdb_layout(self) -> Tuple[Tuple[str, str], ...]:
        """Return ``(column, duckdb_type)`` pairs following storage order."""

        return tuple((field.name, self._DUCKDB_TYPE_MAP[field.stac_type]) for field in self._FIELDS)

    def stac_columns(self) -> Tuple[Mapping[str, str], ...]:
        """Expose the schema as STAC ``table:columns`` style dictionaries."""

        return tuple(field.stac_entry() for field in self._FIELDS)

    @classmethod
    def _pyarrow_type_for(cls, stac_type: str) -> "pa.DataType":
        """Translate a STAC logical type into the corresponding PyArrow type."""

        if pa is None:  # pragma: no cover - guarded in caller.
            raise RuntimeError("PyArrow not available")

        cached = cls._ARROW_TYPE_FACTORIES.get(stac_type)
        if cached is not None:
            return cached

        if stac_type == "datetime":
            arrow_type = pa.timestamp("us", tz="UTC")
        elif stac_type == "string":
            arrow_type = pa.string()
        elif stac_type == "binary":
            arrow_type = pa.binary()
        elif stac_type == "integer":
            arrow_type = pa.int64()
        elif stac_type == "number":
            arrow_type = pa.float64()
        elif stac_type == "boolean":
            arrow_type = pa.bool_()
        else:  # pragma: no cover - unexpected schema entry.
            raise ValueError(f"Unsupported STAC logical type: {stac_type!r}")

        cls._ARROW_TYPE_FACTORIES = dict(cls._ARROW_TYPE_FACTORIES)
        cls._ARROW_TYPE_FACTORIES[stac_type] = arrow_type
        return arrow_type
