"""Schema regression tests for the SAR range-aware GeoParquet snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pytest

from hf_wind_resource.io.schema_registry import SarRangeFinalSchema

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
duckdb = pytest.importorskip("duckdb")

DATASET_PATH = Path("use_case/catalogs/sar_range_final_pivots_joined/assets/data.parquet")
SCHEMA = SarRangeFinalSchema()


def _arrow_type_to_stac(field_type: "pa.DataType") -> str:
    """Map a PyArrow type to the logical STAC type string."""

    import pyarrow.types as pat

    if pat.is_timestamp(field_type):
        return "datetime"
    if pat.is_string(field_type):
        return "string"
    if pat.is_binary(field_type):
        return "binary"
    if pat.is_integer(field_type):
        return "integer"
    if pat.is_floating(field_type):
        return "number"
    if pat.is_boolean(field_type):
        return "boolean"
    raise AssertionError(f"Unexpected PyArrow type encountered: {field_type}")  # pragma: no cover


def _duckdb_type_to_stac(type_name: str) -> str:
    """Normalise a DuckDB type string into the logical STAC category."""

    normalised = str(type_name).upper()
    if normalised.startswith("TIMESTAMP"):
        return "datetime"
    if normalised in {"VARCHAR", "STRING", "TEXT"}:
        return "string"
    if normalised in {"BLOB", "BINARY"}:
        return "binary"
    if normalised in {"BIGINT", "INTEGER", "INT8", "INT16", "INT32", "INT64", "UBIGINT"}:
        return "integer"
    if normalised in {"DOUBLE", "FLOAT", "DECIMAL", "REAL"}:
        return "number"
    if normalised == "BOOLEAN":
        return "boolean"
    raise AssertionError(f"Unexpected DuckDB type encountered: {type_name}")  # pragma: no cover


def _ensure_dataset_available() -> None:
    if not DATASET_PATH.exists():
        pytest.skip("Production GeoParquet snapshot is not available in the repository.")


def test_pyarrow_schema_matches_registry() -> None:
    """The physical GeoParquet schema must remain aligned with the registry."""

    _ensure_dataset_available()

    dataset_schema = pq.read_schema(DATASET_PATH)
    assert tuple(dataset_schema.names) == SCHEMA.field_names()

    mismatches: list[str] = []
    for field, expected in zip(dataset_schema, SCHEMA, strict=True):
        logical = _arrow_type_to_stac(field.type)
        if logical != expected.stac_type:
            mismatches.append(f"{field.name}: expected {expected.stac_type}, got {logical}")
        if logical == "datetime":
            timezone = getattr(field.type, "tz", None)
            if timezone not in (None, "UTC"):
                mismatches.append(f"{field.name}: unsupported timezone {timezone!r}")
    assert not mismatches, "PyArrow schema drift detected:\n- " + "\n- ".join(mismatches)


def test_duckdb_schema_matches_registry() -> None:
    """DuckDB view of the dataset must expose the documented logical types."""

    _ensure_dataset_available()

    relation = duckdb.read_parquet(str(DATASET_PATH))
    actual_columns: Tuple[str, ...] = tuple(relation.columns)
    actual_types: Tuple[str, ...] = tuple(relation.dtypes)

    assert actual_columns == SCHEMA.field_names()

    mismatches: list[str] = []
    for name, dtype, expected in zip(actual_columns, actual_types, SCHEMA, strict=True):
        logical = _duckdb_type_to_stac(dtype)
        if logical != expected.stac_type:
            mismatches.append(f"{name}: expected {expected.stac_type}, got {logical} ({dtype})")
    assert not mismatches, "DuckDB schema drift detected:\n- " + "\n- ".join(mismatches)
