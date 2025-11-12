"""Unit tests for generic buoy ingestion and synchronisation utilities."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from hf_wind_resource.preprocessing import (
    BuoySentinelConfig,
    HeightCorrectionConfig,
    SynchronisationConfig,
    build_geoparquet_table,
    load_buoy_timeseries,
    load_height_correction_from_config,
    prepare_buoy_timeseries,
    synchronise_buoy_and_ann,
)


def _make_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def test_load_buoy_timeseries_filters_speed_and_marks_direction(tmp_path: Path) -> None:
    dataset = tmp_path / "buoy.parquet"
    frame = pd.DataFrame(
        {
            "timestamp": [
                _make_timestamp("2023-01-01T00:00:00"),
                _make_timestamp("2023-01-01T01:00:00"),
                _make_timestamp("2023-01-01T02:00:00"),
                _make_timestamp("2023-01-01T03:00:00"),
            ],
            "wind_speed": [5.2, -9999.9, 6.7, 7.1],
            "wind_dir": [180, 190, -10000, 200],
            "geometry": [b"", b"", b"", b""],
        }
    )
    frame.to_parquet(dataset)

    result = load_buoy_timeseries(dataset, BuoySentinelConfig())

    assert result.total_records == 4
    assert result.dropped_speed_records == 1
    assert result.direction_sentinel_records == 1
    assert len(result.dataframe) == 3
    assert result.dataframe["wind_speed"].min() > 0
    # Direction sentinel should be represented as a missing value but the row kept.
    assert result.dataframe["wind_dir"].isna().sum() == 1
    assert "wind_speed_original_height" in result.dataframe.columns
    factor = math.log(10.0 / 0.0002) / math.log(3.0 / 0.0002)
    ratios = result.dataframe["wind_speed"] / result.dataframe["wind_speed_original_height"]
    assert pytest.approx(ratios.dropna().unique()[0], rel=1e-6) == factor
    assert result.height_correction is not None
    assert result.height_correction.method == "log_profile"
    assert pytest.approx(result.height_correction.scale_factor, rel=1e-6) == factor
    assert result.coverage_start == _make_timestamp("2023-01-01T00:00:00")
    assert result.coverage_end == _make_timestamp("2023-01-01T03:00:00")
    assert result.cadence.nominal == timedelta(hours=1)


def test_load_buoy_timeseries_without_height_correction(tmp_path: Path) -> None:
    dataset = tmp_path / "buoy.parquet"
    frame = pd.DataFrame(
        {
            "timestamp": [
                _make_timestamp("2023-01-01T00:00:00"),
                _make_timestamp("2023-01-01T01:00:00"),
            ],
            "wind_speed": [5.2, 7.1],
            "wind_dir": [180, 182],
            "geometry": [b"", b""],
        }
    )
    frame.to_parquet(dataset)

    result = load_buoy_timeseries(dataset, BuoySentinelConfig(), height_correction=None)

    assert "wind_speed_original_height" not in result.dataframe.columns
    assert result.height_correction is None
    assert result.dataframe["wind_speed"].iloc[0] == pytest.approx(5.2)


def test_load_buoy_timeseries_log_profile_height(tmp_path: Path) -> None:
    dataset = tmp_path / "buoy.parquet"
    frame = pd.DataFrame(
        {
            "timestamp": [
                _make_timestamp("2023-01-01T00:00:00"),
                _make_timestamp("2023-01-01T01:00:00"),
            ],
            "wind_speed": [5.0, 6.0],
            "wind_dir": [170, 185],
            "geometry": [b"", b""],
        }
    )
    frame.to_parquet(dataset)

    config = HeightCorrectionConfig(
        method="log_profile",
        measurement_height_m=3.0,
        target_height_m=10.0,
        roughness_length_m=0.0002,
    )
    result = load_buoy_timeseries(dataset, BuoySentinelConfig(), height_correction=config)

    expected_scale = math.log(10.0 / 0.0002) / math.log(3.0 / 0.0002)
    assert pytest.approx(result.height_correction.scale_factor, rel=1e-6) == expected_scale


def test_load_height_correction_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / "height_config.json"
    config_path.write_text(
        json.dumps(
            {
                "method": "power_law",
                "measurement_height_m": 4.5,
                "target_height_m": 60.0,
                "power_law_alpha": 0.14,
                "roughness_length_m": 0.0002,
            }
        ),
        encoding="utf-8",
    )

    config = load_height_correction_from_config(config_path)
    assert config.method == "power_law"
    assert pytest.approx(config.measurement_height_m, rel=1e-6) == 4.5
    assert pytest.approx(config.target_height_m, rel=1e-6) == 60.0
    assert pytest.approx(config.power_law_alpha, rel=1e-6) == 0.14


def test_synchronise_buoy_and_ann_exact_only() -> None:
    buoy = pd.DataFrame(
        {
            "timestamp": [
                _make_timestamp("2023-01-01T00:00:00"),
                _make_timestamp("2023-01-01T01:00:00"),
                _make_timestamp("2023-01-01T02:00:00"),
            ],
            "wind_speed": [5.0, 6.0, 7.0],
            "wind_dir": [180, 190, 200],
        }
    )
    ann = pd.DataFrame(
        {
            "timestamp": [
                _make_timestamp("2023-01-01T00:00:00"),
                _make_timestamp("2023-01-01T00:45:00"),
                _make_timestamp("2023-01-01T02:00:00"),
                _make_timestamp("2023-01-01T03:05:00"),
            ],
            "node_id": ["Sample_buoy"] * 4,
            "pred_wind_speed": [4.8, 5.5, 7.1, 8.0],
        }
    )

    matched, summary = synchronise_buoy_and_ann(
        buoy,
        ann,
        SynchronisationConfig(tolerance=timedelta(minutes=30), prefer_nearest=False),
    )

    assert summary.matched_rows == 2
    assert summary.unmatched_ann_rows == 2
    assert summary.unmatched_buoy_rows == 1
    assert summary.exact_matches == 2
    assert summary.nearest_matches == 0
    assert pytest.approx(summary.match_ratio_ann, rel=1e-6) == 0.5
    assert pytest.approx(summary.match_ratio_buoy, rel=1e-6) == 2 / 3
    assert (matched["timestamp_ann"] == matched["timestamp_buoy"]).all()


def test_synchronise_buoy_and_ann_counts_exact_and_nearest() -> None:
    buoy = pd.DataFrame(
        {
            "timestamp": [
                _make_timestamp("2023-01-01T00:00:00"),
                _make_timestamp("2023-01-01T01:00:00"),
                _make_timestamp("2023-01-01T02:00:00"),
            ],
            "wind_speed": [5.0, 6.0, 7.0],
            "wind_dir": [180, 190, 200],
        }
    )
    ann = pd.DataFrame(
        {
            "timestamp": [
                _make_timestamp("2023-01-01T00:00:00"),
                _make_timestamp("2023-01-01T00:45:00"),
                _make_timestamp("2023-01-01T02:00:00"),
                _make_timestamp("2023-01-01T03:05:00"),
            ],
            "node_id": ["Sample_buoy"] * 4,
            "pred_wind_speed": [4.8, 5.5, 7.1, 8.0],
        }
    )

    matched, summary = synchronise_buoy_and_ann(
        buoy,
        ann,
        SynchronisationConfig(tolerance=timedelta(minutes=30), prefer_nearest=True),
    )

    assert summary.matched_rows == 3
    assert summary.unmatched_ann_rows == 1
    assert summary.unmatched_buoy_rows == 0
    assert summary.exact_matches == 2
    assert summary.nearest_matches == 1
    assert pytest.approx(summary.match_ratio_ann, rel=1e-6) == 0.75
    assert pytest.approx(summary.match_ratio_buoy, rel=1e-6) == 1.0
    assert matched["is_exact_match"].sum() == 2
    assert set(matched.columns).issuperset({"timestamp_ann", "timestamp_buoy", "wind_speed"})


def test_prepare_buoy_timeseries_end_to_end(tmp_path: Path) -> None:
    buoy_dataset = tmp_path / "buoy.parquet"
    ann_dataset = tmp_path / "sar_range_final.parquet"

    buoy_frame = pd.DataFrame(
        {
            "timestamp": [
                _make_timestamp("2023-01-01T00:00:00"),
                _make_timestamp("2023-01-01T01:00:00"),
                _make_timestamp("2023-01-01T02:00:00"),
            ],
            "wind_speed": [5.0, 6.5, 7.2],
            "wind_dir": [180, 190, 200],
            "geometry": [b"", b"", b""],
        }
    )
    buoy_frame.to_parquet(buoy_dataset)

    ann_frame = pd.DataFrame(
        {
            "timestamp": [
                _make_timestamp("2023-01-01T00:00:00"),
                _make_timestamp("2023-01-01T00:40:00"),
                _make_timestamp("2023-01-01T01:00:00"),
                _make_timestamp("2023-01-01T03:00:00"),
            ],
            "node_id": ["Target_buoy", "Target_buoy", "Other_node", "Target_buoy"],
            "pred_wind_speed": [4.9, 5.6, 5.4, 8.1],
            "pred_wind_direction": [175.0, 182.0, 183.0, 195.0],
            "pred_range_label": ["in", "below", "in", "above"],
            "prob_range_below": [0.1, 0.7, 0.1, 0.2],
            "prob_range_in": [0.8, 0.25, 0.85, 0.3],
            "prob_range_above": [0.1, 0.05, 0.05, 0.5],
            "range_flag": ["in", "below", "in", "above"],
            "range_flag_confident": [True, True, True, False],
        }
    )
    ann_frame.to_parquet(ann_dataset)

    result = prepare_buoy_timeseries(
        buoy_dataset=buoy_dataset,
        ann_dataset=ann_dataset,
        node_id="Target_buoy",
        synchronisation_config=SynchronisationConfig(
            tolerance=timedelta(minutes=30), prefer_nearest=True
        ),
    )

    # Three ANN rows belong to the target node; one exceeds the tolerance window.
    assert len(result.ann_dataframe) == 3
    assert result.buoy.total_records == 3
    assert result.synchronisation.matched_rows == 2
    assert result.synchronisation.exact_matches >= 1
    assert result.matched_dataframe["pred_wind_speed"].notna().all()
    assert result.matched_dataframe["wind_speed"].notna().all()
    assert "wind_speed_original_height" in result.buoy.dataframe.columns
    assert result.buoy.height_correction is not None
    assert "wind_speed_original_height" in result.matched_dataframe.columns


def test_build_geoparquet_table_adds_metadata() -> None:
    df = pd.DataFrame(
        {
            "timestamp_ann": [
                _make_timestamp("2023-01-01T00:00:00"),
                _make_timestamp("2023-01-01T01:00:00"),
            ],
            "geometry": [
                b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?",
                b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@",
            ],
        }
    )

    table = build_geoparquet_table(df, geometry_column="geometry", crs="EPSG:4326")
    metadata = table.schema.metadata or {}
    assert b"geo" in metadata
    geo_meta = json.loads(metadata[b"geo"].decode("utf-8"))
    assert geo_meta["primary_column"] == "geometry"
    assert geo_meta["columns"]["geometry"]["crs"] == "EPSG:4326"
