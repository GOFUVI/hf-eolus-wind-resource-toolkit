"""Unit tests for temporal normalisation helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hf_wind_resource.preprocessing import (
    TemporalNormalizationConfig,
    normalise_temporal_records,
)


pytest.importorskip("pandas")
import pandas as pd  # noqa: E402  (import guarded by importorskip)


def _make_timestamp(value: datetime) -> datetime:
    """Ensure all sample timestamps are timezone-aware UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def test_duplicates_are_removed_and_flagged() -> None:
    """Temporal normalisation should deduplicate on node/timestamp."""
    records = pd.DataFrame(
        {
            "node_id": ["node_a", "node_a", "node_a", "node_b"],
            "timestamp": [
                _make_timestamp(datetime(2024, 1, 1, 0, 0)),
                _make_timestamp(datetime(2024, 1, 1, 0, 0)),  # duplicate
                _make_timestamp(datetime(2024, 1, 1, 0, 30)),
                _make_timestamp(datetime(2024, 1, 1, 0, 30)),
            ],
            "pred_wind_speed": [8.5, 8.5, 9.1, 6.0],
        }
    )

    result = normalise_temporal_records(records)
    node_a = result.per_node["node_a"]

    assert node_a.total_observations == 2
    assert node_a.duplicate_records == 1
    assert node_a.quality_flags.has_duplicates is True
    assert not result.dataframe.duplicated(subset=["node_id", "timestamp"]).any()


def test_gap_detection_and_coverage_metrics() -> None:
    """Gaps larger than the cadence + tolerance should be logged."""
    records = pd.DataFrame(
        {
            "node_id": ["node_gap"] * 4,
            "timestamp": [
                _make_timestamp(datetime(2024, 4, 1, 0, 0)),
                _make_timestamp(datetime(2024, 4, 1, 0, 30)),
                _make_timestamp(datetime(2024, 4, 1, 2, 0)),  # 90 min gap -> 2 missing
                _make_timestamp(datetime(2024, 4, 1, 2, 30)),
            ],
        }
    )

    result = normalise_temporal_records(records)
    summary = result.per_node["node_gap"]

    assert summary.missing_observations == 2
    assert summary.quality_flags.has_gaps is True
    assert summary.quality_flags.irregular_cadence is True
    assert summary.cadence.sample_count == summary.total_observations - 1
    assert result.gap_report.descriptors["node_gap"].missing_windows
    assert summary.coverage_ratio == pytest.approx(
        summary.total_observations / (summary.total_observations + summary.missing_observations)
    )


def test_rounding_strategy_can_be_adjusted() -> None:
    """Validate the rounding strategy configuration knobs."""
    records = pd.DataFrame(
        {
            "node_id": ["rounded"] * 2,
            "timestamp": [
                _make_timestamp(datetime(2024, 5, 1, 0, 14)),
                _make_timestamp(datetime(2024, 5, 1, 0, 45)),
            ],
        }
    )

    config = TemporalNormalizationConfig(expected_cadence=timedelta(minutes=30), rounding="nearest")
    result = normalise_temporal_records(records, config=config)
    timestamps = result.dataframe.loc[result.dataframe["node_id"] == "rounded", "timestamp"]

    assert timestamps.iloc[0] == _make_timestamp(datetime(2024, 5, 1, 0, 0))
    assert timestamps.iloc[1] == _make_timestamp(datetime(2024, 5, 1, 1, 0))


def test_insufficient_coverage_sets_flag_and_notes() -> None:
    """Nodes con cobertura inferior al umbral deben marcar bandera y nota."""
    records = pd.DataFrame(
        {
            "node_id": ["node_low"] * 2,
            "timestamp": [
                _make_timestamp(datetime(2024, 6, 1, 0, 0)),
                _make_timestamp(datetime(2024, 6, 1, 2, 30)),
            ],
        }
    )

    result = normalise_temporal_records(records)
    summary = result.per_node["node_low"]

    assert summary.quality_flags.insufficient_coverage is True
    assert any("Coverage ratio" in note for note in summary.notes)
