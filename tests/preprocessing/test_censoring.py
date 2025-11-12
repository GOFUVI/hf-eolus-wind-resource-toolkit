"""Tests for range-label partitioning and censoring summaries."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hf_wind_resource.preprocessing import (
    RangePartitioningResult,
    RangeThresholds,
    load_range_thresholds,
    partition_range_labels,
)

pytest.importorskip("pandas")
import pandas as pd  # noqa: E402  (import guarded by importorskip)


def _ts(value: str) -> datetime:
    """Build timezone-aware timestamps for the fixtures."""
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def test_partition_counts_and_ratios_per_node() -> None:
    """Ensure counts and proportions are captured per node."""
    frame = pd.DataFrame(
        {
            "timestamp": [
                _ts("2024-01-01T00:00:00"),
                _ts("2024-01-01T00:30:00"),
                _ts("2024-01-01T01:00:00"),
                _ts("2024-01-02T00:00:00"),
                _ts("2024-01-02T00:30:00"),
                _ts("2024-01-02T01:00:00"),
            ],
            "node_id": ["NODE_A", "NODE_A", "NODE_A", "NODE_B", "NODE_B", "NODE_B"],
            "pred_wind_speed": [5.2, 8.3, 18.6, 4.9, 7.1, 17.9],
            "pred_range_label": ["below", "in", "above", "below", "inside", "above"],
            "pred_speed_range_label": ["below", "in", "above", "below", "in", "above"],
        }
    )

    result = partition_range_labels(frame, thresholds=RangeThresholds(lower=5.7, upper=17.8))
    assert isinstance(result, RangePartitioningResult)
    assert len(result.below) == 2
    assert len(result.in_range) == 2
    assert len(result.above) == 2
    assert result.uncertain.empty

    node_a = result.per_node["NODE_A"]
    assert node_a.total_observations == 3
    assert node_a.left_censored_count == 1
    assert node_a.in_range_count == 1
    assert node_a.right_censored_count == 1
    assert node_a.uncertain_count == 0
    assert node_a.left_censored_ratio == pytest.approx(1 / 3)
    assert node_a.in_range_ratio == pytest.approx(1 / 3)
    assert node_a.right_censored_ratio == pytest.approx(1 / 3)
    assert node_a.uncertain_ratio == pytest.approx(0.0)

    node_b = result.per_node["NODE_B"]
    assert node_b.total_observations == 3
    assert node_b.left_censored_count == 1
    assert node_b.in_range_count == 1
    assert node_b.right_censored_count == 1
    assert node_b.uncertain_count == 0
    assert node_b.discrepancy_count == 0
    assert node_b.notes == ()


def test_partition_records_discrepancies_and_notes() -> None:
    """Range mismatches must be logged for auditing."""
    frame = pd.DataFrame(
        {
            "timestamp": [
                _ts("2024-02-01T00:00:00"),
                _ts("2024-02-01T00:30:00"),
                _ts("2024-02-01T01:00:00"),
                _ts("2024-02-01T01:30:00"),
            ],
            "node_id": ["NODE_C"] * 4,
            "pred_wind_speed": [18.2, 4.9, None, 6.6],
            "pred_range_label": ["in", "uncertain", "above", "in"],
            "pred_speed_range_label": ["in", "below", "above", "in"],
        }
    )

    result = partition_range_labels(frame)

    issues = {record.issue for record in result.discrepancies}
    assert issues == {
        "classifier_vs_threshold",
        "uncertain_vs_threshold",
        "missing_speed_value",
        "deterministic_vs_threshold",
    }

    node_summary = result.per_node["NODE_C"]
    assert node_summary.total_observations == 4
    assert node_summary.left_censored_count == 0
    assert node_summary.in_range_count == 2
    assert node_summary.right_censored_count == 1
    assert node_summary.uncertain_count == 1
    assert node_summary.discrepancy_count == 6
    assert any("threshold discrepancies" in note for note in node_summary.notes)
    assert any("uncertain" in note for note in node_summary.notes)
    assert any("missing wind_speed values" in note for note in node_summary.notes)


def test_load_range_thresholds_allows_configuration(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Thresholds should be overridable from a JSON configuration file."""
    config_path = tmp_path_factory.mktemp("cfg") / "thresholds.json"
    config_path.write_text(
        '{"lower": 4.5, "upper": 15.2, "schema_version": "test"}',
        encoding="utf-8",
    )

    thresholds = load_range_thresholds(config_path)
    assert thresholds.lower == pytest.approx(4.5)
    assert thresholds.upper == pytest.approx(15.2)
