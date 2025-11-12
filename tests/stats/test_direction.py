"""Tests for circular wind direction error utilities."""

from __future__ import annotations

import numpy as np
import pytest

from hf_wind_resource.stats import evaluate_direction_pairs


def test_evaluate_direction_pairs_wraps_around_dateline() -> None:
    predicted = [359.0, 1.0, 180.0]
    observed = [1.0, 359.0, 190.0]

    result = evaluate_direction_pairs(predicted, observed)

    expected_errors = np.array([-2.0, 2.0, -10.0])
    np.testing.assert_allclose(result.records["angular_error_deg"], expected_errors)
    np.testing.assert_allclose(
        result.records["absolute_error_deg"], np.abs(expected_errors)
    )
    assert pytest.approx(result.metrics.mean_absolute_error_deg, rel=1e-6) == 4.666666
    assert pytest.approx(result.metrics.root_mean_square_error_deg, rel=1e-6) == pytest.approx(
        np.sqrt(np.mean(np.square(expected_errors)))
    )
    assert result.quality.total_pairs == 3
    assert result.quality.valid_pairs == 3
    assert pytest.approx(result.quality.coverage_ratio, rel=1e-9) == 1.0


def test_evaluate_direction_pairs_filters_sentinels() -> None:
    predicted = [120.0, 240.0, 300.0]
    observed = [130.0, -10000.0, 310.0]

    result = evaluate_direction_pairs(predicted, observed)

    assert result.quality.total_pairs == 3
    assert result.quality.valid_pairs == 2
    assert result.quality.sentinel_pairs == 1
    assert pytest.approx(result.quality.coverage_ratio, rel=1e-9) == 2 / 3
    assert len(result.records) == 2
    assert np.all(result.records["absolute_error_deg"] >= 0.0)


def test_evaluate_direction_pairs_all_invalid_returns_empty_records() -> None:
    predicted = [np.nan, 45.0]
    observed = [np.nan, -10000.0]

    result = evaluate_direction_pairs(predicted, observed)

    assert result.records.empty
    assert result.metrics.mean_absolute_error_deg is None
    assert result.quality.valid_pairs == 0
    assert result.quality.coverage_ratio == 0.0
