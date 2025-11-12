from __future__ import annotations

import pandas as pd
import pytest

from hf_wind_resource.stats.seasonal import compute_seasonal_analysis


def _find_season(result, node_id: str, season: str):
    for slice_ in result.per_season:
        if slice_.node_id == node_id and slice_.season == season:
            return slice_
    raise AssertionError(f"Seasonal slice not found for {(node_id, season)}")


def _find_year(result, node_id: str, year: int):
    for slice_ in result.per_year:
        if slice_.node_id == node_id and slice_.year == year:
            return slice_
    raise AssertionError(f"Annual slice not found for {(node_id, year)}")


def _find_summary(result, node_id: str):
    for summary in result.variation:
        if summary.node_id == node_id:
            return summary
    raise AssertionError(f"Summary not found for {node_id}")


def test_compute_seasonal_analysis_basic() -> None:
    records = [
        # Node A multiple seasons/years
        {"timestamp": "2022-04-01T00:00:00Z", "node_id": "NODE_A", "pred_wind_speed": 6.0, "pred_range_label": "in"},
        {"timestamp": "2022-07-01T00:00:00Z", "node_id": "NODE_A", "pred_wind_speed": 7.0, "pred_range_label": "in"},
        {"timestamp": "2023-01-15T00:00:00Z", "node_id": "NODE_A", "pred_wind_speed": 8.0, "pred_range_label": " in "},
        {"timestamp": "2023-02-10T00:00:00Z", "node_id": "NODE_A", "pred_wind_speed": 4.0, "pred_range_label": "below"},
        {"timestamp": "2023-04-05T00:00:00Z", "node_id": "NODE_A", "pred_wind_speed": 9.0, "pred_range_label": "IN_RANGE"},
        {"timestamp": "2023-07-07T00:00:00Z", "node_id": "NODE_A", "pred_wind_speed": 11.0, "pred_range_label": "in"},
        {"timestamp": "2023-10-12T00:00:00Z", "node_id": "NODE_A", "pred_wind_speed": 12.0, "pred_range_label": "above"},
        {"timestamp": "2024-01-05T00:00:00Z", "node_id": "NODE_A", "pred_wind_speed": 7.0, "pred_range_label": "in"},
        {"timestamp": "2024-04-05T00:00:00Z", "node_id": "NODE_A", "pred_wind_speed": 10.0, "pred_range_label": "in"},
        {"timestamp": "2024-07-07T00:00:00Z", "node_id": "NODE_A", "pred_wind_speed": 12.0, "pred_range_label": "in"},
        {"timestamp": "2024-10-12T00:00:00Z", "node_id": "NODE_A", "pred_wind_speed": 13.0, "pred_range_label": "in"},
        # Node B has uncertain/below labels
        {"timestamp": "2023-03-15T00:00:00Z", "node_id": "NODE_B", "pred_wind_speed": 6.0, "pred_range_label": "in"},
        {"timestamp": "2023-08-20T00:00:00Z", "node_id": "NODE_B", "pred_wind_speed": 8.0, "pred_range_label": "in"},
        {"timestamp": "2023-11-01T00:00:00Z", "node_id": "NODE_B", "pred_wind_speed": 5.0, "pred_range_label": "uncertain"},
        {"timestamp": "2024-03-15T00:00:00Z", "node_id": "NODE_B", "pred_wind_speed": 5.0, "pred_range_label": "below"},
        {"timestamp": "2024-08-20T00:00:00Z", "node_id": "NODE_B", "pred_wind_speed": 9.0, "pred_range_label": "in"},
        # Node C single year/season
        {"timestamp": "2023-05-01T00:00:00Z", "node_id": "NODE_C", "pred_wind_speed": 5.0, "pred_range_label": "in"},
    ]

    frame = pd.DataFrame.from_records(records)
    result = compute_seasonal_analysis(frame)

    djf_a = _find_season(result, "NODE_A", "DJF")
    assert djf_a.sample_count == 2
    assert djf_a.mean_speed == pytest.approx(7.5)
    assert djf_a.below_ratio == pytest.approx(1 / 3)
    assert djf_a.in_ratio == pytest.approx(2 / 3)

    mam_a = _find_season(result, "NODE_A", "MAM")
    assert mam_a.sample_count == 3  # 2022, 2023, 2024 in-range observations
    assert mam_a.mean_speed == pytest.approx((6.0 + 9.0 + 10.0) / 3)

    son_a = _find_season(result, "NODE_A", "SON")
    assert son_a.sample_count == 1
    assert son_a.mean_speed == pytest.approx(13.0)
    assert son_a.above_ratio == pytest.approx(0.5)  # one above vs one in-range

    mam_b = _find_season(result, "NODE_B", "MAM")
    assert mam_b.sample_count == 1
    assert mam_b.mean_speed == pytest.approx(6.0)
    assert mam_b.below_ratio == pytest.approx(0.5)
    assert mam_b.uncertain_ratio == pytest.approx(0.0)

    jja_b = _find_season(result, "NODE_B", "JJA")
    assert jja_b.sample_count == 2
    assert jja_b.mean_speed == pytest.approx((8.0 + 9.0) / 2)

    annual_a_2023 = _find_year(result, "NODE_A", 2023)
    assert annual_a_2023.sample_count == 3
    assert annual_a_2023.mean_speed == pytest.approx((8.0 + 9.0 + 11.0) / 3)

    annual_a_2024 = _find_year(result, "NODE_A", 2024)
    assert annual_a_2024.sample_count == 4
    assert annual_a_2024.p90_speed == pytest.approx(12.6, abs=0.1)

    summary_a = _find_summary(result, "NODE_A")
    assert summary_a.strongest_season == "SON"
    assert summary_a.weakest_season == "DJF"
    assert summary_a.seasonal_amplitude == pytest.approx(5.5)
    assert summary_a.annual_trend_slope == pytest.approx(2.0, abs=1e-6)
    assert summary_a.annual_samples == 3
    assert summary_a.trend_note is None

    summary_b = _find_summary(result, "NODE_B")
    assert summary_b.annual_samples == 2
    assert summary_b.annual_trend_slope == pytest.approx(2.0)
    assert summary_b.trend_note is None

    summary_c = _find_summary(result, "NODE_C")
    assert summary_c.seasonal_coverage == 1
    assert summary_c.annual_trend_slope is None
    assert summary_c.trend_note == "Only one year available."
