from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from hf_wind_resource.stats import (
    TimeSeriesSegment,
    compute_acf_pacf,
    fit_ets_seasonal,
    fit_sarima_auto,
    prepare_monthly_series,
    split_series_by_gaps,
)


def _build_rows(values: list[float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ts = datetime(2015, 1, 1, tzinfo=timezone.utc)
    for value in values:
        rows.append(
            {
                "period_start": ts.isoformat(),
                "turbine_mean_power_kw": value,
                "node_id": "TEST_NODE",
            }
        )
        ts = (pd.Timestamp(ts) + pd.DateOffset(months=1)).to_pydatetime()
    return rows


def test_prepare_monthly_series_builds_sorted_series() -> None:
    rows = _build_rows([100.0, 105.0, 110.0])
    series = prepare_monthly_series(rows)
    assert list(series.index) == [
        pd.Timestamp("2015-01-01 00:00:00"),
        pd.Timestamp("2015-02-01 00:00:00"),
        pd.Timestamp("2015-03-01 00:00:00"),
    ]
    assert series.index.freqstr == "MS"
    assert series.iloc[-1] == 110.0


def test_fit_sarima_auto_returns_model() -> None:
    values = []
    base = 200.0
    for i in range(1, 73):
        seasonal = 20.0 * math.sin(2 * math.pi * (i % 12) / 12.0)
        values.append(base + seasonal + 5 * np.random.RandomState(i).randn())
    series = prepare_monthly_series(_build_rows(values))
    result = fit_sarima_auto(series, seasonal_periods=12, max_order=1, forecast_steps=6)
    assert result is not None
    assert len(result.forecast) == 6
    assert result.config.order[1] in (0, 1)
    assert result.config.seasonal_order[-1] == 12


def test_fit_ets_seasonal_returns_forecast() -> None:
    values = [150 + 10 * math.sin(2 * math.pi * (i % 12) / 12.0) for i in range(60)]
    series = prepare_monthly_series(_build_rows(values))
    result = fit_ets_seasonal(series, seasonal_periods=12, forecast_steps=4)
    assert result is not None
    assert len(result.forecast) == 4
    assert result.seasonal_periods == 12


def test_compute_acf_pacf_shapes() -> None:
    arr = np.arange(1, 25, dtype=float)
    acf, pacf = compute_acf_pacf(arr, nlags=10)
    assert len(acf) == 11  # includes lag zero
    assert len(pacf) == 11


def test_split_series_by_gaps_detects_segments() -> None:
    rows = _build_rows([100.0, 105.0, None, None, 120.0, 130.0, None, None, None, None, None, None, 140.0, 150.0])
    series = prepare_monthly_series(rows)
    segments = split_series_by_gaps(series, max_gap_months=3, min_length=2)
    assert len(segments) == 2
    first, second = segments
    assert isinstance(first, TimeSeriesSegment)
    assert first.series.index.freqstr == "MS"
    assert second.series.index[0] > first.series.index[-1]
    assert bool(first.gap_filled)  # small gap interpolated
    assert not bool(second.gap_filled)  # starts after large gap
