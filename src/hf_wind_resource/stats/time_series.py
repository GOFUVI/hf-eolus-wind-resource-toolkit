"""Time-series modelling utilities for monthly wind-power aggregates."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - availability checked at runtime
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
    from statsmodels.tsa.stattools import acf, pacf
    from statsmodels.tools.sm_exceptions import ValueWarning as StatsmodelsValueWarning
    _STATS_MODELS_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover
    acorr_ljungbox = None  # type: ignore[assignment]
    ExponentialSmoothing = None  # type: ignore[assignment]
    SARIMAX = None  # type: ignore[assignment]
    SARIMAXResults = None  # type: ignore[assignment]
    acf = None  # type: ignore[assignment]
    pacf = None  # type: ignore[assignment]
    StatsmodelsValueWarning = Warning
    _STATS_MODELS_ERROR = exc


_TIME_SERIES_CONFIG_PATH = Path("config") / "time_series.json"


def _require_statsmodels() -> None:
    if _STATS_MODELS_ERROR is not None:  # pragma: no cover - simple guard
        raise ImportError(
            "statsmodels is required for time-series modelling. "
            "Install statsmodels or remove time-series functionality from the call."
        ) from _STATS_MODELS_ERROR

__all__ = [
    "SarimaConfig",
    "SarimaResult",
    "EtsResult",
    "TimeSeriesSegment",
    "fit_sarima_auto",
    "fit_ets_seasonal",
    "compute_acf_pacf",
    "prepare_monthly_series",
    "split_series_by_gaps",
    "load_time_series_config",
]


def load_time_series_config(path: str | Path | None = None) -> dict[str, int]:
    """Return configuration values governing time-series segmentation."""

    defaults = {"max_gap_months": 6, "min_segment_months": 36}
    target = Path(path) if path is not None else _TIME_SERIES_CONFIG_PATH
    if not target.exists():
        return defaults
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Invalid JSON in time-series configuration: {target}") from exc

    for key in defaults:
        value = payload.get(key)
        if value is not None:
            defaults[key] = int(value)
    return defaults


@dataclass(frozen=True)
class SarimaConfig:
    """SARIMA order configuration."""

    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]


@dataclass(frozen=True)
class SarimaResult:
    """Selected SARIMA model with diagnostics."""

    config: SarimaConfig
    aic: float
    aicc: float
    bic: float
    nobs: int
    params: Mapping[str, float]
    residuals: np.ndarray
    ljung_box_pvalue: float | None
    forecast: np.ndarray


@dataclass(frozen=True)
class EtsResult:
    """Holt-Winters exponential smoothing result."""

    seasonal: str
    trend: str | None
    seasonal_periods: int
    aic: float
    bic: float
    aicc: float
    residuals: np.ndarray
    ljung_box_pvalue: float | None
    forecast: np.ndarray


@dataclass(frozen=True)
class TimeSeriesSegment:
    """Continuous monthly segment without large gaps."""

    series: pd.Series
    start: pd.Timestamp
    end: pd.Timestamp
    gap_filled: bool


def prepare_monthly_series(
    rows: Iterable[Mapping[str, object]],
    *,
    value_field: str = "turbine_mean_power_kw",
) -> pd.Series:
    """Return a sorted UTC-indexed monthly series from CSV rows."""

    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return pd.Series(dtype=float)

    if "period_start" not in frame.columns:
        raise KeyError("Input rows must include 'period_start' column.")

    frame["timestamp"] = pd.to_datetime(frame["period_start"], utc=True)
    frame.sort_values("timestamp", inplace=True)
    values = pd.to_numeric(frame[value_field], errors="coerce")
    idx = frame["timestamp"].dt.tz_convert(None)
    series = pd.Series(values.values, index=idx).sort_index()
    series = series.dropna()
    if series.empty:
        return series

    inferred = pd.infer_freq(series.index)
    if inferred is None and len(series) >= 2:
        deltas = (series.index.to_series().diff().dropna().dt.days.abs())
        if deltas.mode().isin([28, 29, 30, 31]).any():
            inferred = "MS"

    if inferred:
        full_index = pd.date_range(series.index.min(), series.index.max(), freq=inferred)
        series = series.reindex(full_index)
        series.index = pd.DatetimeIndex(series.index, freq=inferred)
    else:
        series.index = pd.DatetimeIndex(series.index)

    return series


def split_series_by_gaps(
    series: pd.Series,
    *,
    max_gap_months: int | None = None,
    min_length: int | None = None,
) -> list[TimeSeriesSegment]:
    """Split a monthly series into continuous segments, interpolating small gaps."""

    if series.empty:
        return []

    config = load_time_series_config()
    max_gap = max_gap_months if max_gap_months is not None else config["max_gap_months"]
    min_len = min_length if min_length is not None else config["min_segment_months"]

    freq = series.index.freq or pd.infer_freq(series.index)
    if freq is None:
        freq = "MS"
        full_index = pd.date_range(series.index.min(), series.index.max(), freq=freq)
        series = series.reindex(full_index)
        series.index = pd.DatetimeIndex(series.index, freq=freq)
    else:
        if series.index.freq is None:
            series.index = pd.DatetimeIndex(series.index, freq=freq)

    original = series.copy()

    limit = max(0, max_gap - 1)
    interpolated = series.copy()
    if limit > 0:
        interpolated = interpolated.interpolate(method="time", limit=limit, limit_direction="both")

    segments: list[TimeSeriesSegment] = []
    current_index: list[pd.Timestamp] = []
    current_values: list[float] = []
    gap_filled = False

    def flush_segment() -> None:
        nonlocal current_index, current_values, gap_filled
        if current_index:
            seg_series = pd.Series(current_values, index=current_index)
            seg_series.index = pd.DatetimeIndex(seg_series.index, freq=freq)
            original_subset = original.reindex(seg_series.index)
            mask_valid = ~original_subset.isna()
            if mask_valid.any():
                first_valid = mask_valid[mask_valid].index[0]
                last_valid = mask_valid[mask_valid].index[-1]
                seg_series = seg_series.loc[first_valid:last_valid]
                original_subset = original_subset.loc[first_valid:last_valid]
            else:
                seg_series = pd.Series(dtype=float)

            if seg_series.dropna().shape[0] >= min_len:
                segments.append(
                    TimeSeriesSegment(
                        series=seg_series,
                        start=seg_series.index.min(),
                        end=seg_series.index.max(),
                        gap_filled=bool(original_subset.isna().any()),
                    )
                )
        current_index = []
        current_values = []
        gap_filled = False

    for timestamp, value in interpolated.items():
        if pd.notna(value):
            current_index.append(timestamp)
            current_values.append(value)
            if pd.isna(series.get(timestamp)):
                gap_filled = True
        else:
            flush_segment()

    flush_segment()
    return segments


def _aicc(aic: float, nobs: int, nparams: int) -> float:
    if nobs - nparams - 1 <= 0:
        return float("inf")
    return aic + (2 * nparams * (nparams + 1)) / (nobs - nparams - 1)


def _fit_sarima(series: pd.Series, config: SarimaConfig) -> SARIMAXResults | None:
    _require_statsmodels()
    if series.empty or series.size < 12:
        return None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="A date index has been provided, but it has no associated frequency information",
                category=StatsmodelsValueWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="No supported index is available.",
                category=StatsmodelsValueWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="No supported index is available.",
                category=FutureWarning,
            )
            model = SARIMAX(
                series,
                order=config.order,
                seasonal_order=config.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
                trend="c",
            )
            result = model.fit(disp=False)
        return result
    except (ValueError, np.linalg.LinAlgError):
        return None


def fit_sarima_auto(
    series: pd.Series,
    *,
    seasonal_periods: int = 12,
    max_order: int = 2,
    d: int | None = None,
    D: int | None = None,
    forecast_steps: int = 12,
) -> SarimaResult | None:
    """Run a limited grid-search over SARIMA configurations."""

    _require_statsmodels()
    if series.size < seasonal_periods * 2:
        return None

    if d is None:
        diff = series.diff().dropna()
        d = 1 if diff.std() < series.std() else 0
    if D is None:
        seas_diff = series.diff(seasonal_periods).dropna()
        D = 1 if seas_diff.std() < series.std() else 0

    candidates: list[SarimaResult] = []
    for p in range(0, max_order + 1):
        for q in range(0, max_order + 1):
            order = (p, d, q)
            for P in range(0, 2):
                for Q in range(0, 2):
                    seasonal_order = (P, D, Q, seasonal_periods)
                    config = SarimaConfig(order=order, seasonal_order=seasonal_order)
                    result = _fit_sarima(series, config)
                    if result is None:
                        continue
                    nobs = result.nobs
                    nparams = result.params.size
                    aic = float(result.aic)
                    bic = float(result.bic)
                    aicc_value = _aicc(aic, nobs, nparams)
                    residuals = result.resid
                    ljung_box = acorr_ljungbox(residuals, lags=[min(12, nobs - 1)], return_df=True)
                    pvalue = float(ljung_box["lb_pvalue"].iloc[0]) if not ljung_box.empty else None
                    forecast = result.forecast(forecast_steps)
                    candidates.append(
                        SarimaResult(
                            config=config,
                            aic=aic,
                            aicc=aicc_value,
                            bic=bic,
                            nobs=nobs,
                            params=result.params.to_dict(),
                            residuals=residuals,
                            ljung_box_pvalue=pvalue,
                            forecast=forecast.to_numpy(),
                        )
                    )

    if not candidates:
        return None
    candidates.sort(key=lambda item: item.aicc)
    return candidates[0]


def fit_ets_seasonal(
    series: pd.Series,
    *,
    seasonal_periods: int = 12,
    trend: str | None = "add",
    seasonal: str = "add",
    forecast_steps: int = 12,
) -> EtsResult | None:
    """Fit a multiplicative/additive Holt-Winters model."""

    _require_statsmodels()
    if series.size < seasonal_periods * 2:
        return None

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="A date index has been provided, but it has no associated frequency information",
                category=StatsmodelsValueWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="No supported index is available.",
                category=StatsmodelsValueWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="No supported index is available.",
                category=FutureWarning,
            )
            model = ExponentialSmoothing(
                series,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            )
            result = model.fit()
    except (ValueError, np.linalg.LinAlgError):
        return None

    residuals = series - result.fittedvalues
    nobs = residuals.size
    nparams = len(result.params)
    aic = float(result.aic)
    bic = float(result.bic)
    aicc_value = _aicc(aic, nobs, nparams)
    ljung_box = acorr_ljungbox(residuals, lags=[min(12, nobs - 1)], return_df=True)
    pvalue = float(ljung_box["lb_pvalue"].iloc[0]) if not ljung_box.empty else None
    forecast = result.forecast(forecast_steps)
    return EtsResult(
        seasonal=seasonal,
        trend=trend,
        seasonal_periods=seasonal_periods,
        aic=aic,
        bic=bic,
        aicc=aicc_value,
        residuals=residuals.to_numpy(),
        ljung_box_pvalue=pvalue,
        forecast=forecast.to_numpy(),
    )


def compute_acf_pacf(
    series: Sequence[float],
    *,
    nlags: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ACF and PACF arrays for diagnostic plots."""

    _require_statsmodels()
    arr = np.asarray(series, dtype=float)
    if arr.size == 0:
        return np.array([]), np.array([])
    max_stat_nlags = max(1, (arr.size - 1) // 2)
    nlags = min(nlags, arr.size - 1, max_stat_nlags)
    if nlags <= 0:
        return np.array([]), np.array([])
    acf_values = acf(arr, nlags=nlags, fft=True)
    pacf_values = pacf(arr, nlags=nlags, method="yw")
    return acf_values, pacf_values
