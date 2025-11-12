"""Seasonal and interannual diagnostics for ANN-derived wind speeds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import math

import numpy as np
import pandas as pd

__all__ = [
    "SeasonalSlice",
    "AnnualSlice",
    "NodeVariationSummary",
    "SeasonalAnalysisResult",
    "compute_seasonal_analysis",
]

_LABEL_ALIASES: Mapping[str, str] = {
    "inside": "in",
    "within": "in",
    "in_range": "in",
    "below_range": "below",
    "under": "below",
    "over": "above",
    "upper": "above",
    "right": "above",
    "left": "below",
    "unknown": "uncertain",
    "nan": "uncertain",
}

_CANONICAL_LABELS: Tuple[str, ...] = ("below", "in", "above", "uncertain")
_SEASON_ORDER: Tuple[str, ...] = ("DJF", "MAM", "JJA", "SON")
_MONTH_TO_SEASON: Mapping[int, str] = {
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
    12: "DJF",
}


@dataclass(frozen=True)
class SeasonalSlice:
    """Aggregated statistics for a node/season combination."""

    node_id: str
    season: str
    sample_count: int
    mean_speed: float | None
    std_speed: float | None
    p50_speed: float | None
    p90_speed: float | None
    p99_speed: float | None
    below_ratio: float | None
    in_ratio: float | None
    above_ratio: float | None
    uncertain_ratio: float | None


@dataclass(frozen=True)
class AnnualSlice:
    """Aggregated statistics for a node/year combination."""

    node_id: str
    year: int
    sample_count: int
    mean_speed: float | None
    std_speed: float | None
    p50_speed: float | None
    p90_speed: float | None
    p99_speed: float | None


@dataclass(frozen=True)
class NodeVariationSummary:
    """Seasonal amplitude and interannual trend diagnostics per node."""

    node_id: str
    strongest_season: str | None
    weakest_season: str | None
    seasonal_amplitude: float | None
    seasonal_mean_std: float | None
    seasonal_coverage: int
    annual_trend_slope: float | None
    annual_trend_units: str
    annual_samples: int
    trend_note: str | None


@dataclass(frozen=True)
class SeasonalAnalysisResult:
    """Container with detailed slices and per-node summaries."""

    per_season: Tuple[SeasonalSlice, ...]
    per_year: Tuple[AnnualSlice, ...]
    variation: Tuple[NodeVariationSummary, ...]


def _normalise_labels(raw: pd.Series) -> pd.Series:
    """Map raw classifier labels to the canonical four-class vocabulary."""

    series = raw.astype("string").str.strip().str.lower()
    series = series.replace(_LABEL_ALIASES)
    series = series.fillna("uncertain")
    series = series.where(series.isin(_CANONICAL_LABELS), "uncertain")
    return series


def _safe_ratio(count: float, total: float) -> float | None:
    if total <= 0:
        return None
    return count / total


def _to_optional_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return float(value)


def _quantile(series: pd.Series, q: float) -> float | None:
    if series.empty:
        return None
    value = series.quantile(q)
    return _to_optional_float(float(value))


def _season_sort_key(season: str) -> int:
    try:
        return _SEASON_ORDER.index(season)
    except ValueError:
        return len(_SEASON_ORDER)


def _build_seasonal_slices(
    frame: pd.DataFrame,
    *,
    node_column: str,
    speed_column: str,
    label_column: str,
    timestamp_column: str,
) -> Tuple[SeasonalSlice, ...]:
    data = frame.copy(deep=True)
    data[timestamp_column] = pd.to_datetime(data[timestamp_column], utc=True)
    data["_season"] = data[timestamp_column].dt.month.map(_MONTH_TO_SEASON)
    data["_label"] = _normalise_labels(data[label_column])

    label_counts = (
        data.groupby([node_column, "_season", "_label"])
        .size()
        .unstack(fill_value=0)
    )
    for label in _CANONICAL_LABELS:
        if label not in label_counts.columns:
            label_counts[label] = 0
    label_counts = label_counts.reindex(columns=_CANONICAL_LABELS, fill_value=0)
    totals = label_counts.sum(axis=1)

    in_range = data.loc[
        (data["_label"] == "in") & data[speed_column].notna(),
        [node_column, "_season", speed_column],
    ]

    grouped: Dict[Tuple[str, str], pd.Series] = {}
    if not in_range.empty:
        for (node_value, season_value), group in in_range.groupby([node_column, "_season"]):
            key = (str(node_value), str(season_value))
            grouped[key] = group[speed_column]

    seasonal_rows: List[SeasonalSlice] = []
    nodes = sorted(data[node_column].astype("string").unique())

    for node_id in nodes:
        season_values = sorted(
            data.loc[data[node_column] == node_id, "_season"].dropna().unique(),
            key=_season_sort_key,
        )
        for season in season_values:
            key = (node_id, str(season))
            series = grouped.get(key, pd.Series(dtype=float))

            sample_count = int(series.count()) if not series.empty else 0
            mean_speed = _to_optional_float(series.mean()) if sample_count else None
            std_speed = _to_optional_float(series.std(ddof=0)) if sample_count else None
            p50_speed = _quantile(series, 0.5)
            p90_speed = _quantile(series, 0.9)
            p99_speed = _quantile(series, 0.99)

            if key in label_counts.index:
                counts = label_counts.loc[key]
                total = totals.loc[key]
            else:
                counts = pd.Series({label: 0 for label in _CANONICAL_LABELS}, dtype=float)
                total = 0

            seasonal_rows.append(
                SeasonalSlice(
                    node_id=str(node_id),
                    season=str(season),
                    sample_count=sample_count,
                    mean_speed=mean_speed,
                    std_speed=std_speed,
                    p50_speed=p50_speed,
                    p90_speed=p90_speed,
                    p99_speed=p99_speed,
                    below_ratio=_safe_ratio(float(counts.get("below", 0)), float(total)),
                    in_ratio=_safe_ratio(float(counts.get("in", 0)), float(total)),
                    above_ratio=_safe_ratio(float(counts.get("above", 0)), float(total)),
                    uncertain_ratio=_safe_ratio(
                        float(counts.get("uncertain", 0)), float(total)
                    ),
                )
            )

    seasonal_rows.sort(key=lambda row: (row.node_id, _season_sort_key(row.season)))
    return tuple(seasonal_rows)


def _build_annual_slices(
    frame: pd.DataFrame,
    *,
    node_column: str,
    speed_column: str,
    label_column: str,
    timestamp_column: str,
) -> Tuple[AnnualSlice, ...]:
    data = frame.copy(deep=True)
    data[timestamp_column] = pd.to_datetime(data[timestamp_column], utc=True)
    data["_label"] = _normalise_labels(data[label_column])
    data["_year"] = data[timestamp_column].dt.year

    in_range = data.loc[
        (data["_label"] == "in") & data[speed_column].notna(),
        [node_column, "_year", speed_column],
    ]

    if in_range.empty:
        return tuple()

    grouped = in_range.groupby([node_column, "_year"])[speed_column]
    annual_rows: List[AnnualSlice] = []

    for (node_id, year), series in grouped:
        sample_count = int(series.count())
        mean_speed = _to_optional_float(series.mean()) if sample_count else None
        std_speed = _to_optional_float(series.std(ddof=0)) if sample_count else None
        p50_speed = _quantile(series, 0.5)
        p90_speed = _quantile(series, 0.9)
        p99_speed = _quantile(series, 0.99)

        annual_rows.append(
            AnnualSlice(
                node_id=str(node_id),
                year=int(year),
                sample_count=sample_count,
                mean_speed=mean_speed,
                std_speed=std_speed,
                p50_speed=p50_speed,
                p90_speed=p90_speed,
                p99_speed=p99_speed,
            )
        )

    annual_rows.sort(key=lambda row: (row.node_id, row.year))
    return tuple(annual_rows)


def _summarise_variation(
    seasonal: Iterable[SeasonalSlice],
    annual: Iterable[AnnualSlice],
) -> Tuple[NodeVariationSummary, ...]:
    seasonal_by_node: Dict[str, List[SeasonalSlice]] = {}
    for slice_ in seasonal:
        seasonal_by_node.setdefault(slice_.node_id, []).append(slice_)

    annual_by_node: Dict[str, List[AnnualSlice]] = {}
    for slice_ in annual:
        annual_by_node.setdefault(slice_.node_id, []).append(slice_)

    summaries: List[NodeVariationSummary] = []
    node_ids = sorted(set(seasonal_by_node) | set(annual_by_node))

    for node_id in node_ids:
        seasonal_slices = seasonal_by_node.get(node_id, [])
        annual_slices = annual_by_node.get(node_id, [])

        seasonal_means = [
            slice_.mean_speed for slice_ in seasonal_slices if slice_.mean_speed is not None
        ]
        seasonal_mean_std: float | None = None
        seasonal_amplitude: float | None = None
        strongest_season: str | None = None
        weakest_season: str | None = None

        if seasonal_means:
            numeric_means = np.array(seasonal_means, dtype=float)
            seasonal_mean_std = float(np.std(numeric_means, ddof=0))
            seasonal_amplitude = float(np.max(numeric_means) - np.min(numeric_means))

            ordered = sorted(
                (slice_ for slice_ in seasonal_slices if slice_.mean_speed is not None),
                key=lambda item: item.mean_speed,  # type: ignore[arg-type]
            )
            if ordered:
                weakest_season = ordered[0].season
                strongest_season = ordered[-1].season

        trend_note: str | None = None
        annual_trend_slope: float | None = None
        annual_samples = len(annual_slices)

        if annual_samples >= 2:
            years = np.array([slice_.year for slice_ in annual_slices], dtype=float)
            means = np.array(
                [
                    slice_.mean_speed if slice_.mean_speed is not None else np.nan
                    for slice_ in annual_slices
                ],
                dtype=float,
            )
            mask = ~np.isnan(means)
            if mask.sum() >= 2:
                slope = np.polyfit(years[mask], means[mask], deg=1)[0]
                annual_trend_slope = float(slope)
            else:
                trend_note = "Insufficient annual mean values for regression."
        elif annual_samples == 1:
            trend_note = "Only one year available."
        else:
            trend_note = "No annual samples available."

        summaries.append(
            NodeVariationSummary(
                node_id=node_id,
                strongest_season=strongest_season,
                weakest_season=weakest_season,
                seasonal_amplitude=seasonal_amplitude,
                seasonal_mean_std=seasonal_mean_std,
                seasonal_coverage=len(seasonal_slices),
                annual_trend_slope=annual_trend_slope,
                annual_trend_units="m/s per year",
                annual_samples=annual_samples,
                trend_note=trend_note,
            )
        )

    summaries.sort(key=lambda row: row.node_id)
    return tuple(summaries)


def compute_seasonal_analysis(
    frame: pd.DataFrame,
    *,
    node_column: str = "node_id",
    speed_column: str = "pred_wind_speed",
    label_column: str = "pred_range_label",
    timestamp_column: str = "timestamp",
) -> SeasonalAnalysisResult:
    """Compute seasonal and interannual wind-speed diagnostics per node.

    Parameters
    ----------
    frame:
        ``pandas.DataFrame`` containing at least node identifiers, timestamps,
        predicted wind speeds, and range labels.
    node_column:
        Name of the column holding node identifiers. Defaults to ``node_id``.
    speed_column:
        Column name with the predicted wind speed (m/s). Defaults to
        ``pred_wind_speed``.
    label_column:
        Column with the raw classifier labels used for censoring diagnostics.
    timestamp_column:
        UTC timestamp column. The function normalises values with
        :func:`pandas.to_datetime`.

    Returns
    -------
    SeasonalAnalysisResult
        Structured result with per-season slices, per-year slices, and
        per-node summaries describing seasonal amplitude and annual trends.
    """

    required = {node_column, speed_column, label_column, timestamp_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Frame is missing required columns: {sorted(missing)}")

    seasonal = _build_seasonal_slices(
        frame,
        node_column=node_column,
        speed_column=speed_column,
        label_column=label_column,
        timestamp_column=timestamp_column,
    )
    annual = _build_annual_slices(
        frame,
        node_column=node_column,
        speed_column=speed_column,
        label_column=label_column,
        timestamp_column=timestamp_column,
    )
    variation = _summarise_variation(seasonal, annual)
    return SeasonalAnalysisResult(per_season=seasonal, per_year=annual, variation=variation)
