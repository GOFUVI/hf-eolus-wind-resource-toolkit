"""Temporal normalisation and QA utilities for HF radar ANN datasets.

This module prepares the ``sar_range_final_pivots_joined`` snapshot for
statistical processing by enforcing a consistent temporal grid, removing
duplicate observations, and extracting quality indicators per mesh node.
It generates structured gap logs, sampling summaries, and coverage metrics
that downstream components can consume when computing wind resource
statistics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Dict, Mapping, Sequence, Tuple

import pandas as pd

from hf_wind_resource.io import GapDescriptor, GapReport, TimeWindow

__all__ = [
    "CadenceStats",
    "NodeTemporalSummary",
    "TemporalNormalizationConfig",
    "TemporalNormalizationResult",
    "TemporalQualityFlags",
    "normalise_temporal_records",
]


@dataclass(frozen=True)
class TemporalQualityFlags:
    """Boolean indicators summarising temporal QA checks."""

    has_duplicates: bool
    has_gaps: bool
    irregular_cadence: bool
    insufficient_coverage: bool


@dataclass(frozen=True)
class CadenceStats:
    """Describes cadence behaviour for a node."""

    expected: timedelta
    min_observed: timedelta | None
    max_observed: timedelta | None
    mean_observed_seconds: float | None
    median_observed_seconds: float | None
    std_observed_seconds: float | None
    sample_count: int
    distribution_seconds: Mapping[int, int]


@dataclass(frozen=True)
class NodeTemporalSummary:
    """Aggregated temporal metrics for a single node."""

    node_id: str
    sampling_start: datetime | None
    sampling_end: datetime | None
    total_observations: int
    duplicate_records: int
    missing_observations: int
    expected_observations: int | None
    coverage_ratio: float | None
    cadence: CadenceStats
    gap_windows: Tuple[TimeWindow, ...]
    quality_flags: TemporalQualityFlags
    notes: Tuple[str, ...]


@dataclass(frozen=True)
class TemporalNormalizationConfig:
    """Configuration knobs for temporal normalisation."""

    expected_cadence: timedelta = timedelta(minutes=30)
    rounding: str = "nearest"
    tolerance: timedelta = timedelta(minutes=5)
    coverage_threshold: float = 0.8

    def __post_init__(self) -> None:
        allowed = {"nearest", "floor", "ceiling"}
        if self.rounding not in allowed:
            raise ValueError(f"rounding must be one of {allowed}, received {self.rounding!r}")
        if self.expected_cadence <= timedelta(0):
            raise ValueError("expected_cadence must be positive")
        if self.tolerance < timedelta(0):
            raise ValueError("tolerance must be non-negative")
        if not (0.0 < self.coverage_threshold <= 1.0):
            raise ValueError("coverage_threshold must lie in (0, 1]")


@dataclass(frozen=True)
class TemporalNormalizationResult:
    """Holds the artefacts produced by ``normalise_temporal_records``."""

    dataframe: pd.DataFrame
    per_node: Mapping[str, NodeTemporalSummary]
    gap_report: GapReport


def normalise_temporal_records(
    frame: pd.DataFrame,
    timestamp_column: str = "timestamp",
    node_column: str = "node_id",
    config: TemporalNormalizationConfig | None = None,
) -> TemporalNormalizationResult:
    """Normalise timestamps and compute temporal QA summaries per node.

    Parameters
    ----------
    frame:
        ``pandas.DataFrame`` containing the ANN inference snapshot. Must
        expose at least ``timestamp`` and ``node_id`` columns matching the
        schema recorded in ``docs/sar_range_final_schema.md``.
    timestamp_column:
        Name of the column holding UTC timestamps.
    node_column:
        Name of the column identifying mesh nodes.
    config:
        Optional :class:`TemporalNormalizationConfig` adjusting cadence,
        rounding strategy, tolerance for irregular intervals, and coverage
        thresholds.

    Returns
    -------
    TemporalNormalizationResult
        Structured summary with the normalised DataFrame, per-node metrics,
        and a :class:`GapReport` storing all detected gap windows.
    """

    if config is None:
        config = TemporalNormalizationConfig()

    required_columns = {timestamp_column, node_column}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise KeyError(f"Frame is missing required columns: {sorted(missing_columns)}")

    df = frame.copy(deep=True)
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], utc=True)
    df[timestamp_column] = df[timestamp_column].dt.tz_convert("UTC")

    cadence_delta = pd.Timedelta(config.expected_cadence)
    tolerance_delta = pd.Timedelta(config.tolerance)

    if config.rounding == "nearest":
        df[timestamp_column] = df[timestamp_column].dt.round(cadence_delta)
    elif config.rounding == "floor":
        df[timestamp_column] = df[timestamp_column].dt.floor(cadence_delta)
    else:
        df[timestamp_column] = df[timestamp_column].dt.ceil(cadence_delta)

    duplicates_mask = df.duplicated(subset=[node_column, timestamp_column], keep="first")
    duplicate_counts = Counter(df.loc[duplicates_mask, node_column])
    df = df.loc[~duplicates_mask].copy()

    df.sort_values(by=[node_column, timestamp_column], inplace=True)
    df.reset_index(drop=True, inplace=True)

    per_node: Dict[str, NodeTemporalSummary] = {}
    gap_report = GapReport(descriptors={})

    grouped = df.groupby(node_column, sort=False)
    for node_id, group in grouped:
        group = group.sort_values(timestamp_column)
        n_rows = len(group)
        sampling_start: datetime | None
        sampling_end: datetime | None
        if n_rows:
            sampling_start = group.iloc[0][timestamp_column].to_pydatetime()
            sampling_end = group.iloc[-1][timestamp_column].to_pydatetime()
        else:
            sampling_start = None
            sampling_end = None

        deltas = group[timestamp_column].diff().iloc[1:]
        delta_seconds = deltas.dt.total_seconds().dropna()
        distribution = Counter(int(value) for value in delta_seconds)

        mean_seconds = float(delta_seconds.mean()) if not delta_seconds.empty else None
        median_seconds = float(delta_seconds.median()) if not delta_seconds.empty else None
        std_seconds = float(delta_seconds.std(ddof=0)) if not delta_seconds.empty else None
        min_delta = deltas.min() if not deltas.empty else None
        max_delta = deltas.max() if not deltas.empty else None

        expected_observations: int | None
        missing_observations = 0
        coverage_ratio: float | None
        notes: list[str] = []
        gap_windows: list[TimeWindow] = []

        if n_rows:
            duration = group.iloc[-1][timestamp_column] - group.iloc[0][timestamp_column]
            expected_observations = int(duration // cadence_delta) + 1
            expected_observations = max(expected_observations, 1)
            observed_seconds: Sequence[float] = list(delta_seconds)
            irregular = any(abs(value - cadence_delta.total_seconds()) > tolerance_delta.total_seconds() for value in observed_seconds)
        else:
            expected_observations = None
            coverage_ratio = None
            irregular = False

        if n_rows:
            previous_timestamp = group.iloc[0][timestamp_column]
            for current_timestamp in group.iloc[1:][timestamp_column]:
                delta = current_timestamp - previous_timestamp
                if delta > cadence_delta + tolerance_delta:
                    step_ratio = float(delta / cadence_delta)
                    missing_steps = max(math.ceil(step_ratio) - 1, 0)
                    if missing_steps:
                        missing_observations += missing_steps
                        gap_start = previous_timestamp + cadence_delta
                        gap_end = previous_timestamp + cadence_delta * missing_steps
                        gap_windows.append(
                            TimeWindow(
                                start=gap_start.to_pydatetime(),
                                end=gap_end.to_pydatetime(),
                            )
                        )
                previous_timestamp = current_timestamp

            observed_total = n_rows
            if expected_observations is not None:
                missing_from_expectation = max(expected_observations - observed_total, 0)
                missing_observations = max(missing_observations, missing_from_expectation)
                denominator = observed_total + missing_observations
                coverage_ratio = observed_total / denominator if denominator else None
            else:
                coverage_ratio = None
        else:
            coverage_ratio = None
            irregular = False

        if duplicate_counts.get(node_id, 0):
            notes.append(
                f"Removed {duplicate_counts[node_id]} duplicate records after temporal normalisation"
            )
        if missing_observations:
            notes.append(f"Identified {missing_observations} missing observations")

        quality_flags = TemporalQualityFlags(
            has_duplicates=duplicate_counts.get(node_id, 0) > 0,
            has_gaps=bool(gap_windows),
            irregular_cadence=irregular,
            insufficient_coverage=(
                coverage_ratio is not None and coverage_ratio < config.coverage_threshold
            ),
        )

        if quality_flags.irregular_cadence:
            notes.append(
                "Observed cadence deviates from expected interval beyond configured tolerance"
            )
        if quality_flags.insufficient_coverage:
            ratio_display = coverage_ratio if coverage_ratio is not None else 0.0
            notes.append(
                f"Coverage ratio {ratio_display:.3f} below threshold {config.coverage_threshold:.3f}"
            )

        cadence_stats = CadenceStats(
            expected=config.expected_cadence,
            min_observed=min_delta.to_pytimedelta() if min_delta is not None else None,
            max_observed=max_delta.to_pytimedelta() if max_delta is not None else None,
            mean_observed_seconds=mean_seconds,
            median_observed_seconds=median_seconds,
            std_observed_seconds=std_seconds,
            sample_count=len(delta_seconds),
            distribution_seconds=dict(distribution),
        )

        summary = NodeTemporalSummary(
            node_id=node_id,
            sampling_start=sampling_start,
            sampling_end=sampling_end,
            total_observations=n_rows,
            duplicate_records=duplicate_counts.get(node_id, 0),
            missing_observations=missing_observations,
            expected_observations=expected_observations,
            coverage_ratio=coverage_ratio,
            cadence=cadence_stats,
            gap_windows=tuple(gap_windows),
            quality_flags=quality_flags,
            notes=tuple(notes),
        )
        per_node[node_id] = summary

        if gap_windows:
            gap_report.register(
                GapDescriptor(
                    node_id=node_id,
                    expected_cadence=config.expected_cadence,
                    missing_windows=tuple(gap_windows),
                )
            )

    return TemporalNormalizationResult(dataframe=df, per_node=per_node, gap_report=gap_report)
