"""Circular wind direction comparison utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "DirectionErrorMetrics",
    "DirectionErrorResult",
    "DirectionQualitySummary",
    "evaluate_direction_pairs",
]


@dataclass(frozen=True)
class DirectionQualitySummary:
    """Diagnostic counts describing how many direction pairs remain usable."""

    total_pairs: int
    missing_pairs: int
    sentinel_pairs: int
    nonfinite_pairs: int
    valid_pairs: int

    @property
    def coverage_ratio(self) -> float:
        """Return the proportion of pairs that remain after quality filtering."""

        if self.total_pairs == 0:
            return 0.0
        return self.valid_pairs / self.total_pairs


@dataclass(frozen=True)
class DirectionErrorMetrics:
    """Aggregated statistics summarising the angular discrepancies."""

    mean_absolute_error_deg: float | None
    root_mean_square_error_deg: float | None
    circular_bias_deg: float | None
    absolute_error_p50_deg: float | None
    absolute_error_p90_deg: float | None
    absolute_error_p99_deg: float | None
    max_absolute_error_deg: float | None


@dataclass(frozen=True)
class DirectionErrorResult:
    """Container for per-sample angular errors and their aggregate metrics."""

    quality: DirectionQualitySummary
    metrics: DirectionErrorMetrics
    records: pd.DataFrame


def evaluate_direction_pairs(
    predicted_deg: Sequence[float],
    observed_deg: Sequence[float],
    *,
    sentinel_threshold: float = -9000.0,
) -> DirectionErrorResult:
    """Compute angular discrepancies between modelled and observed directions.

    Parameters
    ----------
    predicted_deg:
        Sequence of model-derived wind directions in degrees (clockwise from
        geographic north).
    observed_deg:
        Sequence of buoy wind directions in degrees (clockwise from geographic
        north).
    sentinel_threshold:
        Values less than or equal to this threshold are treated as invalid
        sentinel markers and removed from the comparison. The default matches
        the Vilano buoy encoding (-10000 for missing directions).

    Returns
    -------
    DirectionErrorResult
        Per-sample error records, aggregate metrics, and quality filtering
        diagnostics.
    """

    predicted_series = pd.Series(predicted_deg, copy=False, dtype="float64")
    observed_series = pd.Series(observed_deg, copy=False, dtype="float64")

    if len(predicted_series) != len(observed_series):
        raise ValueError(
            "predicted_deg and observed_deg must contain the same number of elements"
        )

    total_pairs = len(predicted_series)
    missing_mask = predicted_series.isna() | observed_series.isna()

    finite_mask = np.isfinite(predicted_series.to_numpy()) & np.isfinite(
        observed_series.to_numpy()
    )
    finite_mask = pd.Series(finite_mask, index=predicted_series.index)

    sentinel_mask = (
        (predicted_series <= sentinel_threshold)
        | (observed_series <= sentinel_threshold)
    )

    valid_mask = (~missing_mask) & finite_mask & (~sentinel_mask)
    valid_pairs = int(valid_mask.sum())

    quality = DirectionQualitySummary(
        total_pairs=total_pairs,
        missing_pairs=int(missing_mask.sum()),
        sentinel_pairs=int(sentinel_mask.sum()),
        nonfinite_pairs=int((~finite_mask & ~missing_mask).sum()),
        valid_pairs=valid_pairs,
    )

    if valid_pairs == 0:
        empty_frame = pd.DataFrame(
            columns=[
                "predicted_direction_deg",
                "observed_direction_deg",
                "angular_error_deg",
                "absolute_error_deg",
            ]
        )
        metrics = DirectionErrorMetrics(
            mean_absolute_error_deg=None,
            root_mean_square_error_deg=None,
            circular_bias_deg=None,
            absolute_error_p50_deg=None,
            absolute_error_p90_deg=None,
            absolute_error_p99_deg=None,
            max_absolute_error_deg=None,
        )
        return DirectionErrorResult(quality=quality, metrics=metrics, records=empty_frame)

    predicted_valid_series = predicted_series.loc[valid_mask]
    observed_valid_series = observed_series.loc[valid_mask]

    # Normalise directions to [0, 360) before computing angular differences.
    predicted_norm = np.mod(predicted_valid_series.to_numpy(), 360.0)
    observed_norm = np.mod(observed_valid_series.to_numpy(), 360.0)

    angular_error = _circular_difference(predicted_norm, observed_norm)
    absolute_error = np.abs(angular_error)

    records = pd.DataFrame(
        {
            "predicted_direction_deg": predicted_norm,
            "observed_direction_deg": observed_norm,
            "angular_error_deg": angular_error,
            "absolute_error_deg": absolute_error,
        },
        index=predicted_valid_series.index,
    )
    records.index.name = "source_index"

    metrics = _compute_metrics(angular_error, absolute_error)

    return DirectionErrorResult(quality=quality, metrics=metrics, records=records)


def _circular_difference(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Return the minimal signed angular difference (degrees)."""

    return ((predicted - observed + 180.0) % 360.0) - 180.0


def _compute_metrics(
    angular_error: np.ndarray, absolute_error: np.ndarray
) -> DirectionErrorMetrics:
    """Aggregate descriptive statistics for the provided error arrays."""

    mae = float(np.mean(absolute_error))
    rmse = float(np.sqrt(np.mean(np.square(angular_error))))
    max_error = float(np.max(absolute_error))

    percentiles = _safe_quantiles(absolute_error, [0.5, 0.9, 0.99])
    bias_deg = _circular_mean(angular_error)

    return DirectionErrorMetrics(
        mean_absolute_error_deg=mae,
        root_mean_square_error_deg=rmse,
        circular_bias_deg=bias_deg,
        absolute_error_p50_deg=percentiles[0],
        absolute_error_p90_deg=percentiles[1],
        absolute_error_p99_deg=percentiles[2],
        max_absolute_error_deg=max_error,
    )


def _circular_mean(angular_error: np.ndarray) -> float:
    """Compute the circular mean of the error distribution in degrees."""

    radians = np.deg2rad(angular_error)
    sin_mean = float(np.mean(np.sin(radians)))
    cos_mean = float(np.mean(np.cos(radians)))

    if math.isclose(sin_mean, 0.0, abs_tol=1e-12) and math.isclose(
        cos_mean, 0.0, abs_tol=1e-12
    ):
        return 0.0
    return math.degrees(math.atan2(sin_mean, cos_mean))


def _safe_quantiles(
    absolute_error: np.ndarray, quantiles: Sequence[float]
) -> tuple[float, ...]:
    """Return the requested quantiles guarding against numerical edge cases."""

    if absolute_error.size == 0:
        return tuple(float("nan") for _ in quantiles)
    values = np.quantile(absolute_error, quantiles, method="linear")
    return tuple(float(x) for x in np.atleast_1d(values))
