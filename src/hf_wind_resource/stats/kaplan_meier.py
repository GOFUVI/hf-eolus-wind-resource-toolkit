"""Weighted Kaplan–Meier estimator tailored to ANN range-aware outputs."""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

from .weibull import CensoredWeibullData

__all__ = [
    "KaplanMeierResult",
    "KaplanMeierSelectionCriteria",
    "load_kaplan_meier_selection_criteria",
    "evaluate_kaplan_meier_selection",
    "evaluate_step_cdf",
    "run_weighted_kaplan_meier",
]


_EPS = 1e-12


@dataclass(frozen=True)
class KaplanMeierSelectionCriteria:
    """Thresholds that trigger the non-parametric fallback."""

    min_total_observations: int = 200
    min_total_censored_ratio: float = 0.20
    min_below_ratio: float = 0.15
    max_valid_share: float = 0.55
    min_uncensored_weight: float = 150.0


_DEFAULT_CRITERIA_PATH = Path("config") / "kaplan_meier_thresholds.json"


@lru_cache(maxsize=None)
def _load_cached_criteria(path_key: str) -> KaplanMeierSelectionCriteria:
    target_path = Path(path_key)
    if not target_path.exists():
        return KaplanMeierSelectionCriteria()
    try:
        payload = json.loads(target_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in Kaplan–Meier criteria file: {target_path}") from exc

    allowed_keys = {
        "min_total_observations",
        "min_total_censored_ratio",
        "min_below_ratio",
        "max_valid_share",
        "min_uncensored_weight",
    }
    parsed: dict[str, object] = {}
    for key in allowed_keys:
        if key not in payload:
            continue
        value = payload[key]
        if key == "min_total_observations":
            parsed[key] = int(value)
        elif key == "min_uncensored_weight":
            parsed[key] = float(value)
        else:
            parsed[key] = float(value)
    return KaplanMeierSelectionCriteria(**parsed)


def load_kaplan_meier_selection_criteria(
    path: Path | str | None = None,
) -> KaplanMeierSelectionCriteria:
    """Load selection criteria from configuration or fall back to defaults."""

    target_path = Path(path) if path is not None else _DEFAULT_CRITERIA_PATH
    return _load_cached_criteria(str(target_path.resolve()))


@dataclass(frozen=True)
class KaplanMeierResult:
    """Kaplan–Meier step function and ancillary summary statistics."""

    support: Tuple[float, ...]
    cdf: Tuple[float, ...]
    survival: Tuple[float, ...]
    total_weight: float
    left_censored_weight: float
    right_censored_weight: float

    def cdf_at(self, speed: float) -> float:
        """Return the cumulative probability up to ``speed``."""

        if not self.support:
            return 0.0

        idx = bisect_right(self.support, speed) - 1
        if idx < 0:
            return 0.0
        return self.cdf[idx]

    def quantile(self, probability: float) -> float | None:
        """Return the left-most speed with CDF >= ``probability``."""

        if probability < 0.0 or probability > 1.0:
            raise ValueError("Probability must lie inside [0, 1].")
        if not self.support:
            return None

        for speed, cdf_value in zip(self.support, self.cdf):
            if cdf_value + _EPS >= probability:
                return speed
        # Right tail remains unresolved (probability mass above max support).
        return None

    @property
    def right_tail_probability(self) -> float:
        """Return the residual survival probability after the last event."""

        if not self.survival:
            return 1.0
        return self.survival[-1]

    def to_mapping(self) -> Mapping[str, object]:
        """Encode the result as primitives suitable for JSON export."""

        return {
            "support": list(self.support),
            "cdf": list(self.cdf),
            "survival": list(self.survival),
            "total_weight": self.total_weight,
            "left_censored_weight": self.left_censored_weight,
            "right_censored_weight": self.right_censored_weight,
            "right_tail_probability": self.right_tail_probability,
        }


def evaluate_kaplan_meier_selection(
    summary_row: Mapping[str, object],
    *,
    criteria: KaplanMeierSelectionCriteria | None = None,
) -> tuple[bool, Tuple[str, ...]]:
    """Return whether the fallback should run and why."""

    if criteria is None:
        criteria = load_kaplan_meier_selection_criteria()

    total_observations = float(summary_row.get("total_observations") or 0.0)
    if total_observations < criteria.min_total_observations:
        return False, ()

    censored_ratio = float(summary_row.get("censored_ratio") or 0.0)
    below_ratio = float(summary_row.get("below_ratio") or 0.0)
    valid_count = float(summary_row.get("valid_count") or 0.0)
    in_ratio = float(summary_row.get("in_ratio") or 0.0)

    triggers: list[str] = []

    if censored_ratio >= criteria.min_total_censored_ratio:
        triggers.append(
            f"censored ratio {censored_ratio:.3f} ≥ {criteria.min_total_censored_ratio:.2f}"
        )
    if below_ratio >= criteria.min_below_ratio:
        triggers.append(f"below ratio {below_ratio:.3f} ≥ {criteria.min_below_ratio:.2f}")

    if total_observations > 0.0:
        valid_share = valid_count / total_observations
        if valid_share <= criteria.max_valid_share:
            triggers.append(
                f"in-range share {valid_share:.3f} ≤ {criteria.max_valid_share:.2f}"
            )

    if valid_count <= criteria.min_uncensored_weight:
        triggers.append(
            f"uncensored weight {valid_count:.0f} ≤ {criteria.min_uncensored_weight:.0f}"
        )

    return (len(triggers) > 0, tuple(triggers))


def run_weighted_kaplan_meier(data: CensoredWeibullData) -> KaplanMeierResult:
    """Compute the Kaplan–Meier estimator for weighted censored samples."""

    if data.total_weight <= 0.0:
        raise ValueError("Kaplan–Meier estimator requires positive weight.")

    event_weights = _coalesce_weights(data.in_values, data.in_weights)
    left_mass = _sum_positive(data.left_weights)
    if left_mass > 0.0 and data.left_limits:
        # Treat left-censored mass as an event at the minimum censoring limit.
        left_time = min(value for value in data.left_limits if isfinite(value))
        event_weights[left_time] = event_weights.get(left_time, 0.0) + left_mass

    right_censor_weights = _coalesce_weights(data.right_limits, data.right_weights)

    timeline = sorted(set(event_weights).union(right_censor_weights))
    if not timeline:
        # All mass is left-censored or undefined speeds.
        left_time = min(data.left_limits) if data.left_limits else 0.0
        return KaplanMeierResult(
            support=(left_time,),
            cdf=(1.0,),
            survival=(0.0,),
            total_weight=data.total_weight,
            left_censored_weight=left_mass,
            right_censored_weight=_sum_positive(data.right_weights),
        )

    support: list[float] = []
    cdf_points: list[float] = []
    survival_points: list[float] = []

    at_risk = float(data.total_weight)
    survival = 1.0

    for time in timeline:
        events = max(0.0, event_weights.get(time, 0.0))
        censors = max(0.0, right_censor_weights.get(time, 0.0))

        if events > 0.0 and at_risk > 0.0:
            frac = min(1.0, events / max(at_risk, _EPS))
            survival *= max(0.0, 1.0 - frac)
            support.append(time)
            survival_points.append(survival)
            cdf_points.append(1.0 - survival)

        at_risk = max(0.0, at_risk - events - censors)

    return KaplanMeierResult(
        support=tuple(support),
        cdf=tuple(cdf_points),
        survival=tuple(survival_points),
        total_weight=data.total_weight,
        left_censored_weight=left_mass,
        right_censored_weight=_sum_positive(data.right_weights),
    )


def evaluate_step_cdf(result: KaplanMeierResult, speeds: Iterable[float]) -> Tuple[float, ...]:
    """Evaluate the CDF at each speed in ``speeds``."""

    return tuple(result.cdf_at(speed) for speed in speeds)


def _coalesce_weights(values: Sequence[float], weights: Sequence[float]) -> dict[float, float]:
    combined: dict[float, float] = {}
    for value, weight in zip(values, weights):
        if weight is None or value is None:
            continue
        if weight <= 0.0:
            continue
        if not isfinite(value):
            continue
        combined[value] = combined.get(value, 0.0) + float(weight)
    return combined


def _sum_positive(weights: Sequence[float]) -> float:
    total = 0.0
    for weight in weights:
        if weight is None:
            continue
        if weight > 0.0:
            total += float(weight)
    return total
