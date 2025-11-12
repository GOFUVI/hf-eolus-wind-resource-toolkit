"""Shared helpers for computing wind power distributions.

This module centralises the logic that turns censored ANN-derived wind
records into power-density and power-curve estimates. The
``generate_power_estimates`` CLI and the bootstrap uncertainty workflow both
need to evaluate the same decision tree (Weibull vs. Kaplan–Meier) under the
same height-correction notes and censoring summaries. Extracting the helpers
here ensures both paths remain consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import math

from .kaplan_meier import (
    KaplanMeierResult,
    KaplanMeierSelectionCriteria,
    evaluate_kaplan_meier_selection,
    run_weighted_kaplan_meier,
)
from .power import (
    PowerCurve,
    PowerCurveEstimate,
    PowerDensityEstimate,
    compute_kaplan_meier_power_density,
    compute_weibull_power_density,
    estimate_power_curve_from_kaplan_meier,
    estimate_power_curve_from_weibull,
)
from .weibull import CensoredWeibullData, WeibullFitResult, fit_censored_weibull

__all__ = [
    "HeightCorrection",
    "format_height_note",
    "summarise_records_for_selection",
    "compute_power_distribution",
]


@dataclass(frozen=True)
class HeightCorrection:
    """Configuration and scaling factor for vertical wind extrapolation."""

    method: str
    source_height_m: float
    target_height_m: float
    speed_scale: float
    power_law_alpha: float | None = None
    roughness_length_m: float | None = None

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "method": self.method,
            "source_height_m": self.source_height_m,
            "target_height_m": self.target_height_m,
            "speed_scale": self.speed_scale,
            "power_law_alpha": self.power_law_alpha,
            "roughness_length_m": self.roughness_length_m,
        }


def format_height_note(height: HeightCorrection) -> str | None:
    """Return a human-readable note describing the applied height correction."""

    if math.isclose(height.speed_scale, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        return None
    if height.method == "log":
        roughness = height.roughness_length_m if height.roughness_length_m is not None else 0.0
        return (
            f"Height correction log-law {height.source_height_m:.1f}→{height.target_height_m:.1f} m "
            f"(scale {height.speed_scale:.3f}, z0={roughness:.4f} m)."
        )
    if height.method == "power":
        alpha = height.power_law_alpha if height.power_law_alpha is not None else 0.0
        return (
            f"Height correction power-law α={alpha:.3f} {height.source_height_m:.1f}→{height.target_height_m:.1f} m "
            f"(scale {height.speed_scale:.3f})."
        )
    return None


def summarise_records_for_selection(
    records: Sequence[Mapping[str, object]],
    *,
    min_confidence: float,
) -> Mapping[str, float]:
    """Aggregate censoring weights required by the Kaplan–Meier selector."""

    total_weight = 0.0
    below_weight = 0.0
    in_weight = 0.0
    above_weight = 0.0

    for record in records:
        prob_below = float(record.get("prob_range_below") or 0.0)
        prob_in = float(record.get("prob_range_in") or 0.0)
        prob_above = float(record.get("prob_range_above") or 0.0)
        total_prob = prob_below + prob_in + prob_above
        if total_prob <= 0.0:
            continue

        prob_below /= total_prob
        prob_in /= total_prob
        prob_above /= total_prob

        flag = str(record.get("range_flag") or "").strip().lower()
        flag_confident = bool(record.get("range_flag_confident"))
        confident_weight_applied = False
        if flag_confident:
            if flag == "in" and prob_in >= min_confidence:
                in_weight += 1.0
                confident_weight_applied = True
            elif flag == "below" and prob_below >= min_confidence:
                below_weight += 1.0
                confident_weight_applied = True
            elif flag == "above" and prob_above >= min_confidence:
                above_weight += 1.0
                confident_weight_applied = True

        if not confident_weight_applied:
            below_weight += prob_below
            in_weight += prob_in
            above_weight += prob_above

        total_weight += 1.0

    if total_weight <= 0.0:
        return {
            "total_observations": 0.0,
            "censored_ratio": 0.0,
            "below_ratio": 0.0,
            "in_ratio": 0.0,
        }

    censored_weight = below_weight + above_weight
    return {
        "total_observations": total_weight,
        "censored_ratio": censored_weight / total_weight,
        "below_ratio": below_weight / total_weight,
        "in_ratio": in_weight / total_weight,
    }


def compute_power_distribution(
    *,
    data: CensoredWeibullData,
    summary_row: Mapping[str, object],
    power_curve: PowerCurve,
    air_density: float,
    tail_surrogate: float,
    min_in_range: float,
    km_criteria: KaplanMeierSelectionCriteria,
    height: HeightCorrection,
) -> tuple[
    str,
    PowerDensityEstimate,
    PowerCurveEstimate,
    WeibullFitResult,
    KaplanMeierResult | None,
    Tuple[str, ...],
    list[str],
]:
    """Return the power metrics and diagnostics for a node."""

    weibull = fit_censored_weibull(
        data,
        min_in_count=min_in_range,
    )
    selection_required, selection_reasons = evaluate_kaplan_meier_selection(
        summary_row, criteria=km_criteria
    )

    method = "weibull"
    method_notes: list[str] = []
    if not (weibull.success and weibull.reliable):
        method = "kaplan_meier"
    if selection_required:
        method_notes.append("Kaplan–Meier fallback triggered by censoring criteria.")
        method = "kaplan_meier"

    power_density: PowerDensityEstimate
    power_curve_estimate: PowerCurveEstimate
    km_result: KaplanMeierResult | None = None

    if method == "weibull":
        power_density = compute_weibull_power_density(
            weibull,
            air_density=air_density,
            speed_scale=height.speed_scale,
        )
        power_curve_estimate = estimate_power_curve_from_weibull(
            weibull,
            power_curve,
            air_density=air_density,
            speed_scale=height.speed_scale,
        )
    else:
        km_result = run_weighted_kaplan_meier(data)
        power_density = compute_kaplan_meier_power_density(
            km_result,
            air_density=air_density,
            right_tail_surrogate=tail_surrogate,
            speed_scale=height.speed_scale,
        )
        power_curve_estimate = estimate_power_curve_from_kaplan_meier(
            km_result,
            power_curve,
            air_density=air_density,
            right_tail_surrogate=tail_surrogate,
            speed_scale=height.speed_scale,
        )

    return method, power_density, power_curve_estimate, weibull, km_result, selection_reasons, method_notes
