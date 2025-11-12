"""Power-related utilities built on top of fitted wind distributions.

This module derives average wind power density (W/m²) and expected turbine
production from statistical models such as censored Weibull fits and
Kaplan–Meier survival functions. Results record the air-density assumptions
and any approximations applied outside the ANN regression range so downstream
reporting layers can disclose the theoretical nature of these estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gamma
import math
from typing import Mapping, Sequence, Tuple

import numpy as np

from .kaplan_meier import KaplanMeierResult
from .weibull import WeibullFitResult

__all__ = [
    "PowerCurve",
    "PowerCurveEstimate",
    "PowerDensityEstimate",
    "compute_kaplan_meier_power_density",
    "compute_weibull_power_density",
    "estimate_power_curve_from_kaplan_meier",
    "estimate_power_curve_from_weibull",
]


@dataclass(frozen=True)
class PowerCurve:
    """Tabulated turbine power curve expressed in kilo-watts."""

    name: str
    speeds: Tuple[float, ...]
    power_kw: Tuple[float, ...]
    reference_air_density: float = 1.225
    hub_height_m: float | None = None
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.speeds) != len(self.power_kw):
            raise ValueError("speeds and power_kw must share the same length.")
        if not self.speeds:
            raise ValueError("Power curves require at least one sample.")
        if any(speed < 0.0 for speed in self.speeds):
            raise ValueError("Wind speeds must be non-negative.")
        if any(power < 0.0 for power in self.power_kw):
            raise ValueError("Power values must be non-negative.")
        if any(self.speeds[i] >= self.speeds[i + 1] for i in range(len(self.speeds) - 1)):
            raise ValueError("Wind speeds must be strictly increasing.")
        if self.reference_air_density <= 0.0:
            raise ValueError("reference_air_density must be positive.")

    @property
    def rated_power_kw(self) -> float:
        """Return the maximum power attainable on the tabulated curve."""

        return max(self.power_kw)

    @property
    def cutout_speed(self) -> float:
        """Return the highest wind speed covered by the tabulated curve."""

        return self.speeds[-1]

    def evaluate_kw(self, speeds: Sequence[float], *, air_density: float | None = None) -> np.ndarray:
        """Return interpolated power values for *speeds* at the requested density."""

        if air_density is None:
            density_factor = 1.0
        else:
            if air_density <= 0.0:
                raise ValueError("air_density must be positive when provided.")
            density_factor = air_density / self.reference_air_density

        arr = np.asarray(list(speeds), dtype=float)
        interpolated = np.interp(arr, self.speeds, self.power_kw, left=0.0, right=self.power_kw[-1])
        # Speeds above cut-out often map to zero power; honour the tabulated value.
        return interpolated * density_factor

    def to_mapping(self) -> Mapping[str, object]:
        """Encode the power-curve metadata as primitives suitable for JSON."""

        return {
            "name": self.name,
            "reference_air_density": self.reference_air_density,
            "hub_height_m": self.hub_height_m,
            "speeds": list(self.speeds),
            "power_kw": list(self.power_kw),
            "rated_power_kw": self.rated_power_kw,
            "cutout_speed": self.cutout_speed,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class PowerDensityEstimate:
    """Average wind power density inferred from a statistical model."""

    method: str
    estimate_w_per_m2: float | None
    air_density: float
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PowerCurveEstimate:
    """Expected turbine generation derived from a wind distribution."""

    curve: PowerCurve
    mean_power_kw: float | None
    capacity_factor: float | None
    air_density: float
    notes: Tuple[str, ...] = ()


def compute_weibull_power_density(
    result: WeibullFitResult,
    *,
    air_density: float = 1.225,
    speed_scale: float = 1.0,
) -> PowerDensityEstimate:
    """Return the mean power density implied by a censored Weibull fit."""

    notes: list[str] = []
    if air_density <= 0.0:
        raise ValueError("air_density must be positive.")

    if not (result.success and result.reliable and result.shape and result.scale):
        notes.append("Weibull fit is unavailable or flagged as unreliable.")
        return PowerDensityEstimate(
            method="weibull",
            estimate_w_per_m2=None,
            air_density=air_density,
            notes=tuple(notes),
        )

    shape = result.shape
    scale = result.scale
    effective_scale = scale * max(speed_scale, 0.0)
    estimate = _weibull_power_density(shape, effective_scale, air_density)
    notes.append("Computed from censored Weibull parameters assuming full-range validity.")
    if not math.isclose(speed_scale, 1.0):
        notes.append(f"Speeds scaled by factor {speed_scale:.3f} before computing power density.")
    return PowerDensityEstimate(
        method="weibull",
        estimate_w_per_m2=estimate,
        air_density=air_density,
        notes=tuple(notes),
    )


def compute_kaplan_meier_power_density(
    result: KaplanMeierResult,
    *,
    air_density: float = 1.225,
    right_tail_surrogate: float | None = None,
    speed_scale: float = 1.0,
) -> PowerDensityEstimate:
    """Return the power density using the Kaplan–Meier step distribution."""

    if air_density <= 0.0:
        raise ValueError("air_density must be positive.")

    support, masses = _kaplan_meier_probabilities(result)
    if support.size == 0 and result.right_tail_probability <= 0.0:
        notes = ("Kaplan–Meier estimator returned no events; cannot derive power density.",)
        return PowerDensityEstimate(
            method="kaplan_meier",
            estimate_w_per_m2=None,
            air_density=air_density,
            notes=notes,
        )

    scaled_support = support * speed_scale
    moment = float(np.sum(masses * np.power(scaled_support, 3)))
    notes: list[str] = []

    tail = max(0.0, float(result.right_tail_probability))
    if tail > 0.0:
        if right_tail_surrogate is None:
            notes.append(
                "Right-tail probability present without surrogate speed; returning lower-bound estimate."
            )
            tail_contrib = 0.0
        else:
            tail_speed = right_tail_surrogate * speed_scale
            tail_contrib = tail * tail_speed**3
            notes.append(
                f"Right-tail probability {tail:.3f} approximated at {tail_speed:.2f} m/s."
            )
    else:
        tail_contrib = 0.0

    total_moment = moment + tail_contrib
    if total_moment <= 0.0:
        estimate = None
        notes.append("No finite contribution to v^3 moment detected.")
    else:
        estimate = 0.5 * air_density * total_moment

    if estimate is None and not notes:
        notes.append("Kaplan–Meier distribution had zero support and tail contribution.")
    if not math.isclose(speed_scale, 1.0):
        notes.append(f"Speeds scaled by factor {speed_scale:.3f} before computing power density.")

    return PowerDensityEstimate(
        method="kaplan_meier",
        estimate_w_per_m2=estimate,
        air_density=air_density,
        notes=tuple(notes),
    )


def estimate_power_curve_from_weibull(
    result: WeibullFitResult,
    curve: PowerCurve,
    *,
    air_density: float = 1.225,
    integration_points: int = 2000,
    speed_scale: float = 1.0,
) -> PowerCurveEstimate:
    """Return expected turbine production using a censored Weibull fit."""

    if air_density <= 0.0:
        raise ValueError("air_density must be positive.")
    if integration_points <= 10:
        raise ValueError("integration_points must exceed 10 for numerical stability.")

    notes: list[str] = []

    if not (result.success and result.reliable and result.shape and result.scale):
        notes.append("Weibull fit is unavailable or flagged as unreliable.")
        return PowerCurveEstimate(
            curve=curve,
            mean_power_kw=None,
            capacity_factor=None,
            air_density=air_density,
            notes=tuple(notes),
    )

    shape = result.shape
    scale = result.scale

    effective_scale = scale * max(speed_scale, 0.0)

    speeds = _integration_grid(max(curve.cutout_speed, effective_scale * 5.0), integration_points)
    pdf = _weibull_pdf(speeds, shape, effective_scale)

    # Restrict contribution beyond cut-out if the curve terminates with zero.
    curve_power = curve.evaluate_kw(speeds, air_density=air_density)
    if curve.power_kw[-1] == 0.0:
        mask = speeds > curve.cutout_speed
        curve_power = curve_power.copy()
        curve_power[mask] = 0.0

    mean_power_kw = float(np.trapz(curve_power * pdf, speeds))

    capacity = None
    rated = curve.rated_power_kw
    if rated > 0.0:
        capacity = mean_power_kw / rated

    notes.append(
        "Expected turbine output derived via numerical integration of the Weibull PDF over the tabulated power curve."
    )
    if air_density != curve.reference_air_density:
        notes.append(
            f"Power curve scaled linearly by air-density ratio {air_density / curve.reference_air_density:.3f}."
        )
    if not math.isclose(speed_scale, 1.0):
        notes.append(
            f"Speeds scaled by factor {speed_scale:.3f} before evaluating the power curve."
        )

    return PowerCurveEstimate(
        curve=curve,
        mean_power_kw=mean_power_kw,
        capacity_factor=capacity,
        air_density=air_density,
        notes=tuple(notes),
    )


def estimate_power_curve_from_kaplan_meier(
    result: KaplanMeierResult,
    curve: PowerCurve,
    *,
    air_density: float = 1.225,
    right_tail_surrogate: float | None = None,
    speed_scale: float = 1.0,
) -> PowerCurveEstimate:
    """Return expected turbine output using the Kaplan–Meier estimate."""

    if air_density <= 0.0:
        raise ValueError("air_density must be positive.")

    support, masses = _kaplan_meier_probabilities(result)
    notes: list[str] = []

    if support.size == 0 and result.right_tail_probability <= 0.0:
        notes.append("Kaplan–Meier estimator returned no events; cannot derive turbine output.")
        return PowerCurveEstimate(
            curve=curve,
            mean_power_kw=None,
            capacity_factor=None,
            air_density=air_density,
            notes=tuple(notes),
        )

    scaled_support = support * speed_scale
    power_values = curve.evaluate_kw(scaled_support, air_density=air_density)
    mean_power_kw = float(np.sum(power_values * masses))

    tail = max(0.0, float(result.right_tail_probability))
    if tail > 0.0:
        if right_tail_surrogate is None:
            notes.append(
                "Right-tail probability present without surrogate speed; turbine output treated as lower-bound."
            )
        else:
            surrogate_speed = right_tail_surrogate * speed_scale
            surrogate_power = float(curve.evaluate_kw([surrogate_speed], air_density=air_density)[0])
            mean_power_kw += tail * surrogate_power
            notes.append(
                f"Right-tail probability {tail:.3f} approximated at {surrogate_speed:.2f} m/s "
                "for turbine output."
            )

    capacity = None
    rated = curve.rated_power_kw
    if rated > 0.0 and mean_power_kw is not None:
        capacity = mean_power_kw / rated

    if air_density != curve.reference_air_density:
        notes.append(
            f"Power curve scaled linearly by air-density ratio {air_density / curve.reference_air_density:.3f}."
        )

    if not notes:
        notes.append("Derived by discrete summation over Kaplan–Meier support points.")
    if not math.isclose(speed_scale, 1.0):
        notes.append(
            f"Speeds scaled by factor {speed_scale:.3f} before evaluating the power curve."
        )

    return PowerCurveEstimate(
        curve=curve,
        mean_power_kw=mean_power_kw,
        capacity_factor=capacity,
        air_density=air_density,
        notes=tuple(notes),
    )


def _weibull_power_density(shape: float, scale: float, air_density: float) -> float:
    if shape <= 0.0 or scale <= 0.0:
        raise ValueError("Weibull shape and scale must be positive.")
    third_moment = scale**3 * gamma(1.0 + 3.0 / shape)
    return 0.5 * air_density * third_moment


def _integration_grid(max_speed: float, points: int) -> np.ndarray:
    lower = 1e-6
    upper = max(max_speed, lower * 10.0)
    return np.linspace(lower, upper, points, dtype=float)


def _weibull_pdf(speeds: np.ndarray, shape: float, scale: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scaled = speeds / scale
        base = (shape / scale) * np.power(scaled, shape - 1.0) * np.exp(-np.power(scaled, shape))
    base = np.where(np.isfinite(base), base, 0.0)
    return base


def _kaplan_meier_probabilities(result: KaplanMeierResult) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(result.support, dtype=float)
    if support.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    cdf = np.asarray(result.cdf, dtype=float)
    masses = np.empty_like(cdf)
    masses[0] = max(0.0, cdf[0])
    if masses.size > 1:
        diffs = np.diff(cdf)
        masses[1:] = np.maximum(0.0, diffs)
    masses = np.asarray(masses, dtype=float)
    total = float(np.sum(masses))
    if total > 1.0:
        masses = masses / total
    return support, masses
