"""Maximum-likelihood Weibull fitting with explicit censoring support."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, gamma, isfinite, log, log1p, sqrt
from typing import Iterable, Mapping, Sequence

__all__ = [
    "CensoredWeibullData",
    "WeibullFitDiagnostics",
    "WeibullFitResult",
    "build_censored_data_from_records",
    "compute_censored_weibull_log_likelihood",
    "fit_censored_weibull",
]


_SMALL = 1e-12
_LOG_SHAPE_MIN = log(0.3)
_LOG_SHAPE_MAX = log(12.0)
_LOG_SCALE_MIN = log(0.5)
_LOG_SCALE_MAX = log(40.0)


@dataclass(frozen=True)
class CensoredWeibullData:
    """Container holding weighted samples for censored Weibull fitting."""

    in_values: tuple[float, ...]
    in_weights: tuple[float, ...]
    left_limits: tuple[float, ...]
    left_weights: tuple[float, ...]
    right_limits: tuple[float, ...]
    right_weights: tuple[float, ...]

    def __post_init__(self) -> None:
        _ensure_same_length(self.in_values, self.in_weights, "in_values", "in_weights")
        _ensure_same_length(self.left_limits, self.left_weights, "left_limits", "left_weights")
        _ensure_same_length(self.right_limits, self.right_weights, "right_limits", "right_weights")

    @property
    def in_count(self) -> float:
        """Return the total weight of the uncensored observations."""

        return _sum_weights(self.in_weights)

    @property
    def left_count(self) -> float:
        """Return the total weight of the left-censored observations."""

        return _sum_weights(self.left_weights)

    @property
    def right_count(self) -> float:
        """Return the total weight of the right-censored observations."""

        return _sum_weights(self.right_weights)

    @property
    def total_weight(self) -> float:
        """Return the total observation weight (censored + uncensored)."""

        return self.in_count + self.left_count + self.right_count

    def is_empty(self) -> bool:
        """Return ``True`` when no observations are present."""

        return self.total_weight <= 0.0


@dataclass(frozen=True)
class WeibullFitDiagnostics:
    """Diagnostic metadata describing the optimisation outcome."""

    iterations: int
    gradient_norm: float
    last_step_size: float
    message: str


@dataclass(frozen=True)
class WeibullFitResult:
    """Encapsulate the outcome of the censored Weibull fit."""

    shape: float | None
    scale: float | None
    log_likelihood: float | None
    success: bool
    diagnostics: WeibullFitDiagnostics
    used_gradients: bool
    reliable: bool
    in_count: float
    left_count: float
    right_count: float


def build_censored_data_from_records(
    records: Iterable[Mapping[str, object]],
    *,
    lower_threshold: float,
    upper_threshold: float,
    min_confidence: float = 0.5,
) -> CensoredWeibullData:
    """Convert ANN range-aware records into weighted censored samples.

    Parameters
    ----------
    records:
        Sequence of mappings exposing at least the fields documented in
        ``docs/sar_range_final_schema.md``: ``pred_wind_speed``,
        ``prob_range_below``, ``prob_range_in``, ``prob_range_above``,
        ``range_flag`` and ``range_flag_confident``.
    lower_threshold / upper_threshold:
        Physical operating limits of the HF radar inversion (m/s). Samples
        labelled ``below`` or ``above`` are treated as left- and
        right-censored at these thresholds.
    min_confidence:
        Minimum confidence for ``range_flag`` to be treated as a hard label.
        When the flag is not confident, the posterior probabilities are used
        as fractional weights instead of the discrete class.
    """

    in_values: list[float] = []
    in_weights: list[float] = []
    left_limits: dict[float, float] = {}
    right_limits: dict[float, float] = {}

    lower_threshold = max(lower_threshold, _SMALL)
    upper_threshold = max(upper_threshold, lower_threshold + _SMALL)

    for record in records:
        speed = _to_positive_float(record.get("pred_wind_speed"))
        prob_below = _clamp_probability(record.get("prob_range_below"))
        prob_in = _clamp_probability(record.get("prob_range_in"))
        prob_above = _clamp_probability(record.get("prob_range_above"))
        total_prob = prob_below + prob_in + prob_above
        if total_prob <= 0.0:
            continue
        # Normalise probabilities if they suffer from floating-point drift.
        prob_below /= total_prob
        prob_in /= total_prob
        prob_above /= total_prob

        flag = str(record.get("range_flag") or "").strip().lower()
        flag_confident = _to_bool(record.get("range_flag_confident"))
        confident = False
        if flag_confident:
            if flag == "in" and prob_in >= min_confidence:
                confident = True
            elif flag == "below" and prob_below >= min_confidence:
                confident = True
            elif flag == "above" and prob_above >= min_confidence:
                confident = True

        if confident and flag in {"below", "in", "above"}:
            if flag == "in" and speed is not None:
                in_values.append(speed)
                in_weights.append(1.0)
            elif flag == "below":
                left_limits[lower_threshold] = left_limits.get(lower_threshold, 0.0) + 1.0
            elif flag == "above":
                right_limits[upper_threshold] = right_limits.get(upper_threshold, 0.0) + 1.0
            continue

        if speed is not None and prob_in > 0.0:
            in_values.append(speed)
            in_weights.append(prob_in)
        if prob_below > 0.0:
            left_limits[lower_threshold] = left_limits.get(lower_threshold, 0.0) + prob_below
        if prob_above > 0.0:
            right_limits[upper_threshold] = right_limits.get(upper_threshold, 0.0) + prob_above

    left_limits_list, left_weights_list = _dict_to_lists(left_limits)
    right_limits_list, right_weights_list = _dict_to_lists(right_limits)

    return CensoredWeibullData(
        in_values=tuple(in_values),
        in_weights=tuple(in_weights if in_weights else [1.0] * len(in_values)),
        left_limits=tuple(left_limits_list),
        left_weights=tuple(left_weights_list),
        right_limits=tuple(right_limits_list),
        right_weights=tuple(right_weights_list),
    )


def compute_censored_weibull_log_likelihood(
    shape: float,
    scale: float,
    data: CensoredWeibullData,
    *,
    compute_gradients: bool = False,
) -> tuple[float, tuple[float, float] | None]:
    """Evaluate the log-likelihood (and optionally its gradients)."""

    if shape <= 0.0 or scale <= 0.0:
        raise ValueError("Shape and scale parameters must be positive.")

    log_shape = log(shape)
    log_scale = log(scale)

    total_log_likelihood = 0.0
    grad_shape = 0.0
    grad_scale = 0.0

    for value, weight in zip(data.in_values, data.in_weights):
        if value <= 0.0 or weight <= 0.0:
            continue
        log_value = log(value)
        log_ratio = log_value - log_scale
        exponent = shape * log_ratio
        power_term = exp(exponent)
        contribution = log_shape + (shape - 1.0) * log_value - shape * log_scale - power_term
        total_log_likelihood += weight * contribution
        if compute_gradients:
            grad_shape += weight * ((1.0 / shape) + log_value - log_scale - power_term * log_ratio)
            grad_scale += weight * ((shape / scale) * (power_term - 1.0))

    for limit, weight in zip(data.left_limits, data.left_weights):
        if limit <= 0.0 or weight <= 0.0:
            continue
        log_ratio = log(limit) - log_scale
        exponent = shape * log_ratio
        u_value = exp(exponent)
        survival = exp(-u_value)
        cdf = 1.0 - survival
        if cdf < _SMALL:
            cdf = _SMALL
        log_cdf = log1p(-survival) if survival > 0.1 else log(cdf)
        total_log_likelihood += weight * log_cdf
        if compute_gradients:
            grad_shape += weight * ((survival * u_value * log_ratio) / cdf)
            grad_scale += weight * (-(survival * u_value * shape) / (scale * cdf))

    for limit, weight in zip(data.right_limits, data.right_weights):
        if limit <= 0.0 or weight <= 0.0:
            continue
        log_ratio = log(limit) - log_scale
        exponent = shape * log_ratio
        u_value = exp(exponent)
        total_log_likelihood -= weight * u_value
        if compute_gradients:
            grad_shape -= weight * (u_value * log_ratio)
            grad_scale += weight * (u_value * shape / scale)

    gradients = (grad_shape, grad_scale) if compute_gradients else None
    return total_log_likelihood, gradients


def fit_censored_weibull(
    data: CensoredWeibullData,
    *,
    min_in_count: float = 500.0,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
    base_step_size: float = 0.05,
) -> WeibullFitResult:
    """Fit a censored Weibull distribution via gradient-ascent MLE."""

    if data.in_count < min_in_count or data.in_count <= 0.0:
        diagnostics = WeibullFitDiagnostics(
            iterations=0,
            gradient_norm=0.0,
            last_step_size=0.0,
            message="Insufficient in-range support for the parametric fit.",
        )
        return WeibullFitResult(
            shape=None,
            scale=None,
            log_likelihood=None,
            success=False,
            diagnostics=diagnostics,
            used_gradients=False,
            reliable=False,
            in_count=data.in_count,
            left_count=data.left_count,
            right_count=data.right_count,
        )

    initial_shape, initial_scale = _initial_parameters(data)
    log_shape = _clamp(initial_shape, _LOG_SHAPE_MIN, _LOG_SHAPE_MAX, log_space=True)
    log_scale = _clamp(initial_scale, _LOG_SCALE_MIN, _LOG_SCALE_MAX, log_space=True)

    shape = exp(log_shape)
    scale = exp(log_scale)
    log_likelihood, gradients = compute_censored_weibull_log_likelihood(
        shape, scale, data, compute_gradients=True
    )
    if gradients is None:
        raise RuntimeError("Gradients are required for the optimisation loop.")
    grad_shape, grad_scale = gradients
    grad_log_shape = grad_shape * shape
    grad_log_scale = grad_scale * scale
    gradient_norm = sqrt(grad_log_shape * grad_log_shape + grad_log_scale * grad_log_scale)

    iterations = 0
    last_step_size = 0.0
    current_step = base_step_size

    while iterations < max_iterations:
        iterations += 1
        if gradient_norm < tolerance:
            diagnostics = WeibullFitDiagnostics(
                iterations=iterations,
                gradient_norm=gradient_norm,
                last_step_size=last_step_size,
                message="Gradient norm below tolerance.",
            )
            return WeibullFitResult(
                shape=shape,
                scale=scale,
                log_likelihood=log_likelihood,
                success=True,
                diagnostics=diagnostics,
                used_gradients=True,
                reliable=True,
                in_count=data.in_count,
                left_count=data.left_count,
                right_count=data.right_count,
            )

        accepted = False
        step_size = current_step
        while step_size >= 1e-6:
            candidate_log_shape = log_shape + step_size * grad_log_shape
            candidate_log_scale = log_scale + step_size * grad_log_scale
            candidate_log_shape = min(max(candidate_log_shape, _LOG_SHAPE_MIN), _LOG_SHAPE_MAX)
            candidate_log_scale = min(max(candidate_log_scale, _LOG_SCALE_MIN), _LOG_SCALE_MAX)
            candidate_shape = exp(candidate_log_shape)
            candidate_scale = exp(candidate_log_scale)
            candidate_ll, _ = compute_censored_weibull_log_likelihood(
                candidate_shape, candidate_scale, data, compute_gradients=False
            )
            if candidate_ll >= log_likelihood:
                log_shape = candidate_log_shape
                log_scale = candidate_log_scale
                shape = candidate_shape
                scale = candidate_scale
                log_likelihood = candidate_ll
                last_step_size = step_size
                accepted = True
                break
            step_size *= 0.5

        if not accepted:
            current_step *= 0.5
            if current_step < 1e-6:
                diagnostics = WeibullFitDiagnostics(
                    iterations=iterations,
                    gradient_norm=gradient_norm,
                    last_step_size=last_step_size,
                    message="Line search stalled; returning last iterate.",
                )
                return WeibullFitResult(
                    shape=shape,
                    scale=scale,
                    log_likelihood=log_likelihood,
                    success=True,
                    diagnostics=diagnostics,
                    used_gradients=True,
                    reliable=True,
                    in_count=data.in_count,
                    left_count=data.left_count,
                    right_count=data.right_count,
                )
            continue

        log_likelihood, gradients = compute_censored_weibull_log_likelihood(
            shape, scale, data, compute_gradients=True
        )
        if gradients is None:
            raise RuntimeError("Gradients unexpectedly missing mid-optimisation.")
        grad_shape, grad_scale = gradients
        grad_log_shape = grad_shape * shape
        grad_log_scale = grad_scale * scale
        gradient_norm = sqrt(grad_log_shape * grad_log_shape + grad_log_scale * grad_log_scale)

    diagnostics = WeibullFitDiagnostics(
        iterations=iterations,
        gradient_norm=gradient_norm,
        last_step_size=last_step_size,
        message="Maximum iterations reached without convergence.",
    )
    return WeibullFitResult(
        shape=shape,
        scale=scale,
        log_likelihood=log_likelihood,
        success=False,
        diagnostics=diagnostics,
        used_gradients=True,
        reliable=False,
        in_count=data.in_count,
        left_count=data.left_count,
        right_count=data.right_count,
    )


def _ensure_same_length(
    first: Sequence[float],
    second: Sequence[float],
    first_name: str,
    second_name: str,
) -> None:
    if len(first) != len(second):
        raise ValueError(f"{first_name} and {second_name} must have matching lengths.")


def _sum_weights(weights: Sequence[float]) -> float:
    return sum(value for value in weights if value > 0.0)


def _dict_to_lists(mapping: Mapping[float, float]) -> tuple[list[float], list[float]]:
    values: list[float] = []
    weights: list[float] = []
    for key, weight in mapping.items():
        if weight > 0.0:
            values.append(float(key))
            weights.append(float(weight))
    return values, weights


def _to_positive_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(numeric) or numeric <= 0.0:
        return None
    return numeric


def _clamp_probability(value: object) -> float:
    numeric = _to_positive_float(value)
    if numeric is None:
        return 0.0
    return min(max(numeric, 0.0), 1.0)


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _initial_parameters(data: CensoredWeibullData) -> tuple[float, float]:
    values = data.in_values
    weights = data.in_weights
    if not values or not weights:
        # Fallback to a moderate wind regime.
        return 2.0, 9.0
    total_weight = sum(weights)
    if total_weight <= 0.0:
        return 2.0, 9.0
    mean = sum(value * weight for value, weight in zip(values, weights)) / total_weight
    variance = (
        sum(
            weight * (value - mean) * (value - mean)
            for value, weight in zip(values, weights)
        )
        / max(total_weight, 1.0)
    )
    std_dev = sqrt(max(variance, _SMALL))
    if mean <= 0.0:
        return 2.0, max(max(values, default=8.0), 1.0)
    ratio = std_dev / mean
    if ratio <= 0.0:
        shape = 3.0
    else:
        shape = ratio ** (-1.086)
    shape = min(max(shape, 0.5), 6.0)
    scale = mean / max(gamma(1.0 + 1.0 / shape), _SMALL)
    scale = max(scale, 1.0)
    return shape, scale


def _clamp(value: float, log_min: float, log_max: float, *, log_space: bool) -> float:
    if log_space:
        return min(max(log(value), log_min), log_max)
    return min(max(value, exp(log_min)), exp(log_max))
