"""Parametric model comparison utilities for censored wind-speed data."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, inf, isfinite, lgamma, log, pi, sqrt
from typing import Callable, Mapping, Sequence

from .weibull import CensoredWeibullData, WeibullFitResult

try:  # pragma: no cover - optional dependency detection
    from scipy.special import gammainc as _scipy_gammainc  # type: ignore
    from scipy.special import gammaincc as _scipy_gammaincc  # type: ignore

    _SCIPY_SPECIAL_AVAILABLE = True
except Exception:  # pragma: no cover - scipy absent in certain runtimes
    _scipy_gammainc = None
    _scipy_gammaincc = None
    _SCIPY_SPECIAL_AVAILABLE = False

_SQRT_TWO = sqrt(2.0)
_LOG_TWO_PI = log(2.0 * pi)
_PROB_EPS = 1e-15


@dataclass(frozen=True)
class ParametricModelSummary:
    """Diagnostic summary for a specific parametric candidate."""

    name: str
    parameters: Mapping[str, float | None]
    log_likelihood: float | None
    aic: float | None
    bic: float | None
    ks_statistic: float | None
    ks_pvalue: float | None
    success: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ParametricComparison:
    """Container with all evaluated candidates and the preferred choice."""

    selection_metric: str
    preferred_model: str | None
    preferred_metric_value: float | None
    candidates: tuple[ParametricModelSummary, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ParametricComparisonConfig:
    """Configuration controlling how alternative parametric fits are evaluated."""

    min_in_weight: float = 200.0
    ks_min_weight: float = 75.0
    selection_metric: str = "bic"
    enable_gamma: bool = True

    def __post_init__(self) -> None:
        if self.min_in_weight < 0.0:
            raise ValueError("min_in_weight must be non-negative.")
        if self.ks_min_weight < 0.0:
            raise ValueError("ks_min_weight must be non-negative.")
        metric = self.selection_metric.lower()
        if metric not in {"aic", "bic"}:
            raise ValueError("selection_metric must be either 'aic' or 'bic'.")
        object.__setattr__(self, "selection_metric", metric)


def evaluate_parametric_models(
    data: CensoredWeibullData,
    weibull: WeibullFitResult,
    *,
    config: ParametricComparisonConfig,
) -> ParametricComparison:
    """Evaluate Weibull vs. alternative parametric models on a censored dataset."""

    candidates: list[ParametricModelSummary] = []
    notes: list[str] = []

    candidates.append(_summarise_weibull_candidate(data, weibull, config))

    if data.in_count < config.min_in_weight:
        message = (
            f"Alternative fits skipped: in-range weight {data.in_count:.1f} "
            f"< minimum {config.min_in_weight:.1f}."
        )
        notes.append(message)
        candidates.append(
            ParametricModelSummary(
                name="lognormal",
                parameters={},
                log_likelihood=None,
                aic=None,
                bic=None,
                ks_statistic=None,
                ks_pvalue=None,
                success=False,
                notes=(message,),
            )
        )
        candidates.append(_gamma_unavailable_summary("gamma", reason=message))
    else:
        candidates.append(_summarise_lognormal_candidate(data, config))
        candidates.append(_summarise_gamma_candidate(data, config))

    metric_name = config.selection_metric
    preferred_name: str | None = None
    preferred_value: float | None = None

    for candidate in candidates:
        value = candidate.aic if metric_name == "aic" else candidate.bic
        if value is None or not candidate.success or not isfinite(value):
            continue
        if preferred_value is None or value < preferred_value:
            preferred_value = value
            preferred_name = candidate.name

    return ParametricComparison(
        selection_metric=metric_name,
        preferred_model=preferred_name,
        preferred_metric_value=preferred_value,
        candidates=tuple(candidates),
        notes=tuple(notes),
    )


def _summarise_weibull_candidate(
    data: CensoredWeibullData,
    weibull: WeibullFitResult,
    config: ParametricComparisonConfig,
) -> ParametricModelSummary:
    shape = weibull.shape
    scale = weibull.scale
    log_likelihood = weibull.log_likelihood
    total_obs = max(data.total_weight, 0.0)
    aic, bic = _information_criteria(log_likelihood, total_obs, 2)

    ks_statistic: float | None = None
    ks_pvalue: float | None = None
    if shape is not None and scale is not None and data.in_count >= config.ks_min_weight:
        ks_statistic, ks_pvalue = _weighted_ks(
            data.in_values,
            data.in_weights,
            lambda value: _weibull_cdf(value, shape, scale),
        )

    return ParametricModelSummary(
        name="weibull",
        parameters={"shape": shape, "scale": scale},
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        ks_statistic=ks_statistic,
        ks_pvalue=ks_pvalue,
        success=bool(shape is not None and scale is not None and weibull.reliable),
        notes=(weibull.diagnostics.message,) if weibull.diagnostics.message else (),
    )


def _summarise_lognormal_candidate(
    data: CensoredWeibullData,
    config: ParametricComparisonConfig,
) -> ParametricModelSummary:
    positive_pairs = [
        (value, weight)
        for value, weight in zip(data.in_values, data.in_weights)
        if value > 0.0 and weight > 0.0
    ]
    if not positive_pairs:
        note = "Log-normal fit unavailable: missing positive in-range observations."
        return ParametricModelSummary(
            name="lognormal",
            parameters={},
            log_likelihood=None,
            aic=None,
            bic=None,
            ks_statistic=None,
            ks_pvalue=None,
            success=False,
            notes=(note,),
        )

    total_weight = sum(weight for _, weight in positive_pairs)
    log_values = [log(value) for value, _ in positive_pairs]
    mean_log = _weighted_mean(log_values, [weight for _, weight in positive_pairs])
    variance_log = _weighted_variance(log_values, [weight for _, weight in positive_pairs], mean_log)

    if variance_log is None or variance_log <= 0.0:
        note = "Log-normal fit unavailable: zero variance in logarithmic domain."
        return ParametricModelSummary(
            name="lognormal",
            parameters={"mu": mean_log, "sigma": None},
            log_likelihood=None,
            aic=None,
            bic=None,
            ks_statistic=None,
            ks_pvalue=None,
            success=False,
            notes=(note,),
        )

    sigma = sqrt(max(variance_log, 1e-9))
    mu = mean_log

    log_pdf = lambda value: _log_pdf_lognormal(value, mu, sigma)
    cdf = lambda value: _cdf_lognormal(value, mu, sigma)
    log_likelihood = _censored_log_likelihood(data, log_pdf, cdf)
    aic, bic = _information_criteria(log_likelihood, data.total_weight, 2)

    ks_statistic: float | None = None
    ks_pvalue: float | None = None
    if data.in_count >= config.ks_min_weight:
        ks_statistic, ks_pvalue = _weighted_ks(
            data.in_values,
            data.in_weights,
            cdf,
        )

    return ParametricModelSummary(
        name="lognormal",
        parameters={"mu": mu, "sigma": sigma},
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        ks_statistic=ks_statistic,
        ks_pvalue=ks_pvalue,
        success=log_likelihood is not None,
        notes=("Parameters estimated via weighted log-moment matching.",),
    )


def _summarise_gamma_candidate(
    data: CensoredWeibullData,
    config: ParametricComparisonConfig,
) -> ParametricModelSummary:
    if not config.enable_gamma:
        return _gamma_unavailable_summary("gamma", reason="Gamma candidate disabled via configuration.")

    if not _SCIPY_SPECIAL_AVAILABLE:
        return _gamma_unavailable_summary(
            "gamma", reason="SciPy is required to evaluate the gamma CDF for censored likelihoods."
        )

    positive_pairs = [
        (value, weight)
        for value, weight in zip(data.in_values, data.in_weights)
        if value > 0.0 and weight > 0.0
    ]
    if not positive_pairs:
        note = "Gamma fit unavailable: missing positive in-range observations."
        return ParametricModelSummary(
            name="gamma",
            parameters={},
            log_likelihood=None,
            aic=None,
            bic=None,
            ks_statistic=None,
            ks_pvalue=None,
            success=False,
            notes=(note,),
        )

    weights = [weight for _, weight in positive_pairs]
    values = [value for value, _ in positive_pairs]
    mean_value = _weighted_mean(values, weights)
    variance_value = _weighted_variance(values, weights, mean_value)

    if mean_value is None or mean_value <= 0.0 or variance_value is None or variance_value <= 0.0:
        note = "Gamma fit unavailable: invalid weighted moments."
        return ParametricModelSummary(
            name="gamma",
            parameters={},
            log_likelihood=None,
            aic=None,
            bic=None,
            ks_statistic=None,
            ks_pvalue=None,
            success=False,
            notes=(note,),
        )

    shape = max((mean_value * mean_value) / variance_value, 1e-6)
    scale = variance_value / mean_value

    log_pdf = lambda value: _log_pdf_gamma(value, shape, scale)
    cdf = lambda value: _cdf_gamma(value, shape, scale)
    log_likelihood = _censored_log_likelihood(data, log_pdf, cdf)
    aic, bic = _information_criteria(log_likelihood, data.total_weight, 2)

    ks_statistic: float | None = None
    ks_pvalue: float | None = None
    if data.in_count >= config.ks_min_weight:
        ks_statistic, ks_pvalue = _weighted_ks(data.in_values, data.in_weights, cdf)

    return ParametricModelSummary(
        name="gamma",
        parameters={"shape": shape, "scale": scale},
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        ks_statistic=ks_statistic,
        ks_pvalue=ks_pvalue,
        success=log_likelihood is not None,
        notes=("Parameters estimated via weighted moment matching.",),
    )


def _gamma_unavailable_summary(name: str, *, reason: str) -> ParametricModelSummary:
    """Return a placeholder summary when the gamma candidate cannot be evaluated."""

    return ParametricModelSummary(
        name=name,
        parameters={},
        log_likelihood=None,
        aic=None,
        bic=None,
        ks_statistic=None,
        ks_pvalue=None,
        success=False,
        notes=(reason,),
    )


def _censored_log_likelihood(
    data: CensoredWeibullData,
    log_pdf: Callable[[float], float | None],
    cdf: Callable[[float], float | None],
) -> float | None:
    total = 0.0

    for value, weight in zip(data.in_values, data.in_weights):
        if weight <= 0.0:
            continue
        log_density = log_pdf(value)
        if log_density is None or not isfinite(log_density):
            return None
        total += weight * log_density

    for limit, weight in zip(data.left_limits, data.left_weights):
        if weight <= 0.0:
            continue
        prob = cdf(limit)
        if prob is None:
            return None
        prob = _clamp_probability(prob)
        if prob <= 0.0:
            return None
        total += weight * log(prob)

    for limit, weight in zip(data.right_limits, data.right_weights):
        if weight <= 0.0:
            continue
        prob = cdf(limit)
        if prob is None:
            return None
        survival = _clamp_probability(1.0 - prob)
        if survival <= 0.0:
            return None
        total += weight * log(survival)

    return total


def _information_criteria(
    log_likelihood: float | None,
    total_weight: float,
    num_parameters: int,
) -> tuple[float | None, float | None]:
    if log_likelihood is None or total_weight <= 0.0 or num_parameters <= 0:
        return None, None
    aic = 2.0 * num_parameters - 2.0 * log_likelihood
    bic = log(total_weight) * num_parameters - 2.0 * log_likelihood
    return aic, bic


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for value, weight in zip(values, weights):
        if weight <= 0.0:
            continue
        numerator += value * weight
        denominator += weight
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _weighted_variance(
    values: Sequence[float],
    weights: Sequence[float],
    mean_value: float | None,
) -> float | None:
    if mean_value is None:
        return None
    numerator = 0.0
    denominator = 0.0
    for value, weight in zip(values, weights):
        if weight <= 0.0:
            continue
        diff = value - mean_value
        numerator += weight * diff * diff
        denominator += weight
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _log_pdf_lognormal(value: float, mu: float, sigma: float) -> float | None:
    if value <= 0.0 or sigma <= 0.0:
        return None
    log_value = log(value)
    exponent = (log_value - mu) / sigma
    return -log_value - log(sigma) - 0.5 * _LOG_TWO_PI - 0.5 * exponent * exponent


def _cdf_lognormal(value: float, mu: float, sigma: float) -> float:
    if value <= 0.0:
        return 0.0
    exponent = (log(value) - mu) / (sigma * _SQRT_TWO)
    return 0.5 * (1.0 + erf(exponent))


def _log_pdf_gamma(value: float, shape: float, scale: float) -> float | None:
    if value <= 0.0 or shape <= 0.0 or scale <= 0.0:
        return None
    return ((shape - 1.0) * log(value)) - (value / scale) - shape * log(scale) - lgamma(shape)


def _cdf_gamma(value: float, shape: float, scale: float) -> float:
    if value <= 0.0:
        return 0.0
    if not _SCIPY_SPECIAL_AVAILABLE or _scipy_gammainc is None:
        raise RuntimeError("Gamma CDF requested but SciPy is unavailable.")
    argument = max(value / scale, 0.0)
    return float(_scipy_gammainc(shape, argument))


def _weibull_cdf(value: float, shape: float, scale: float) -> float:
    if value <= 0.0:
        return 0.0
    ratio = value / scale
    exponent = -pow(max(ratio, 0.0), shape)
    return 1.0 - exp(exponent)


def _weighted_ks(
    values: Sequence[float],
    weights: Sequence[float],
    cdf: Callable[[float], float],
) -> tuple[float | None, float | None]:
    pairs = [
        (value, weight)
        for value, weight in sorted(zip(values, weights), key=lambda item: item[0])
        if weight > 0.0
    ]
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0.0:
        return None, None

    cumulative = 0.0
    max_diff = 0.0
    for value, weight in pairs:
        cumulative += weight
        empirical = cumulative / total_weight
        model = _clamp_probability(cdf(value))
        diff = abs(empirical - model)
        if diff > max_diff:
            max_diff = diff

    pvalue = _kolmogorov_pvalue(max_diff, total_weight)
    return max_diff, pvalue


def _kolmogorov_pvalue(statistic: float | None, effective_weight: float) -> float | None:
    if statistic is None or effective_weight <= 0.0:
        return None
    if statistic <= 0.0:
        return 1.0

    sqrt_weight = sqrt(effective_weight)
    lam = (sqrt_weight + 0.12 + 0.11 / sqrt_weight) * statistic
    series_sum = 0.0
    for k in range(1, 200):
        term = (-1) ** (k - 1) * exp(-2.0 * (lam * lam) * (k * k))
        series_sum += term
        if abs(term) < 1e-8:
            break
    pvalue = 2.0 * series_sum
    return max(0.0, min(1.0, pvalue))


def _clamp_probability(value: float) -> float:
    if value <= 0.0:
        return _PROB_EPS
    if value >= 1.0:
        return 1.0 - _PROB_EPS
    if not isfinite(value):
        return _PROB_EPS
    return value


__all__ = [
    "ParametricComparison",
    "ParametricComparisonConfig",
    "ParametricModelSummary",
    "evaluate_parametric_models",
]
