"""Tests for wind power density and turbine output utilities."""

from __future__ import annotations

import math

import numpy as np

import pytest

from hf_wind_resource.stats import (
    KaplanMeierResult,
    PowerCurve,
    compute_kaplan_meier_power_density,
    compute_weibull_power_density,
    estimate_power_curve_from_kaplan_meier,
    estimate_power_curve_from_weibull,
)
from hf_wind_resource.stats.weibull import WeibullFitDiagnostics, WeibullFitResult


def _make_weibull_result(shape: float, scale: float) -> WeibullFitResult:
    diagnostics = WeibullFitDiagnostics(iterations=10, gradient_norm=0.0, last_step_size=0.0, message="ok")
    return WeibullFitResult(
        shape=shape,
        scale=scale,
        log_likelihood=0.0,
        success=True,
        diagnostics=diagnostics,
        used_gradients=True,
        reliable=True,
        in_count=2000.0,
        left_count=500.0,
        right_count=150.0,
    )


def test_weibull_power_density_matches_analytic() -> None:
    result = _make_weibull_result(shape=2.0, scale=10.0)
    estimate = compute_weibull_power_density(result, air_density=1.225)
    expected = 0.5 * 1.225 * (10.0**3) * math.gamma(1.0 + 3.0 / 2.0)
    assert estimate.estimate_w_per_m2 == pytest.approx(expected, rel=1e-6)


def test_power_curve_evaluate_scales_by_density() -> None:
    curve = PowerCurve(
        name="Synthetic 2MW",
        speeds=(0.0, 5.0, 10.0, 15.0, 20.0, 25.0),
        power_kw=(0.0, 200.0, 900.0, 1900.0, 2000.0, 0.0),
        reference_air_density=1.225,
        hub_height_m=100.0,
    )

    sample_speeds = [2.5, 12.0, 18.0]
    base = curve.evaluate_kw(sample_speeds)
    expected_base = np.interp(sample_speeds, curve.speeds, curve.power_kw, left=0.0, right=curve.power_kw[-1])
    assert np.allclose(base, expected_base)

    scaled = curve.evaluate_kw(sample_speeds, air_density=1.15)
    ratio = 1.15 / curve.reference_air_density
    assert np.allclose(scaled, expected_base * ratio)


def test_weibull_power_curve_expectation_matches_density_based_reference() -> None:
    air_density = 1.225
    area = 120.0  # m²
    cp = 0.42

    speeds = tuple(float(v) for v in range(0, 31))
    power_kw = tuple(
        0.5 * air_density * area * cp * (v**3) / 1000.0
        for v in speeds
    )
    curve = PowerCurve(
        name="Idealised Betz-limited",
        speeds=speeds,
        power_kw=power_kw,
        reference_air_density=air_density,
        hub_height_m=110.0,
        notes=("Ideal cubic scaling using constant Cp.",),
    )

    result = _make_weibull_result(shape=2.1, scale=9.5)
    estimate = estimate_power_curve_from_weibull(result, curve, air_density=air_density, integration_points=5000)

    expected_power_density = 0.5 * air_density * (result.scale**3) * math.gamma(1.0 + 3.0 / result.shape)
    expected_mean_kw = (cp * area * expected_power_density) / 1000.0

    assert estimate.mean_power_kw == pytest.approx(expected_mean_kw, rel=0.02)
    assert estimate.capacity_factor == pytest.approx(estimate.mean_power_kw / curve.rated_power_kw, rel=1e-9)


def test_kaplan_meier_power_density_handles_tail_surrogate() -> None:
    result = KaplanMeierResult(
        support=(5.7, 8.0, 10.0),
        cdf=(0.2, 0.7, 0.9),
        survival=(0.8, 0.3, 0.1),
        total_weight=1000.0,
        left_censored_weight=200.0,
        right_censored_weight=100.0,
    )

    air_density = 1.225
    surrogate = 17.8
    estimate = compute_kaplan_meier_power_density(result, air_density=air_density, right_tail_surrogate=surrogate)

    probabilities = np.array([0.2, 0.5, 0.2])
    moment = np.sum(probabilities * np.array([5.7, 8.0, 10.0])**3)
    tail_moment = 0.1 * surrogate**3
    expected = 0.5 * air_density * (moment + tail_moment)

    assert estimate.estimate_w_per_m2 == pytest.approx(expected, rel=1e-6)
    assert any("Right-tail probability" in note for note in estimate.notes)


def test_kaplan_meier_power_density_without_surrogate_returns_lower_bound() -> None:
    result = KaplanMeierResult(
        support=(5.7, 8.0, 10.0),
        cdf=(0.2, 0.7, 0.9),
        survival=(0.8, 0.3, 0.1),
        total_weight=1000.0,
        left_censored_weight=200.0,
        right_censored_weight=100.0,
    )

    estimate = compute_kaplan_meier_power_density(result, air_density=1.225, right_tail_surrogate=None)

    probabilities = np.array([0.2, 0.5, 0.2])
    moment = np.sum(probabilities * np.array([5.7, 8.0, 10.0])**3)
    expected = 0.5 * 1.225 * moment

    assert estimate.estimate_w_per_m2 == pytest.approx(expected, rel=1e-6)
    assert any("lower-bound" in note for note in estimate.notes)


def test_power_curve_from_kaplan_meier_includes_tail_surrogate() -> None:
    result = KaplanMeierResult(
        support=(5.7, 8.0, 10.0),
        cdf=(0.2, 0.7, 0.9),
        survival=(0.8, 0.3, 0.1),
        total_weight=1000.0,
        left_censored_weight=200.0,
        right_censored_weight=100.0,
    )

    curve = PowerCurve(
        name="Triangular 2MW",
        speeds=(0.0, 5.0, 10.0, 15.0, 20.0, 25.0),
        power_kw=(0.0, 200.0, 900.0, 1900.0, 2000.0, 0.0),
        reference_air_density=1.225,
    )

    surrogate = 17.8
    estimate = estimate_power_curve_from_kaplan_meier(result, curve, air_density=1.225, right_tail_surrogate=surrogate)

    probs = np.array([0.2, 0.5, 0.2])
    support = np.array([5.7, 8.0, 10.0])
    power_vals = curve.evaluate_kw(support)
    tail_power = curve.evaluate_kw([surrogate])[0]
    expected_kw = float(np.sum(probs * power_vals) + 0.1 * tail_power)

    assert estimate.mean_power_kw == pytest.approx(expected_kw, rel=1e-6)
    assert any("Right-tail probability" in note for note in estimate.notes)


def test_weibull_height_scaling_modifies_power_density() -> None:
    result = _make_weibull_result(shape=2.5, scale=9.0)
    base = compute_weibull_power_density(result, air_density=1.225)
    scaled = compute_weibull_power_density(result, air_density=1.225, speed_scale=1.15)

    factor = 1.15 ** 3
    assert scaled.estimate_w_per_m2 == pytest.approx(base.estimate_w_per_m2 * factor, rel=1e-6)
    assert any("scaled" in note for note in scaled.notes)


def test_kaplan_meier_height_scaling_modifies_power_density() -> None:
    result = KaplanMeierResult(
        support=(5.7, 8.0, 10.0),
        cdf=(0.2, 0.7, 0.9),
        survival=(0.8, 0.3, 0.1),
        total_weight=1000.0,
        left_censored_weight=200.0,
        right_censored_weight=100.0,
    )

    scaled = compute_kaplan_meier_power_density(
        result,
        air_density=1.225,
        right_tail_surrogate=17.8,
        speed_scale=1.1,
    )
    base = compute_kaplan_meier_power_density(
        result,
        air_density=1.225,
        right_tail_surrogate=17.8,
    )

    assert scaled.estimate_w_per_m2 == pytest.approx(base.estimate_w_per_m2 * 1.1 ** 3, rel=1e-6)
    assert any("scaled" in note for note in scaled.notes)


def test_power_curve_height_scaling_propagates_to_expectation() -> None:
    result = _make_weibull_result(shape=2.2, scale=8.5)
    curve = PowerCurve(
        name="Reference",
        speeds=(0.0, 5.0, 10.0, 15.0, 20.0, 25.0),
        power_kw=(0.0, 300.0, 1200.0, 2100.0, 2400.0, 0.0),
        reference_air_density=1.225,
        hub_height_m=100.0,
    )

    base = estimate_power_curve_from_weibull(result, curve, air_density=1.225)
    scaled = estimate_power_curve_from_weibull(
        result,
        curve,
        air_density=1.225,
        speed_scale=1.1,
    )

    assert scaled.mean_power_kw > base.mean_power_kw
    assert scaled.capacity_factor is not None and base.capacity_factor is not None
    assert scaled.capacity_factor > base.capacity_factor
    assert any("scaled" in note for note in scaled.notes)


def test_kaplan_meier_power_curve_height_scaling() -> None:
    result = KaplanMeierResult(
        support=(5.7, 8.0, 10.0),
        cdf=(0.2, 0.7, 0.9),
        survival=(0.8, 0.3, 0.1),
        total_weight=1000.0,
        left_censored_weight=200.0,
        right_censored_weight=100.0,
    )
    curve = PowerCurve(
        name="Reference",
        speeds=(0.0, 5.0, 10.0, 15.0, 20.0, 25.0),
        power_kw=(0.0, 200.0, 900.0, 1900.0, 2000.0, 0.0),
        reference_air_density=1.225,
    )

    base = estimate_power_curve_from_kaplan_meier(result, curve, air_density=1.225, right_tail_surrogate=17.8)
    scaled = estimate_power_curve_from_kaplan_meier(
        result,
        curve,
        air_density=1.225,
        right_tail_surrogate=17.8,
        speed_scale=1.05,
    )

    assert scaled.mean_power_kw > base.mean_power_kw
    assert any("scaled" in note for note in scaled.notes)
