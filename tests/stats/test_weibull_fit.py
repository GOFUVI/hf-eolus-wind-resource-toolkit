"""Tests for the censored Weibull fitting utilities."""

from __future__ import annotations

import random

import pytest

from hf_wind_resource.stats import (
    CensoredWeibullData,
    build_censored_data_from_records,
    compute_censored_weibull_log_likelihood,
    fit_censored_weibull,
)


@pytest.fixture
def synthetic_censored_data() -> CensoredWeibullData:
    """Return synthetic samples generated from a known Weibull model."""

    random.seed(20251021)
    shape = 2.4
    scale = 11.5
    lower_threshold = 5.7
    upper_threshold = 17.8

    in_values: list[float] = []
    in_weights: list[float] = []
    left_count = 0.0
    right_count = 0.0

    for _ in range(1600):
        sample = random.weibullvariate(scale, shape)
        if sample < lower_threshold:
            left_count += 1.0
        elif sample > upper_threshold:
            right_count += 1.0
        else:
            in_values.append(sample)
            in_weights.append(1.0)

    left_limits = (lower_threshold,) if left_count > 0 else ()
    left_weights = (left_count,) if left_count > 0 else ()
    right_limits = (upper_threshold,) if right_count > 0 else ()
    right_weights = (right_count,) if right_count > 0 else ()

    return CensoredWeibullData(
        in_values=tuple(in_values),
        in_weights=tuple(in_weights),
        left_limits=left_limits,
        left_weights=left_weights,
        right_limits=right_limits,
        right_weights=right_weights,
    )


def test_log_likelihood_prefers_true_parameters(synthetic_censored_data: CensoredWeibullData) -> None:
    true_shape = 2.4
    true_scale = 11.5

    ll_true, _ = compute_censored_weibull_log_likelihood(
        true_shape, true_scale, synthetic_censored_data, compute_gradients=False
    )
    ll_shifted, _ = compute_censored_weibull_log_likelihood(
        true_shape * 0.7, true_scale * 1.3, synthetic_censored_data, compute_gradients=False
    )

    assert ll_true > ll_shifted


def test_fit_recovers_parameters_with_censoring(synthetic_censored_data: CensoredWeibullData) -> None:
    result = fit_censored_weibull(
        synthetic_censored_data,
        min_in_count=200.0,
        max_iterations=300,
        tolerance=1e-5,
    )

    assert result.success is True
    assert result.reliable is True
    assert result.shape == pytest.approx(2.4, rel=0.15)
    assert result.scale == pytest.approx(11.5, rel=0.10)
    assert result.log_likelihood is not None


def test_fit_enforces_minimum_in_range_observations() -> None:
    data = CensoredWeibullData(
        in_values=(10.0,) * 10,
        in_weights=(1.0,) * 10,
        left_limits=(5.7,),
        left_weights=(120.0,),
        right_limits=(17.8,),
        right_weights=(60.0,),
    )

    result = fit_censored_weibull(data, min_in_count=50.0)

    assert result.success is False
    assert result.reliable is False
    assert "Insufficient in-range support" in result.diagnostics.message


def test_builder_respects_range_probabilities() -> None:
    records = [
        {
            "pred_wind_speed": 9.5,
            "prob_range_below": 0.1,
            "prob_range_in": 0.8,
            "prob_range_above": 0.1,
            "range_flag": "in",
            "range_flag_confident": True,
        },
        {
            "pred_wind_speed": 18.4,
            "prob_range_below": 0.02,
            "prob_range_in": 0.03,
            "prob_range_above": 0.95,
            "range_flag": "above",
            "range_flag_confident": True,
        },
        {
            "pred_wind_speed": 6.6,
            "prob_range_below": 0.6,
            "prob_range_in": 0.3,
            "prob_range_above": 0.1,
            "range_flag": "uncertain",
            "range_flag_confident": False,
        },
        {
            "pred_wind_speed": 4.2,
            "prob_range_below": 0.92,
            "prob_range_in": 0.05,
            "prob_range_above": 0.03,
            "range_flag": "below",
            "range_flag_confident": True,
        },
    ]

    data = build_censored_data_from_records(
        records,
        lower_threshold=5.7,
        upper_threshold=17.8,
        min_confidence=0.5,
    )

    assert data.in_count == pytest.approx(1.3)
    assert data.left_count == pytest.approx(1.6)
    assert data.right_count == pytest.approx(1.1)
    assert len(data.in_values) == 2
    assert data.left_limits == (5.7,)
    assert data.right_limits == (17.8,)
