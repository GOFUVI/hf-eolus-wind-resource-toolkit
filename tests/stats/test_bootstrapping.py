"""Tests for the stratified bootstrap uncertainty module."""

from __future__ import annotations

import math
import random

import numpy as np

import pytest

from hf_wind_resource.stats import bootstrapping as boot_module

from hf_wind_resource.stats import (
    HeightCorrection,
    NodeBootstrapInput,
    PowerCurve,
    StratifiedBootstrapConfig,
    compute_stratified_bootstrap_uncertainty,
    load_kaplan_meier_selection_criteria,
)


def _make_config(**overrides) -> StratifiedBootstrapConfig:
    power_curve = PowerCurve(
        name="test_curve",
        speeds=(0.0, 5.0, 10.0, 15.0, 25.0),
        power_kw=(0.0, 200.0, 800.0, 2000.0, 2000.0),
        reference_air_density=1.225,
        notes=("synthetic",),
    )
    km_criteria = load_kaplan_meier_selection_criteria(None)
    return StratifiedBootstrapConfig(
        power_curve=power_curve,
        km_criteria=km_criteria,
        lower_threshold=3.0,
        upper_threshold=25.0,
        min_in_range_weight=0.0,
        **overrides,
    )


def _make_height() -> HeightCorrection:
    return HeightCorrection(method="none", source_height_m=10.0, target_height_m=10.0, speed_scale=1.0)


def test_bootstrap_generates_confidence_intervals_without_noise() -> None:
    records = [
        {
            "pred_wind_speed": float(value),
            "prob_range_below": 0.05,
            "prob_range_in": 0.9,
            "prob_range_above": 0.05,
            "range_flag": "in",
            "range_flag_confident": True,
        }
        for value in (8.0, 9.0, 10.0, 11.0, 12.0)
    ] + [
        {
            "pred_wind_speed": 2.5,
            "prob_range_below": 0.9,
            "prob_range_in": 0.05,
            "prob_range_above": 0.05,
            "range_flag": "below",
            "range_flag_confident": True,
        },
        {
            "pred_wind_speed": 27.0,
            "prob_range_below": 0.05,
            "prob_range_in": 0.05,
            "prob_range_above": 0.9,
            "range_flag": "above",
            "range_flag_confident": True,
        },
    ]

    data = NodeBootstrapInput(node_id="TEST_NODE", records=records, height=_make_height())

    config = _make_config(replicas=128, random_seed=20251022, apply_rmse_noise=False)

    result = compute_stratified_bootstrap_uncertainty(data, config=config)

    mean_interval = result.metrics["mean_speed"]
    assert math.isclose(mean_interval.estimate or 0.0, 10.0, rel_tol=1e-2)
    assert mean_interval.lower is not None and mean_interval.upper is not None
    assert mean_interval.lower <= mean_interval.estimate <= mean_interval.upper
    assert mean_interval.replicates == config.replicas

    assert sum(result.label_counts.values()) == len(records)
    assert sum(result.label_proportions.values()) == pytest.approx(1.0, rel=1e-12)
    assert result.power_diagnostics is not None
    assert isinstance(result.power_diagnostics.replicate_method_counts, dict)


def test_bootstrap_without_samples_reports_notes() -> None:
    data = NodeBootstrapInput(node_id="EMPTY", records=[], height=_make_height())
    config = _make_config(replicas=0)
    result = compute_stratified_bootstrap_uncertainty(data, config=config)

    assert "No samples available" in " ".join(result.notes)
    for interval in result.metrics.values():
        assert interval.estimate is None
        assert interval.lower is None
        assert interval.upper is None
        assert interval.replicates == 0


def test_min_confidence_influences_primary_metrics() -> None:
    random.seed(20251022)
    records = []
    for speed in (7.5, 8.0, 10.5, 11.0, 12.5):
        records.append(
            {
                "pred_wind_speed": float(speed),
                "prob_range_below": 0.2,
                "prob_range_in": 0.6,
                "prob_range_above": 0.2,
                "range_flag": "in",
                "range_flag_confident": False,
            }
        )
    for _ in range(6):
        records.append(
            {
                "pred_wind_speed": 2.0,
                "prob_range_below": 0.75,
                "prob_range_in": 0.15,
                "prob_range_above": 0.10,
                "range_flag": "below",
                "range_flag_confident": False,
            }
        )

    data = NodeBootstrapInput(node_id="MIXED", records=records, height=_make_height())

    strict = compute_stratified_bootstrap_uncertainty(
        data,
        config=_make_config(replicas=64, random_seed=1, apply_rmse_noise=False, min_confidence=0.8),
    )
    relaxed = compute_stratified_bootstrap_uncertainty(
        data,
        config=_make_config(replicas=64, random_seed=1, apply_rmse_noise=False, min_confidence=0.1),
    )

    assert strict.label_counts != relaxed.label_counts
    assert strict.label_proportions["uncertain"] > relaxed.label_proportions["uncertain"]


def test_prepare_sample_clamps_to_thresholds() -> None:
    rng = np.random.default_rng(123)
    records = [
        {
            "pred_wind_speed": 8.0,
            "prob_range_below": 0.1,
            "prob_range_in": 0.8,
            "prob_range_above": 0.1,
            "range_flag": "in",
            "range_flag_confident": True,
        }
        for _ in range(20)
    ]

    sample = boot_module._apply_noise_to_records(  # type: ignore[attr-defined]
        records,
        rng=rng,
        rmse_value=5.0,
        apply_noise=True,
        truncation_multiplier=4.0,
        lower_threshold=3.0,
        upper_threshold=12.0,
    )

    speeds = [entry["pred_wind_speed"] for entry in sample]
    assert all(3.0 <= speed <= 12.0 for speed in speeds)


def test_label_resample_strategy_adds_note_and_confident_flags() -> None:
    records = [
        {
            "pred_wind_speed": 9.5,
            "prob_range_below": 0.15,
            "prob_range_in": 0.7,
            "prob_range_above": 0.15,
            "range_flag": "in",
            "range_flag_confident": False,
        }
        for _ in range(12)
    ]
    data = NodeBootstrapInput(node_id="RESAMPLE", records=records, height=_make_height())

    config = _make_config(
        replicas=64,
        random_seed=2025,
        apply_rmse_noise=False,
        label_strategy="label_resample",
        min_confidence=0.1,
    )

    result = compute_stratified_bootstrap_uncertainty(data, config=config)

    assert any("Label resampling" in note for note in result.notes)

    rng = np.random.default_rng(321)
    sample = boot_module._draw_label_resample_sample(  # type: ignore[attr-defined]
        records,
        rng,
        min_confidence=0.1,
    )
    assert len(sample) == len(records)
    confident = [entry for entry in sample if entry.get("range_flag_confident")]
    assert confident, "Expected at least one imputed confident label"
    for entry in confident:
        label = str(entry.get("range_flag"))
        assert label in {"below", "in", "above"}
        assert entry[f"prob_range_{label}"] == pytest.approx(1.0)


def test_power_mode_noise_is_unbiased() -> None:
    records = [
        {
            "pred_wind_speed": 9.0,
            "prob_range_below": 0.1,
            "prob_range_in": 0.8,
            "prob_range_above": 0.1,
            "range_flag": "in",
            "range_flag_confident": True,
        }
        for _ in range(30)
    ]
    data = NodeBootstrapInput(node_id="POWER", records=records, height=_make_height())

    config = _make_config(
        replicas=300,
        random_seed=2024,
        apply_rmse_noise=True,
        rmse_mode="power",
    )

    result = compute_stratified_bootstrap_uncertainty(data, config=config)
    interval = result.metrics["power_density"]
    assert interval.lower is not None and interval.upper is not None
    baseline = interval.estimate
    assert baseline is not None
    bootstrap_mean = result.bootstrap_means["power_density"]
    assert bootstrap_mean is not None
    assert abs(bootstrap_mean - baseline) < 50.0


def test_invalid_label_strategy_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _make_config(label_strategy="invalid")


def test_moving_block_bootstrap_adds_note() -> None:
    records = [
        {
            "timestamp": f"2025-01-01T00:{idx:02d}:00Z",
            "pred_wind_speed": 8.0 + idx,
            "prob_range_below": 0.1,
            "prob_range_in": 0.8,
            "prob_range_above": 0.1,
            "range_flag": "in",
            "range_flag_confident": True,
        }
        for idx in range(6)
    ]
    data = NodeBootstrapInput(node_id="BLOCK", records=records, height=_make_height())

    config = _make_config(
        replicas=10,
        random_seed=123,
        apply_rmse_noise=False,
        resampling_mode="moving_block",
        block_length=2,
        node_block_lengths={"BLOCK": 5},
    )

    result = compute_stratified_bootstrap_uncertainty(data, config=config)

    assert any("Moving-block bootstrap" in note for note in result.notes)


def test_stationary_bootstrap_adds_note() -> None:
    records = [
        {
            "timestamp": f"2025-01-02T00:{idx:02d}:00Z",
            "pred_wind_speed": 6.0,
            "prob_range_below": 0.2,
            "prob_range_in": 0.7,
            "prob_range_above": 0.1,
            "range_flag": "in",
            "range_flag_confident": True,
        }
        for idx in range(4)
    ]
    data = NodeBootstrapInput(node_id="STAT", records=records, height=_make_height())

    config = _make_config(
        replicas=8,
        random_seed=321,
        apply_rmse_noise=False,
        resampling_mode="stationary",
        block_length=3,
    )

    result = compute_stratified_bootstrap_uncertainty(data, config=config)

    assert any("Stationary bootstrap" in note for note in result.notes)


def test_bca_fallback_when_jackknife_unavailable() -> None:
    records = [
        {
            "pred_wind_speed": 7.5,
            "prob_range_below": 0.2,
            "prob_range_in": 0.7,
            "prob_range_above": 0.1,
            "range_flag": "in",
            "range_flag_confident": False,
        }
        for _ in range(10)
    ]
    data = NodeBootstrapInput(node_id="BCA", records=records, height=_make_height())

    config = _make_config(
        replicas=50,
        random_seed=123,
        ci_method="bca",
        jackknife_max_samples=5,
    )

    result = compute_stratified_bootstrap_uncertainty(data, config=config)
    assert any("falling back" in note.lower() for note in result.notes)


def test_percentile_t_fallback_when_jackknife_zero_variance() -> None:
    records = [
        {
            "pred_wind_speed": 8.0,
            "prob_range_below": 0.1,
            "prob_range_in": 0.8,
            "prob_range_above": 0.1,
            "range_flag": "in",
            "range_flag_confident": True,
        }
        for _ in range(3)
    ]
    data = NodeBootstrapInput(node_id="T", records=records, height=_make_height())

    config = _make_config(
        replicas=20,
        random_seed=99,
        ci_method="percentile_t",
        jackknife_max_samples=3,
    )

    result = compute_stratified_bootstrap_uncertainty(data, config=config)
    assert any("percentile-t" in note.lower() for note in result.notes)
