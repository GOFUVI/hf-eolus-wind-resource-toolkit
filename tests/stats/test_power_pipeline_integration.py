"""Integration-style tests for the statistical power pipeline.

These tests exercise the end-to-end flow using deterministic synthetic
datasets that mirror the ANN-derived records leveraged by the project.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hf_wind_resource.stats import (
    HeightCorrection,
    KaplanMeierSelectionCriteria,
    NodeBootstrapInput,
    PowerCurve,
    StratifiedBootstrapConfig,
    build_censored_data_from_records,
    compute_power_distribution,
    compute_stratified_bootstrap_uncertainty,
    summarise_records_for_selection,
)


_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "stats_synthetic_datasets.json"


@pytest.fixture(scope="module")
def synthetic_stats_dataset() -> dict[str, dict[str, object]]:
    """Return the synthetic datasets used across integration tests."""

    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    return payload


@pytest.fixture
def reference_power_curve() -> PowerCurve:
    """Return a lightweight power curve for deterministic expectations."""

    return PowerCurve(
        name="Synthetic 2MW",
        speeds=(0.0, 5.0, 10.0, 15.0, 25.0),
        power_kw=(0.0, 200.0, 900.0, 2000.0, 2000.0),
        reference_air_density=1.225,
        notes=("deterministic-fixture",),
    )


@pytest.fixture
def neutral_height() -> HeightCorrection:
    """Return a neutral height correction (identity scaling)."""

    return HeightCorrection(method="none", source_height_m=10.0, target_height_m=10.0, speed_scale=1.0)


@pytest.fixture
def relaxed_km_criteria() -> KaplanMeierSelectionCriteria:
    """Return permissive criteria to keep the pipeline in Weibull mode unless it fails."""

    return KaplanMeierSelectionCriteria(
        min_total_observations=1_000_000,
        min_total_censored_ratio=1.0,
        min_below_ratio=1.0,
        max_valid_share=0.0,
        min_uncensored_weight=1_000_000.0,
    )


def test_power_distribution_recovers_weibull_parameters(
    synthetic_stats_dataset: dict[str, dict[str, object]],
    reference_power_curve: PowerCurve,
    neutral_height: HeightCorrection,
    relaxed_km_criteria: KaplanMeierSelectionCriteria,
) -> None:
    payload = synthetic_stats_dataset["weibull_reference"]
    records = tuple(payload["records"])
    lower = float(payload["lower_threshold"])
    upper = float(payload["upper_threshold"])
    min_confidence = float(payload["min_confidence"])

    data = build_censored_data_from_records(
        records,
        lower_threshold=lower,
        upper_threshold=upper,
        min_confidence=min_confidence,
    )
    assert data.in_count > 200.0  # Ensure the sample has enough support.

    summary = summarise_records_for_selection(records, min_confidence=min_confidence)
    tail_surrogate = float(payload["upper_threshold"])

    method, power_density, power_curve_estimate, weibull, km_result, selection_reasons, method_notes = (
        compute_power_distribution(
            data=data,
            summary_row=summary,
            power_curve=reference_power_curve,
            air_density=1.225,
            tail_surrogate=tail_surrogate,
            min_in_range=150.0,
            km_criteria=relaxed_km_criteria,
            height=neutral_height,
        )
    )

    assert method == "weibull"
    assert not selection_reasons
    assert method_notes == []
    assert km_result is None
    assert weibull.success is True and weibull.reliable is True
    assert weibull.shape == pytest.approx(float(payload["shape"]), rel=0.20)
    assert weibull.scale == pytest.approx(float(payload["scale"]), rel=0.10)
    assert power_density.estimate_w_per_m2 is not None
    assert power_curve_estimate.mean_power_kw is not None


def test_power_distribution_switches_to_kaplan_meier_when_in_range_missing(
    synthetic_stats_dataset: dict[str, dict[str, object]],
    reference_power_curve: PowerCurve,
    neutral_height: HeightCorrection,
    relaxed_km_criteria: KaplanMeierSelectionCriteria,
) -> None:
    payload = synthetic_stats_dataset["fully_censored"]
    records = tuple(payload["records"])
    lower = float(payload["lower_threshold"])
    upper = float(payload["upper_threshold"])
    min_confidence = float(payload["min_confidence"])

    data = build_censored_data_from_records(
        records,
        lower_threshold=lower,
        upper_threshold=upper,
        min_confidence=min_confidence,
    )
    assert data.in_count == pytest.approx(0.0)

    summary = summarise_records_for_selection(records, min_confidence=min_confidence)
    method, power_density, power_curve_estimate, weibull, km_result, selection_reasons, method_notes = (
        compute_power_distribution(
            data=data,
            summary_row=summary,
            power_curve=reference_power_curve,
            air_density=1.225,
            tail_surrogate=upper,
            min_in_range=50.0,
            km_criteria=relaxed_km_criteria,
            height=neutral_height,
        )
    )

    assert method == "kaplan_meier"
    assert km_result is not None
    assert km_result.total_weight == pytest.approx(data.total_weight)
    assert power_density.estimate_w_per_m2 is not None
    assert power_curve_estimate.mean_power_kw is not None
    # The Weibull fit is expected to fail due to missing in-range weight.
    assert weibull.success is False or weibull.reliable is False


def test_bootstrap_rmse_noise_widens_confidence_interval(
    synthetic_stats_dataset: dict[str, dict[str, object]],
    reference_power_curve: PowerCurve,
    neutral_height: HeightCorrection,
    relaxed_km_criteria: KaplanMeierSelectionCriteria,
) -> None:
    payload = synthetic_stats_dataset["rmse_stress"]
    records = tuple(payload["records"])
    lower = float(payload["lower_threshold"])
    upper = float(payload["upper_threshold"])
    min_confidence = float(payload["min_confidence"])

    base_kwargs = dict(
        replicas=160,
        random_seed=20251024,
        rmse_mode="velocity",
        power_curve=reference_power_curve,
        km_criteria=relaxed_km_criteria,
        lower_threshold=lower,
        upper_threshold=upper,
        tail_surrogate=upper,
        min_confidence=min_confidence,
        min_in_range_weight=20.0,
    )

    data = NodeBootstrapInput(
        node_id="SYNTH_NODE",
        records=records,
        height=neutral_height,
    )

    baseline_kwargs = dict(base_kwargs)
    baseline_kwargs["apply_rmse_noise"] = False
    baseline = compute_stratified_bootstrap_uncertainty(
        data,
        config=StratifiedBootstrapConfig(**baseline_kwargs),
    )
    noisy_kwargs = dict(base_kwargs)
    noisy_kwargs["apply_rmse_noise"] = True
    noisy = compute_stratified_bootstrap_uncertainty(
        data,
        config=StratifiedBootstrapConfig(**noisy_kwargs),
    )

    base_interval = baseline.metrics["mean_speed"]
    noisy_interval = noisy.metrics["mean_speed"]

    assert base_interval.lower is not None and base_interval.upper is not None
    assert noisy_interval.lower is not None and noisy_interval.upper is not None

    base_span = base_interval.upper - base_interval.lower
    noisy_span = noisy_interval.upper - noisy_interval.lower

    assert noisy_span > base_span
    assert noisy.bootstrap_means["mean_speed"] is not None
    assert len(noisy.notes) >= len(baseline.notes)
