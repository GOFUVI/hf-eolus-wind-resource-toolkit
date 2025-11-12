"""Tests for parametric model comparison diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hf_wind_resource.stats import (
    ParametricComparisonConfig,
    build_censored_data_from_records,
    evaluate_parametric_models,
    fit_censored_weibull,
)

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "stats_synthetic_datasets.json"


@pytest.fixture(scope="module")
def synthetic_dataset() -> dict[str, object]:
    """Load the deterministic statistical fixtures used across parametric tests."""

    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _build_reference_data(payload: dict[str, object]) -> tuple:
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
    weibull = fit_censored_weibull(data, min_in_count=10.0)
    return data, weibull


def test_parametric_comparison_reports_lognormal_metrics(synthetic_dataset: dict[str, object]) -> None:
    payload = synthetic_dataset["weibull_reference"]
    data, weibull = _build_reference_data(payload)  # type: ignore[arg-type]
    config = ParametricComparisonConfig(min_in_weight=10.0, ks_min_weight=5.0, selection_metric="aic")

    result = evaluate_parametric_models(data, weibull, config=config)
    assert result.selection_metric == "aic"
    lognormal = next(candidate for candidate in result.candidates if candidate.name == "lognormal")
    assert lognormal.success is True
    assert lognormal.parameters["mu"] is not None
    assert lognormal.parameters["sigma"] is not None
    assert lognormal.ks_statistic is not None
    assert result.preferred_model in {"weibull", "lognormal", "gamma"}


def test_gamma_candidate_can_be_disabled(synthetic_dataset: dict[str, object]) -> None:
    payload = synthetic_dataset["weibull_reference"]
    data, weibull = _build_reference_data(payload)  # type: ignore[arg-type]
    config = ParametricComparisonConfig(min_in_weight=10.0, ks_min_weight=5.0, enable_gamma=False)

    result = evaluate_parametric_models(data, weibull, config=config)
    gamma = next(candidate for candidate in result.candidates if candidate.name == "gamma")
    assert gamma.success is False
    assert "disabled" in " ".join(gamma.notes).lower()


def test_gamma_candidate_reports_metrics_when_available(synthetic_dataset: dict[str, object]) -> None:
    pytest.importorskip("scipy", reason="Gamma diagnostic requires SciPy.")
    payload = synthetic_dataset["weibull_reference"]
    data, weibull = _build_reference_data(payload)  # type: ignore[arg-type]
    config = ParametricComparisonConfig(min_in_weight=10.0, ks_min_weight=5.0)

    result = evaluate_parametric_models(data, weibull, config=config)
    gamma = next(candidate for candidate in result.candidates if candidate.name == "gamma")
    assert gamma.parameters["shape"] is not None
    assert gamma.parameters["scale"] is not None
    assert gamma.log_likelihood is not None
    assert gamma.aic is not None
    assert gamma.bic is not None


def test_alternative_fits_skipped_when_insufficient_in_range_weight() -> None:
    data = build_censored_data_from_records(
        [
            {
                "pred_wind_speed": 7.5,
                "prob_range_in": 0.9,
                "prob_range_below": 0.1,
                "prob_range_above": 0.0,
                "range_flag": "in",
                "range_flag_confident": True,
            }
        ],
        lower_threshold=5.7,
        upper_threshold=17.8,
        min_confidence=0.5,
    )
    weibull = fit_censored_weibull(data, min_in_count=0.1)
    config = ParametricComparisonConfig(min_in_weight=100.0, ks_min_weight=50.0)

    result = evaluate_parametric_models(data, weibull, config=config)
    # Both alternative candidates should be marked as unavailable.
    skipped = [candidate for candidate in result.candidates if candidate.name in {"lognormal", "gamma"}]
    assert all(candidate.success is False for candidate in skipped)
    assert "skipped" in " ".join(result.notes).lower()
