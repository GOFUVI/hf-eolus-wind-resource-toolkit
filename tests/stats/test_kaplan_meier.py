from __future__ import annotations

import json
from pathlib import Path

import pytest

from hf_wind_resource.stats import (
    CensoredWeibullData,
    KaplanMeierSelectionCriteria,
    evaluate_kaplan_meier_selection,
    evaluate_step_cdf,
    load_kaplan_meier_selection_criteria,
    run_weighted_kaplan_meier,
)


def _make_sample_data() -> CensoredWeibullData:
    return CensoredWeibullData(
        in_values=(6.0, 8.0, 10.0),
        in_weights=(1.0, 1.0, 1.0),
        left_limits=(5.7,),
        left_weights=(2.0,),
        right_limits=(17.8,),
        right_weights=(1.0,),
    )


def test_run_weighted_kaplan_meier_produces_step_function() -> None:
    data = _make_sample_data()
    result = run_weighted_kaplan_meier(data)

    assert result.support == (5.7, 6.0, 8.0, 10.0)
    assert result.left_censored_weight == 2.0
    assert result.right_censored_weight == 1.0
    assert result.total_weight == 6.0

    expected_cdf = (1 / 3, 0.5, 2 / 3, 5 / 6)
    for idx, value in enumerate(result.cdf):
        assert value == pytest.approx(expected_cdf[idx], rel=1e-6)

    assert result.cdf_at(5.6) == 0.0
    assert result.cdf_at(7.0) == pytest.approx(0.5, rel=1e-6)
    assert result.cdf_at(20.0) == pytest.approx(5 / 6, rel=1e-6)

    assert result.quantile(0.5) == pytest.approx(6.0, rel=1e-6)
    assert result.quantile(0.7) == pytest.approx(10.0, rel=1e-6)
    assert result.quantile(0.95) is None
    assert result.right_tail_probability == pytest.approx(1 / 6, rel=1e-6)


def test_evaluate_step_cdf_returns_vector() -> None:
    result = run_weighted_kaplan_meier(_make_sample_data())
    points = (5.7, 8.0, 12.0)
    evaluated = evaluate_step_cdf(result, points)
    assert evaluated[0] == pytest.approx(1 / 3, rel=1e-6)
    assert evaluated[1] == pytest.approx(2 / 3, rel=1e-6)
    assert evaluated[2] == pytest.approx(5 / 6, rel=1e-6)


def test_evaluate_kaplan_meier_selection_triggers_on_censoring() -> None:
    summary = {
        "total_observations": 1200,
        "censored_ratio": 0.28,
        "below_ratio": 0.10,
        "valid_count": 500,
        "in_ratio": 0.42,
    }

    use, reasons = evaluate_kaplan_meier_selection(
        summary,
        criteria=KaplanMeierSelectionCriteria(
            min_total_observations=200,
            min_total_censored_ratio=0.2,
            min_below_ratio=0.15,
            max_valid_share=0.55,
            min_uncensored_weight=300.0,
        ),
    )

    assert use is True
    assert any("censored ratio" in reason for reason in reasons)


def test_evaluate_kaplan_meier_selection_skips_when_below_thresholds() -> None:
    summary = {
        "total_observations": 800,
        "censored_ratio": 0.12,
        "below_ratio": 0.08,
        "valid_count": 600,
        "in_ratio": 0.75,
    }

    use, reasons = evaluate_kaplan_meier_selection(summary)
    assert use is False
    assert reasons == ()


def test_load_kaplan_meier_selection_criteria_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / "criteria.json"
    config_path.write_text(
        json.dumps(
            {
                "min_total_observations": 300,
                "min_total_censored_ratio": 0.25,
                "min_below_ratio": 0.2,
                "max_valid_share": 0.6,
                "min_uncensored_weight": 200.0,
            }
        ),
        encoding="utf-8",
    )

    criteria = load_kaplan_meier_selection_criteria(config_path)
    assert criteria.min_total_observations == 300
    assert criteria.min_total_censored_ratio == pytest.approx(0.25)
    assert criteria.min_below_ratio == pytest.approx(0.2)
    assert criteria.max_valid_share == pytest.approx(0.6)
    assert criteria.min_uncensored_weight == pytest.approx(200.0)
