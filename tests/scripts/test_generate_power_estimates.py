"""Unit tests for helper utilities in generate_power_estimates."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hf_wind_resource.preprocessing.censoring import RangeThresholds
from hf_wind_resource.stats import KaplanMeierResult, PowerCurve, PowerCurveEstimate, PowerDensityEstimate, load_kaplan_meier_selection_criteria
from hf_wind_resource.stats.weibull import WeibullFitDiagnostics, WeibullFitResult
from scripts.generate_power_estimates import (
    _aggregate_temporal_power,
    _build_summary_row,
    _load_height_config,
    _load_power_curve,
    _parse_records,
    _power_curve_estimate_to_mapping,
    _power_density_to_mapping,
    _resolve_height_correction,
    HeightCorrection,
)


def _make_weibull(shape: float = 2.1, scale: float = 9.4) -> WeibullFitResult:
    diagnostics = WeibullFitDiagnostics(iterations=12, gradient_norm=0.01, last_step_size=0.005, message="ok")
    return WeibullFitResult(
        shape=shape,
        scale=scale,
        log_likelihood=-123.4,
        success=True,
        diagnostics=diagnostics,
        used_gradients=True,
        reliable=True,
        in_count=620.0,
        left_count=180.0,
        right_count=40.0,
    )


def test_parse_records_converts_to_numeric() -> None:
    csv_payload = "pred_wind_speed,prob_range_below,range_flag,range_flag_confident\n8.5,0.1,in,true\n"
    records = _parse_records(csv_payload)
    assert len(records) == 1
    row = records[0]
    assert row["pred_wind_speed"] == pytest.approx(8.5)
    assert row["prob_range_below"] == pytest.approx(0.1)
    assert row["range_flag"] == "in"
    assert row["range_flag_confident"] is True


def test_load_power_curve_from_config(tmp_path: Path) -> None:
    config = tmp_path / "power_curves.json"
    config.write_text(
        json.dumps(
            {
                "example": {
                    "name": "Example Curve",
                    "reference_air_density": 1.225,
                    "hub_height_m": 100.0,
                    "speeds": [0.0, 5.0, 10.0],
                    "power_kw": [0.0, 500.0, 2000.0],
                    "notes": ["Synthetic reference"],
                }
            }
        ),
        encoding="utf-8",
    )

    curve = _load_power_curve(config, "example")
    assert curve.name == "Example Curve"
    assert curve.rated_power_kw == pytest.approx(2000.0)
    assert curve.hub_height_m == pytest.approx(100.0)
    assert curve.notes == ("Synthetic reference",)


def test_power_density_and_curve_mappings() -> None:
    curve = PowerCurve(
        name="Reference 2MW",
        speeds=(0.0, 5.0, 10.0, 15.0),
        power_kw=(0.0, 200.0, 1000.0, 2000.0),
    )
    density = PowerDensityEstimate(method="weibull", estimate_w_per_m2=480.0, air_density=1.225, notes=("note",))
    curve_estimate = PowerCurveEstimate(
        curve=curve,
        mean_power_kw=950.0,
        capacity_factor=0.475,
        air_density=1.225,
        notes=("curve_note",),
    )

    density_map = _power_density_to_mapping(density)
    curve_map = _power_curve_estimate_to_mapping(curve_estimate)

    assert density_map["estimate_w_per_m2"] == pytest.approx(480.0)
    assert curve_map["mean_power_kw"] == pytest.approx(950.0)
    assert curve_map["capacity_factor"] == pytest.approx(0.475)


def test_build_summary_row_combines_notes_and_diagnostics() -> None:
    curve = PowerCurve(
        name="Reference 2MW",
        speeds=(0.0, 5.0, 10.0, 15.0),
        power_kw=(0.0, 200.0, 1000.0, 2000.0),
    )
    density = PowerDensityEstimate(
        method="weibull",
        estimate_w_per_m2=520.0,
        air_density=1.225,
        notes=("density_note",),
    )
    curve_estimate = PowerCurveEstimate(
        curve=curve,
        mean_power_kw=980.0,
        capacity_factor=0.49,
        air_density=1.225,
        notes=("curve_note",),
    )
    weibull = _make_weibull()

    height = HeightCorrection(method="none", source_height_m=10.0, target_height_m=10.0, speed_scale=1.0)

    summary = _build_summary_row(
        node_id="NODE_A",
        method="weibull",
        air_density=1.225,
        power_density=density,
        power_curve_estimate=curve_estimate,
        power_curve=curve,
        weibull=weibull,
        km_result=None,
        selection_reasons=("censored ratio 0.35 ≥ 0.20",),
        extra_notes=("extra",),
        height=height,
    )

    assert summary["node_id"] == "NODE_A"
    assert summary["method"] == "weibull"
    assert summary["power_density_w_m2"] == pytest.approx(520.0)
    assert summary["turbine_mean_power_kw"] == pytest.approx(980.0)
    assert "extra" in summary["power_density_notes"]
    assert "density_note" in summary["power_density_notes"]
    assert "curve_note" in summary["power_density_notes"]
    assert summary["weibull_shape"] == pytest.approx(2.1)
    assert summary["kaplan_meier_selection_reasons"] == "censored ratio 0.35 ≥ 0.20"
    assert summary["height_method"] == "none"
    assert summary["height_speed_scale"] == pytest.approx(1.0)


def test_build_summary_row_records_kaplan_meier_tail() -> None:
    curve = PowerCurve(
        name="Reference 2MW",
        speeds=(0.0, 5.0, 10.0),
        power_kw=(0.0, 500.0, 2000.0),
    )
    density = PowerDensityEstimate("kaplan_meier", 410.0, 1.225, ("km_note",))
    curve_estimate = PowerCurveEstimate(curve=curve, mean_power_kw=820.0, capacity_factor=0.41, air_density=1.225, notes=())
    weibull = _make_weibull()
    km = KaplanMeierResult(
        support=(5.7, 8.0),
        cdf=(0.3, 0.9),
        survival=(0.7, 0.1),
        total_weight=1000.0,
        left_censored_weight=300.0,
        right_censored_weight=100.0,
    )

    height = HeightCorrection(method="log", source_height_m=10.0, target_height_m=80.0, speed_scale=1.5, roughness_length_m=0.0002)

    summary = _build_summary_row(
        node_id="NODE_B",
        method="kaplan_meier",
        air_density=1.225,
        power_density=density,
        power_curve_estimate=curve_estimate,
        power_curve=curve,
        weibull=weibull,
        km_result=km,
        selection_reasons=(),
        extra_notes=(),
        height=height,
    )

    assert summary["kaplan_meier_tail_probability"] == pytest.approx(0.1)
    assert summary["method"] == "kaplan_meier"
    assert summary["height_method"] == "log"


def test_resolve_height_correction_uses_config(tmp_path: Path) -> None:
    config_path = tmp_path / "height.json"
    config_path.write_text(
        json.dumps(
            {
                "method": "power",
                "source_height_m": 12.0,
                "target_height_m": 90.0,
                "power_law_alpha": 0.14,
                "roughness_length_m": 0.001,
            }
        ),
        encoding="utf-8",
    )

    defaults, resolved = _load_height_config(config_path)
    curve = PowerCurve(
        name="Ref",
        speeds=(0.0, 5.0, 10.0),
        power_kw=(0.0, 500.0, 2000.0),
        reference_air_density=1.225,
        hub_height_m=100.0,
    )

    args = SimpleNamespace(
        height_method=None,
        source_height_m=None,
        target_height_m=None,
        power_law_alpha=None,
        roughness_length_m=None,
    )

    correction = _resolve_height_correction(args, curve, defaults)
    assert resolved == config_path.resolve()
    assert correction.method == "power"
    assert correction.source_height_m == pytest.approx(12.0)
    assert correction.target_height_m == pytest.approx(90.0)
    expected_scale = (90.0 / 12.0) ** 0.14
    assert correction.speed_scale == pytest.approx(expected_scale)


def test_aggregate_temporal_power_returns_monthly_and_seasonal_rows() -> None:
    thresholds = RangeThresholds(lower=5.0, upper=20.0)
    records = [
        {
            "timestamp": "2023-01-05T00:00:00Z",
            "pred_wind_speed": 9.0,
            "prob_range_below": 0.0,
            "prob_range_in": 1.0,
            "prob_range_above": 0.0,
            "range_flag": "in",
            "range_flag_confident": True,
        },
        {
            "timestamp": "2023-02-05T00:00:00Z",
            "pred_wind_speed": 10.0,
            "prob_range_below": 0.0,
            "prob_range_in": 1.0,
            "prob_range_above": 0.0,
            "range_flag": "in",
            "range_flag_confident": True,
        },
        {
            "timestamp": "2023-06-05T00:00:00Z",
            "pred_wind_speed": 11.0,
            "prob_range_below": 0.0,
            "prob_range_in": 1.0,
            "prob_range_above": 0.0,
            "range_flag": "in",
            "range_flag_confident": True,
        },
    ]

    power_curve = PowerCurve(
        name="Reference 2MW",
        speeds=(0.0, 5.0, 10.0, 15.0),
        power_kw=(0.0, 200.0, 1000.0, 2000.0),
        reference_air_density=1.225,
        hub_height_m=100.0,
    )
    height = HeightCorrection(method="none", source_height_m=10.0, target_height_m=10.0, speed_scale=1.0)
    km_criteria = load_kaplan_meier_selection_criteria()

    seasonal_rows, monthly_rows = _aggregate_temporal_power(
        node_id="NODE_X",
        records=records,
        thresholds=thresholds,
        min_confidence=0.5,
        power_curve=power_curve,
        air_density=1.225,
        tail_surrogate=thresholds.upper,
        min_in_range=0.1,
        km_criteria=km_criteria,
        height=height,
    )

    assert len(monthly_rows) == 3
    assert all(row["period_type"] == "monthly" for row in monthly_rows)
    jan_row = next(row for row in monthly_rows if row["month"] == 1)
    assert jan_row["power_density_w_m2"] is not None
    assert jan_row["period_start"].startswith("2023-01-01")

    assert len(seasonal_rows) == 2
    djf_row = next(row for row in seasonal_rows if row["season"] == "DJF")
    assert djf_row["season_year"] == 2023
    assert djf_row["power_density_method"] in {"weibull", "kaplan_meier"}
