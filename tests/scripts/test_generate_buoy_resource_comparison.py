"""Tests for bootstrap integration in generate_buoy_resource_comparison."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from scripts.generate_buoy_resource_comparison import (
    _build_buoy_height_alignment,
    _compute_differences,
    _load_bootstrap_intervals,
    _write_bias_svg,
    _write_markdown_table,
)
from hf_wind_resource.preprocessing.buoy_timeseries import HeightCorrectionConfig
from hf_wind_resource.stats.power_pipeline import HeightCorrection


def test_load_bootstrap_intervals_reads_csv(tmp_path: Path) -> None:
    summary = tmp_path / "bootstrap_summary.csv"
    summary.write_text(
        "node_id,rmse_value,rmse_source,power_method,power_method_notes,power_selection_reasons,"
        "mean_speed_estimate,mean_speed_lower,mean_speed_upper,mean_speed_replicates,mean_speed_bootstrap_estimate,"
        "p50_estimate,p50_lower,p50_upper,p50_replicates,p50_bootstrap_estimate,"
        "p90_estimate,p90_lower,p90_upper,p90_replicates,p90_bootstrap_estimate,"
        "p99_estimate,p99_lower,p99_upper,p99_replicates,p99_bootstrap_estimate,"
        "power_density_estimate,power_density_lower,power_density_upper,power_density_replicates,power_density_bootstrap_estimate\n"
        "NODE_A,2.8,docs/example.md:10,kaplan_meier,notes,criteria,"
        "10.0,9.5,10.5,200,10.1,"
        "9.8,9.5,10.2,200,9.9,"
        "12.5,12.0,13.0,200,12.4,"
        "14.0,13.5,14.5,200,14.1,"
        "600.0,560.0,640.0,200,602.0\n",
        encoding="utf-8",
    )

    result = _load_bootstrap_intervals(summary, "NODE_A")
    assert result is not None
    assert result["power_density"]["estimate"] == pytest.approx(600.0)
    assert result["power_density"]["replicates"] == 200
    assert result["rmse_value"] == pytest.approx(2.8)
    assert result["power_method"] == "kaplan_meier"
    assert result["power_notes"] == "notes"
    assert result["selection_reasons"] == "criteria"


def test_load_bootstrap_intervals_reads_json_metrics(tmp_path: Path) -> None:
    summary = tmp_path / "bootstrap_summary.jsonl"
    record = {
        "node_id": "NODE_A",
        "rmse_record": {"value": 3.1, "source": "docs/rmse.md:5"},
        "metrics": {
            "mean_speed": {
                "estimate": 11.0,
                "lower": 10.7,
                "upper": 11.3,
                "replicates": 180,
            },
            "power_density": {
                "estimate": 650.0,
                "lower": 600.0,
                "upper": 700.0,
                "bootstrap_estimate": 652.0,
                "replicates": 180,
                "confidence": 0.9,
            },
        },
        "power_diagnostics": {
            "method": "kaplan_meier",
            "method_notes": ["note a", "note b"],
            "selection_reasons": ["reason1", "reason2"],
        },
    }
    summary.write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = _load_bootstrap_intervals(summary, "NODE_A")
    assert result is not None
    assert result["power_density"]["estimate"] == pytest.approx(650.0)
    assert result["power_density"]["bootstrap_estimate"] == pytest.approx(652.0)
    assert result["power_density"]["replicates"] == 180
    assert result["power_density"]["confidence"] == pytest.approx(0.9)
    assert result["mean_speed"]["lower"] == pytest.approx(10.7)
    assert result["rmse_value"] == pytest.approx(3.1)
    assert result["rmse_source"] == "docs/rmse.md:5"
    assert result["power_method"] == "kaplan_meier"
    assert result["power_notes"] == "note a | note b"
    assert result["selection_reasons"] == "reason1 | reason2"


def test_build_buoy_height_alignment_scales_to_ann_target() -> None:
    ann_height = HeightCorrection(
        method="log",
        source_height_m=10.0,
        target_height_m=110.0,
        speed_scale=1.0,
        roughness_length_m=0.0002,
    )
    config = HeightCorrectionConfig(
        method="log_profile",
        measurement_height_m=3.0,
        target_height_m=10.0,
        roughness_length_m=0.0002,
    )
    alignment, metadata = _build_buoy_height_alignment(ann_height, config)
    expected = math.log(110.0 / 0.0002) / math.log(10.0 / 0.0002)
    assert alignment is not None
    assert alignment.speed_scale == pytest.approx(expected)
    assert metadata is not None
    assert metadata["paired_series_height_m"] == pytest.approx(10.0)
    assert "10.0→110.0 m" in (metadata["note"] or "")


def test_build_buoy_height_alignment_returns_identity_when_heights_match() -> None:
    ann_height = HeightCorrection(
        method="log",
        source_height_m=10.0,
        target_height_m=110.0,
        speed_scale=1.0,
        roughness_length_m=0.0002,
    )
    config = HeightCorrectionConfig(
        method="log_profile",
        measurement_height_m=3.0,
        target_height_m=110.0,
        roughness_length_m=0.0002,
    )
    alignment, metadata = _build_buoy_height_alignment(ann_height, config)
    assert alignment is not None
    assert alignment.speed_scale == pytest.approx(1.0)
    assert metadata is not None
    assert metadata["scale_applied"] == pytest.approx(1.0)
    assert metadata["paired_series_height_m"] == pytest.approx(110.0)


def test_build_buoy_height_alignment_handles_missing_config() -> None:
    ann_height = HeightCorrection(
        method="log",
        source_height_m=10.0,
        target_height_m=110.0,
        speed_scale=1.0,
        roughness_length_m=0.0002,
    )
    alignment, metadata = _build_buoy_height_alignment(ann_height, None)
    assert alignment is None
    assert metadata is None


def _sample_rows() -> list[dict[str, object]]:
    return [
        {
            "dataset": "ANN",
            "scope": "paired",
            "mean_speed": 11.2,
            "mean_speed_lower": 11.0,
            "mean_speed_upper": 11.4,
            "p90": 15.4,
            "p90_lower": 15.0,
            "p90_upper": 15.8,
            "power_density_model": 640.0,
            "power_density_model_lower": 610.0,
            "power_density_model_upper": 670.0,
        },
        {
            "dataset": "Buoy",
            "scope": "paired",
            "mean_speed": 6.3,
            "mean_speed_lower": 6.1,
            "mean_speed_upper": 6.5,
            "p90": 11.1,
            "p90_lower": 10.8,
            "p90_upper": 11.4,
            "power_density_model": 520.0,
            "power_density_model_lower": 500.0,
            "power_density_model_upper": 540.0,
        },
        {
            "dataset": "Buoy",
            "scope": "global",
            "mean_speed": 6.4,
            "p90": 11.3,
            "power_density_model": 525.0,
        },
    ]


def test_write_markdown_table_builds_summary(tmp_path: Path) -> None:
    rows = _sample_rows()
    table = tmp_path / "resource_metrics_table.md"
    _write_markdown_table(
        table,
        rows,
        sample_count=11158,
        main_report="docs/empirical_metrics_summary.md",
    )
    payload = table.read_text(encoding="utf-8")
    assert "Vilano buoy validation snapshot" in payload
    assert "| ANN | paired |" in payload
    assert "docs/empirical_metrics_summary.md" in payload


def test_write_bias_svg_generates_svg(tmp_path: Path) -> None:
    rows = _sample_rows()
    ann_row = rows[0]
    buoy_row = rows[1]
    svg_path = tmp_path / "resource_bias.svg"
    differences = {
        "mean_speed": 4.9,
        "p90": 4.3,
        "power_density_model": 120.0,
    }
    _write_bias_svg(
        svg_path,
        differences=differences,
        paired_ann=ann_row,
        paired_buoy=buoy_row,
        sample_count=11412,
    )
    svg = svg_path.read_text(encoding="utf-8")
    assert "<svg" in svg
    assert "Wind-speed bias" in svg
    assert "11,412 paired hours" in svg


def test_compute_differences_handles_model_only_metrics() -> None:
    ann_metrics = {
        "mean_speed": 11.3,
        "p90": 15.2,
        "power_density_model": 640.0,
    }
    buoy_metrics = {
        "mean_speed_model": 7.8,
        "p90_model": 13.6,
        "power_density_model": 520.0,
    }

    differences = _compute_differences(ann_metrics, buoy_metrics)

    assert differences["mean_speed"] == pytest.approx(3.5)
    assert differences["p90"] == pytest.approx(1.6)
    assert differences["power_density_model"] == pytest.approx(120.0)
