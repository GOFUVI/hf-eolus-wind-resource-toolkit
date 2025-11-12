"""Helper tests for the seasonal-variation generation script."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hf_wind_resource.stats import compute_seasonal_analysis
from scripts.generate_seasonal_variations import (
    _analysis_to_tables,
    _format_height_lines,
    _summarise_variation,
    _write_markdown,
)


def test_summarise_variation_creates_payload_and_markdown(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {"timestamp": "2023-01-01T00:00:00Z", "node_id": "NODE_A", "pred_wind_speed": 8.0, "pred_range_label": "in"},
            {"timestamp": "2023-07-01T00:00:00Z", "node_id": "NODE_A", "pred_wind_speed": 12.0, "pred_range_label": "in"},
            {"timestamp": "2024-01-01T00:00:00Z", "node_id": "NODE_A", "pred_wind_speed": 7.0, "pred_range_label": "in"},
            {"timestamp": "2024-07-01T00:00:00Z", "node_id": "NODE_A", "pred_wind_speed": 11.0, "pred_range_label": "in"},
        ]
    )

    result = compute_seasonal_analysis(frame)
    seasonal_df, annual_df, summary_df = _analysis_to_tables(result)
    assert not seasonal_df.empty
    assert not annual_df.empty

    payload = _summarise_variation(summary_df)
    assert payload["total_nodes"] == 1
    assert payload["top_amplitudes"][0]["seasonal_amplitude"] == pytest.approx(4.0)
    assert payload["strongest_negative_trends"][0]["annual_trend_slope"] == pytest.approx(-1.0)

    height_lines = _format_height_lines(
        {
            "method": "log",
            "source_height_m": 10.0,
            "target_height_m": 110.0,
            "speed_scale": 1.2,
            "roughness_length_m": 0.0002,
        }
    )
    assert len(height_lines) == 2
    assert "method=log" in height_lines[0]

    report_path = tmp_path / "seasonal_report.md"
    _write_markdown(
        report_path,
        dataset=Path("catalogs/dataset.parquet"),
        output_dir=tmp_path / "outputs",
        summary_df=summary_df,
        payload=payload,
        height_lines=height_lines,
        height_source=Path("artifacts/power_estimates/metadata.json"),
    )

    content = report_path.read_text(encoding="utf-8")
    assert "Seasonal and Interannual Variations" in content
    assert "dataset.parquet" in content
    assert "method=log" in content
