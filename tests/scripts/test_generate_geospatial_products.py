"""Tests covering the bootstrap integration in generate_geospatial_products."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from scripts.generate_geospatial_products import (
    PowerUncertaintyInterval,
    load_node_metrics,
    _load_power_uncertainty_summary,
    summarise_power_uncertainty,
    write_metadata,
)


def test_load_node_metrics_prefers_taxonomy_flags(tmp_path: Path) -> None:
    csv_path = tmp_path / "nodes.csv"
    geometry_bytes = struct.pack("<BIdd", 1, 1, -8.5, 43.3)
    geometry = "".join(f"\\x{byte:02x}" for byte in geometry_bytes)
    csv_path.write_text(
        "node_id,geometry,low_coverage,coverage_band,continuity_band\n"
        f"NODE_A,{geometry},false,moderate,regular\n",
        encoding="utf-8",
    )

    taxonomy = {
        "NODE_A": {
            "low_coverage": True,
            "coverage_band": "sparse",
            "continuity_band": "long_gaps",
        }
    }

    nodes = load_node_metrics(csv_path, taxonomy=taxonomy)
    assert nodes[0].low_coverage is True
    assert nodes[0].coverage_band == "sparse"
    assert nodes[0].continuity_band == "long_gaps"


def test_load_power_uncertainty_summary_reads_extended_columns(tmp_path: Path) -> None:
    summary = tmp_path / "bootstrap_summary.csv"
    summary.write_text(
        "node_id,power_density_estimate,power_density_lower,power_density_upper,"
        "power_density_bootstrap_estimate,power_density_replicates\n"
        "NODE_A,610.0,580.0,650.0,612.0,200\n"
        "NODE_B,720.0,,,719.0,\n",
        encoding="utf-8",
    )

    lookup = _load_power_uncertainty_summary(
        summary,
        estimate_field="power_density_estimate",
        lower_field="power_density_lower",
        upper_field="power_density_upper",
        bootstrap_field="power_density_bootstrap_estimate",
        replicates_field="power_density_replicates",
        confidence_field=None,
    )

    assert "NODE_A" in lookup
    interval = lookup["NODE_A"]
    assert interval.estimate == pytest.approx(610.0)
    assert interval.lower == pytest.approx(580.0)
    assert interval.upper == pytest.approx(650.0)
    assert interval.bootstrap_estimate == pytest.approx(612.0)
    assert interval.replicates == 200
    assert lookup["NODE_B"].lower is None


def test_load_power_uncertainty_summary_supports_jsonl(tmp_path: Path) -> None:
    summary = tmp_path / "bootstrap_summary.jsonl"
    records = [
        {
            "node_id": "NODE_A",
            "power_density_estimate": 500.0,
            "power_density_lower": 470.0,
            "power_density_upper": 540.0,
            "power_density_bootstrap_estimate": 505.0,
            "power_density_replicates": 150,
        },
        {
            "node_id": "NODE_B",
            "power_density_estimate": 620.0,
            "power_density_lower": 600.0,
            "power_density_upper": 650.0,
        },
    ]
    summary.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")

    lookup = _load_power_uncertainty_summary(
        summary,
        estimate_field="power_density_estimate",
        lower_field="power_density_lower",
        upper_field="power_density_upper",
        bootstrap_field="power_density_bootstrap_estimate",
        replicates_field="power_density_replicates",
        confidence_field=None,
    )
    assert lookup["NODE_A"].replicates == 150
    assert lookup["NODE_B"].bootstrap_estimate is None


def test_load_power_uncertainty_summary_supports_nested_metrics(tmp_path: Path) -> None:
    summary = tmp_path / "bootstrap_summary.jsonl"
    records = [
        {
            "node_id": "NODE_A",
            "metrics": {
                "power_density": {
                    "estimate": 515.0,
                    "lower": 480.0,
                    "upper": 545.0,
                    "bootstrap_estimate": 512.0,
                    "replicates": 180,
                    "confidence": 0.9,
                }
            },
        }
    ]
    summary.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")

    lookup = _load_power_uncertainty_summary(
        summary,
        estimate_field="power_density_estimate",
        lower_field="power_density_lower",
        upper_field="power_density_upper",
        bootstrap_field="power_density_bootstrap_estimate",
        replicates_field="power_density_replicates",
        confidence_field=None,
    )

    interval = lookup["NODE_A"]
    assert interval.estimate == pytest.approx(515.0)
    assert interval.lower == pytest.approx(480.0)
    assert interval.upper == pytest.approx(545.0)
    assert interval.bootstrap_estimate == pytest.approx(512.0)
    assert interval.replicates == 180
    assert interval.confidence == pytest.approx(0.9)


def test_summarise_power_uncertainty_counts_missing_nodes() -> None:
    intervals = {
        "NODE_A": PowerUncertaintyInterval(
            estimate=610.0,
            lower=580.0,
            upper=640.0,
            bootstrap_estimate=612.0,
            replicates=200,
            confidence=0.95,
        ),
        "NODE_B": PowerUncertaintyInterval(
            estimate=720.0,
            lower=None,
            upper=None,
            bootstrap_estimate=None,
            replicates=None,
            confidence=None,
        ),
    }

    stats = summarise_power_uncertainty(intervals, total_nodes=3)
    assert stats["nodes_with_ci"] == 1
    assert stats["nodes_without_ci"] == 2
    assert stats["nodes_without_summary"] == 1
    assert stats["replicates_min"] == pytest.approx(200.0)
    assert stats["replicates_mean"] == pytest.approx(200.0)


def test_write_metadata_includes_power_uncertainty_section(tmp_path: Path) -> None:
    destination = tmp_path / "metadata.json"
    node_summary = tmp_path / "nodes.csv"
    dataset = tmp_path / "dataset.parquet"
    geo_parquet = tmp_path / "map.parquet"
    geojson = tmp_path / "map.geojson"
    power_map = tmp_path / "power.svg"
    uncertainty_map = tmp_path / "uncertainty.svg"
    roses = tmp_path / "roses.svg"
    histogram = tmp_path / "histogram.csv"
    taxonomy = tmp_path / "taxonomy.json"

    for path in (node_summary, dataset, geo_parquet, geojson, power_map, uncertainty_map, roses, histogram, taxonomy):
        path.write_text("", encoding="utf-8")

    power_uncertainty_section = {
        "summary_source": "artifacts/bootstrap_velocity_block/bootstrap_summary.csv",
        "stats": {"nodes_with_ci": 42},
        "replicas": 200,
        "confidence_level": 0.95,
    }

    write_metadata(
        destination,
        node_summary=node_summary,
        dataset=dataset,
        geo_parquet=geo_parquet,
        geojson=geojson,
        power_map=power_map,
        uncertainty_map=uncertainty_map,
        roses=roses,
        histogram=histogram,
        selected_nodes=("NODE_A",),
        power_uncertainty=power_uncertainty_section,
        taxonomy=taxonomy,
        low_coverage_nodes=("NODE_A", "NODE_B"),
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["power_uncertainty"]["summary_source"] == power_uncertainty_section["summary_source"]
    assert payload["power_uncertainty"]["stats"]["nodes_with_ci"] == 42
    assert payload["low_coverage_nodes"] == ["NODE_A", "NODE_B"]
    assert payload["taxonomy_source"].endswith("taxonomy.json")
