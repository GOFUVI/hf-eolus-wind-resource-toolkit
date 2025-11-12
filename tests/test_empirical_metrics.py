from __future__ import annotations

from hf_wind_resource.stats import BiasThresholds, assemble_per_node_metrics, compute_label_ratios


def test_compute_label_ratios_basic() -> None:
    label_rows = [
        {
            "node_id": "NODE_A",
            "total_observations": 10,
            "in_count": 6,
            "below_count": 3,
            "above_count": 1,
            "uncertain_count": 0,
        }
    ]

    result = compute_label_ratios(label_rows)
    assert len(result) == 1
    enriched = result[0]
    assert enriched["in_ratio"] == 0.6
    assert enriched["below_ratio"] == 0.3
    assert enriched["above_ratio"] == 0.1
    assert enriched["uncertain_ratio"] == 0.0
    assert enriched["censored_ratio"] == 0.4


def test_assemble_per_node_metrics_flags_bias() -> None:
    label_counts = [
        {
            "node_id": "NODE_A",
            "total_observations": 10,
            "in_count": 4,
            "below_count": 5,
            "above_count": 1,
            "uncertain_count": 0,
        }
    ]
    valid_metrics = [
        {
            "node_id": "NODE_A",
            "valid_count": 4,
            "mean_speed": 9.5,
            "p90": 12.0,
            "p99": 14.0,
        }
    ]
    taxonomy = {
        "NODE_A": {
            "total_observations": 10,
            "low_coverage": True,
            "coverage_band": "sparse",
            "continuity_band": "long_gaps",
        }
    }

    summary = assemble_per_node_metrics(valid_metrics, label_counts, taxonomy, thresholds=BiasThresholds())
    assert len(summary) == 1
    node_summary = summary[0]
    assert node_summary["any_bias"] is True
    assert node_summary["censoring_bias"] is True
    assert node_summary["coverage_bias"] is True
    assert node_summary["sample_bias"] is True
    assert "censoring" in node_summary["bias_notes"].lower()
