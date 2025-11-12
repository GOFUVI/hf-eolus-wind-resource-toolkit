"""Tests for the temporal dependence diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hf_wind_resource.stats import compute_node_dependence_metrics, summarise_dependence_levels


def _make_dataset(path: Path) -> None:
    rng = np.random.default_rng(1234)
    timestamps = pd.date_range("2022-01-01", periods=200, freq="h", tz="UTC")

    # Node A: AR(1) with strong positive autocorrelation
    ar1 = 0.8
    noise = rng.normal(scale=0.5, size=len(timestamps))
    series_a = np.empty_like(noise)
    series_a[0] = noise[0]
    for idx in range(1, len(series_a)):
        series_a[idx] = ar1 * series_a[idx - 1] + noise[idx]

    # Node B: white noise
    series_b = rng.normal(scale=1.0, size=len(timestamps))

    df = pd.DataFrame(
        {
            "timestamp": np.tile(timestamps, 2),
            "node_id": ["NODE_A"] * len(timestamps) + ["NODE_B"] * len(timestamps),
            "pred_wind_speed": np.concatenate([series_a, series_b]) + 10.0,
        }
    )

    df.to_parquet(path)


def test_compute_node_dependence_metrics_detects_autocorrelation(tmp_path: Path) -> None:
    dataset_path = tmp_path / "synthetic.parquet"
    _make_dataset(dataset_path)

    metrics = compute_node_dependence_metrics(dataset_path, lags=(1, 2), min_pairs=20)
    by_node = {item.node_id: item for item in metrics}

    assert set(by_node.keys()) == {"NODE_A", "NODE_B"}

    node_a = by_node["NODE_A"]
    node_b = by_node["NODE_B"]

    assert node_a.autocorrelations[1] is not None
    assert node_a.autocorrelations[1] > 0.7
    assert node_a.suggested_block_length is not None
    assert node_a.suggested_block_length > 1

    assert node_b.autocorrelations[1] is not None
    assert abs(node_b.autocorrelations[1]) < 0.2
    assert node_b.suggested_block_length == 1


def test_min_pairs_filters_autocorrelation(tmp_path: Path) -> None:
    dataset_path = tmp_path / "synthetic_small.parquet"
    _make_dataset(dataset_path)

    metrics = compute_node_dependence_metrics(dataset_path, lags=(1,), min_pairs=500)
    for item in metrics:
        assert item.autocorrelations[1] is None


def test_summarise_dependence_levels_reports_shares(tmp_path: Path) -> None:
    dataset_path = tmp_path / "synthetic_summary.parquet"
    _make_dataset(dataset_path)

    metrics = compute_node_dependence_metrics(dataset_path, lags=(1,), min_pairs=20)
    summary = summarise_dependence_levels(metrics, acf_thresholds=(0.5,))

    assert summary["node_count"] == 2
    share = summary["share_by_threshold"]["0.5"]
    assert 0.4 <= share <= 0.6
    assert summary["mean_block_length"] is not None
