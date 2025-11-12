"""Unit tests for the bootstrap uncertainty generation script helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from types import SimpleNamespace

import pytest

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hf_wind_resource.stats import (
    BootstrapConfidenceInterval,
    BootstrapPowerDiagnostics,
    BootstrapUncertaintyResult,
    GlobalRmseRecord,
    HeightCorrection,
    NodeBootstrapInput,
    PowerCurve,
    StratifiedBootstrapConfig,
    load_kaplan_meier_selection_criteria,
)
from scripts.generate_bootstrap_uncertainty import (
    _process_nodes,
    _append_partial_row,
    _load_inputs_from_json,
    _load_partial_rows,
    _summarise_results,
    _write_metadata,
    _write_summary_csv,
)


def _make_rmse_record() -> GlobalRmseRecord:
    now = datetime(2025, 10, 22, tzinfo=timezone.utc)
    return GlobalRmseRecord(
        version="test",
        value=2.5,
        unit="m/s",
        effective_from=now,
        effective_until=None,
        source="docs/sar_range_final_schema.md:319",
        computed_at=now,
        notes=(),
    )


def test_summarise_results_formats_metrics() -> None:
    rmse_record = _make_rmse_record()
    metrics = {
        "mean_speed": BootstrapConfidenceInterval(estimate=10.0, lower=9.5, upper=10.4, confidence_level=0.95, replicates=200),
        "p50": BootstrapConfidenceInterval(estimate=9.8, lower=9.1, upper=10.2, confidence_level=0.95, replicates=200),
        "p90": BootstrapConfidenceInterval(estimate=12.1, lower=11.5, upper=12.8, confidence_level=0.95, replicates=200),
        "p99": BootstrapConfidenceInterval(estimate=14.0, lower=13.0, upper=15.5, confidence_level=0.95, replicates=200),
        "power_density": BootstrapConfidenceInterval(estimate=450.0, lower=420.0, upper=480.0, confidence_level=0.95, replicates=200),
    }
    result = BootstrapUncertaintyResult(
        node_id="NODE_A",
        metrics=metrics,
        rmse_record=rmse_record,
        label_counts={"below": 2, "in": 5, "above": 1, "uncertain": 0},
        label_proportions={"below": 0.25, "in": 0.625, "above": 0.125, "uncertain": 0.0},
        total_samples=8,
        bootstrap_means={name: interval.estimate for name, interval in metrics.items()},
        power_diagnostics=BootstrapPowerDiagnostics(
            method="weibull",
            selection_reasons=("reason-1",),
            method_notes=("note-1",),
            weibull=None,
            replicate_method_counts={"weibull": 150, "kaplan_meier": 50},
        ),
        notes=("Sample note",),
    )

    rows = _summarise_results([result])
    assert len(rows) == 1
    row = rows[0]
    assert row["node_id"] == "NODE_A"
    assert row["rmse_value"] == pytest.approx(2.5)
    assert row["mean_speed_estimate"] == pytest.approx(10.0)
    assert row["p90_upper"] == pytest.approx(12.8)
    assert row["power_density_replicates"] == 200
    assert "Sample note" in row["notes"]
    assert row["power_method"] == "weibull"
    assert row["replicate_method_weibull"] == 150


def test_write_summary_csv(tmp_path: Path) -> None:
    rows = [
        {
            "node_id": "NODE_A",
            "rmse_version": "test",
            "rmse_value": 2.5,
            "rmse_unit": "m/s",
            "rmse_source": "docs/sar_range_final_schema.md:319",
            "label_count_below": 2,
            "label_count_in": 5,
            "label_count_above": 1,
            "label_count_uncertain": 0,
            "label_ratio_below": 0.2,
            "label_ratio_in": 0.5,
            "label_ratio_above": 0.3,
            "label_ratio_uncertain": 0.0,
            "notes": "",
            "mean_speed_estimate": 10.0,
            "mean_speed_lower": 9.5,
            "mean_speed_upper": 10.4,
            "mean_speed_replicates": 200,
            "p50_estimate": 9.8,
            "p50_lower": 9.0,
            "p50_upper": 10.1,
            "p50_replicates": 200,
            "p90_estimate": 12.0,
            "p90_lower": 11.4,
            "p90_upper": 12.6,
            "p90_replicates": 200,
            "p99_estimate": 14.0,
            "p99_lower": 13.0,
            "p99_upper": 15.0,
            "p99_replicates": 200,
            "power_density_estimate": 450.0,
            "power_density_lower": 425.0,
            "power_density_upper": 470.0,
            "power_density_replicates": 200,
        }
    ]

    path = tmp_path / "summary.csv"
    _write_summary_csv(rows, path)

    content = path.read_text(encoding="utf-8")
    assert "node_id" in content.splitlines()[0]
    assert "NODE_A" in content


def test_load_inputs_from_json(tmp_path: Path) -> None:
    path = tmp_path / "inputs.jsonl"
    payload = {
        "node_id": "A",
        "records": [
            {
                "pred_wind_speed": 8.0,
                "prob_range_below": 0.2,
                "prob_range_in": 0.6,
                "prob_range_above": 0.2,
                "range_flag": "in",
                "range_flag_confident": True,
            }
        ],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    default_height = HeightCorrection(method="none", source_height_m=10.0, target_height_m=10.0, speed_scale=1.0)
    inputs = _load_inputs_from_json(path, height_corrections={}, default_height=default_height)

    assert len(inputs) == 1
    item = inputs[0]
    assert item.node_id == "A"
    assert item.records[0]["pred_wind_speed"] == pytest.approx(8.0)
    assert item.height.speed_scale == pytest.approx(1.0)


def test_partial_row_helpers(tmp_path: Path) -> None:
    partial_path = tmp_path / "partial.jsonl"
    row_a = {"node_id": "A", "value": 1}
    row_b = {"node_id": "B", "value": 2}

    _append_partial_row(partial_path, row_a)
    _append_partial_row(partial_path, row_b)

    rows, processed = _load_partial_rows(partial_path)
    assert len(rows) == 2
    assert processed == {"A", "B"}


def test_write_metadata_includes_progress(tmp_path: Path) -> None:
    destination = tmp_path / "meta.json"
    partial_path = tmp_path / "partial.jsonl"

    power_curve_mapping = {
        "name": "Example",
        "speeds": [0.0, 10.0],
        "power_kw": [0.0, 1000.0],
        "reference_air_density": 1.225,
        "hub_height_m": 100.0,
        "notes": ["test"],
    }

    km_mapping = {
        "min_total_observations": 10,
        "max_censored_ratio": 0.3,
        "max_below_ratio": 0.3,
        "min_in_ratio": 0.5,
    }

    class DummyCurve:
        def __init__(self, mapping: Mapping[str, object]) -> None:
            self._mapping = mapping

        def to_mapping(self) -> Mapping[str, object]:
            return self._mapping

    class DummyKm:
        def __init__(self, mapping: Mapping[str, object]) -> None:
            self.__dict__ = dict(mapping)

    config = SimpleNamespace(
        replicas=100,
        confidence_level=0.95,
        apply_rmse_noise=True,
        min_confidence=0.5,
        min_in_range_weight=500.0,
        tail_surrogate=None,
        random_seed=42,
        air_density=1.225,
        power_curve=DummyCurve(power_curve_mapping),
        km_criteria=DummyKm(km_mapping),
        rmse_mode="velocity",
        resampling_mode="moving_block",
        block_length=6,
        node_block_lengths={"NODE_A": 8},
        label_strategy="fixed",
        ci_method="percentile",
        jackknife_max_samples=200,
    )

    args = SimpleNamespace(
        image="duckdb/duckdb:latest",
        max_nodes=None,
        resume=True,
        progress_interval=10,
        workers=2,
        rmse_mode="velocity",
        resampling_mode="moving_block",
        block_length=6,
        block_lengths_csv=Path("artifacts/bootstrap_uncertainty/block_bootstrap_diagnostics.csv"),
        max_block_length=10,
        ci_method="percentile",
        jackknife_max_samples=200,
    )

    _write_metadata(
        destination,
        config=config,
        dataset=Path("/data/test.parquet"),
        args=args,
        partial_path=partial_path,
        processed_nodes=12,
        workers=2,
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["resume"] is True
    assert payload["processed_nodes"] == 12
    assert payload["partial_results_path"].endswith("partial.jsonl")
    assert payload["ci_method"] == "percentile"
    assert payload["jackknife_max_samples"] == 200
    assert payload["label_strategy"] == "fixed"
    assert payload["resampling_mode"] == "moving_block"
    assert payload["block_length"] == 6
    assert payload["node_block_lengths_loaded"] == 1


def test_process_nodes_parallel(tmp_path: Path) -> None:
    power_curve = PowerCurve(
        name="TestCurve",
        speeds=(0.0, 10.0, 25.0),
        power_kw=(0.0, 600.0, 600.0),
        reference_air_density=1.225,
        notes=("test",),
    )
    km_criteria = load_kaplan_meier_selection_criteria(None)
    config = StratifiedBootstrapConfig(
        replicas=10,
        confidence_level=0.9,
        random_seed=123,
        apply_rmse_noise=False,
        air_density=1.225,
        min_confidence=0.5,
        min_in_range_weight=0.0,
        tail_surrogate=None,
        power_curve=power_curve,
        km_criteria=km_criteria,
    )

    height = HeightCorrection(method="none", source_height_m=10.0, target_height_m=10.0, speed_scale=1.0)

    records_a = [
        {
            "pred_wind_speed": 8.0,
            "prob_range_below": 0.1,
            "prob_range_in": 0.8,
            "prob_range_above": 0.1,
            "range_flag": "in",
            "range_flag_confident": True,
        },
        {
            "pred_wind_speed": 12.0,
            "prob_range_below": 0.05,
            "prob_range_in": 0.7,
            "prob_range_above": 0.25,
            "range_flag": "in",
            "range_flag_confident": False,
        },
    ]

    records_b = [
        {
            "pred_wind_speed": 6.5,
            "prob_range_below": 0.2,
            "prob_range_in": 0.7,
            "prob_range_above": 0.1,
            "range_flag": "in",
            "range_flag_confident": False,
        },
        {
            "pred_wind_speed": 2.5,
            "prob_range_below": 0.85,
            "prob_range_in": 0.1,
            "prob_range_above": 0.05,
            "range_flag": "below",
            "range_flag_confident": True,
        },
    ]

    inputs = [
        NodeBootstrapInput(node_id="A", records=records_a, height=height),
        NodeBootstrapInput(node_id="B", records=records_b, height=height),
    ]

    partial_seq = tmp_path / "partial_seq.jsonl"
    rows_seq, workers_seq = _process_nodes(
        inputs,
        config=config,
        partial_path=partial_seq,
        resume=False,
        progress_interval=0,
        workers=1,
    )

    partial_par = tmp_path / "partial_par.jsonl"
    rows_par, workers_par = _process_nodes(
        inputs,
        config=config,
        partial_path=partial_par,
        resume=False,
        progress_interval=0,
        workers=2,
    )

    assert workers_seq == 1
    assert workers_par == 2

    sorted_seq = sorted(rows_seq, key=lambda row: row["node_id"])
    sorted_par = sorted(rows_par, key=lambda row: row["node_id"])
    assert sorted_seq == sorted_par

    assert partial_seq.read_text(encoding="utf-8").count("\n") == len(inputs)
    assert partial_par.read_text(encoding="utf-8").count("\n") == len(inputs)
