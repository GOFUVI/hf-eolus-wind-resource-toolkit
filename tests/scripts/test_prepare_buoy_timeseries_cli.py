"""Tests for the CLI wrapper preparing buoy time series."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "prepare_buoy_timeseries.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("_prepare_buoy_cli", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_main_resolves_ann_dataset_from_stac(monkeypatch, tmp_path):
    module = _load_cli_module()

    dummy_ann_path = tmp_path / "ann.parquet"
    dummy_ann_path.touch()

    captured_args = {}

    class DummyResult:
        def __init__(self):
            self.buoy = SimpleNamespace(
                total_records=10,
                dataframe=[1, 2, 3],
                dropped_speed_records=1,
                direction_sentinel_records=0,
                coverage_start=None,
                coverage_end=None,
                cadence=SimpleNamespace(nominal=None, unique_intervals=[]),
                height_correction=None,
            )
            self.synchronisation = SimpleNamespace(
                matched_rows=3,
                unmatched_ann_rows=1,
                unmatched_buoy_rows=0,
                exact_matches=3,
                nearest_matches=0,
                match_ratio_ann=None,
                match_ratio_buoy=None,
            )
            self.matched_dataframe = []

    def fake_prepare_buoy_timeseries(*, buoy_dataset, ann_dataset, node_id, **kwargs):
        captured_args["buoy_dataset"] = buoy_dataset
        captured_args["ann_dataset"] = ann_dataset
        captured_args["node_id"] = node_id
        captured_args["kwargs"] = kwargs
        return DummyResult()

    def fake_resolve_catalog_asset(dataset, *, config_path, root, **kwargs):
        assert dataset == module.DEFAULT_STAC_DATASET
        assert root == module.REPO_ROOT
        assert config_path == (module.REPO_ROOT / module.DEFAULT_STAC_CONFIG).resolve()
        return SimpleNamespace(require_local_path=lambda: dummy_ann_path)

    monkeypatch.setattr(module, "prepare_buoy_timeseries", fake_prepare_buoy_timeseries)
    monkeypatch.setattr(module, "resolve_catalog_asset", fake_resolve_catalog_asset)

    buoy_dataset = tmp_path / "buoy.parquet"

    argv = [
        "prepare_buoy_timeseries.py",
        "--buoy-dataset",
        str(buoy_dataset),
        "--node-id",
        "Vilano_buoy",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    assert captured_args["buoy_dataset"] == buoy_dataset
    assert captured_args["ann_dataset"] == dummy_ann_path
    assert captured_args["node_id"] == "Vilano_buoy"
