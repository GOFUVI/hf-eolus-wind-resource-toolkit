"""Tests for IOFilters configuration handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hf_wind_resource.io import IOFilters, load_range_flag_threshold


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Ensure each test runs with a clean threshold cache."""

    load_range_flag_threshold.cache_clear()


def test_load_range_flag_threshold_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / "thresholds.json"
    config_path.write_text(json.dumps({"range_flag_threshold": 0.62}), encoding="utf-8")

    threshold = load_range_flag_threshold(config_path)
    assert threshold == pytest.approx(0.62)


def test_load_range_flag_threshold_alias(tmp_path: Path) -> None:
    config_path = tmp_path / "thresholds.json"
    payload = {"classifier_confidence_threshold": "0.7"}
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    threshold = load_range_flag_threshold(config_path)
    assert threshold == pytest.approx(0.7)


def test_load_range_flag_threshold_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.json"

    threshold = load_range_flag_threshold(missing_path)
    assert threshold == pytest.approx(0.5)


def test_load_range_flag_threshold_invalid_range(tmp_path: Path) -> None:
    config_path = tmp_path / "thresholds.json"
    config_path.write_text(json.dumps({"range_flag_threshold": 1.5}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_range_flag_threshold(config_path)


def test_iofilters_resolved_min_confidence_uses_config(tmp_path: Path) -> None:
    config_path = tmp_path / "thresholds.json"
    config_path.write_text(json.dumps({"range_flag_threshold": 0.66}), encoding="utf-8")

    filters = IOFilters(require_in_range=True)
    threshold = filters.resolved_min_confidence(config_path=config_path)
    assert threshold == pytest.approx(0.66)


def test_iofilters_resolved_min_confidence_prefers_explicit_value(tmp_path: Path) -> None:
    config_path = tmp_path / "thresholds.json"
    config_path.write_text(json.dumps({"range_flag_threshold": 0.3}), encoding="utf-8")

    filters = IOFilters(require_in_range=True, min_confidence=0.8)
    threshold = filters.resolved_min_confidence(config_path=config_path)
    assert threshold == pytest.approx(0.8)
