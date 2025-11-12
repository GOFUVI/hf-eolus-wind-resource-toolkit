"""Tests for the global RMSE provider and taxonomy integration."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hf_wind_resource.stats import GlobalRmseProvider, GlobalRmseRecord


def test_default_loader_exposes_expected_global_rmse() -> None:
    """The bundled RMSE record should match the Vilano buoy analysis."""

    provider = GlobalRmseProvider()
    record = provider.get_global_rmse()

    assert record.value == pytest.approx(2.802399)
    assert record.unit == "m/s"
    assert record.source == "docs/sar_inference_on_vilano_10m_all.md:42"
    assert record.version.startswith("2025-10-18_vilano_global_rmse")
    assert "classification-matched" in " ".join(record.notes)


def test_node_assessment_reports_limitation_and_taxonomy() -> None:
    """Nodes should return taxonomy metadata and a limitation message."""

    provider = GlobalRmseProvider()
    assessment = provider.get_node_assessment("VILA_PRIO11")

    assert assessment.rmse is None
    assert "global RMSE" in assessment.limitation
    assert assessment.taxonomy.total_observations == 814
    assert assessment.taxonomy.coverage_band == "sparse"
    assert assessment.taxonomy.continuity_band in {"extreme_gaps", "long_gaps"}
    assert assessment.taxonomy.is_low_coverage is True


def test_refresh_allows_loader_to_publish_new_versions() -> None:
    """Refreshing should pick up records exposed by the user-supplied loader."""

    timestamp_v1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamp_v2 = datetime(2024, 6, 1, tzinfo=timezone.utc)

    record_v1 = GlobalRmseRecord(
        version="v1",
        value=3.1,
        unit="m/s",
        effective_from=timestamp_v1,
        effective_until=timestamp_v2,
        source="local://test",
        computed_at=timestamp_v1,
        notes=("first draft",),
    )
    record_v2 = GlobalRmseRecord(
        version="v2",
        value=2.9,
        unit="m/s",
        effective_from=timestamp_v2,
        effective_until=None,
        source="local://test",
        computed_at=timestamp_v2,
        notes=("second draft",),
    )

    class MutableLoader:
        def __init__(self) -> None:
            self.records: tuple[GlobalRmseRecord, ...] = (record_v1,)

        def __call__(self) -> tuple[GlobalRmseRecord, ...]:
            return self.records

    loader = MutableLoader()
    provider = GlobalRmseProvider(loader=loader)

    assert provider.get_global_rmse(as_of=datetime(2024, 3, 1, tzinfo=timezone.utc)) == record_v1
    assert provider.version == "v1+0"

    loader.records = (record_v1, record_v2)
    provider.refresh()
    active = provider.get_global_rmse(as_of=datetime(2024, 7, 1, tzinfo=timezone.utc))

    assert active == record_v2
    assert provider.version == "v2+1"
