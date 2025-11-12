from __future__ import annotations

from hf_wind_resource.preprocessing.censoring import NodeRangeSummary
from hf_wind_resource.qa import (
    RangeQaThresholds,
    TemporalQaMetrics,
    evaluate_range_quality,
)


def _make_summary(
    *,
    node_id: str,
    total: int,
    below: int,
    in_range: int,
    above: int,
    uncertain: int,
    notes: tuple[str, ...] = (),
) -> NodeRangeSummary:
    return NodeRangeSummary(
        node_id=node_id,
        total_observations=total,
        left_censored_count=below,
        left_censored_ratio=below / total if total else None,
        in_range_count=in_range,
        in_range_ratio=in_range / total if total else None,
        right_censored_count=above,
        right_censored_ratio=above / total if total else None,
        uncertain_count=uncertain,
        uncertain_ratio=uncertain / total if total else None,
        discrepancy_count=0,
        notes=notes,
    )


def test_evaluate_range_quality_flags_and_reliability() -> None:
    thresholds = RangeQaThresholds(
        min_in_range_count=300,
        min_in_range_ratio=0.4,
        max_total_censored_ratio=0.6,
        max_left_censored_ratio=0.5,
        max_right_censored_ratio=0.2,
        max_uncertain_ratio=0.05,
        min_temporal_coverage=0.95,
        max_gap_hours=24,
    )

    range_summary = _make_summary(
        node_id="atlantic-001",
        total=1000,
        below=520,
        in_range=160,
        above=250,
        uncertain=70,
        notes=("legacy note",),
    )
    temporal_metrics = TemporalQaMetrics(
        node_id="atlantic-001",
        coverage_ratio=0.8,
        expected_observations=1200,
        distinct_observations=990,
        duplicate_records=10,
        max_gap_hours=36.0,
    )

    assessment = evaluate_range_quality(
        {"atlantic-001": range_summary},
        {"atlantic-001": temporal_metrics},
        thresholds=thresholds,
    )
    status = assessment.per_node["atlantic-001"]

    assert set(status.flags) == {
        "excessive_total_censoring",
        "excessive_left_censoring",
        "excessive_right_censoring",
        "low_in_range_share",
        "excessive_uncertain_share",
        "insufficient_temporal_coverage",
        "temporal_gap_exceeds_threshold",
        "duplicate_timestamps_detected",
    }
    assert status.parametric_reliable is False
    assert any("in-range sample count" in reason for reason in status.reliability_reasons)
    assert any("coverage" in reason for reason in status.reliability_reasons)
    assert any("legacy note" in note for note in status.notes)
    assert any("210 expected timestamps missing" in note for note in status.notes)
    assert any("duplicate timestamps detected" in note for note in status.notes)


def test_assessment_dataframe_serialises_flags() -> None:
    thresholds = RangeQaThresholds(
        min_in_range_count=0,
        min_in_range_ratio=0.0,
        min_temporal_coverage=0.0,
    )
    range_summary = _make_summary(
        node_id="atlantic-002",
        total=100,
        below=10,
        in_range=80,
        above=5,
        uncertain=5,
    )

    assessment = evaluate_range_quality({"atlantic-002": range_summary}, thresholds=thresholds)
    dataframe = assessment.to_dataframe()

    assert list(dataframe.columns) == [
        "node_id",
        "total_observations",
        "in_range_count",
        "in_range_ratio",
        "below_count",
        "below_ratio",
        "above_count",
        "above_ratio",
        "uncertain_count",
        "uncertain_ratio",
        "total_censored_ratio",
        "coverage_ratio",
        "max_gap_hours",
        "duplicate_records",
        "parametric_reliable",
        "flags",
        "reliability_reasons",
        "notes",
    ]
    assert dataframe.at[0, "node_id"] == "atlantic-002"
    assert dataframe.at[0, "flags"] == ""
    assert bool(dataframe.at[0, "parametric_reliable"])
