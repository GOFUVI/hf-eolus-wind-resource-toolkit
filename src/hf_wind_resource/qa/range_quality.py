"""Range-label QA evaluation producing structured per-node assessments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import pandas as pd

from hf_wind_resource.preprocessing.censoring import NodeRangeSummary

__all__ = [
    "NodeRangeQaStatus",
    "RangeQualityAssessment",
    "RangeQaThresholds",
    "TemporalQaMetrics",
    "evaluate_range_quality",
    "load_range_qa_thresholds",
]

_DEFAULT_THRESHOLD_PATH = Path("config") / "range_quality_thresholds.json"


def _safe_ratio(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return count / total


@dataclass(frozen=True)
class RangeQaThresholds:
    """Configurable thresholds steering the QA assessment."""

    min_in_range_count: int = 500
    min_in_range_ratio: float = 0.2
    max_total_censored_ratio: float = 0.8
    max_left_censored_ratio: float = 0.75
    max_right_censored_ratio: float = 0.25
    max_uncertain_ratio: float = 0.1
    min_temporal_coverage: float = 0.65
    max_gap_hours: float = 720.0

    def __post_init__(self) -> None:
        if self.min_in_range_count < 0:
            raise ValueError("min_in_range_count must be non-negative.")
        for attr in (
            "min_in_range_ratio",
            "max_total_censored_ratio",
            "max_left_censored_ratio",
            "max_right_censored_ratio",
            "max_uncertain_ratio",
            "min_temporal_coverage",
        ):
            value = getattr(self, attr)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{attr} must lie within [0, 1]; received {value!r}.")
        if self.max_gap_hours < 0.0:
            raise ValueError("max_gap_hours must be non-negative.")


def load_range_qa_thresholds(path: str | Path | None = None) -> RangeQaThresholds:
    """Load QA thresholds from JSON configuration."""

    target_path = Path(path) if path is not None else _DEFAULT_THRESHOLD_PATH
    if not target_path.exists():
        return RangeQaThresholds()

    try:
        payload = json.loads(target_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in QA-threshold configuration: {target_path}") from exc

    return RangeQaThresholds(
        min_in_range_count=int(payload.get("min_in_range_count", 500)),
        min_in_range_ratio=float(payload.get("min_in_range_ratio", 0.2)),
        max_total_censored_ratio=float(payload.get("max_total_censored_ratio", 0.8)),
        max_left_censored_ratio=float(payload.get("max_left_censored_ratio", 0.75)),
        max_right_censored_ratio=float(payload.get("max_right_censored_ratio", 0.25)),
        max_uncertain_ratio=float(payload.get("max_uncertain_ratio", 0.1)),
        min_temporal_coverage=float(payload.get("min_temporal_coverage", 0.65)),
        max_gap_hours=float(payload.get("max_gap_hours", 720.0)),
    )


@dataclass(frozen=True)
class TemporalQaMetrics:
    """Temporal density indicators used by the QA assessment."""

    node_id: str
    coverage_ratio: float | None
    expected_observations: int | None
    distinct_observations: int
    duplicate_records: int
    max_gap_hours: float | None

    @property
    def has_duplicates(self) -> bool:
        """Return True when duplicated timestamps were detected."""

        return self.duplicate_records > 0


@dataclass(frozen=True)
class NodeRangeQaStatus:
    """Per-node QA outcome combining range and temporal diagnostics."""

    node_id: str
    total_observations: int
    in_range_count: int
    in_range_ratio: float | None
    below_count: int
    below_ratio: float | None
    above_count: int
    above_ratio: float | None
    uncertain_count: int
    uncertain_ratio: float | None
    coverage_ratio: float | None
    max_gap_hours: float | None
    duplicate_records: int
    flags: Tuple[str, ...]
    parametric_reliable: bool
    reliability_reasons: Tuple[str, ...]
    notes: Tuple[str, ...]

    @property
    def total_censored_ratio(self) -> float | None:
        """Return the combined left/right censored ratio."""

        censored_total = self.below_count + self.above_count
        return _safe_ratio(censored_total, self.total_observations)


@dataclass(frozen=True)
class RangeQualityAssessment:
    """Container holding the QA assessment for all nodes."""

    per_node: Mapping[str, NodeRangeQaStatus]
    thresholds: RangeQaThresholds

    def to_records(self) -> Tuple[dict[str, object], ...]:
        """Convert the assessment into serialisable dictionaries."""

        records: list[dict[str, object]] = []
        for status in sorted(self.per_node.values(), key=lambda item: item.node_id):
            record = {
                "node_id": status.node_id,
                "total_observations": status.total_observations,
                "in_range_count": status.in_range_count,
                "in_range_ratio": status.in_range_ratio,
                "below_count": status.below_count,
                "below_ratio": status.below_ratio,
                "above_count": status.above_count,
                "above_ratio": status.above_ratio,
                "uncertain_count": status.uncertain_count,
                "uncertain_ratio": status.uncertain_ratio,
                "coverage_ratio": status.coverage_ratio,
                "max_gap_hours": status.max_gap_hours,
                "duplicate_records": status.duplicate_records,
                "flags": list(status.flags),
                "parametric_reliable": status.parametric_reliable,
                "reliability_reasons": list(status.reliability_reasons),
                "notes": list(status.notes),
            }
            record["total_censored_ratio"] = status.total_censored_ratio
            records.append(record)
        return tuple(records)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the assessment into a pandas DataFrame."""

        records = []
        for status in sorted(self.per_node.values(), key=lambda item: item.node_id):
            records.append(
                {
                    "node_id": status.node_id,
                    "total_observations": status.total_observations,
                    "in_range_count": status.in_range_count,
                    "in_range_ratio": status.in_range_ratio,
                    "below_count": status.below_count,
                    "below_ratio": status.below_ratio,
                    "above_count": status.above_count,
                    "above_ratio": status.above_ratio,
                    "uncertain_count": status.uncertain_count,
                    "uncertain_ratio": status.uncertain_ratio,
                    "total_censored_ratio": status.total_censored_ratio,
                    "coverage_ratio": status.coverage_ratio,
                    "max_gap_hours": status.max_gap_hours,
                    "duplicate_records": status.duplicate_records,
                    "parametric_reliable": status.parametric_reliable,
                    "flags": ";".join(status.flags),
                    "reliability_reasons": ";".join(status.reliability_reasons),
                    "notes": " | ".join(status.notes),
                }
            )
        return pd.DataFrame.from_records(records)


def evaluate_range_quality(
    range_summaries: Mapping[str, NodeRangeSummary],
    temporal_metrics: Mapping[str, TemporalQaMetrics] | None = None,
    *,
    thresholds: RangeQaThresholds | None = None,
) -> RangeQualityAssessment:
    """Evaluate QA rules for the provided range and temporal summaries."""

    if thresholds is None:
        thresholds = load_range_qa_thresholds()

    temporal_metrics = temporal_metrics or {}
    statuses: Dict[str, NodeRangeQaStatus] = {}

    for node_id, summary in range_summaries.items():
        total = summary.total_observations
        below_count = summary.left_censored_count
        above_count = summary.right_censored_count
        in_count = summary.in_range_count
        uncertain_count = summary.uncertain_count

        left_ratio = summary.left_censored_ratio
        right_ratio = summary.right_censored_ratio
        in_ratio = summary.in_range_ratio
        uncertain_ratio = summary.uncertain_ratio

        if left_ratio is None:
            left_ratio = _safe_ratio(below_count, total)
        if right_ratio is None:
            right_ratio = _safe_ratio(above_count, total)
        if in_ratio is None:
            in_ratio = _safe_ratio(in_count, total)
        if uncertain_ratio is None:
            uncertain_ratio = _safe_ratio(uncertain_count, total)

        total_censored_ratio = _safe_ratio(below_count + above_count, total)

        node_temporal = temporal_metrics.get(node_id)
        coverage_ratio = node_temporal.coverage_ratio if node_temporal else None
        max_gap_hours = node_temporal.max_gap_hours if node_temporal else None
        duplicate_records = node_temporal.duplicate_records if node_temporal else 0

        flags: list[str] = []
        notes: list[str] = list(summary.notes)

        if total_censored_ratio is not None and total_censored_ratio > thresholds.max_total_censored_ratio:
            flags.append("excessive_total_censoring")
        if left_ratio is not None and left_ratio > thresholds.max_left_censored_ratio:
            flags.append("excessive_left_censoring")
        if right_ratio is not None and right_ratio > thresholds.max_right_censored_ratio:
            flags.append("excessive_right_censoring")
        if in_ratio is None or in_ratio < thresholds.min_in_range_ratio:
            flags.append("low_in_range_share")
        if uncertain_ratio is not None and uncertain_ratio > thresholds.max_uncertain_ratio:
            flags.append("excessive_uncertain_share")
        if coverage_ratio is not None and coverage_ratio < thresholds.min_temporal_coverage:
            flags.append("insufficient_temporal_coverage")
        if max_gap_hours is not None and max_gap_hours > thresholds.max_gap_hours:
            flags.append("temporal_gap_exceeds_threshold")
        if duplicate_records > 0:
            flags.append("duplicate_timestamps_detected")
            notes.append(f"{duplicate_records} duplicate timestamps detected.")

        if node_temporal and node_temporal.expected_observations is not None:
            missing = node_temporal.expected_observations - node_temporal.distinct_observations
            if missing > 0:
                notes.append(f"{missing} expected timestamps missing from the nominal cadence.")

        parametric_reliable = True
        reliability_reasons: list[str] = []

        if in_count < thresholds.min_in_range_count:
            parametric_reliable = False
            reliability_reasons.append(
                f"in-range sample count {in_count} below minimum {thresholds.min_in_range_count}"
            )
        if in_ratio is None or in_ratio < thresholds.min_in_range_ratio:
            parametric_reliable = False
            ratio_repr = "undefined" if in_ratio is None else f"{in_ratio:.3f}"
            reliability_reasons.append(
                f"in-range ratio {ratio_repr} below minimum {thresholds.min_in_range_ratio}"
            )
        if coverage_ratio is not None and coverage_ratio < thresholds.min_temporal_coverage:
            parametric_reliable = False
            reliability_reasons.append(
                f"temporal coverage {coverage_ratio:.3f} below minimum {thresholds.min_temporal_coverage}"
            )

        statuses[node_id] = NodeRangeQaStatus(
            node_id=node_id,
            total_observations=total,
            in_range_count=in_count,
            in_range_ratio=in_ratio,
            below_count=below_count,
            below_ratio=left_ratio,
            above_count=above_count,
            above_ratio=right_ratio,
            uncertain_count=uncertain_count,
            uncertain_ratio=uncertain_ratio,
            coverage_ratio=coverage_ratio,
            max_gap_hours=max_gap_hours,
            duplicate_records=duplicate_records,
            flags=tuple(dict.fromkeys(flags)),
            parametric_reliable=parametric_reliable,
            reliability_reasons=tuple(reliability_reasons),
            notes=tuple(dict.fromkeys(notes)),
        )

    return RangeQualityAssessment(per_node=statuses, thresholds=thresholds)
