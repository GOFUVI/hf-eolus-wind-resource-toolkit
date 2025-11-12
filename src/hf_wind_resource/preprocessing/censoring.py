"""Range-label partitioning and censoring diagnostics.

This module converts the ANN classifier outputs into structures that are
useful for censored statistical modelling. It normalises the range labels
(`below`, `in`, `above`, `uncertain`), keeps per-node counts and proportions
for left/right censoring, validates the physical thresholds defined in
``docs/sar_range_final_schema.md`` (lower/upper bounds configurable via
``config/range_thresholds.json``), and records discrepancies that should be
surfaced during audits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Tuple

import pandas as pd

__all__ = [
    "RangeThresholds",
    "load_range_thresholds",
    "RangeDiscrepancy",
    "NodeRangeSummary",
    "RangePartitioningResult",
    "partition_range_labels",
]


_CANONICAL_LABELS = ("below", "in", "above", "uncertain")
_LABEL_ALIASES = {
    "inside": "in",
    "within": "in",
    "in_range": "in",
    "below_range": "below",
    "under": "below",
    "upper": "above",
    "over": "above",
    "right": "above",
    "left": "below",
    "unknown": "uncertain",
    "nan": "uncertain",
}


_DEFAULT_THRESHOLD_PATH = Path("config") / "range_thresholds.json"


@dataclass(frozen=True)
class RangeThresholds:
    """Physical thresholds delimiting the valid ANN regression range."""

    lower: float = 5.7
    upper: float = 17.8

    def __post_init__(self) -> None:
        if not self.lower < self.upper:
            raise ValueError("The lower threshold must be smaller than the upper threshold.")


def load_range_thresholds(path: Path | str | None = None) -> RangeThresholds:
    """Load range thresholds from JSON configuration.

    Parameters
    ----------
    path:
        Optional path to the configuration file. Defaults to
        ``config/range_thresholds.json``.

    Returns
    -------
    RangeThresholds
        Thresholds declared in the configuration or the dataclass defaults
        when the file is missing.
    """

    target_path = Path(path) if path is not None else _DEFAULT_THRESHOLD_PATH
    default = RangeThresholds()

    if not target_path.exists():
        return default

    try:
        payload = json.loads(target_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in range-threshold configuration: {target_path}") from exc

    lower = float(payload.get("lower", default.lower))
    upper = float(payload.get("upper", default.upper))
    return RangeThresholds(lower=lower, upper=upper)


@dataclass(frozen=True)
class RangeDiscrepancy:
    """Issue detected while validating range labels against speed thresholds."""

    node_id: str
    timestamp: datetime | None
    issue: str
    reported_label: str | None
    expected_label: str | None
    deterministic_label: str | None
    wind_speed: float | None


@dataclass(frozen=True)
class NodeRangeSummary:
    """Aggregated counts, proportions, and notes for a node."""

    node_id: str
    total_observations: int
    left_censored_count: int
    left_censored_ratio: float | None
    in_range_count: int
    in_range_ratio: float | None
    right_censored_count: int
    right_censored_ratio: float | None
    uncertain_count: int
    uncertain_ratio: float | None
    discrepancy_count: int
    notes: Tuple[str, ...]


@dataclass(frozen=True)
class RangePartitioningResult:
    """Return type produced by :func:`partition_range_labels`."""

    below: pd.DataFrame
    in_range: pd.DataFrame
    above: pd.DataFrame
    uncertain: pd.DataFrame
    per_node: Mapping[str, NodeRangeSummary]
    discrepancies: Tuple[RangeDiscrepancy, ...]


def _normalise_label(raw: pd.Series) -> pd.Series:
    """Normalise textual labels to the canonical set."""

    series = raw.astype("string").str.strip().str.lower()
    series = series.fillna("uncertain")
    series = series.replace(_LABEL_ALIASES)
    series = series.where(series.isin(_CANONICAL_LABELS), "uncertain")
    return series


def _compute_expected_labels(
    speeds: pd.Series,
    thresholds: RangeThresholds,
) -> pd.Series:
    """Infer the label that should apply given the physical thresholds."""

    expected = pd.Series("uncertain", index=speeds.index, dtype="string")
    valid_mask = speeds.notna()
    expected.loc[valid_mask] = "in"
    expected.loc[valid_mask & (speeds <= thresholds.lower)] = "below"
    expected.loc[valid_mask & (speeds >= thresholds.upper)] = "above"
    return expected


def _to_python_datetime(value: pd.Timestamp | None) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    # ``to_pydatetime`` already returns timezone-aware datetimes when applicable.
    return value.to_pydatetime()


def _safe_ratio(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return count / total


def _aggregate_counts(
    nodes: pd.Series,
    labels: pd.Series,
) -> pd.DataFrame:
    frame = (
        pd.DataFrame({"node_id": nodes, "_label": labels})
        .groupby(["node_id", "_label"])
        .size()
        .unstack(fill_value=0)
    )
    for label in _CANONICAL_LABELS:
        if label not in frame.columns:
            frame[label] = 0
    return frame.reindex(columns=_CANONICAL_LABELS, fill_value=0)


def partition_range_labels(
    frame: pd.DataFrame,
    *,
    node_column: str = "node_id",
    label_column: str = "pred_range_label",
    speed_column: str = "pred_wind_speed",
    timestamp_column: str = "timestamp",
    deterministic_label_column: str | None = "pred_speed_range_label",
    thresholds: RangeThresholds | None = None,
) -> RangePartitioningResult:
    """Split samples by range label and collect censoring diagnostics.

    Parameters
    ----------
    frame:
        DataFrame following the ANN inference schema.
    node_column:
        Column holding node identifiers.
    label_column:
        Column with classifier range labels (below/in/above).
    speed_column:
        Column with predicted wind speeds in m/s.
    timestamp_column:
        Column with sample timestamps (UTC recommended).
    deterministic_label_column:
        Optional column that deterministically maps wind speed to a range
        label (``pred_speed_range_label`` in the ANN schema). When provided,
        discrepancies between this value and the physical thresholds will be
        reported.
    thresholds:
        Optional :class:`RangeThresholds`. Defaults to the values declared in
        ``config/range_thresholds.json`` (falling back to the physical range
        documented in ``docs/sar_range_final_schema.md``).

    Returns
    -------
    RangePartitioningResult
        Partitioned DataFrames alongside per-node censoring summaries and
        recorded discrepancies.
    """

    if thresholds is None:
        thresholds = load_range_thresholds()

    required_columns = {node_column, label_column, speed_column, timestamp_column}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise KeyError(f"Frame is missing required columns: {sorted(missing)}")

    df = frame.copy(deep=True)
    nodes = df[node_column].astype("string").fillna("").astype(str)
    raw_labels = df[label_column]
    normalised_labels = _normalise_label(raw_labels)
    speeds = pd.to_numeric(df[speed_column], errors="coerce")
    timestamps = pd.to_datetime(df[timestamp_column], utc=True, errors="coerce")
    expected_labels = _compute_expected_labels(speeds, thresholds)

    deterministic_labels: pd.Series | None = None
    if deterministic_label_column and deterministic_label_column in df.columns:
        deterministic_labels = _normalise_label(df[deterministic_label_column])

    discrepancies: list[RangeDiscrepancy] = []

    # Missing speeds paired with deterministic labels require attention.
    missing_speed_mask = speeds.isna()
    informative_label_mask = normalised_labels.isin({"below", "in", "above"})
    for idx in df.index[missing_speed_mask & informative_label_mask]:
        discrepancies.append(
            RangeDiscrepancy(
                node_id=str(nodes.at[idx]),
                timestamp=_to_python_datetime(timestamps.at[idx]),
                issue="missing_speed_value",
                reported_label=str(normalised_labels.at[idx]),
                expected_label=None,
                deterministic_label=(
                    str(deterministic_labels.at[idx]) if deterministic_labels is not None else None
                ),
                wind_speed=None,
            )
        )

    # Classifier vs physical thresholds.
    classifier_mismatch_mask = informative_label_mask & (normalised_labels != expected_labels)
    for idx in df.index[classifier_mismatch_mask]:
        discrepancies.append(
            RangeDiscrepancy(
                node_id=str(nodes.at[idx]),
                timestamp=_to_python_datetime(timestamps.at[idx]),
                issue="classifier_vs_threshold",
                reported_label=str(normalised_labels.at[idx]),
                expected_label=str(expected_labels.at[idx]),
                deterministic_label=(
                    str(deterministic_labels.at[idx]) if deterministic_labels is not None else None
                ),
                wind_speed=float(speeds.at[idx]) if pd.notna(speeds.at[idx]) else None,
            )
        )

    # Uncertain labels whose speed clearly belongs to a censored region.
    uncertain_mask = (normalised_labels == "uncertain") & expected_labels.isin({"below", "above"})
    for idx in df.index[uncertain_mask]:
        discrepancies.append(
            RangeDiscrepancy(
                node_id=str(nodes.at[idx]),
                timestamp=_to_python_datetime(timestamps.at[idx]),
                issue="uncertain_vs_threshold",
                reported_label="uncertain",
                expected_label=str(expected_labels.at[idx]),
                deterministic_label=(
                    str(deterministic_labels.at[idx]) if deterministic_labels is not None else None
                ),
                wind_speed=float(speeds.at[idx]) if pd.notna(speeds.at[idx]) else None,
            )
        )

    # Deterministic label vs thresholds.
    if deterministic_labels is not None:
        deterministic_mask = deterministic_labels.isin({"below", "in", "above"})
        deterministic_mismatch = deterministic_mask & (deterministic_labels != expected_labels)
        for idx in df.index[deterministic_mismatch]:
            discrepancies.append(
                RangeDiscrepancy(
                    node_id=str(nodes.at[idx]),
                    timestamp=_to_python_datetime(timestamps.at[idx]),
                    issue="deterministic_vs_threshold",
                    reported_label=str(normalised_labels.at[idx]),
                    expected_label=str(expected_labels.at[idx]),
                    deterministic_label=str(deterministic_labels.at[idx]),
                    wind_speed=float(speeds.at[idx]) if pd.notna(speeds.at[idx]) else None,
                )
            )

    partition_below = df.loc[normalised_labels == "below"].copy()
    partition_in = df.loc[normalised_labels == "in"].copy()
    partition_above = df.loc[normalised_labels == "above"].copy()
    partition_uncertain = df.loc[normalised_labels == "uncertain"].copy()

    counts = _aggregate_counts(nodes, normalised_labels)

    discrepancy_counter: Dict[str, int] = {}
    for record in discrepancies:
        discrepancy_counter[record.node_id] = discrepancy_counter.get(record.node_id, 0) + 1

    notes_by_node: Dict[str, list[str]] = {str(node): [] for node in counts.index}

    # Notes about uncertain/unknown labels.
    uncertain_original = (
        normalised_labels.eq("uncertain")
        & raw_labels.astype("string").str.strip().str.lower().ne("uncertain")
    )
    uncertain_counts = nodes[uncertain_original].value_counts()
    for node_id, value in uncertain_counts.items():
        if node_id in notes_by_node:
            notes_by_node[node_id].append(
                f"{int(value)} samples lack a definitive range label."
            )

    # Notes about missing speeds.
    missing_counts = nodes[missing_speed_mask].value_counts()
    for node_id, value in missing_counts.items():
        if node_id in notes_by_node and int(value) > 0:
            notes_by_node[node_id].append(
                f"{int(value)} samples have missing wind_speed values."
            )

    summaries: Dict[str, NodeRangeSummary] = {}
    for node, row in counts.iterrows():
        node_id = str(node)
        total = int(row.sum())
        below_count = int(row["below"])
        in_count = int(row["in"])
        above_count = int(row["above"])
        uncertain_count = int(row["uncertain"])

        node_notes = notes_by_node.get(node_id, []).copy()
        if uncertain_count:
            node_notes.append(f"{uncertain_count} samples marked as uncertain.")
        discrepancy_count = discrepancy_counter.get(node_id, 0)
        if discrepancy_count:
            node_notes.append(f"{discrepancy_count} threshold discrepancies detected.")

        summaries[node_id] = NodeRangeSummary(
            node_id=node_id,
            total_observations=total,
            left_censored_count=below_count,
            left_censored_ratio=_safe_ratio(below_count, total),
            in_range_count=in_count,
            in_range_ratio=_safe_ratio(in_count, total),
            right_censored_count=above_count,
            right_censored_ratio=_safe_ratio(above_count, total),
            uncertain_count=uncertain_count,
            uncertain_ratio=_safe_ratio(uncertain_count, total),
            discrepancy_count=discrepancy_count,
            notes=tuple(sorted(set(node_notes))),
        )

    return RangePartitioningResult(
        below=partition_below,
        in_range=partition_in,
        above=partition_above,
        uncertain=partition_uncertain,
        per_node=summaries,
        discrepancies=tuple(discrepancies),
    )
