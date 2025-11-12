"""Empirical wind-speed metrics and bias diagnostics without third-party deps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import json

__all__ = [
    "BiasThresholds",
    "assemble_per_node_metrics",
    "compute_label_ratios",
    "load_taxonomy_records",
]


LABEL_KEYS = ("in_count", "below_count", "above_count", "uncertain_count")


@dataclass(frozen=True)
class BiasThresholds:
    """Thresholds applied when flagging potential bias."""

    max_total_censored_ratio: float = 0.20
    max_below_ratio: float = 0.15
    min_valid_samples: int = 500
    min_valid_share: float = 0.40


def load_taxonomy_records(path: str | Path) -> Dict[str, Mapping[str, object]]:
    """Return taxonomy entries indexed by ``node_id``."""

    payload_path = Path(path)
    data = json.loads(payload_path.read_text(encoding="utf-8"))
    nodes: Sequence[Mapping[str, object]] = data.get("nodes", [])  # type: ignore[assignment]
    if not nodes:
        raise ValueError(f"No taxonomy nodes found in {payload_path}")
    return {str(item["node_id"]): item for item in nodes if "node_id" in item}


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def compute_label_ratios(label_rows: Iterable[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Add ratio columns to the label-count dictionaries."""

    enriched: List[Dict[str, object]] = []
    for row in label_rows:
        node_id = str(row.get("node_id"))
        total = float(row.get("total_observations", 0) or 0)
        record = dict(row)
        for key in LABEL_KEYS:
            count = float(row.get(key, 0) or 0)
            ratio = _safe_divide(count, total)
            ratio_key = key.replace("_count", "_ratio")
            record[ratio_key] = ratio
        below_ratio = record.get("below_ratio") or 0.0
        above_ratio = record.get("above_ratio") or 0.0
        record["censored_ratio"] = (below_ratio or 0.0) + (above_ratio or 0.0)
        record["node_id"] = node_id
        enriched.append(record)
    return enriched


def assemble_per_node_metrics(
    valid_metrics: Iterable[Mapping[str, object]],
    label_counts: Iterable[Mapping[str, object]],
    taxonomy: Mapping[str, Mapping[str, object]],
    *,
    thresholds: BiasThresholds | None = None,
) -> List[Dict[str, object]]:
    """Merge metrics, label ratios, and taxonomy diagnostics."""

    if thresholds is None:
        thresholds = BiasThresholds()

    metrics_by_node: Dict[str, Mapping[str, object]] = {
        str(row.get("node_id")): row for row in valid_metrics if "node_id" in row
    }
    ratios = compute_label_ratios(label_counts)

    summaries: List[Dict[str, object]] = []
    for ratio_row in ratios:
        node_id = str(ratio_row.get("node_id"))
        summary: Dict[str, object] = dict(ratio_row)
        metric_row = metrics_by_node.get(node_id, {})
        taxonomy_row = taxonomy.get(node_id, {})

        summary.update(metric_row)
        summary["taxonomy_observations"] = taxonomy_row.get("total_observations")
        summary["coverage_band"] = taxonomy_row.get("coverage_band")
        summary["continuity_band"] = taxonomy_row.get("continuity_band")
        summary["low_coverage"] = bool(taxonomy_row.get("low_coverage", False))

        valid_count = int(metric_row.get("valid_count", 0) or 0)
        summary["valid_count"] = valid_count

        in_ratio = summary.get("in_ratio")
        if in_ratio is None:
            in_count = float(summary.get("in_count", 0) or 0)
            total_observations = float(summary.get("total_observations", 0) or 0)
            summary["in_ratio"] = _safe_divide(in_count, total_observations)

        notes: List[str] = []
        censored_ratio = float(summary.get("censored_ratio") or 0.0)
        below_ratio = float(summary.get("below_ratio") or 0.0)
        in_share = float(summary.get("in_ratio") or 0.0)

        censoring_flag = False
        if censored_ratio is not None and censored_ratio > thresholds.max_total_censored_ratio:
            notes.append(
                f"High censoring ratio {censored_ratio:.3f} exceeds {thresholds.max_total_censored_ratio:.2f}."
            )
            censoring_flag = True
        if below_ratio is not None and below_ratio > thresholds.max_below_ratio:
            notes.append(
                f"Left-censored share {below_ratio:.3f} above {thresholds.max_below_ratio:.2f}."
            )
            censoring_flag = True
        if in_share is not None and in_share < thresholds.min_valid_share:
            notes.append(
                f"In-range share {in_share:.3f} below {thresholds.min_valid_share:.2f}."
            )
            censoring_flag = True

        coverage_flag = False
        if summary.get("low_coverage"):
            notes.append("Marked as low coverage in taxonomy.")
            coverage_flag = True
        if isinstance(summary.get("coverage_band"), str) and str(summary["coverage_band"]).lower() == "sparse":
            notes.append("Coverage band classified as sparse.")
            coverage_flag = True

        sample_flag = False
        if valid_count < thresholds.min_valid_samples:
            notes.append(
                f"Valid sample count {valid_count} below minimum {thresholds.min_valid_samples}."
            )
            sample_flag = True

        summary["censoring_bias"] = censoring_flag
        summary["coverage_bias"] = coverage_flag
        summary["sample_bias"] = sample_flag
        summary["bias_notes"] = "; ".join(notes)
        summary["any_bias"] = censoring_flag or coverage_flag or sample_flag

        summaries.append(summary)

    summaries.sort(key=lambda row: row.get("node_id", ""))
    return summaries

