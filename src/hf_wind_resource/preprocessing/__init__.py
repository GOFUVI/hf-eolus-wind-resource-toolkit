"""Preprocessing utilities for temporal alignment and QA."""

from __future__ import annotations

from .node_alignment import (
    CadenceStats,
    NodeTemporalSummary,
    TemporalNormalizationConfig,
    TemporalNormalizationResult,
    TemporalQualityFlags,
    normalise_temporal_records,
)
from .buoy_timeseries import (
    BuoyPreparation,
    BuoySentinelConfig,
    BuoySeries,
    CadenceSummary,
    HeightCorrectionConfig,
    HeightCorrectionResult,
    load_height_correction_from_config,
    build_geoparquet_table,
    SynchronisationConfig,
    SynchronisationSummary,
    load_ann_node_timeseries,
    load_buoy_timeseries,
    prepare_buoy_timeseries,
    synchronise_buoy_and_ann,
)
from .censoring import (
    NodeRangeSummary,
    RangeDiscrepancy,
    RangePartitioningResult,
    RangeThresholds,
    load_range_thresholds,
    partition_range_labels,
)

__all__ = [
    "CadenceStats",
    "NodeTemporalSummary",
    "TemporalNormalizationConfig",
    "TemporalNormalizationResult",
    "TemporalQualityFlags",
    "normalise_temporal_records",
    "BuoyPreparation",
    "BuoySentinelConfig",
    "BuoySeries",
    "CadenceSummary",
    "HeightCorrectionConfig",
    "HeightCorrectionResult",
    "load_height_correction_from_config",
    "build_geoparquet_table",
    "SynchronisationConfig",
    "SynchronisationSummary",
    "load_ann_node_timeseries",
    "load_buoy_timeseries",
    "prepare_buoy_timeseries",
    "synchronise_buoy_and_ann",
    "NodeRangeSummary",
    "RangeDiscrepancy",
    "RangePartitioningResult",
    "RangeThresholds",
    "load_range_thresholds",
    "partition_range_labels",
]
