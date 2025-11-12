"""IO contracts for the SAR range-aware wind-resource workflow.

This module gathers abstract data-access primitives that all readers of the
`sar_range_final_pivots_joined` GeoParquet snapshot must honour. The schema
is documented in detail in ``docs/sar_range_final_schema.md`` and the package
layout in ``docs/python_architecture.md`` explains where concrete
implementations will live. The intent here is to lock the public API so that
the rest of the stack (preprocessing, statistics, CLI) can rely on stable,
chunk-friendly access patterns even when storage backends evolve.

Performance and memory assumptions
----------------------------------
* Reading is expected to happen in 50k–200k row chunks so that peak memory
  footprint stays below ~1.5 GiB when each chunk is materialised as a pandas
  ``DataFrame`` alongside auxiliary arrays (assuming ≈160 columns of float
  and integer data). Adapt the ``PerformanceBudget`` parameters when working
  in tighter environments.
* The data lives in a single GeoParquet file partitioned logically by
  ``node_id`` and ``timestamp``. Filtering should therefore push predicates
  down to DuckDB or PyArrow to avoid scanning the whole file when only a
  subset of nodes or time windows is required.
* Callers may opt into an ephemeral on-disk cache (e.g. using ``sqlite3`` or
  ``diskcache``) to amortise repeated scans. The cache contract is defined in
  :class:`ChunkCache`.
* Gaps in the nominal 30-minute cadence must be logged through the anomaly
  sink so downstream statistics can decide whether to interpolate or discard
  affected windows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import (
    Iterator,
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
    TYPE_CHECKING,
)

from .stac import (
    ResolvedStacAsset,
    StacCatalogEntry,
    load_catalog_index,
    resolve_ann_asset,
    resolve_catalog_asset,
)

if TYPE_CHECKING:  # pragma: no cover - used purely for typing
    import pandas as pd
    from hf_wind_resource.io import schema_registry  # local dependency

    DataFrameType = pd.DataFrame
    SchemaType = schema_registry.SarRangeFinalSchema
else:
    DataFrameType = object
    SchemaType = object

__all__ = [
    "AnomalyEvent",
    "AnomalySink",
    "CachePolicy",
    "ChunkCache",
    "ChunkMetadata",
    "ChunkReadPlan",
    "GapDescriptor",
    "GapReport",
    "IOFilters",
    "load_range_flag_threshold",
    "NodeFilter",
    "PerformanceBudget",
    "TimeWindow",
    "build_chunk_reader",
    "ResolvedStacAsset",
    "StacCatalogEntry",
    "load_catalog_index",
    "resolve_catalog_asset",
    "resolve_ann_asset",
]

_DEFAULT_FLAG_THRESHOLD = 0.5
_RANGE_THRESHOLD_CONFIG = Path("config") / "range_thresholds.json"


@lru_cache(maxsize=8)
def load_range_flag_threshold(path: str | Path | None = None) -> float:
    """Return the classifier confidence threshold used for range gating.

    Parameters
    ----------
    path:
        Optional location of the configuration file. Defaults to
        ``config/range_thresholds.json`` when omitted.

    Returns
    -------
    float
        The configured confidence threshold or the default (0.5) when the
        configuration is absent.

    Raises
    ------
    ValueError
        If the configuration file exists but cannot be decoded as JSON or
        defines a value outside the [0, 1] interval.
    """

    target_path = Path(path) if path is not None else _RANGE_THRESHOLD_CONFIG
    if not target_path.exists():
        return _DEFAULT_FLAG_THRESHOLD

    try:
        payload = json.loads(target_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in range-threshold configuration: {target_path}") from exc

    raw_value = payload.get("range_flag_threshold")
    if raw_value is None:
        raw_value = payload.get("classifier_confidence_threshold")

    if raw_value is None:
        return _DEFAULT_FLAG_THRESHOLD

    threshold = float(raw_value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"Range flag threshold must lie within [0, 1]; received {threshold!r} from {target_path}"
        )
    return threshold


@dataclass(frozen=True)
class TimeWindow:
    """Represents an inclusive temporal window in UTC."""

    start: datetime
    end: datetime

    def contains(self, instant: datetime) -> bool:
        """Return True when *instant* lies within the window boundaries."""
        return self.start <= instant <= self.end


@dataclass(frozen=True)
class NodeFilter:
    """Allows inclusion or exclusion of node identifiers."""

    include: Tuple[str, ...] | None = None
    exclude: Tuple[str, ...] | None = None


@dataclass(frozen=True)
class IOFilters:
    """Compound filter set applied when scanning the GeoParquet dataset."""

    nodes: NodeFilter | None = None
    window: TimeWindow | None = None
    require_in_range: bool = False
    min_confidence: float | None = None

    def resolved_min_confidence(
        self,
        *,
        config_path: str | Path | None = None,
    ) -> float | None:
        """Return the effective confidence threshold for range filtering.

        When :attr:`require_in_range` is ``True`` and :attr:`min_confidence`
        is unset, the loader falls back to the configured ANN classifier
        threshold declared in ``config/range_thresholds.json``. Callers that
        want to bypass confidence-based filtering can keep both fields unset.
        """

        if self.min_confidence is not None:
            return self.min_confidence
        if not self.require_in_range:
            return None
        return load_range_flag_threshold(config_path)


@dataclass(frozen=True)
class PerformanceBudget:
    """Captures memory and throughput expectations for a reader instance."""

    max_bytes_in_memory: int
    target_rows_per_chunk: int
    max_latency_per_chunk: timedelta | None = None
    notes: str | None = None


@dataclass(frozen=True)
class CachePolicy:
    """Configuration for optional chunk caching."""

    enabled: bool
    cache_dir: Path | None = None
    max_disk_usage_bytes: int | None = None
    ttl: timedelta | None = None


@dataclass(frozen=True)
class ChunkMetadata:
    """Describes the contents of a materialised chunk."""

    node_ids: Tuple[str, ...]
    start: datetime
    end: datetime
    n_rows: int
    estimated_bytes: int


@dataclass(frozen=True)
class ChunkReadPlan:
    """Pre-computed instructions a reader uses to iterate over chunks."""

    dataset_path: Path
    schema_version: str
    filters: IOFilters
    performance: PerformanceBudget
    cache_policy: CachePolicy | None
    partitions: Sequence[ChunkMetadata]


@dataclass(frozen=True)
class AnomalyEvent:
    """Information about anomalies spotted during IO."""

    description: str
    affected_nodes: Tuple[str, ...]
    window: TimeWindow
    severity: str = "warning"


@dataclass(frozen=True)
class GapDescriptor:
    """Representation of temporal gaps detected while iterating chunks."""

    node_id: str
    expected_cadence: timedelta
    missing_windows: Tuple[TimeWindow, ...]


@dataclass
class GapReport:
    """Aggregates gap information for later inspection."""

    descriptors: MutableMapping[str, GapDescriptor]

    def register(self, descriptor: GapDescriptor) -> None:
        """Store or merge *descriptor* information."""
        self.descriptors[descriptor.node_id] = descriptor

    def summarise(self) -> Mapping[str, GapDescriptor]:
        """Return an immutable view of the collected gap descriptors."""
        return dict(self.descriptors)


@runtime_checkable
class ChunkCache(Protocol):
    """Protocol for cache backends storing chunk-level materialisations."""

    def get(self, key: str) -> DataFrameType | None:
        """Retrieve a cached chunk identified by *key*."""

    def set(self, key: str, value: DataFrameType, metadata: ChunkMetadata) -> None:
        """Persist *value* using *key*, retaining relevant *metadata*."""

    def invalidate(self, key: str | None = None) -> None:
        """Drop a specific cached chunk or flush the whole cache."""


@runtime_checkable
class AnomalySink(Protocol):
    """Receives anomaly events emitted during IO operations."""

    def emit(self, event: AnomalyEvent) -> None:
        """Persist or display *event* appropriately."""


@runtime_checkable
class ChunkedDatasetReader(Protocol):
    """High-level contract for chunked access to the inference dataset."""

    schema: SchemaType

    def plan(self, filters: IOFilters, budget: PerformanceBudget) -> ChunkReadPlan:
        """Derive a deterministic :class:`ChunkReadPlan` for given filters."""

    def iter_chunks(
        self,
        plan: ChunkReadPlan,
        *,
        cache: ChunkCache | None = None,
        anomalies: AnomalySink | None = None,
        gap_report: GapReport | None = None,
    ) -> Iterator[Tuple[ChunkMetadata, DataFrameType]]:
        """Iterate over ``(metadata, chunk)`` tuples, respecting *plan* constraints."""


def build_chunk_reader(
    dataset_path: Path,
    *,
    schema: SchemaType,
    performance_budget: PerformanceBudget,
    cache_policy: CachePolicy | None = None,
) -> ChunkedDatasetReader:
    """Factory returning a chunk-capable dataset reader.

    Parameters
    ----------
    dataset_path:
        Absolute or project-relative path to ``data.parquet`` (see
        ``use_case/catalogs/sar_range_final_pivots_joined``).
    schema:
        ``SarRangeFinalSchema`` instance describing expected columns, dtypes
        and range-awareness semantics. Consumers should obtain this from
        ``hf_wind_resource.io.schema_registry`` to avoid drift from the
        documentation.
    performance_budget:
        Memory and throughput constraints negotiated with the caller.
    cache_policy:
        Optional cache instructions. When ``None`` no cache layer will be
        created, even if the underlying implementation supports it.

    Notes
    -----
    Concrete implementations are responsible for choosing between DuckDB,
    PyArrow, or any other backend that can honour the requested filters
    without breaching the budget. Implementations should emit anomalies
    whenever gaps exceed 90 minutes or when ``prob_range_*`` columns do not
    form a simplex, as these conditions violate the ANN guarantees described
    in ``docs/hf_dev_plan.md``.
    """
    raise NotImplementedError("Chunk reader factory must be provided by IO backends.")


def describe_gap_expectations(expected_cadence_minutes: int = 30) -> GapReport:
    """Return an empty :class:`GapReport` seeded with cadence assumptions."""

    cadence = timedelta(minutes=expected_cadence_minutes)
    return GapReport(descriptors={})  # consumers fill descriptors as gaps appear


def iter_in_memory(
    dataset: DataFrameType,
    *,
    filters: IOFilters,
    performance_budget: PerformanceBudget,
) -> Iterator[Tuple[ChunkMetadata, DataFrameType]]:
    """Utility helper that yields in-memory DataFrame slices matching *filters*.

    This function primarily exists to support unit tests before the DuckDB
    backed reader is implemented. It should not be used in production because
    it bypasses lazy loading and offers no caching.
    """
    raise NotImplementedError("In-memory iteration helper is pending implementation.")
