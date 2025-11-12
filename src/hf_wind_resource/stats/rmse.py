"""Global RMSE registry and node taxonomy integration."""

from __future__ import annotations

import json

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence, Tuple


__all__ = [
    "GlobalRmseProvider",
    "GlobalRmseRecord",
    "NodeRmseAssessment",
    "NodeTaxonomyEntry",
]


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "config"
TAXONOMY_CONFIG = CONFIG_DIR / "node_taxonomy.json"
GLOBAL_RMSE_CONFIG = CONFIG_DIR / "global_rmse.json"
TAXONOMY_BANDS_CONFIG = CONFIG_DIR / "taxonomy_bands.json"


@dataclass(frozen=True)
class TaxonomyBandThresholds:
    """Thresholds controlling coverage and continuity band classification."""

    coverage_sparse_upper: int
    coverage_moderate_upper: int
    continuity_long_gap_min: float
    continuity_extreme_gap_min: float


def _load_taxonomy_bands_config() -> TaxonomyBandThresholds:
    """Read taxonomy band thresholds from disk."""

    try:
        payload = json.loads(TAXONOMY_BANDS_CONFIG.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - defensive fallback
        raise RuntimeError(
            "Cannot load taxonomy band thresholds because "
            f"{TAXONOMY_BANDS_CONFIG} is missing"
        ) from exc

    coverage = payload.get("coverage", {})
    continuity = payload.get("continuity", {})

    try:
        sparse_upper = int(coverage["sparse_upper"])
        moderate_upper = int(coverage["moderate_upper"])
        long_gap_min = float(continuity["long_gap_min_days"])
        extreme_gap_min = float(continuity["extreme_gap_min_days"])
    except KeyError as exc:  # pragma: no cover - invalid config
        raise ValueError(
            "taxonomy_bands.json must define coverage.sparse_upper, coverage.moderate_upper, "
            "continuity.long_gap_min_days and continuity.extreme_gap_min_days"
        ) from exc

    return TaxonomyBandThresholds(
        coverage_sparse_upper=sparse_upper,
        coverage_moderate_upper=moderate_upper,
        continuity_long_gap_min=long_gap_min,
        continuity_extreme_gap_min=extreme_gap_min,
    )


_TAXONOMY_BANDS = _load_taxonomy_bands_config()


def _reload_taxonomy_bands() -> None:
    """Refresh cached taxonomy band thresholds from disk."""

    global _TAXONOMY_BANDS
    _TAXONOMY_BANDS = _load_taxonomy_bands_config()


def _classify_coverage_band(total_observations: int) -> str:
    """Return the coverage band name for ``total_observations``."""

    thresholds = _TAXONOMY_BANDS
    if total_observations < thresholds.coverage_sparse_upper:
        return "sparse"
    if total_observations < thresholds.coverage_moderate_upper:
        return "moderate"
    return "dense"


def _classify_continuity_band(max_gap_days: float | None) -> str:
    """Return the continuity band name based on ``max_gap_days``."""

    if max_gap_days is None:
        return "unknown"

    thresholds = _TAXONOMY_BANDS
    if max_gap_days >= thresholds.continuity_extreme_gap_min:
        return "extreme_gaps"
    if max_gap_days >= thresholds.continuity_long_gap_min:
        return "long_gaps"
    return "regular"


def _ensure_utc(moment: datetime) -> datetime:
    """Return *moment* with an explicit UTC timezone."""

    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _coerce_instant(value: datetime | None) -> datetime:
    """Normalise ``value`` (or now) to an aware UTC timestamp."""

    if value is None:
        return datetime.now(tz=timezone.utc)
    return _ensure_utc(value)


@dataclass(frozen=True)
class NodeTaxonomyEntry:
    """Metadata snapshot for a node sourced from ``config/node_taxonomy.json``."""

    node_id: str
    total_observations: int
    on_cadence_transitions: int | None
    short_gaps: int | None
    long_gaps: int | None
    total_intervals: int | None
    max_gap_days: float | None
    low_coverage: bool
    coverage_band_value: str | None = None
    continuity_band_value: str | None = None

    @property
    def coverage_band(self) -> str:
        """Return coverage category for the node using configured thresholds."""

        if self.coverage_band_value:
            return self.coverage_band_value
        return _classify_coverage_band(self.total_observations)

    @property
    def continuity_band(self) -> str:
        """Return temporal continuity category using configured thresholds."""

        if self.continuity_band_value:
            return self.continuity_band_value
        return _classify_continuity_band(self.max_gap_days)

    @property
    def is_low_coverage(self) -> bool:
        """Return ``True`` when the node fails the coverage audit thresholds."""

        return self.low_coverage


def _load_taxonomy_config() -> Dict[str, NodeTaxonomyEntry]:
    """Load taxonomy entries from the JSON configuration file."""

    try:
        payload = json.loads(TAXONOMY_CONFIG.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - defensive fallback
        raise RuntimeError(
            f"Cannot load node taxonomy because {TAXONOMY_CONFIG} is missing"
        ) from exc

    nodes = payload.get("nodes", [])
    taxonomy: Dict[str, NodeTaxonomyEntry] = {}

    for entry in nodes:
        node_id = entry["node_id"]
        taxonomy[node_id] = NodeTaxonomyEntry(
            node_id=node_id,
            total_observations=int(entry.get("total_observations", 0)),
            on_cadence_transitions=_maybe_int(entry.get("on_cadence_transitions")),
            short_gaps=_maybe_int(entry.get("short_gaps")),
            long_gaps=_maybe_int(entry.get("long_gaps")),
            total_intervals=_maybe_int(entry.get("total_intervals")),
            max_gap_days=_maybe_float(entry.get("max_gap_days")),
            low_coverage=bool(entry.get("low_coverage", False)),
            coverage_band_value=entry.get("coverage_band"),
            continuity_band_value=entry.get("continuity_band"),
        )

    return taxonomy


@dataclass(frozen=True)
class GlobalRmseRecord:
    """Versioned RMSE metadata for the ANN vs. Vilano buoy comparison."""

    version: str
    value: float
    unit: str
    effective_from: datetime
    effective_until: datetime | None
    source: str
    computed_at: datetime
    notes: Tuple[str, ...] = ()

    def is_effective(self, instant: datetime) -> bool:
        """Return ``True`` when ``instant`` falls within the record validity window."""

        moment = _ensure_utc(instant)
        start = _ensure_utc(self.effective_from)
        if moment < start:
            return False
        if self.effective_until is None:
            return True
        end = _ensure_utc(self.effective_until)
        return moment < end


@dataclass(frozen=True)
class NodeRmseAssessment:
    """Outcome of requesting per-node RMSE information."""

    node_id: str
    rmse: float | None
    unit: str
    limitation: str
    global_record: GlobalRmseRecord
    taxonomy: NodeTaxonomyEntry


class GlobalRmseProvider:
    """Registry exposing global RMSE records and taxonomy-aware fallbacks."""

    def __init__(
        self,
        loader: Callable[[], Sequence[GlobalRmseRecord]] | None = None,
    ) -> None:
        self._loader = loader or _default_global_rmse_loader
        self._records: Tuple[GlobalRmseRecord, ...] = self._prepare_records(self._loader())
        if not self._records:
            raise ValueError("Global RMSE loader produced no records")
        self._taxonomy: Dict[str, NodeTaxonomyEntry] = _load_taxonomy_config()
        self._revision = 0

    @staticmethod
    def _prepare_records(records: Sequence[GlobalRmseRecord]) -> Tuple[GlobalRmseRecord, ...]:
        """Sort records by effective date and drop duplicates."""

        unique: Dict[Tuple[str, datetime], GlobalRmseRecord] = {}
        for record in records:
            key = (record.version, _ensure_utc(record.effective_from))
            unique[key] = record
        sorted_records = sorted(unique.values(), key=lambda item: _ensure_utc(item.effective_from))
        return tuple(sorted_records)

    @property
    def version(self) -> str:
        """Return a simple monotonic revision identifier."""

        latest = self._records[-1]
        return f"{latest.version}+{self._revision}"

    def refresh(self) -> None:
        """Reload RMSE records and taxonomy metadata."""

        self._records = self._prepare_records(self._loader())
        if not self._records:
            raise ValueError("Global RMSE loader produced no records")
        self._taxonomy = _load_taxonomy_config()
        self._revision += 1
        _reload_taxonomy_bands()

    def get_global_rmse(self, as_of: datetime | None = None) -> GlobalRmseRecord:
        """Return the active global RMSE record for ``as_of`` (defaults to now)."""

        instant = _coerce_instant(as_of)
        for record in reversed(self._records):
            if record.is_effective(instant):
                return record
        raise LookupError(f"No RMSE record available for {instant.isoformat()}")  # pragma: no cover

    def get_node_assessment(self, node_id: str, as_of: datetime | None = None) -> NodeRmseAssessment:
        """Return node metadata highlighting the absence of per-node RMSE values."""

        if node_id not in self._taxonomy:
            raise KeyError(f"Unknown node_id {node_id!r}")
        record = self.get_global_rmse(as_of=as_of)
        taxonomy_entry = self._taxonomy[node_id]
        limitation = (
            "Node-level RMSE estimates are not available in the delivered datasets; "
            f"defaulting to the global RMSE {record.value:.6f} {record.unit} "
            f"(source {record.source})."
        )
        return NodeRmseAssessment(
            node_id=node_id,
            rmse=None,
            unit=record.unit,
            limitation=limitation,
            global_record=record,
            taxonomy=taxonomy_entry,
        )

    def taxonomy_snapshot(self) -> Mapping[str, NodeTaxonomyEntry]:
        """Return a shallow copy of the current node taxonomy."""

        return dict(self._taxonomy)


def _maybe_int(value: object) -> int | None:
    """Cast ``value`` to ``int`` when possible."""

    if value is None:
        return None
    return int(value)


def _maybe_float(value: object) -> float | None:
    """Cast ``value`` to ``float`` when possible."""

    if value is None:
        return None
    return float(value)


def _default_global_rmse_loader() -> Tuple[GlobalRmseRecord, ...]:
    """Read RMSE records from the JSON configuration file."""

    try:
        payload = json.loads(GLOBAL_RMSE_CONFIG.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - defensive fallback
        raise RuntimeError(
            f"Cannot load global RMSE configuration because {GLOBAL_RMSE_CONFIG} is missing"
        ) from exc

    records = []
    for entry in payload.get("records", []):
        effective_from = _parse_timestamp(entry["effective_from"])
        effective_until_raw = entry.get("effective_until")
        effective_until = _parse_timestamp(effective_until_raw) if effective_until_raw else None
        computed_at = _parse_timestamp(entry.get("computed_at"))

        records.append(
            GlobalRmseRecord(
                version=str(entry["version"]),
                value=float(entry["value"]),
                unit=str(entry["unit"]),
                effective_from=effective_from,
                effective_until=effective_until,
                source=str(entry["source"]),
                computed_at=computed_at,
                notes=tuple(entry.get("notes", ()) or ()),
            )
        )

    return tuple(records)


def _parse_timestamp(value: str | None) -> datetime:
    """Parse ISO8601 timestamps stored in the configuration."""

    if value is None:
        return datetime.now(tz=timezone.utc)
    if value.endswith("Z"):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return datetime.fromisoformat(value)
