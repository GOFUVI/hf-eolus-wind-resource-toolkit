"""Regenerate the node taxonomy configuration from the GeoParquet snapshot.

The utility queries the `sar_range_final_pivots_joined` dataset using the
official DuckDB container, computes per-node observation totals together with
cadence gap statistics, and serialises the result to `config/node_taxonomy.json`.
It additionally derives a reusable `low_coverage` indicator combining the
minimum observation count and the presence of multi-year gaps, aligning with
audit recommendations for downstream reporting. Thresholds are sourced from
`config/low_coverage_rules.json` by default and may be overridden via CLI
arguments when required.

By default the script discovers the ANN GeoParquet snapshot using the STAC
index declared in ``config/stac_catalogs.json``. Override the dataset path
explicitly when working with alternate exports.

Example (default configuration):

    python scripts/update_node_taxonomy.py

Specify a custom dataset or output location if needed:

    python scripts/update_node_taxonomy.py \\
        --dataset use_case/catalogs/sar_range_final_pivots_joined/assets/data.parquet \\
        --output config/node_taxonomy.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hf_wind_resource.io import resolve_catalog_asset  # noqa: E402


DEFAULT_LOW_COVERAGE_MIN_OBSERVATIONS = 1_000
DEFAULT_LOW_COVERAGE_MIN_GAP_DAYS = 730.0  # two years expressed in days
DEFAULT_LOW_COVERAGE_CONFIG = Path("config/low_coverage_rules.json")
DEFAULT_TAXONOMY_BANDS_CONFIG = Path("config/taxonomy_bands.json")
DEFAULT_STAC_CONFIG = Path("config/stac_catalogs.json")
DEFAULT_STAC_DATASET = "sar_range_final_pivots_joined"


def _project_root() -> Path:
    """Return the repository root (one level above the scripts directory)."""

    return REPO_ROOT


def _relativise(path: Path, root: Path) -> Path:
    """Return *path* relative to *root*, raising if it lies outside."""

    try:
        return path.relative_to(root)
    except ValueError as exc:  # pragma: no cover - defensive fallback
        raise ValueError(f"Path {path} must reside within project root {root}") from exc


def _resolve_with_root(path: Path, root: Path) -> Path:
    """Resolve *path* against *root* when it is not absolute."""

    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _escape_sql(value: str) -> str:
    """Escape single quotes for embedding in SQL strings."""

    return value.replace("'", "''")


def _run_duckdb(sql_statements: Iterable[str], root: Path) -> None:
    """Execute *sql_statements* inside the DuckDB container."""

    joined = " ".join(statement.strip() for statement in sql_statements)
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{root}:/workspace",
        "-w",
        "/workspace",
        "duckdb/duckdb:latest",
        "duckdb",
        "-cmd",
        joined,
    ]
    subprocess.run(command, check=True)


def _load_json(path: Path) -> list[Dict[str, object]]:
    """Load a JSON array from *path* and delete the file afterwards."""

    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    finally:
        path.unlink(missing_ok=True)
    return content


def _prepare_tmp(path: Path) -> Path:
    """Return a temporary JSON path alongside *path*."""

    return path.with_name(f".tmp_{uuid4().hex}.json")


def _generate_taxonomy(
    dataset: Path,
    output: Path,
    schema_version: str,
    source: str,
    low_coverage_min_observations: int,
    low_coverage_min_gap_days: float,
    taxonomy_bands: Dict[str, Dict[str, object]],
) -> None:
    """Compute taxonomy metrics and write *output* JSON file."""

    root = _project_root()
    dataset_rel = _relativise(dataset, root)
    output_rel = _relativise(output, root)

    tmp_counts = _prepare_tmp(output)
    tmp_gaps = _prepare_tmp(output)
    tmp_counts_rel = _relativise(tmp_counts, root)
    tmp_gaps_rel = _relativise(tmp_gaps, root)

    dataset_sql = _escape_sql(dataset_rel.as_posix())
    counts_sql = (
        f"COPY (SELECT node_id, COUNT(*) AS total_observations "
        f"FROM read_parquet('{dataset_sql}') "
        "GROUP BY node_id ORDER BY node_id) "
        f"TO '{_escape_sql(tmp_counts_rel.as_posix())}' (FORMAT JSON, ARRAY TRUE);"
    )

    gaps_sql = (
        "WITH diffs AS ( "
        f"SELECT node_id, "
        f"CAST(EXTRACT(EPOCH FROM timestamp - LAG(timestamp) OVER (PARTITION BY node_id ORDER BY timestamp)) AS BIGINT) AS dt "
        f"FROM read_parquet('{dataset_sql}') "
        ") "
        "SELECT node_id, "
        "COUNT(*) FILTER (WHERE dt IS NULL OR dt = 1800) AS on_cadence_transitions, "
        "COUNT(*) FILTER (WHERE dt > 1800 AND dt <= 7200) AS short_gaps, "
        "COUNT(*) FILTER (WHERE dt > 7200) AS long_gaps, "
        "COUNT(*) AS total_intervals, "
        "MAX(COALESCE(dt, 0)) / 86400.0 AS max_gap_days "
        "FROM diffs GROUP BY node_id ORDER BY node_id"
    )
    gaps_copy = (
        f"COPY ({gaps_sql}) TO '{_escape_sql(tmp_gaps_rel.as_posix())}' "
        "(FORMAT JSON, ARRAY TRUE);"
    )

    _run_duckdb([counts_sql, gaps_copy], root)

    counts = {entry["node_id"]: entry for entry in _load_json(tmp_counts)}
    gaps = {entry["node_id"]: entry for entry in _load_json(tmp_gaps)}

    nodes = []
    for node_id in sorted(set(counts) | set(gaps)):
        count_data = counts.get(node_id, {})
        gap_data = gaps.get(node_id, {})

        total_observations = int(count_data.get("total_observations", 0))
        max_gap_days = _maybe_float(gap_data.get("max_gap_days"))

        entry: Dict[str, object] = {
            "node_id": node_id,
            "total_observations": total_observations,
            "on_cadence_transitions": _maybe_int(gap_data.get("on_cadence_transitions")),
            "short_gaps": _maybe_int(gap_data.get("short_gaps")),
            "long_gaps": _maybe_int(gap_data.get("long_gaps")),
            "total_intervals": _maybe_int(gap_data.get("total_intervals")),
            "max_gap_days": max_gap_days,
            "low_coverage": (
                total_observations < low_coverage_min_observations
                and (
                    max_gap_days is not None
                    and max_gap_days > low_coverage_min_gap_days
                )
            ),
            "coverage_band": _classify_coverage(total_observations, taxonomy_bands),
            "continuity_band": _classify_continuity(max_gap_days, taxonomy_bands),
        }
        nodes.append(entry)

    payload = {
        "schema_version": schema_version,
        "source": source,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "nodes": nodes,
        "low_coverage_rules": {
            "min_observations": low_coverage_min_observations,
            "min_consecutive_gap_days": low_coverage_min_gap_days,
            "description": (
                "Node flagged when total_observations < min_observations and max_gap_days > "
                "min_consecutive_gap_days"
            ),
        },
        "taxonomy_bands": taxonomy_bands,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _maybe_int(value: object) -> int | None:
    """Convert *value* to int when possible."""

    if value is None:
        return None
    return int(value)


def _maybe_float(value: object) -> float | None:
    """Convert *value* to float when possible."""

    if value is None:
        return None
    return float(value)


def _load_low_coverage_settings(path: Path) -> Tuple[int, float]:
    """Extract low-coverage thresholds from *path* or return defaults."""

    min_observations = DEFAULT_LOW_COVERAGE_MIN_OBSERVATIONS
    min_gap_days = DEFAULT_LOW_COVERAGE_MIN_GAP_DAYS

    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - invalid config
            raise ValueError(f"Cannot parse low-coverage configuration {path}: {exc}") from exc
        if "min_observations" in payload:
            min_observations = int(payload["min_observations"])
        if "min_consecutive_gap_days" in payload:
            min_gap_days = float(payload["min_consecutive_gap_days"])

    return min_observations, min_gap_days


def _load_taxonomy_bands(path: Path) -> Dict[str, Dict[str, object]]:
    """Return taxonomy band thresholds from *path* raising on invalid payload."""

    if not path.exists():
        raise FileNotFoundError(
            "Taxonomy band configuration not found: "
            f"{path}. Regenerate it or provide an alternate path with --taxonomy-bands-config"
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - invalid config
        raise ValueError(f"Cannot parse taxonomy band configuration {path}: {exc}") from exc

    if "coverage" not in payload or "continuity" not in payload:
        raise ValueError(
            "Taxonomy bands configuration must define both 'coverage' and 'continuity' sections"
        )
    return payload


def _classify_coverage(total_observations: int, bands: Dict[str, Dict[str, object]]) -> str:
    """Return the coverage band for *total_observations* using *bands* thresholds."""

    coverage_cfg = bands.get("coverage", {})
    sparse_upper = int(coverage_cfg.get("sparse_upper", 5_000))
    moderate_upper = int(coverage_cfg.get("moderate_upper", 20_000))

    if total_observations < sparse_upper:
        return "sparse"
    if total_observations < moderate_upper:
        return "moderate"
    return "dense"


def _classify_continuity(max_gap_days: float | None, bands: Dict[str, Dict[str, object]]) -> str:
    """Categorise nodes based on *max_gap_days* using *bands* thresholds."""

    if max_gap_days is None:
        return "unknown"

    continuity_cfg = bands.get("continuity", {})
    long_gap_min = float(continuity_cfg.get("long_gap_min_days", 1_150.0))
    extreme_gap_min = float(continuity_cfg.get("extreme_gap_min_days", 1_200.0))

    if max_gap_days >= extreme_gap_min:
        return "extreme_gaps"
    if max_gap_days >= long_gap_min:
        return "long_gaps"
    return "regular"


def parse_args() -> argparse.Namespace:
    """Create the CLI parser and return parsed arguments."""

    default_output = Path("config/node_taxonomy.json")

    parser = argparse.ArgumentParser(description="Regenerate node taxonomy config.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Override the GeoParquet snapshot path. When omitted the asset is "
            "resolved through the STAC catalog configuration."
        ),
    )
    parser.add_argument(
        "--stac-config",
        type=Path,
        default=DEFAULT_STAC_CONFIG,
        help=(
            "Path to the STAC catalog index JSON. Ignored when --dataset is provided. "
            "Defaults to config/stac_catalogs.json."
        ),
    )
    parser.add_argument(
        "--stac-dataset",
        default=DEFAULT_STAC_DATASET,
        help=(
            "Dataset key within the STAC catalog index used to resolve the ANN snapshot. "
            "Ignored when --dataset is provided."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output JSON file (relative to project root).",
    )
    parser.add_argument(
        "--schema-version",
        default=datetime.now(tz=timezone.utc).date().isoformat(),
        help="Schema version string stored in the output (default: today).",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional source description; defaults to the dataset path.",
    )
    parser.add_argument(
        "--low-coverage-config",
        type=Path,
        default=DEFAULT_LOW_COVERAGE_CONFIG,
        help=(
            "Optional JSON file with low-coverage thresholds (min_observations, "
            "min_consecutive_gap_days). Defaults to config/low_coverage_rules.json."
        ),
    )
    parser.add_argument(
        "--taxonomy-bands-config",
        type=Path,
        default=DEFAULT_TAXONOMY_BANDS_CONFIG,
        help=(
            "JSON file with coverage and continuity band thresholds. "
            "Defaults to config/taxonomy_bands.json."
        ),
    )
    parser.add_argument(
        "--low-coverage-min-observations",
        type=int,
        default=None,
        help=(
            "Override the minimum number of observations required to avoid the "
            "low-coverage flag. Takes precedence over the configuration file."
        ),
    )
    parser.add_argument(
        "--low-coverage-min-gap-days",
        type=float,
        default=None,
        help=(
            "Override the minimum multi-year gap (in days) that contributes to the "
            "low-coverage flag. Takes precedence over the configuration file."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the script."""

    args = parse_args()
    root = _project_root()

    resolved_asset = None
    if args.dataset is not None:
        dataset = _resolve_with_root(args.dataset, root)
    else:
        stac_config_path = _resolve_with_root(args.stac_config, root)
        resolved_asset = resolve_catalog_asset(
            args.stac_dataset,
            config_path=stac_config_path,
            root=root,
        )
        dataset = resolved_asset.require_local_path()

    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    output = _resolve_with_root(args.output, root)
    if args.source is not None:
        source = args.source
    elif resolved_asset is not None:
        try:
            source = _relativise(dataset, root).as_posix()
        except ValueError:
            source = resolved_asset.href
    elif args.dataset is not None:
        source = args.dataset.as_posix()
    else:
        source = dataset.as_posix()

    coverage_config_path = _resolve_with_root(args.low_coverage_config, root)
    min_observations, min_gap_days = _load_low_coverage_settings(coverage_config_path)

    if args.low_coverage_min_observations is not None:
        min_observations = args.low_coverage_min_observations
    if args.low_coverage_min_gap_days is not None:
        min_gap_days = args.low_coverage_min_gap_days

    taxonomy_bands_path = _resolve_with_root(args.taxonomy_bands_config, root)
    taxonomy_bands = _load_taxonomy_bands(taxonomy_bands_path)

    _generate_taxonomy(
        dataset=dataset,
        output=output,
        schema_version=args.schema_version,
        source=source,
        low_coverage_min_observations=min_observations,
        low_coverage_min_gap_days=min_gap_days,
        taxonomy_bands=taxonomy_bands,
    )


if __name__ == "__main__":
    main()
