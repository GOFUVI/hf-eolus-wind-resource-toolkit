#!/usr/bin/env python3
"""Generate range QA reports combining censoring and temporal diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Mapping

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hf_wind_resource.io import resolve_catalog_asset
from hf_wind_resource.preprocessing.censoring import NodeRangeSummary
from hf_wind_resource.qa import (
    RangeQaThresholds,
    TemporalQaMetrics,
    evaluate_range_quality,
    load_range_qa_thresholds,
)


DEFAULT_IMAGE = "duckdb/duckdb:latest"
DEFAULT_ENGINE = "docker"
DEFAULT_STAC_CONFIG = Path("config/stac_catalogs.json")
DEFAULT_DATASET_KEY = "sar_range_final_pivots_joined"
DEFAULT_OUTPUT_DIR = Path("artifacts") / "range_quality"
LABEL_REPORT = "range_quality_summary.csv"
JSON_REPORT = "range_quality_summary.json"
LOG_REPORT = "range_quality_summary.log"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QA checks on range-aware ANN outputs and emit structured reports.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Optional path to the GeoParquet dataset. Overrides STAC resolution when provided.",
    )
    parser.add_argument(
        "--stac-config",
        type=Path,
        default=DEFAULT_STAC_CONFIG,
        help="Path to the STAC catalog index JSON file.",
    )
    parser.add_argument(
        "--stac-dataset",
        default=DEFAULT_DATASET_KEY,
        help="Dataset key inside the STAC catalog index.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the QA artefacts will be written.",
    )
    parser.add_argument(
        "--threshold-config",
        type=Path,
        default=Path("config") / "range_quality_thresholds.json",
        help="QA threshold configuration file.",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="Docker image used to run DuckDB queries.",
    )
    parser.add_argument(
        "--engine",
        choices=("docker", "python"),
        default=DEFAULT_ENGINE,
        help=(
            "Execution engine for DuckDB queries. "
            "`docker` launches duckdb/duckdb:latest (default); "
            "`python` runs queries via the local DuckDB Python module."
        ),
    )
    return parser.parse_args()


def _docker_duckdb(sql: str, *, workdir: Path, image: str) -> str:
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{workdir}:/workspace",
        "-w",
        "/workspace",
        image,
        "duckdb",
        "-csv",
        "-header",
        "-cmd",
        sql,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout


def _python_duckdb(sql: str) -> str:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:  # pragma: no cover - defensive fallback
        raise RuntimeError(
            "DuckDB Python module not available. Install `duckdb` or use --engine docker."
        ) from exc

    with duckdb.connect(database=":memory:") as conn:
        conn.execute("SET threads TO 4")
        result = conn.execute(sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()

    output = ",".join(columns) + "\n"
    for row in rows:
        output += ",".join("" if value is None else str(value) for value in row) + "\n"
    return output.rstrip("\n")


def _read_csv_dicts(payload: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    reader = csv.DictReader(payload.splitlines())
    for raw_row in reader:
        row: dict[str, object] = {}
        for key, value in raw_row.items():
            if value is None or value == "":
                row[key] = None
                continue
            try:
                if any(char in value for char in (".", "e", "E")):
                    row[key] = float(value)
                else:
                    row[key] = int(value)
            except ValueError:
                row[key] = value
        rows.append(row)
    return rows


def _escape_path_for_sql(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _label_counts_sql(dataset: Path) -> str:
    escaped = _escape_path_for_sql(dataset)
    return textwrap.dedent(
        f"""
        WITH raw AS (
            SELECT
              node_id,
              timestamp,
              pred_wind_speed,
              pred_range_label
            FROM read_parquet('{escaped}')
        ),
        normalised AS (
            SELECT
              node_id,
              timestamp,
              pred_wind_speed,
              CASE
                WHEN pred_range_label IS NULL THEN 'uncertain'
                WHEN lower(pred_range_label) IN ('in', 'inside', 'within', 'in_range') THEN 'in'
                WHEN lower(pred_range_label) IN ('below', 'under', 'below_range', 'left') THEN 'below'
                WHEN lower(pred_range_label) IN ('above', 'over', 'upper', 'right') THEN 'above'
                ELSE 'uncertain'
              END AS label,
              CASE
                WHEN pred_range_label IS NULL THEN 1
                WHEN lower(pred_range_label) NOT IN ('uncertain', 'in', 'inside', 'within', 'in_range', 'below', 'under', 'below_range', 'left', 'above', 'over', 'upper', 'right')
                  THEN 1 ELSE 0
              END AS uncertain_from_other_label
            FROM raw
        )
        SELECT
            node_id,
            COUNT(*) AS total_observations,
            SUM(CASE WHEN label = 'in' THEN 1 ELSE 0 END) AS in_count,
            SUM(CASE WHEN label = 'below' THEN 1 ELSE 0 END) AS below_count,
            SUM(CASE WHEN label = 'above' THEN 1 ELSE 0 END) AS above_count,
            SUM(CASE WHEN label = 'uncertain' THEN 1 ELSE 0 END) AS uncertain_count,
            SUM(CASE WHEN label = 'uncertain' AND uncertain_from_other_label = 1 THEN 1 ELSE 0 END) AS uncertain_from_other_count,
            SUM(CASE WHEN pred_wind_speed IS NULL THEN 1 ELSE 0 END) AS missing_speed_count
        FROM normalised
        GROUP BY node_id
        ORDER BY node_id;
        """
    )


def _temporal_metrics_sql(dataset: Path) -> str:
    escaped = _escape_path_for_sql(dataset)
    return textwrap.dedent(
        f"""
        WITH base AS (
            SELECT
              node_id,
              timestamp
            FROM read_parquet('{escaped}')
            WHERE timestamp IS NOT NULL
        ),
        ordered AS (
            SELECT
              node_id,
              timestamp,
              LAG(timestamp) OVER (PARTITION BY node_id ORDER BY timestamp) AS prev_timestamp
            FROM base
        ),
        gaps AS (
            SELECT
              node_id,
              GREATEST(EXTRACT(EPOCH FROM (timestamp - prev_timestamp)) - 1800, 0) AS missing_seconds
            FROM ordered
            WHERE prev_timestamp IS NOT NULL
        ),
        coverage AS (
            SELECT
              node_id,
              COUNT(*) AS total_observations,
              COUNT(DISTINCT timestamp) AS distinct_observations,
              MIN(timestamp) AS start_ts,
              MAX(timestamp) AS end_ts
            FROM base
            GROUP BY node_id
        )
        SELECT
            c.node_id,
            c.total_observations,
            c.distinct_observations,
            c.total_observations - c.distinct_observations AS duplicate_records,
            CASE
                WHEN c.start_ts IS NULL OR c.end_ts IS NULL THEN NULL
                ELSE CAST(floor(EXTRACT(EPOCH FROM (c.end_ts - c.start_ts)) / 1800.0) AS BIGINT) + 1
            END AS expected_observations,
            CASE
                WHEN c.start_ts IS NULL OR c.end_ts IS NULL THEN NULL
                WHEN c.distinct_observations = 0 THEN NULL
                ELSE c.distinct_observations :: DOUBLE
                     / (CAST(floor(EXTRACT(EPOCH FROM (c.end_ts - c.start_ts)) / 1800.0) AS BIGINT) + 1)
            END AS coverage_ratio,
            COALESCE(g.max_missing_seconds, 0) AS max_missing_seconds
        FROM coverage c
        LEFT JOIN (
            SELECT
              node_id,
              MAX(missing_seconds) AS max_missing_seconds
            FROM gaps
            GROUP BY node_id
        ) g
        ON g.node_id = c.node_id
        ORDER BY c.node_id;
        """
    )


def _build_range_summaries(rows: Iterable[Mapping[str, object]]) -> Dict[str, NodeRangeSummary]:
    summaries: Dict[str, NodeRangeSummary] = {}
    for row in rows:
        node_id = str(row["node_id"])
        total = int(row.get("total_observations", 0) or 0)
        in_count = int(row.get("in_count", 0) or 0)
        below_count = int(row.get("below_count", 0) or 0)
        above_count = int(row.get("above_count", 0) or 0)
        uncertain_count = int(row.get("uncertain_count", 0) or 0)
        uncertain_from_other = int(row.get("uncertain_from_other_count", 0) or 0)
        missing_speed_count = int(row.get("missing_speed_count", 0) or 0)

        def _ratio(value: int) -> float | None:
            if total <= 0:
                return None
            return value / total

        notes: list[str] = []
        if uncertain_from_other > 0:
            notes.append(f"{uncertain_from_other} samples lack a definitive range label.")
        if missing_speed_count > 0:
            notes.append(f"{missing_speed_count} samples have missing wind_speed values.")

        summaries[node_id] = NodeRangeSummary(
            node_id=node_id,
            total_observations=total,
            left_censored_count=below_count,
            left_censored_ratio=_ratio(below_count),
            in_range_count=in_count,
            in_range_ratio=_ratio(in_count),
            right_censored_count=above_count,
            right_censored_ratio=_ratio(above_count),
            uncertain_count=uncertain_count,
            uncertain_ratio=_ratio(uncertain_count),
            discrepancy_count=0,
            notes=tuple(notes),
        )
    return summaries


def _build_temporal_metrics(rows: Iterable[Mapping[str, object]]) -> Dict[str, TemporalQaMetrics]:
    metrics: Dict[str, TemporalQaMetrics] = {}
    for row in rows:
        node_id = str(row["node_id"])
        expected_raw = row.get("expected_observations")
        expected = int(expected_raw) if expected_raw is not None else None
        metrics[node_id] = TemporalQaMetrics(
            node_id=node_id,
            coverage_ratio=float(row["coverage_ratio"]) if row.get("coverage_ratio") is not None else None,
            expected_observations=expected,
            distinct_observations=int(row.get("distinct_observations", 0) or 0),
            duplicate_records=int(row.get("duplicate_records", 0) or 0),
            max_gap_hours=(
                float(row.get("max_missing_seconds", 0) or 0) / 3600.0
                if row.get("max_missing_seconds") is not None
                else None
            ),
        )
    return metrics


def _resolve_dataset(args: argparse.Namespace) -> Path:
    if args.dataset:
        return args.dataset.resolve()
    resolved = resolve_catalog_asset(
        args.stac_dataset,
        config_path=args.stac_config,
        root=REPO_ROOT,
    )
    return resolved.require_local_path()


def main() -> None:
    args = _parse_args()
    dataset_path = _resolve_dataset(args)

    try:
        relative_dataset = dataset_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Dataset path {dataset_path} lies outside the repository root {REPO_ROOT}; "
            "mount the location inside the workspace or provide a repository-relative path."
        ) from exc

    if args.engine == "python":
        def run_query(sql: str) -> str:
            return _python_duckdb(sql)
    else:
        def run_query(sql: str) -> str:
            return _docker_duckdb(sql, workdir=REPO_ROOT, image=args.image)

    label_rows = _read_csv_dicts(run_query(_label_counts_sql(relative_dataset)))
    temporal_rows = _read_csv_dicts(run_query(_temporal_metrics_sql(relative_dataset)))

    range_summaries = _build_range_summaries(label_rows)
    temporal_metrics = _build_temporal_metrics(temporal_rows)
    thresholds: RangeQaThresholds = load_range_qa_thresholds(args.threshold_config)

    assessment = evaluate_range_quality(
        range_summaries,
        temporal_metrics,
        thresholds=thresholds,
    )

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = assessment.to_dataframe()
    dataframe.sort_values("node_id", inplace=True)
    dataframe.to_csv(output_dir / LABEL_REPORT, index=False)

    records = assessment.to_records()
    (output_dir / JSON_REPORT).write_text(json.dumps(records, indent=2), encoding="utf-8")

    flagged = [status for status in assessment.per_node.values() if status.flags]
    unreliable = [status for status in assessment.per_node.values() if not status.parametric_reliable]

    log_lines = [
        "Range QA assessment",
        f"Dataset: {relative_dataset}",
        f"Nodes analysed: {len(assessment.per_node)}",
        f"Thresholds: {json.dumps(asdict(thresholds), sort_keys=True)}",
        "",
        f"Nodes with QA flags: {len(flagged)}",
    ]
    for status in sorted(flagged, key=lambda item: item.node_id):
        joined_flags = ", ".join(status.flags)
        log_lines.append(f"- {status.node_id}: {joined_flags}")
    log_lines.append("")
    log_lines.append(f"Nodes marked unreliable for parametric analysis: {len(unreliable)}")
    for status in sorted(unreliable, key=lambda item: item.node_id):
        reasons = "; ".join(status.reliability_reasons) or "no reasons recorded"
        log_lines.append(f"- {status.node_id}: {reasons}")

    (output_dir / LOG_REPORT).write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
