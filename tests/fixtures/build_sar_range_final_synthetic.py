"""Generate synthetic SAR range-aware fixtures for automated tests.

The script materialises a CSV snapshot that mirrors the minimal subset of
columns required by the IO layer. Values are deterministic so the fixtures
stay reproducible offline. After writing the CSV, the helper also computes a
coverage summary to document how many samples exist for each range-label
combination and, optionally, invokes DuckDB inside the official Docker image
to emit a Parquet file whose ``geometry`` column is stored as a WKB BLOB—
matching the layout of the production GeoParquet snapshot.

Example (default) regeneration:

    python3 tests/fixtures/build_sar_range_final_synthetic.py

To skip the Parquet conversion (e.g. when Docker is unavailable):

    python3 tests/fixtures/build_sar_range_final_synthetic.py --skip-parquet

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List


SEED = 20241019
LOWER_LIMIT = 5.7
UPPER_LIMIT = 17.8

CSV_PATH = Path(__file__).with_name("sar_range_final_synthetic.csv")
SUMMARY_PATH = Path(__file__).with_name("sar_range_final_synthetic_summary.json")
PARQUET_PATH = Path(__file__).with_name("sar_range_final_synthetic.parquet")

DUCKDB_IMAGE = "duckdb/duckdb:latest"


@dataclass(frozen=True)
class RecordSpec:
    """Blueprint for a synthetic observation."""

    offset_minutes: int
    pred_wind_speed: float
    pred_range_label: str
    pred_speed_range_label: str
    pred_range_confidence: float
    prob_range_below: float
    prob_range_in: float
    prob_range_above: float
    range_flag: str
    range_flag_confident: bool
    range_near_lower_margin: bool
    range_near_upper_margin: bool
    wind_direction: int


@dataclass(frozen=True)
class NodeSpec:
    """Defines the temporal cadence and records emitted for a node."""

    node_id: str
    geometry_wkb_hex: str
    start: datetime
    cadence: timedelta | None
    records: List[RecordSpec]


def _round(value: float) -> float:
    """Round `value` to eight decimal places for CSV parity."""

    return round(value, 8)


def _rows_for_node(node: NodeSpec) -> Iterable[dict[str, object]]:
    """Convert node specifications into serialisable dictionaries."""

    current = node.start
    for idx, record in enumerate(node.records):
        if idx > 0:
            if node.cadence is None:
                current = node.start + timedelta(minutes=record.offset_minutes)
            else:
                current += node.cadence

        timestamp = (current if node.cadence is not None else node.start + timedelta(minutes=record.offset_minutes))
        direction = record.wind_direction
        cos_dir = _round(math.cos(math.radians(direction)))
        sin_dir = _round(math.sin(math.radians(direction)))
        range_any = record.range_near_lower_margin or record.range_near_upper_margin

        yield {
            "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "node_id": node.node_id,
            "geometry": node.geometry_wkb_hex,
            "pred_wind_speed": record.pred_wind_speed,
            "pred_range_label": record.pred_range_label,
            "pred_speed_range_label": record.pred_speed_range_label,
            "pred_range_confidence": record.pred_range_confidence,
            "prob_range_below": record.prob_range_below,
            "prob_range_in": record.prob_range_in,
            "prob_range_above": record.prob_range_above,
            "range_flag": record.range_flag,
            "range_flag_confident": str(record.range_flag_confident).lower(),
            "range_near_lower_margin": str(record.range_near_lower_margin).lower(),
            "range_near_upper_margin": str(record.range_near_upper_margin).lower(),
            "range_near_any_margin": str(range_any).lower(),
            "pred_wind_direction": direction,
            "pred_cos_wind_dir": f"{cos_dir:.8f}",
            "pred_sin_wind_dir": f"{sin_dir:.8f}",
        }


def build_specs() -> List[NodeSpec]:
    """Return deterministic fixture specifications."""

    rng = random.Random(SEED)

    node1 = NodeSpec(
        node_id="VILA_PRIO01",
        geometry_wkb_hex="010100000033333333333322c03333333333b34540",
        start=datetime(2024, 1, 1, 0, 0, 0),
        cadence=timedelta(minutes=30),
        records=[
            RecordSpec(0, 5.3, "below", "below", 0.86, 0.86, 0.12, 0.02, "below", True, True, False, 45),
            RecordSpec(30, 6.1, "in", "in", 0.90, 0.08, 0.90, 0.02, "in", True, False, False, 135),
            RecordSpec(60, 17.2, "in", "above", 0.70, 0.05, 0.70, 0.25, "uncertain", False, False, True, 225),
            RecordSpec(90, 18.5, "above", "above", 0.70, 0.10, 0.20, 0.70, "above", True, False, True, 315),
        ],
    )

    node2_records = [
        RecordSpec(0, LOWER_LIMIT, "in", "below", 0.34, 0.34, 0.33, 0.33, "uncertain", False, True, False, 90),
        RecordSpec(60, 10.0, "in", "in", 0.86, 0.08, 0.86, 0.06, "in", True, False, False, 180),
        RecordSpec(120, UPPER_LIMIT, "above", "above", 0.50, 0.15, 0.35, 0.50, "above", False, False, True, 210),
        RecordSpec(210, 17.0, "above", "in", 0.45, 0.20, 0.45, 0.35, "uncertain", False, False, True, 240),
    ]
    rng.shuffle(node2_records)
    node2_records.sort(key=lambda spec: spec.offset_minutes)

    node2 = NodeSpec(
        node_id="VILA_PRIO20",
        geometry_wkb_hex="0101000000cdcccccccccc21c09a99999999994540",
        start=datetime(2024, 1, 1, 0, 0, 0),
        cadence=None,
        records=node2_records,
    )

    node3 = NodeSpec(
        node_id="VILA_PRIO35",
        geometry_wkb_hex="010100000000000000000021c09a99999999d94540",
        start=datetime(2024, 1, 1, 0, 10, 0),
        cadence=None,
        records=[
            RecordSpec(0, 4.8, "below", "below", 0.88, 0.88, 0.10, 0.02, "below", True, False, False, 30),
            RecordSpec(30, 14.5, "in", "in", 0.90, 0.05, 0.90, 0.05, "in", True, False, False, 60),
            RecordSpec(65, 19.0, "above", "above", 0.80, 0.05, 0.15, 0.80, "above", True, False, True, 330),
            RecordSpec(95, 6.5, "below", "in", 0.40, 0.40, 0.45, 0.15, "uncertain", False, True, False, 15),
        ],
    )

    return [node1, node2, node3]


def emit_csv(rows: Iterable[dict[str, object]]) -> None:
    """Write CSV file with deterministic ordering."""

    fieldnames = [
        "timestamp",
        "node_id",
        "geometry",
        "pred_wind_speed",
        "pred_range_label",
        "pred_speed_range_label",
        "pred_range_confidence",
        "prob_range_below",
        "prob_range_in",
        "prob_range_above",
        "range_flag",
        "range_flag_confident",
        "range_near_lower_margin",
        "range_near_upper_margin",
        "range_near_any_margin",
        "pred_wind_direction",
        "pred_cos_wind_dir",
        "pred_sin_wind_dir",
    ]

    with CSV_PATH.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def emit_summary(rows: Iterable[dict[str, object]]) -> None:
    """Persist a JSON summary describing coverage per range label."""

    label_counts: dict[str, int] = {}
    combo_counts: dict[str, int] = {}

    for row in rows:
        label = row["pred_range_label"]
        label_counts[label] = label_counts.get(label, 0) + 1
        combo = f"{row['pred_range_label']}::{row['pred_speed_range_label']}"
        combo_counts[combo] = combo_counts.get(combo, 0) + 1

    payload = {
        "seed": SEED,
        "lower_limit": LOWER_LIMIT,
        "upper_limit": UPPER_LIMIT,
        "label_counts": label_counts,
        "combo_counts": combo_counts,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)


def run_duckdb_parquet(csv_path: Path, parquet_path: Path, image: str) -> None:
    """Convert the CSV fixture into Parquet with binary geometry using DuckDB."""

    if shutil.which("docker") is None:
        raise RuntimeError("Docker CLI not found; cannot build Parquet fixture.")

    mount_dir = csv_path.parent.resolve()
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{mount_dir}:/workspace",
        "-w",
        "/workspace",
        image,
        "duckdb",
        "-cmd",
        (
            "COPY (SELECT timestamp, node_id, from_hex(geometry) AS geometry, "
            "pred_wind_speed, pred_range_label, pred_speed_range_label, "
            "pred_range_confidence, prob_range_below, prob_range_in, "
            "prob_range_above, range_flag, range_flag_confident, "
            "range_near_lower_margin, range_near_upper_margin, range_near_any_margin, "
            "pred_wind_direction, pred_cos_wind_dir, pred_sin_wind_dir "
            "FROM read_csv_auto('" + csv_path.name + "', SAMPLE_SIZE=-1)) "
            "TO '" + parquet_path.name + "' (FORMAT 'parquet');"
        ),
    ]

    subprocess.run(command, check=True)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic SAR range fixtures.")
    parser.add_argument(
        "--skip-parquet",
        action="store_true",
        help="Do not invoke DuckDB/Docker to write the Parquet fixture.",
    )
    parser.add_argument(
        "--duckdb-image",
        default=DUCKDB_IMAGE,
        help="Docker image providing the DuckDB CLI (default: %(default)s).",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or [])
    specs = build_specs()
    all_rows = [row for spec in specs for row in _rows_for_node(spec)]
    emit_csv(all_rows)
    emit_summary(all_rows)

    if args.skip_parquet:
        return 0

    try:
        run_duckdb_parquet(CSV_PATH, PARQUET_PATH, args.duckdb_image)
    except (subprocess.CalledProcessError, RuntimeError) as exc:  # pragma: no cover - side effect
        print(
            "⚠️  Failed to refresh Parquet fixture: {}\n"
            "    Re-run manually with Docker when available.".format(exc),
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
