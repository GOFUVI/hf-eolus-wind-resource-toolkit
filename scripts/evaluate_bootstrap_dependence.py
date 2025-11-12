#!/usr/bin/env python3
"""Evaluate temporal dependence to motivate block/stationary bootstrap usage."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hf_wind_resource.io import resolve_catalog_asset
from hf_wind_resource.stats import (
    NodeDependenceMetrics,
    compute_node_dependence_metrics,
    summarise_dependence_levels,
)


DEFAULT_OUTPUT_DIR = Path("artifacts/bootstrap_uncertainty")
DEFAULT_STAC_CONFIG = Path("config/stac_catalogs.json")
DEFAULT_STAC_DATASET = "sar_range_final_pivots_joined"


logger = logging.getLogger("evaluate_bootstrap_dependence")


def _metrics_to_rows(
    metrics: Iterable[NodeDependenceMetrics],
    *,
    lags: Sequence[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in metrics:
        row: dict[str, object] = {
            "node_id": item.node_id,
            "sample_count": item.sample_count,
            "mean_gap_seconds": item.mean_gap_seconds,
            "std_gap_seconds": item.std_gap_seconds,
            "effective_sample_size": item.effective_sample_size,
            "suggested_block_length": item.suggested_block_length,
        }
        for lag in lags:
            row[f"pair_count_lag_{lag}"] = item.pair_counts.get(lag, 0)
            row[f"acf_lag_{lag}"] = item.autocorrelations.get(lag)
        rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Path to the GeoParquet dataset", default=None)
    parser.add_argument("--stac-config", type=Path, default=DEFAULT_STAC_CONFIG, help="STAC configuration JSON")
    parser.add_argument("--stac-dataset", default=DEFAULT_STAC_DATASET, help="Dataset key within the STAC config")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for diagnostics")
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 2, 3], help="Positive integer lags to analyse")
    parser.add_argument("--min-pairs", type=int, default=30, help="Minimum pairs to report an autocorrelation")
    parser.add_argument("--max-block-length", type=int, default=None, help="Optional cap for suggested block length")
    parser.add_argument(
        "--acf-thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
        help="Autocorrelation thresholds for the summary report",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.dataset is not None:
        dataset = args.dataset if args.dataset.is_absolute() else (Path.cwd() / args.dataset)
    else:
        asset = resolve_catalog_asset(args.stac_dataset, config_path=args.stac_config)
        dataset = asset.require_local_path()

    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    output_dir = args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Analysing dataset %s", dataset)

    metrics = compute_node_dependence_metrics(
        dataset,
        lags=tuple(args.lags),
        min_pairs=args.min_pairs,
        max_block_length=args.max_block_length,
    )

    logger.info("Computed dependence metrics for %d nodes", len(metrics))

    rows = _metrics_to_rows(metrics, lags=args.lags)
    depend_csv = output_dir / "block_bootstrap_diagnostics.csv"
    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.sort_values(by="acf_lag_1", ascending=False, inplace=True)
        df.to_csv(depend_csv, index=False)
        rows = df.to_dict("records")
        logger.info("Saved node diagnostics to %s", depend_csv)
    else:
        depend_csv.write_text("", encoding="utf-8")
        logger.warning("No diagnostics written; dataset may be empty")

    summary = summarise_dependence_levels(metrics, acf_thresholds=tuple(args.acf_thresholds))
    summary["dataset"] = dataset.as_posix()
    summary_path = output_dir / "block_bootstrap_summary.json"
    _write_json(summary_path, summary)
    logger.info("Summary written to %s", summary_path)

    if rows:
        top = rows[0]
        logger.info(
            "Top node %s: acf_lag_1=%.3f, suggested_block_length=%s",
            top.get("node_id"),
            top.get("acf_lag_1"),
            top.get("suggested_block_length"),
        )


if __name__ == "__main__":
    main()
