#!/usr/bin/env python3
"""Evaluate temporal dependence for buoy wind-speed records."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


logger = logging.getLogger("evaluate_buoy_dependence")


def _safe_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def _safe_std(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def _compute_autocorrelation(values: np.ndarray, lag: int, *, min_pairs: int) -> tuple[int, float | None]:
    if values.size <= lag:
        return 0, None
    current = values[lag:]
    previous = values[:-lag]
    mask = np.isfinite(current) & np.isfinite(previous)
    pair_count = int(mask.sum())
    if pair_count < min_pairs:
        return pair_count, None
    if pair_count <= 1:
        return pair_count, None
    corr = np.corrcoef(current[mask], previous[mask])[0, 1]
    if np.isnan(corr):
        return pair_count, None
    return pair_count, float(corr)


def evaluate_series_dependence(
    dataset_path: Path,
    *,
    lags: Sequence[int],
    min_pairs: int,
    max_block_length: int | None,
) -> dict[str, object]:
    table = pd.read_parquet(dataset_path, columns=["timestamp", "wind_speed"])
    table = table.dropna(subset=["wind_speed"]).copy()
    if table.empty:
        raise ValueError("Buoy dataset contains no valid wind_speed samples")

    table["timestamp"] = pd.to_datetime(table["timestamp"], utc=True)
    table.sort_values("timestamp", inplace=True)

    timestamps = table["timestamp"].view("int64") // 1_000_000_000
    speeds = table["wind_speed"].astype(float).to_numpy()

    sample_count = int(speeds.size)

    gaps = np.diff(timestamps) if sample_count > 1 else np.array([], dtype=float)

    mean_gap = _safe_mean(gaps)
    std_gap = _safe_std(gaps)

    acf_map: dict[int, float | None] = {}
    pair_map: dict[int, int] = {}

    for lag in lags:
        pairs, corr = _compute_autocorrelation(speeds, lag, min_pairs=min_pairs)
        pair_map[lag] = pairs
        acf_map[lag] = corr

    first_lag = lags[0]
    acf1 = acf_map.get(first_lag)

    effective_sample_size: float | None
    suggested_block_length: int | None

    if acf1 is None or np.isclose(1.0 + acf1, 0.0):
        effective_sample_size = None
        suggested_block_length = None
    else:
        ratio = (1.0 - acf1) / (1.0 + acf1)
        if ratio <= 0.0:
            effective_sample_size = None
            suggested_block_length = None
        else:
            effective_sample_size = sample_count * ratio
            raw_block = sample_count / effective_sample_size if effective_sample_size > 0.0 else None
            if raw_block is None:
                suggested_block_length = None
            else:
                block = max(1, int(round(raw_block)))
                if max_block_length is not None:
                    block = min(block, max_block_length)
                block = min(block, sample_count)
                suggested_block_length = block

    resampling_mode = "iid"
    if suggested_block_length is not None and suggested_block_length > 1:
        resampling_mode = "moving_block"

    return {
        "dataset": dataset_path.as_posix(),
        "sample_count": sample_count,
        "mean_gap_seconds": mean_gap,
        "std_gap_seconds": std_gap,
        "lags": {
            str(lag): {
                "autocorrelation": acf_map.get(lag),
                "pair_count": pair_map.get(lag, 0),
            }
            for lag in lags
        },
        "effective_sample_size": effective_sample_size,
        "suggested_block_length": suggested_block_length,
        "resampling_mode": resampling_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Path to the buoy GeoParquet dataset")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/buoy_validation/buoy_block_bootstrap.json"),
        help="Where the dependency diagnostics will be written",
    )
    parser.add_argument("--lags", nargs="+", type=int, default=[1, 2, 3], help="Positive lags to analyse")
    parser.add_argument("--min-pairs", type=int, default=30, help="Minimum number of pairs for autocorrelation")
    parser.add_argument("--max-block-length", type=int, default=None, help="Optional cap for suggested block length")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dataset = args.dataset if args.dataset.is_absolute() else (Path.cwd() / args.dataset)
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    logger.info("Evaluating temporal dependence for %s", dataset)

    metrics = evaluate_series_dependence(
        dataset,
        lags=tuple(args.lags),
        min_pairs=args.min_pairs,
        max_block_length=args.max_block_length,
    )

    output_path = args.output_json if args.output_json.is_absolute() else (Path.cwd() / args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Dependence metrics written to %s", output_path)


if __name__ == "__main__":
    main()
