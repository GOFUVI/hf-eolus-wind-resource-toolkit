"""Utilities to diagnose temporal dependence in ANN inference records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import math

import duckdb
import pandas as pd

__all__ = [
    "NodeDependenceMetrics",
    "compute_node_dependence_metrics",
    "summarise_dependence_levels",
]


@dataclass(frozen=True)
class NodeDependenceMetrics:
    """Temporal-dependence diagnostics computed for a node."""

    node_id: str
    sample_count: int
    mean_gap_seconds: float | None
    std_gap_seconds: float | None
    autocorrelations: Mapping[int, float | None]
    pair_counts: Mapping[int, int]
    effective_sample_size: float | None
    suggested_block_length: int | None


def compute_node_dependence_metrics(
    dataset_path: Path,
    *,
    lags: Sequence[int] = (1, 2, 3),
    min_pairs: int = 30,
    max_block_length: int | None = None,
) -> list[NodeDependenceMetrics]:
    """Estimate autocorrelations and block-bootstrap hints for each node.

    Parameters
    ----------
    dataset_path:
        Path to the GeoParquet file containing ANN inference records.
    lags:
        Positive integer lags (in record units) for which autocorrelations are
        calculated. Defaults to the first three lags.
    min_pairs:
        Minimum number of paired observations required to report an
        autocorrelation value. Nodes with fewer pairs will receive ``None`` for
        that lag.
    max_block_length:
        Optional cap for the suggested moving-block length. When ``None`` the
        suggestion is only bounded by the available sample size.

    Returns
    -------
    list[NodeDependenceMetrics]
        One entry per node with temporal diagnostics.
    """

    if not lags:
        raise ValueError("lags must not be empty")
    if any(lag <= 0 for lag in lags):
        raise ValueError("lags must contain positive integers")

    parquet_path = dataset_path.resolve().as_posix()

    con = duckdb.connect()
    base_query = f"""
        SELECT node_id, timestamp, pred_wind_speed
        FROM read_parquet('{parquet_path}')
        WHERE pred_wind_speed IS NOT NULL
    """
    con.execute(f"CREATE OR REPLACE TEMP VIEW ann_base AS {base_query}")

    counts_df = con.execute(
        """
        SELECT node_id, COUNT(*) AS sample_count
        FROM ann_base
        GROUP BY node_id
        ORDER BY node_id
        """
    ).fetchdf()

    gap_df = con.execute(
        """
        SELECT node_id,
               AVG(DATEDIFF('second', lag_ts, timestamp)) AS mean_gap_seconds,
               STDDEV_SAMP(DATEDIFF('second', lag_ts, timestamp)) AS std_gap_seconds
        FROM (
            SELECT node_id,
                   timestamp,
                   LAG(timestamp) OVER (PARTITION BY node_id ORDER BY timestamp) AS lag_ts
            FROM ann_base
        )
        WHERE lag_ts IS NOT NULL
        GROUP BY node_id
        ORDER BY node_id
        """
    ).fetchdf()

    metrics_df = counts_df.merge(gap_df, on="node_id", how="left")

    for lag in lags:
        lag_df = con.execute(
            f"""
            SELECT node_id,
                   COUNT(*) AS pair_count_lag_{lag},
                   CORR(current_speed, lag_speed) AS acf_lag_{lag}
            FROM (
                SELECT node_id,
                       pred_wind_speed AS current_speed,
                       LAG(pred_wind_speed, {lag})
                           OVER (PARTITION BY node_id ORDER BY timestamp) AS lag_speed
                FROM ann_base
            )
            WHERE lag_speed IS NOT NULL
            GROUP BY node_id
            ORDER BY node_id
            """
        ).fetchdf()
        metrics_df = metrics_df.merge(lag_df, on="node_id", how="left")

    result: list[NodeDependenceMetrics] = []

    for row in metrics_df.itertuples(index=False):
        autocorr: dict[int, float | None] = {}
        pair_counts: dict[int, int] = {}

        acf1 = None
        for lag in lags:
            corr_value = getattr(row, f"acf_lag_{lag}") if hasattr(row, f"acf_lag_{lag}") else None
            pair_value = getattr(row, f"pair_count_lag_{lag}") if hasattr(row, f"pair_count_lag_{lag}") else None

            if pair_value is None or pd.isna(pair_value):
                pair_counts[lag] = 0
                autocorr[lag] = None
                continue

            pair_counts[lag] = int(pair_value)
            if pair_value < min_pairs:
                autocorr[lag] = None
            elif corr_value is None or pd.isna(corr_value):
                autocorr[lag] = None
            else:
                autocorr[lag] = float(corr_value)

            if lag == lags[0] and autocorr[lag] is not None:
                acf1 = autocorr[lag]

        sample_count = int(row.sample_count)
        n_eff: float | None
        block_length: int | None

        if acf1 is None or math.isclose(1.0 + acf1, 0.0):
            n_eff = None
            block_length = None
        else:
            ratio = (1.0 - acf1) / (1.0 + acf1)
            if ratio <= 0.0:
                n_eff = None
                block_length = None
            else:
                n_eff = sample_count * ratio
                raw_block = sample_count / n_eff if n_eff > 0.0 else None
                if raw_block is None:
                    block_length = None
                else:
                    block = max(1, int(round(raw_block)))
                    if max_block_length is not None:
                        block = min(block, max_block_length)
                    block = min(block, sample_count)
                    block_length = block

        mean_gap = None
        if getattr(row, "mean_gap_seconds", None) is not None and not pd.isna(row.mean_gap_seconds):
            mean_gap = float(row.mean_gap_seconds)

        std_gap = None
        if getattr(row, "std_gap_seconds", None) is not None and not pd.isna(row.std_gap_seconds):
            std_gap = float(row.std_gap_seconds)

        result.append(
            NodeDependenceMetrics(
                node_id=row.node_id,
                sample_count=sample_count,
                mean_gap_seconds=mean_gap,
                std_gap_seconds=std_gap,
                autocorrelations=autocorr,
                pair_counts=pair_counts,
                effective_sample_size=n_eff,
                suggested_block_length=block_length,
            )
        )

    return result


def summarise_dependence_levels(
    metrics: Sequence[NodeDependenceMetrics],
    *,
    acf_thresholds: Sequence[float] = (0.3, 0.5, 0.7),
) -> dict[str, object]:
    """Reduce node-level diagnostics into aggregated statistics."""

    summary: dict[str, object] = {
        "node_count": len(metrics),
        "acf_thresholds": list(acf_thresholds),
    }

    if not metrics:
        summary["share_by_threshold"] = {str(th): 0.0 for th in acf_thresholds}
        summary["mean_block_length"] = None
        summary["median_block_length"] = None
        return summary

    lag1_values = [item.autocorrelations.get(1) for item in metrics]
    block_lengths = [item.suggested_block_length for item in metrics if item.suggested_block_length is not None]

    shares: dict[str, float] = {}
    for threshold in acf_thresholds:
        count = sum(1 for value in lag1_values if value is not None and value >= threshold)
        shares[str(threshold)] = count / len(metrics)
    summary["share_by_threshold"] = shares

    if block_lengths:
        summary["mean_block_length"] = float(sum(block_lengths) / len(block_lengths))
        summary["median_block_length"] = float(pd.Series(block_lengths).median())
    else:
        summary["mean_block_length"] = None
        summary["median_block_length"] = None

    return summary
