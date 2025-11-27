#!/usr/bin/env python3
"""Compute wind-resource diagnostics comparing ANN predictions with buoy observations.

This helper consumes the GeoParquet dataset produced by
``scripts/prepare_buoy_timeseries.py`` (ANN node matched to the Vilano buoy)
and derives side-by-side metrics:

* Weighted ANN wind-speed statistics inside the regression range.
* Buoy wind-speed statistics at the corrected reference height.
* Power-density estimates derived from the ANN censored distribution
  (re-using the Weibull/Kaplan–Meier pipeline) and matching buoy models on
  the paired and full datasets.
* Censoring diagnostics (share of samples flagged below/above range).
* Bootstrap confidence intervals when the ANN node appears in the existing
  ``artifacts/bootstrap_velocity_block/bootstrap_summary.csv`` outputs.

Results are materialised as JSON/CSV artefacts under the requested output
directory so downstream notebooks can embed the quantitative comparison
without re-running the full pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hf_wind_resource.preprocessing import load_buoy_timeseries
from hf_wind_resource.preprocessing.buoy_timeseries import BuoySentinelConfig, HeightCorrectionConfig
from hf_wind_resource.preprocessing.censoring import load_range_thresholds
from hf_wind_resource.stats import (
    HeightCorrection,
    KaplanMeierSelectionCriteria,
    PowerCurve,
    StratifiedBootstrapConfig,
    NodeBootstrapInput,
    compute_stratified_bootstrap_uncertainty,
    GlobalRmseProvider,
    build_censored_data_from_records,
    compute_power_distribution,
    format_height_note,
    load_kaplan_meier_selection_criteria,
    summarise_records_for_selection,
)
from hf_wind_resource.stats.weibull import CensoredWeibullData


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATCHED_DATASET = REPO_ROOT / "artifacts" / "processed" / "vilano_buoy_synced.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "buoy_validation"
DEFAULT_RANGE_THRESHOLDS = REPO_ROOT / "config" / "range_thresholds.json"
DEFAULT_POWER_CURVE_CONFIG = REPO_ROOT / "config" / "power_curves.json"
DEFAULT_POWER_CURVE_KEY = "reference_offshore_6mw"
DEFAULT_HEIGHT_CONFIG = REPO_ROOT / "config" / "power_height.json"
DEFAULT_BOOTSTRAP_SUMMARY = REPO_ROOT / "artifacts" / "bootstrap_velocity_block" / "bootstrap_summary.csv"
DEFAULT_BOOTSTRAP_METADATA = REPO_ROOT / "artifacts" / "bootstrap_velocity_block" / "bootstrap_metadata.json"
DEFAULT_BUOY_BLOCK_CONFIG = REPO_ROOT / "artifacts" / "buoy_validation" / "buoy_block_bootstrap.json"
DEFAULT_BUOY_BLOCK_LENGTHS = REPO_ROOT / "artifacts" / "buoy_block_diagnostics" / "block_bootstrap_diagnostics.csv"
DEFAULT_ANN_LABEL = "ANN"
DEFAULT_BUOY_LABEL = "Buoy"
DEFAULT_BUOY_BOOTSTRAP_REPLICATES = 50
DEFAULT_BUOY_BOOTSTRAP_CONFIDENCE = 0.95
DEFAULT_COMPARISON_CONFIG = REPO_ROOT / "config" / "buoy_comparison.json"
IDENTITY_HEIGHT = HeightCorrection(method="none", source_height_m=1.0, target_height_m=1.0, speed_scale=1.0)


def _load_ann_bootstrap_metadata(path: Path) -> tuple[dict[str, object], dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(
            f"ANN bootstrap metadata not found at {path}. "
            "Run scripts/generate_bootstrap_uncertainty.py to generate it."
        )
    metadata = json.loads(path.read_text(encoding="utf-8"))
    block_lengths_relative = metadata.get("block_lengths_csv", "")
    block_lengths_path = (path.parent / block_lengths_relative).resolve()
    if not block_lengths_path.exists():
        block_lengths_path = (REPO_ROOT / block_lengths_relative).resolve()
    node_block_lengths: dict[str, int] = {}
    if block_lengths_path.exists():
        with block_lengths_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                node_id = row.get("node_id")
                value = row.get("suggested_block_length")
                if node_id and value:
                    try:
                        node_block_lengths[node_id] = int(float(value))
                    except ValueError:
                        continue
    return metadata, node_block_lengths


def _build_ann_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    required = [
        "timestamp_ann",
        "pred_wind_speed",
        "prob_range_below",
        "prob_range_in",
        "prob_range_above",
        "range_flag",
        "range_flag_confident",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Matched dataset is missing ANN columns: {missing}")
    subset = frame[required].copy()
    subset["timestamp_ann"] = pd.to_datetime(subset["timestamp_ann"], utc=True)
    subset.rename(columns={"timestamp_ann": "timestamp"}, inplace=True)
    return subset.to_dict("records")


def _bootstrap_result_to_summary(result) -> dict[str, object]:
    metrics: dict[str, dict[str, object]] = {}
    for name, ci in result.metrics.items():
        metrics[name] = {
            "estimate": float(ci.estimate) if ci.estimate is not None else None,
            "lower": float(ci.lower) if ci.lower is not None else None,
            "upper": float(ci.upper) if ci.upper is not None else None,
            "confidence": float(ci.confidence_level),
            "replicates": int(ci.replicates),
            "notes": list(ci.notes),
        }
    diagnostics: dict[str, object] | None = None
    if result.power_diagnostics is not None:
        power = result.power_diagnostics
        diagnostics = {
            "method": power.method,
            "selection_reasons": list(power.selection_reasons),
            "method_notes": list(power.method_notes),
            "replicate_method_counts": dict(power.replicate_method_counts),
        }
        if power.weibull is not None:
            diagnostics["weibull"] = {
                "success": power.weibull.success,
                "reliable": power.weibull.reliable,
                "shape": float(power.weibull.shape) if power.weibull.shape is not None else None,
                "scale": float(power.weibull.scale) if power.weibull.scale is not None else None,
                "left_weight": float(power.weibull.left_count),
                "right_weight": float(power.weibull.right_count),
                "in_weight": float(power.weibull.in_count),
            }
    return {
        "metrics": metrics,
        "bootstrap_means": {k: (float(v) if v is not None else None) for k, v in result.bootstrap_means.items()},
        "label_counts": {k: float(v) for k, v in result.label_counts.items()},
        "label_proportions": {k: float(v) for k, v in result.label_proportions.items()},
        "total_samples": int(result.total_samples),
        "notes": list(result.notes),
        "power_diagnostics": diagnostics,
    }


def _normalize_ann_global_bootstrap(raw: dict[str, object], *, metadata: dict[str, object]) -> dict[str, object]:
    confidence = float(metadata.get("confidence_level", 0.95))
    metrics: dict[str, dict[str, object]] = {}
    for key, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        estimate = payload.get("estimate")
        lower = payload.get("lower")
        upper = payload.get("upper")
        replicates = payload.get("replicates")
        bootstrap_estimate = payload.get("bootstrap_estimate", estimate)
        metrics[key] = {
            "estimate": float(estimate) if estimate is not None else None,
            "bootstrap_estimate": float(bootstrap_estimate) if bootstrap_estimate is not None else None,
            "lower": float(lower) if lower is not None else None,
            "upper": float(upper) if upper is not None else None,
            "confidence": confidence,
            "replicates": int(replicates) if replicates is not None else None,
        }
    return {"metrics": metrics}


def _ci_estimate(ci_map: Mapping[str, Mapping[str, object]] | None, key: str) -> float | None:
    if not ci_map:
        return None
    data = ci_map.get(key)
    if not isinstance(data, Mapping):
        return None
    value = data.get("estimate")
    if value is None:
        value = data.get("bootstrap_estimate")
    return float(value) if value is not None else None


def _ci_map_to_metrics(ci_map: Mapping[str, Mapping[str, object]] | None) -> dict[str, float | None]:
    return {
        "mean_speed": _ci_estimate(ci_map, "mean_speed"),
        "p50": _ci_estimate(ci_map, "p50"),
        "p90": _ci_estimate(ci_map, "p90"),
        "p99": _ci_estimate(ci_map, "p99"),
        "power_density_model": _ci_estimate(ci_map, "power_density"),
    }


def _to_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _interval_lookup(ci_map: Mapping[str, Mapping[str, object]] | None, key: str) -> tuple[float | None, float | None]:
    if not ci_map:
        return (None, None)
    entry = ci_map.get(key)
    if not isinstance(entry, Mapping):
        return (None, None)
    return _to_float(entry.get("lower")), _to_float(entry.get("upper"))


def _bootstrap_lookup(ci_map: Mapping[str, Mapping[str, object]] | None, key: str) -> float | None:
    if not ci_map:
        return None
    entry = ci_map.get(key)
    if not isinstance(entry, Mapping):
        return None
    return _to_float(entry.get("bootstrap_estimate"))


def _difference(value_a: float | None, value_b: float | None) -> float | None:
    if value_a is None or value_b is None:
        return None
    return float(value_a) - float(value_b)


def _metric_lookup(metrics: Mapping[str, float | None] | None, key: str) -> float | None:
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get(key)
    if value is None:
        value = metrics.get(f"{key}_model")
    return _to_float(value)


def _compute_differences(ann_metrics: Mapping[str, float | None], buoy_metrics: Mapping[str, float | None]) -> dict[str, float | None]:
    return {
        "mean_speed": _difference(_metric_lookup(ann_metrics, "mean_speed"), _metric_lookup(buoy_metrics, "mean_speed")),
        "p50": _difference(_metric_lookup(ann_metrics, "p50"), _metric_lookup(buoy_metrics, "p50")),
        "p90": _difference(_metric_lookup(ann_metrics, "p90"), _metric_lookup(buoy_metrics, "p90")),
        "p99": _difference(_metric_lookup(ann_metrics, "p99"), _metric_lookup(buoy_metrics, "p99")),
        "power_density_model": _difference(
            _metric_lookup(ann_metrics, "power_density_model"),
            _metric_lookup(buoy_metrics, "power_density_model"),
        ),
    }


def _build_row(
    *,
    dataset: str,
    scope: str,
    sample_metrics: Mapping[str, float | None],
    sample_ci: Mapping[str, Mapping[str, object]] | None,
    model_metrics: Mapping[str, float | None],
    model_ci: Mapping[str, Mapping[str, object]] | None,
    censoring: Mapping[str, object] | None,
) -> dict[str, object]:
    mean_lower, mean_upper = _interval_lookup(sample_ci, "mean_speed")
    p50_lower, p50_upper = _interval_lookup(sample_ci, "p50")
    p90_lower, p90_upper = _interval_lookup(sample_ci, "p90")
    p99_lower, p99_upper = _interval_lookup(sample_ci, "p99")
    power_lower, power_upper = _interval_lookup(model_ci, "power_density_model")
    mean_model_lower, mean_model_upper = _interval_lookup(model_ci, "mean_speed_model")
    p50_model_lower, p50_model_upper = _interval_lookup(model_ci, "p50_model")
    p90_model_lower, p90_model_upper = _interval_lookup(model_ci, "p90_model")
    p99_model_lower, p99_model_upper = _interval_lookup(model_ci, "p99_model")
    power_bootstrap = _to_float(_bootstrap_lookup(model_ci, "power_density_model"))

    row = {
        "dataset": dataset,
        "scope": scope,
        "mean_speed": _to_float(sample_metrics.get("mean_speed")),
        "mean_speed_lower": mean_lower,
        "mean_speed_upper": mean_upper,
        "p50": _to_float(sample_metrics.get("p50")),
        "p50_lower": p50_lower,
        "p50_upper": p50_upper,
        "p90": _to_float(sample_metrics.get("p90")),
        "p90_lower": p90_lower,
        "p90_upper": p90_upper,
        "p99": _to_float(sample_metrics.get("p99")),
        "p99_lower": p99_lower,
        "p99_upper": p99_upper,
        "power_density_model": _to_float(model_metrics.get("power_density_model")),
        "power_density_model_bootstrap": power_bootstrap,
        "power_density_model_lower": power_lower,
        "power_density_model_upper": power_upper,
        "mean_speed_model": _to_float(model_metrics.get("mean_speed_model")),
        "mean_speed_model_lower": mean_model_lower,
        "mean_speed_model_upper": mean_model_upper,
        "p50_model": _to_float(model_metrics.get("p50_model")),
        "p50_model_lower": p50_model_lower,
        "p50_model_upper": p50_model_upper,
        "p90_model": _to_float(model_metrics.get("p90_model")),
        "p90_model_lower": p90_model_lower,
        "p90_model_upper": p90_model_upper,
        "p99_model": _to_float(model_metrics.get("p99_model")),
        "p99_model_lower": p99_model_lower,
        "p99_model_upper": p99_model_upper,
        "censored_ratio": _to_float((censoring or {}).get("censored_ratio")),
        "below_ratio": _to_float((censoring or {}).get("below_ratio")),
        "in_ratio": _to_float((censoring or {}).get("in_ratio")),
    }

    return row


def _extract_sample_metrics(section: Mapping[str, object]) -> dict[str, float | None]:
    return {
        "mean_speed": _to_float(section.get("mean_speed")),
        "p50": _to_float(section.get("p50")),
        "p90": _to_float(section.get("p90")),
        "p99": _to_float(section.get("p99")),
    }


def _extract_model_metrics(section: Mapping[str, object], *, fallback: Mapping[str, float | None] | None = None) -> dict[str, float | None]:
    metrics = {
        "power_density_model": _to_float(section.get("power_density_model")),
        "mean_speed_model": _to_float(section.get("mean_speed_model")),
        "p50_model": _to_float(section.get("p50_model")),
        "p90_model": _to_float(section.get("p90_model")),
        "p99_model": _to_float(section.get("p99_model")),
    }
    if fallback:
        if metrics["mean_speed_model"] is None:
            metrics["mean_speed_model"] = _to_float(fallback.get("mean_speed"))
        if metrics["p50_model"] is None:
            metrics["p50_model"] = _to_float(fallback.get("p50"))
        if metrics["p90_model"] is None:
            metrics["p90_model"] = _to_float(fallback.get("p90"))
        if metrics["p99_model"] is None:
            metrics["p99_model"] = _to_float(fallback.get("p99"))
    return metrics


def _build_rows_from_summary(summary: Mapping[str, object]) -> list[dict[str, object]]:
    ann = summary.get("ann", {})
    ann_bootstrap = summary.get("ann_bootstrap", {})
    buoy = summary.get("buoy", {})
    buoy_bootstrap = summary.get("buoy_bootstrap", {})

    ann_paired = ann.get("paired", {})
    ann_global = ann.get("global", {})
    ann_paired_sample = _extract_sample_metrics(ann_paired)
    ann_global_sample = _extract_sample_metrics(ann_global)
    ann_paired_model = _extract_model_metrics(ann_paired, fallback=ann_paired_sample)
    ann_global_model = _extract_model_metrics(ann_global, fallback=ann_global_sample)
    ann_paired_ci = (ann_bootstrap.get("paired") or {}).get("metrics")
    ann_global_ci = (ann_bootstrap.get("global") or {}).get("metrics")
    ann_paired_ci_model = _prepare_model_ci(ann_paired_ci)
    ann_global_ci_model = _prepare_model_ci(ann_global_ci)

    buoy_paired = buoy.get("paired", {})
    buoy_global = buoy.get("global", {})
    buoy_global_censored = buoy.get("global_censored", {})
    buoy_paired_sample = _extract_sample_metrics(buoy_paired.get("sample", {}))
    buoy_global_sample = _extract_sample_metrics(buoy_global.get("sample", {}))
    buoy_global_censored_sample = _extract_sample_metrics(buoy_global_censored.get("sample", {}))
    buoy_paired_model = _extract_model_metrics(buoy_paired.get("model", {}), fallback=buoy_paired_sample)
    buoy_global_model = _extract_model_metrics(buoy_global.get("model", {}), fallback=buoy_global_sample)
    buoy_global_censored_model = _extract_model_metrics(
        buoy_global_censored.get("model", {}),
        fallback=buoy_global_censored_sample,
    )

    buoy_paired_ci_sample = (buoy_bootstrap.get("paired") or {}).get("sample")
    buoy_paired_ci_model = _prepare_model_ci((buoy_bootstrap.get("paired") or {}).get("model"))
    buoy_global_ci_sample = (buoy_bootstrap.get("global") or {}).get("sample")
    buoy_global_ci_model = _prepare_model_ci((buoy_bootstrap.get("global") or {}).get("model"))
    buoy_global_censored_ci_sample = (buoy_bootstrap.get("global_censored") or {}).get("sample")
    buoy_global_censored_ci_model = _prepare_model_ci((buoy_bootstrap.get("global_censored") or {}).get("model"))

    censoring_summary = summary.get("censoring_summary_paired")
    buoy_global_censoring = (buoy_global_censored or {}).get("censoring")

    rows = [
        _build_row(
            dataset="ANN",
            scope="paired",
            sample_metrics=ann_paired_sample,
            sample_ci=ann_paired_ci,
            model_metrics=ann_paired_model,
            model_ci=ann_paired_ci_model,
            censoring=censoring_summary,
        ),
        _build_row(
            dataset="ANN",
            scope="global",
            sample_metrics=ann_global_sample,
            sample_ci=ann_global_ci,
            model_metrics=ann_global_model,
            model_ci=ann_global_ci_model,
            censoring=None,
        ),
        _build_row(
            dataset="Buoy",
            scope="paired",
            sample_metrics=buoy_paired_sample,
            sample_ci=buoy_paired_ci_sample,
            model_metrics=buoy_paired_model,
            model_ci=buoy_paired_ci_model,
            censoring=None,
        ),
        _build_row(
            dataset="Buoy",
            scope="global",
            sample_metrics=buoy_global_sample,
            sample_ci=buoy_global_ci_sample,
            model_metrics=buoy_global_model,
            model_ci=buoy_global_ci_model,
            censoring=None,
        ),
        _build_row(
            dataset="Buoy",
            scope="global_censored",
            sample_metrics=buoy_global_censored_sample,
            sample_ci=buoy_global_censored_ci_sample,
            model_metrics=buoy_global_censored_model,
            model_ci=buoy_global_censored_ci_model,
            censoring=buoy_global_censoring,
        ),
    ]

    return rows


def _find_row(rows: Sequence[Mapping[str, object]], dataset: str, scope: str) -> Mapping[str, object] | None:
    """Return the first row matching (dataset, scope) if present."""

    for row in rows:
        if str(row.get("dataset")) == dataset and str(row.get("scope")) == scope:
            return row
    return None


def _format_interval_string(
    estimate: float | None,
    lower: float | None,
    upper: float | None,
    *,
    decimals: int = 2,
) -> str:
    if estimate is None:
        return "—"
    formatted_estimate = f"{estimate:.{decimals}f}"
    if lower is None or upper is None:
        return formatted_estimate
    lower_str = f"{lower:.{decimals}f}"
    upper_str = f"{upper:.{decimals}f}"
    return f"{formatted_estimate} ({lower_str}–{upper_str})"


def _format_interval_value(row: Mapping[str, object], key: str, *, decimals: int = 2) -> str:
    estimate = _to_float(row.get(key))
    lower = _to_float(row.get(f"{key}_lower"))
    upper = _to_float(row.get(f"{key}_upper"))
    return _format_interval_string(estimate, lower, upper, decimals=decimals)


def _format_value(value: object | None, *, decimals: int = 2, suffix: str = "") -> str:
    number = _to_float(value)
    if number is None:
        return "—"
    formatted = f"{number:.{decimals}f}"
    return f"{formatted}{suffix}"


def _write_markdown_table(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    sample_count: int,
    main_report: str | None,
    ann_label: str,
    buoy_label: str,
) -> None:
    """Render a Markdown table summarising paired/global buoy validation metrics."""

    selection: list[tuple[str, str]] = [
        (ann_label, "paired"),
        (buoy_label, "paired"),
        (buoy_label, "global"),
    ]
    table_rows: list[str] = []
    header = [
        "# Vilano buoy validation snapshot",
        "",
        "| Dataset | Scope | Mean speed (m/s) | P90 (m/s) | Power density (W/m^2) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for dataset, scope in selection:
        row = _find_row(rows, dataset, scope)
        if row is None:
            continue
        mean_value = _format_interval_value(row, "mean_speed", decimals=2)
        p90_value = _format_interval_value(row, "p90", decimals=2)
        power_value = _format_interval_value(row, "power_density_model", decimals=1)
        table_rows.append(f"| {dataset} | {scope} | {mean_value} | {p90_value} | {power_value} |")

    if not table_rows:
        return

    lines = header + table_rows
    lines.append("")
    lines.append(f"*Paired sample size*: {sample_count:,} hourly records.")
    if main_report:
        lines.append(f"*Context*: See [{main_report}]({main_report}) for the full resource report.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_bias_svg(
    path: Path,
    *,
    differences: Mapping[str, object] | None,
    paired_ann: Mapping[str, object] | None,
    paired_buoy: Mapping[str, object] | None,
    sample_count: int,
) -> None:
    """Generate a lightweight SVG comparing ANN vs. buoy biases."""

    differences = differences or {}
    speed_series: list[tuple[str, float]] = []
    for key, label in (("mean_speed", "Mean speed"), ("p90", "P90")):
        value = _to_float(differences.get(key))
        if value is None:
            continue
        speed_series.append((label, value))
    power_value = _to_float(differences.get("power_density_model"))

    width, height = 880, 520
    margin = 40
    panel_height = (height - margin * 2) / 2 - 20
    panel_width = width - margin * 2
    speed_panel_top = margin
    power_panel_top = margin + panel_height + 40
    baseline_speed = speed_panel_top + panel_height / 2
    baseline_power = power_panel_top + panel_height / 2

    def _bars_to_svg(series: Sequence[tuple[str, float]], *, baseline: float, top: float) -> str:
        if not series:
            return (
                f'<text x="{width / 2:.1f}" y="{top + panel_height / 2:.1f}" '
                'text-anchor="middle" fill="#666666" font-size="16">No data available</text>'
            )
        max_value = max(abs(value) for _, value in series)
        if math.isclose(max_value, 0.0):
            max_value = 1.0
        scale = (panel_height / 2 - 30) / max_value
        bar_width = panel_width / max(len(series) * 2, 1)
        elements: list[str] = []
        for index, (label, value) in enumerate(series):
            bar_height = value * scale
            x = margin + bar_width / 2 + index * 2 * bar_width
            y = baseline - bar_height if value >= 0 else baseline
            height_value = abs(bar_height)
            color = "#1f77b4" if value >= 0 else "#d62728"
            elements.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{height_value:.1f}" '
                f'fill="{color}" opacity="0.85"/>'
            )
            label_y = baseline + 18
            elements.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{label_y:.1f}" '
                'text-anchor="middle" font-size="14" fill="#333333">'
                f'{label}</text>'
            )
            value_y = y - 8 if value >= 0 else y + height_value + 18
            elements.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{value_y:.1f}" text-anchor="middle" '
                'font-size="13" fill="#111111">'
                f'{value:+.2f}</text>'
            )
        axis = (
            f'<line x1="{margin}" y1="{baseline:.1f}" x2="{width - margin}" y2="{baseline:.1f}" '
            'stroke="#444444" stroke-dasharray="4 3" stroke-width="1.2"/>'
        )
        return axis + "".join(elements)

    speed_svg = _bars_to_svg(speed_series, baseline=baseline_speed, top=speed_panel_top)
    power_series = [("Power density", power_value)] if power_value is not None else []
    power_svg = _bars_to_svg(power_series, baseline=baseline_power, top=power_panel_top)

    paired_mean_ann = _format_value((paired_ann or {}).get("mean_speed"), decimals=2, suffix=" m/s")
    paired_mean_buoy = _format_value((paired_buoy or {}).get("mean_speed"), decimals=2, suffix=" m/s")
    paired_power_ann = _format_value((paired_ann or {}).get("power_density_model"), decimals=1, suffix=" W/m^2")
    paired_power_buoy = _format_value((paired_buoy or {}).get("power_density_model"), decimals=1, suffix=" W/m^2")

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{width / 2:.1f}" y="{speed_panel_top - 10:.1f}" text-anchor="middle" font-size="18" fill="#111111">
    Wind-speed bias (ANN - buoy)
  </text>
  {speed_svg}
  <text x="{width / 2:.1f}" y="{power_panel_top - 10:.1f}" text-anchor="middle" font-size="18" fill="#111111">
    Power-density bias (ANN - buoy)
  </text>
  {power_svg}
  <text x="{width / 2:.1f}" y="{height - 50:.1f}" text-anchor="middle" font-size="14" fill="#333333">
    {sample_count:,} paired hours – positive bars = ANN higher than buoy
  </text>
  <text x="{width / 2:.1f}" y="{height - 28:.1f}" text-anchor="middle" font-size="13" fill="#555555">
    ANN mean: {paired_mean_ann} | buoy mean: {paired_mean_buoy}
  </text>
  <text x="{width / 2:.1f}" y="{height - 10:.1f}" text-anchor="middle" font-size="13" fill="#555555">
    ANN power: {paired_power_ann} | buoy power: {paired_power_buoy}
  </text>
</svg>'''

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def _prepare_model_ci(ci_map: Mapping[str, Mapping[str, object]] | None) -> Mapping[str, Mapping[str, object]] | None:
    if not ci_map:
        return None
    prepared = dict(ci_map)
    if "power_density" in prepared and "power_density_model" not in prepared:
        prepared["power_density_model"] = prepared["power_density"]
    if "mean_speed" in prepared and "mean_speed_model" not in prepared:
        prepared["mean_speed_model"] = prepared["mean_speed"]
    if "p50" in prepared and "p50_model" not in prepared:
        prepared["p50_model"] = prepared["p50"]
    if "p90" in prepared and "p90_model" not in prepared:
        prepared["p90_model"] = prepared["p90"]
    if "p99" in prepared and "p99_model" not in prepared:
        prepared["p99_model"] = prepared["p99"]
    return prepared



def _compute_ann_paired_bootstrap(
    *,
    frame: pd.DataFrame,
    node_id: str,
    height: HeightCorrection,
    thresholds,
    power_curve: PowerCurve,
    air_density: float,
    min_confidence: float,
    min_in_range: float,
    tail_surrogate: float | None,
    km_criteria: KaplanMeierSelectionCriteria,
    metadata: dict[str, object],
    node_block_lengths: Mapping[str, int],
    replicates_override: int | None,
    confidence_override: float | None,
    seed_override: int | None,
) -> dict[str, object] | None:
    records = _build_ann_records(frame)
    if not records:
        return None

    default_replicas = int(metadata.get("replicas", 0))
    ann_resampling_mode = str(metadata.get("resampling_mode", "iid"))
    replicas = replicates_override if replicates_override is not None else default_replicas
    if replicas <= 0:
        return None
    confidence = confidence_override if confidence_override is not None else float(metadata.get("confidence_level", 0.95))
    random_seed = seed_override if seed_override is not None else metadata.get("random_seed")
    base_block_length = int(metadata.get("block_length", 1))
    node_block_length = int(node_block_lengths.get(node_id, base_block_length))

    config = StratifiedBootstrapConfig(
        replicas=replicas,
        confidence_level=confidence,
        random_seed=int(random_seed) if random_seed is not None else None,
        apply_rmse_noise=bool(metadata.get("apply_rmse_noise", True)),
        rmse_mode=str(metadata.get("rmse_mode", "velocity")),
        ci_method=str(metadata.get("ci_method", "percentile")),
        jackknife_max_samples=int(metadata.get("jackknife_max_samples", 200)),
        label_strategy=str(metadata.get("label_strategy", "fixed")),
        resampling_mode=ann_resampling_mode,
        block_length=base_block_length,
        node_block_lengths={node_id: node_block_length},
        air_density=air_density,
        lower_threshold=float(thresholds.lower),
        upper_threshold=float(thresholds.upper),
        min_confidence=min_confidence,
        min_in_range_weight=min_in_range,
        tail_surrogate=tail_surrogate,
        noise_truncation_multiplier=float(metadata.get("noise_truncation_multiplier", 4.0)),
        power_curve=power_curve,
        km_criteria=km_criteria,
    )
    input_data = NodeBootstrapInput(node_id=node_id, records=tuple(records), height=height)
    result = compute_stratified_bootstrap_uncertainty(
        input_data,
        config=config,
        rmse_provider=GlobalRmseProvider(),
    )
    summary = _bootstrap_result_to_summary(result)
    summary["config"] = {
        "replicas": replicas,
        "confidence_level": confidence,
        "random_seed": int(random_seed) if random_seed is not None else None,
        "apply_rmse_noise": bool(metadata.get("apply_rmse_noise", True)),
        "rmse_mode": str(metadata.get("rmse_mode", "velocity")),
        "label_strategy": str(metadata.get("label_strategy", "fixed")),
        "resampling_mode": ann_resampling_mode,
        "block_length": node_block_length,
    }
    return summary


def _load_buoy_height_config(summary_path: Path) -> HeightCorrectionConfig | None:
    if not summary_path.exists():
        return None
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    info = (data.get("buoy") or {}).get("height_correction")
    if not info:
        return None
    method = str(info.get("method", "log_profile")).lower()
    measurement = float(info.get("measurement_height_m", 3.0))
    target = float(info.get("target_height_m", 10.0))
    parameters = info.get("parameters", {})
    if method == "power_law":
        alpha = float(parameters.get("power_law_alpha", 0.11))
        return HeightCorrectionConfig(
            method="power_law",
            measurement_height_m=measurement,
            target_height_m=target,
            power_law_alpha=alpha,
        )
    roughness = float(parameters.get("roughness_length_m", 0.0002))
    return HeightCorrectionConfig(
        method="log_profile",
        measurement_height_m=measurement,
        target_height_m=target,
        roughness_length_m=roughness,
    )


def _compute_height_scale_from_profile(
    *,
    method: str,
    source_height: float,
    target_height: float,
    power_law_alpha: float | None,
    roughness_length_m: float | None,
) -> float:
    if source_height <= 0.0 or target_height <= 0.0:
        raise ValueError("Source and target heights must be positive.")
    if math.isclose(source_height, target_height, rel_tol=1e-9, abs_tol=1e-9):
        return 1.0
    if method == "log":
        roughness = roughness_length_m if roughness_length_m and roughness_length_m > 0.0 else 0.0002
        if source_height <= roughness or target_height <= roughness:
            raise ValueError("Heights must exceed the roughness length for log-law scaling.")
        numerator = math.log(target_height / roughness)
        denominator = math.log(source_height / roughness)
        if math.isclose(denominator, 0.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Log-law scaling denominator is zero; adjust heights or roughness.")
        return numerator / denominator
    if method == "power":
        alpha = power_law_alpha if power_law_alpha is not None else 0.11
        return (target_height / source_height) ** alpha
    return target_height / source_height


def _build_buoy_height_alignment(
    ann_height: HeightCorrection,
    buoy_height_config: HeightCorrectionConfig | None,
) -> tuple[HeightCorrection | None, dict[str, object] | None]:
    if buoy_height_config is None:
        return None, None
    dataset_height = float(buoy_height_config.target_height_m)
    if not math.isfinite(dataset_height) or dataset_height <= 0.0:
        return None, None
    target_height = float(ann_height.target_height_m)
    scale = _compute_height_scale_from_profile(
        method=ann_height.method,
        source_height=dataset_height,
        target_height=target_height,
        power_law_alpha=ann_height.power_law_alpha,
        roughness_length_m=ann_height.roughness_length_m,
    )
    alignment_height = HeightCorrection(
        method=ann_height.method,
        source_height_m=dataset_height,
        target_height_m=target_height,
        speed_scale=scale,
        power_law_alpha=ann_height.power_law_alpha,
        roughness_length_m=ann_height.roughness_length_m,
    )
    metadata = {
        "paired_series_height_m": dataset_height,
        "ann_target_height_m": target_height,
        "scale_applied": scale,
        "note": format_height_note(alignment_height),
    }
    return alignment_height, metadata


def _load_buoy_block_config(config_path: Path | None) -> dict[str, object]:
    if config_path is None:
        return {}
    path = config_path if config_path.is_absolute() else (REPO_ROOT / config_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in buoy block config: {path}") from exc
    return dict(payload or {})


def _load_comparison_config(config_path: Path | None) -> dict[str, object]:
    if config_path is None:
        return {}
    path = config_path if config_path.is_absolute() else (REPO_ROOT / config_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in comparison config: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Comparison config must be a JSON object with option keys")
    return dict(payload)


def _load_block_lengths_csv(path: Path | None, *, column: str = "suggested_block_length") -> dict[str, int]:
    if path is None:
        return {}
    resolved = path if path.is_absolute() else (REPO_ROOT / path)
    if not resolved.exists():
        return {}
    lengths: dict[str, int] = {}
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            node_id = row.get("node_id")
            value = row.get(column)
            if not node_id or value in (None, ""):
                continue
            try:
                lengths[node_id] = max(1, int(float(value)))
            except ValueError:
                continue
    return lengths


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_COMPARISON_CONFIG,
        help="Optional JSON file with default options for the comparison (default: config/buoy_comparison.json).",
    )

    preliminary, remaining = config_parser.parse_known_args()
    config_values = _load_comparison_config(preliminary.config)

    parser = argparse.ArgumentParser(
        description="Compare ANN resource estimates against the Vilano buoy reference.",
        parents=[config_parser],
    )
    parser.add_argument(
        "--matched-dataset",
        type=Path,
        default=DEFAULT_MATCHED_DATASET,
        help="GeoParquet with synchronised ANN/buoy records (default: artifacts/processed/vilano_buoy_synced.parquet).",
    )
    parser.add_argument(
        "--node-id",
        default="Vilano_buoy",
        help="ANN node identifier represented in the matched dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where comparison artefacts will be written (default: artifacts/buoy_validation).",
    )
    parser.add_argument(
        "--buoy-dataset",
        type=Path,
        default=Path("use_case/catalogs/pde_vilano_buoy/assets/Vilano.parquet"),
        help="Path to the full buoy dataset (used for global metrics).",
    )
    parser.add_argument(
        "--paired-summary",
        type=Path,
        default=Path("artifacts/processed/vilano_buoy_summary.json"),
        help="Summary JSON produced by scripts/prepare_buoy_timeseries.py (used to replicate height correction).",
    )
    parser.add_argument(
        "--range-thresholds",
        type=Path,
        default=DEFAULT_RANGE_THRESHOLDS,
        help="JSON file declaring the ANN regression range thresholds (default: config/range_thresholds.json).",
    )
    parser.add_argument(
        "--power-curve-config",
        type=Path,
        default=DEFAULT_POWER_CURVE_CONFIG,
        help="JSON with reference power curves (default: config/power_curves.json).",
    )
    parser.add_argument(
        "--power-curve-key",
        default=DEFAULT_POWER_CURVE_KEY,
        help="Key selecting the power curve from --power-curve-config (default: reference_offshore_6mw).",
    )
    parser.add_argument(
        "--height-config",
        type=Path,
        default=DEFAULT_HEIGHT_CONFIG,
        help="JSON with default height-correction parameters for ANN winds (default: config/power_height.json).",
    )
    parser.add_argument(
        "--air-density",
        type=float,
        default=1.225,
        help="Air density in kg/m^3 used for power-density calculations (default: 1.225).",
    )
    parser.add_argument(
        "--right-tail-surrogate",
        type=float,
        help="Optional surrogate wind speed (m/s) assigned to right-censored probability mass. Defaults to ANN upper threshold.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence required to treat ANN range flags as deterministic labels (default: 0.5).",
    )
    parser.add_argument(
        "--km-criteria-config",
        type=Path,
        default=None,
        help="Optional JSON overriding Kaplan–Meier selection thresholds (default: config/kaplan_meier_thresholds.json).",
    )
    parser.add_argument(
        "--min-in-range",
        type=float,
        default=500.0,
        help="Minimum in-range weight required to accept the Weibull fit as reliable (default: 500).",
    )
    parser.add_argument(
        "--bootstrap-summary",
        type=Path,
        default=DEFAULT_BOOTSTRAP_SUMMARY,
        help="CSV with bootstrap confidence intervals for ANN nodes (default: artifacts/bootstrap_velocity_block/bootstrap_summary.csv).",
    )
    parser.add_argument(
        "--bootstrap-metadata",
        type=Path,
        default=DEFAULT_BOOTSTRAP_METADATA,
        help="JSON metadata describing the ANN bootstrap run (default: artifacts/bootstrap_velocity_block/bootstrap_metadata.json).",
    )
    parser.add_argument(
        "--buoy-block-config",
        type=Path,
        default=DEFAULT_BUOY_BLOCK_CONFIG,
        help="JSON with buoy dependence diagnostics (default: artifacts/buoy_validation/buoy_block_bootstrap.json).",
    )
    parser.add_argument(
        "--buoy-resampling-mode",
        choices=("iid", "moving_block", "stationary"),
        default=None,
        help="Resampling strategy for buoy bootstrap (defaults to ANN mode when unset).",
    )
    parser.add_argument(
        "--buoy-block-length",
        type=int,
        default=None,
        help="Explicit block length for buoy bootstrap (overrides CSV/config when set).",
    )
    parser.add_argument(
        "--buoy-block-lengths-csv",
        type=Path,
        default=DEFAULT_BUOY_BLOCK_LENGTHS,
        help="CSV with per-node block-length recommendations (default: artifacts/buoy_block_diagnostics/block_bootstrap_diagnostics.csv).",
    )
    parser.add_argument(
        "--buoy-max-block-length",
        type=int,
        default=None,
        help="Cap the buoy block length when using recommendations (optional).",
    )
    parser.add_argument(
        "--buoy-bootstrap-replicates",
        type=int,
        default=DEFAULT_BUOY_BOOTSTRAP_REPLICATES,
        help="Number of bootstrap replicates used for buoy statistics (default: 500). Set to 0 to disable.",
    )
    parser.add_argument(
        "--buoy-bootstrap-confidence",
        type=float,
        default=DEFAULT_BUOY_BOOTSTRAP_CONFIDENCE,
        help="Confidence level for buoy bootstrap intervals (default: 0.95).",
    )
    parser.add_argument(
        "--buoy-bootstrap-seed",
        type=int,
        help="Optional random seed for buoy bootstrap resampling.",
    )
    parser.add_argument(
        "--ann-paired-bootstrap-replicates",
        type=int,
        help="Override the number of bootstrap replicates for the ANN paired subset (defaults to metadata).",
    )
    parser.add_argument(
        "--ann-paired-bootstrap-confidence",
        type=float,
        help="Override the confidence level for the ANN paired bootstrap (defaults to metadata).",
    )
    parser.add_argument(
        "--ann-paired-bootstrap-seed",
        type=int,
        help="Random seed for the ANN paired bootstrap resampling.",
    )
    parser.add_argument(
        "--ann-label",
        default=DEFAULT_ANN_LABEL,
        help="Dataset label used for ANN rows in the outputs (default: ANN).",
    )
    parser.add_argument(
        "--buoy-label",
        default=DEFAULT_BUOY_LABEL,
        help="Dataset label used for buoy rows in the outputs (default: Buoy).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing artefacts in the output directory.",
    )
    parser.add_argument(
        "--reuse-summary",
        action="store_true",
        help="Skip computations and regenerate resource_metrics.csv from an existing resource_comparison.json in the output directory.",
    )

    if config_values:
        valid_dests = {action.dest for action in parser._actions}
        filtered = {k: v for k, v in config_values.items() if k in valid_dests}
        unknown = sorted(set(config_values) - set(filtered))
        if unknown:
            print(
                f"Warning: ignoring unknown keys in comparison config: {', '.join(unknown)}",
                file=sys.stderr,
            )
        if filtered:
            parser.set_defaults(**filtered)

    args = parser.parse_args(remaining)
    return args


def _resolve_with_root(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _load_json(path: Path) -> Mapping[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in configuration: {path}") from exc


def _load_power_curve(path: Path, key: str) -> PowerCurve:
    payload = _load_json(path)
    if key not in payload:
        raise KeyError(f"Power-curve key '{key}' not present in {path}")
    entry = payload[key]
    return PowerCurve(
        name=str(entry.get("name", key)),
        speeds=tuple(float(value) for value in entry["speeds"]),
        power_kw=tuple(float(value) for value in entry["power_kw"]),
        reference_air_density=float(entry.get("reference_air_density", 1.225)),
        hub_height_m=float(entry["hub_height_m"]) if entry.get("hub_height_m") is not None else None,
        notes=tuple(str(note) for note in entry.get("notes", ())),
    )


def _load_height_defaults(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in height configuration: {path}") from exc
    return dict(payload or {})


def _resolve_height_correction(
    defaults: Mapping[str, object],
    *,
    power_curve: PowerCurve,
) -> HeightCorrection:
    method = str(defaults.get("method", "log")).lower()
    source = float(defaults.get("source_height_m", 10.0))
    target_default = defaults.get("target_height_m")
    if target_default is None:
        target_default = power_curve.hub_height_m if power_curve.hub_height_m is not None else source
    target = float(target_default)

    if source <= 0.0 or target <= 0.0:
        raise ValueError("Source and target heights must be positive.")

    if method == "none" or math.isclose(source, target, rel_tol=1e-9, abs_tol=1e-9):
        return HeightCorrection(method="none", source_height_m=source, target_height_m=target, speed_scale=1.0)

    if method == "log":
        roughness = float(defaults.get("roughness_length_m", 0.0002))
        if roughness <= 0.0:
            raise ValueError("Roughness length must be positive for log-law height correction.")
        if source <= roughness or target <= roughness:
            raise ValueError("Source/target heights must exceed roughness length for log-law correction.")
        numerator = math.log(target / roughness)
        denominator = math.log(source / roughness)
        if denominator == 0.0:
            raise ValueError("Log-law height correction produced zero denominator; adjust configuration.")
        speed_scale = numerator / denominator
        return HeightCorrection(
            method="log",
            source_height_m=source,
            target_height_m=target,
            speed_scale=speed_scale,
            power_law_alpha=None,
            roughness_length_m=roughness,
        )

    if method == "power":
        alpha = float(defaults.get("power_law_alpha", 0.11))
        speed_scale = (target / source) ** alpha
        return HeightCorrection(
            method="power",
            source_height_m=source,
            target_height_m=target,
            speed_scale=speed_scale,
            power_law_alpha=alpha,
            roughness_length_m=None,
        )

    raise ValueError(f"Unsupported height correction method: {method}")


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: Sequence[float],
) -> list[float | None]:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return [None for _ in quantiles]

    values = values[mask]
    weights = weights[mask]

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)
    total = cumulative[-1]
    if total <= 0.0:
        return [None for _ in quantiles]

    results: list[float | None] = []
    for q in quantiles:
        if not 0.0 <= q <= 1.0:
            results.append(None)
            continue
        target = q * total
        idx = np.searchsorted(cumulative, target, side="right")
        idx = min(idx, len(values) - 1)
        results.append(float(values[idx]))
    return results


def _compute_ann_metrics(
    frame: pd.DataFrame,
    *,
    thresholds,
    height: HeightCorrection,
    min_confidence: float,
    min_in_range: float,
    air_density: float,
    power_curve: PowerCurve,
    tail_surrogate: float,
    km_criteria: KaplanMeierSelectionCriteria,
) -> tuple[dict[str, object], dict[str, object]]:
    required_columns = [
        "pred_wind_speed",
        "prob_range_below",
        "prob_range_in",
        "prob_range_above",
        "range_flag",
        "range_flag_confident",
    ]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Matched dataset is missing ANN columns: {', '.join(missing)}")

    records = frame[required_columns].to_dict("records")
    censored = build_censored_data_from_records(
        records,
        lower_threshold=thresholds.lower,
        upper_threshold=thresholds.upper,
        min_confidence=min_confidence,
    )

    summary_row = summarise_records_for_selection(records, min_confidence=min_confidence)

    power_method, power_density, power_curve_estimate, weibull, km_result, method_notes, selection_reasons = (
        compute_power_distribution(
            data=censored,
            summary_row=summary_row,
            power_curve=power_curve,
            air_density=air_density,
            tail_surrogate=tail_surrogate,
            min_in_range=min_in_range,
            km_criteria=km_criteria,
            height=height,
        )
    )

    values = np.asarray(censored.in_values, dtype=float) * height.speed_scale
    weights = np.asarray(censored.in_weights, dtype=float)

    if values.size > 0 and np.sum(weights) > 0.0:
        mean_speed = float(np.average(values, weights=weights))
        p50, p90, p99 = _weighted_quantiles(values, weights, [0.5, 0.9, 0.99])
    else:
        mean_speed = None
        p50 = p90 = p99 = None

    censoring_summary = {
        "total_observations": float(summary_row.get("total_observations") or 0.0),
        "censored_ratio": float(summary_row.get("censored_ratio") or 0.0),
        "below_ratio": float(summary_row.get("below_ratio") or 0.0),
        "in_ratio": float(summary_row.get("in_ratio") or 0.0),
        "left_weight": censored.left_count,
        "right_weight": censored.right_count,
        "in_weight": censored.in_count,
    }

    ann_metrics: dict[str, object] = {
        "mean_speed": mean_speed,
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "power_density_model": power_density.estimate_w_per_m2 if power_density is not None else None,
        "power_method": power_method,
        "power_notes": tuple(method_notes),
        "censoring": censoring_summary,
        "height_correction": asdict(height),
        "height_note": format_height_note(height),
        "tail_surrogate": tail_surrogate,
        "selection_reasons": tuple(selection_reasons),
        "power_curve_estimate": {
            "mean_power_kw": power_curve_estimate.mean_power_kw,
            "capacity_factor": power_curve_estimate.capacity_factor,
            "air_density": power_curve_estimate.air_density,
            "notes": tuple(power_curve_estimate.notes),
        },
    }

    if weibull is not None:
        ann_metrics["weibull"] = {
            "success": weibull.success,
            "reliable": weibull.reliable,
            "shape": weibull.shape,
            "scale": weibull.scale,
            "in_weight": weibull.in_count,
            "left_weight": weibull.left_count,
            "right_weight": weibull.right_count,
        }
    if km_result is not None:
        ann_metrics["kaplan_meier"] = km_result.to_mapping()

    return ann_metrics, censoring_summary


def _compute_buoy_metrics(frame: pd.DataFrame, *, air_density: float, speed_scale: float = 1.0) -> dict[str, object]:
    if "wind_speed" not in frame.columns:
        raise ValueError("Matched dataset is missing the 'wind_speed' column for buoy observations.")

    speeds = frame["wind_speed"].to_numpy(dtype=float)
    speeds = speeds[np.isfinite(speeds)]
    if not math.isclose(speed_scale, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        speeds = speeds * speed_scale

    if speeds.size == 0:
        return {
            "sample_count": 0,
            "mean_speed": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "power_density": None,
        }

    return {
        "sample_count": int(speeds.size),
        "mean_speed": float(np.mean(speeds)),
        "p50": float(np.quantile(speeds, 0.5)),
        "p90": float(np.quantile(speeds, 0.9)),
        "p99": float(np.quantile(speeds, 0.99)),
    }


def _build_uncensored_dataset(speeds: np.ndarray) -> CensoredWeibullData | None:
    values = np.asarray(speeds, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return None
    weights = np.ones_like(values, dtype=float)
    return CensoredWeibullData(
        in_values=tuple(float(v) for v in values),
        in_weights=tuple(float(w) for w in weights),
        left_limits=(),
        left_weights=(),
        right_limits=(),
        right_weights=(),
    )

def _build_censored_dataset_from_thresholds(
    speeds: np.ndarray,
    *,
    lower_threshold: float | None,
    upper_threshold: float | None,
) -> CensoredWeibullData | None:
    values = np.asarray(speeds, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return None

    in_mask = np.ones_like(values, dtype=bool)
    if lower_threshold is not None:
        in_mask &= values >= lower_threshold
    if upper_threshold is not None:
        in_mask &= values <= upper_threshold

    in_values = values[in_mask]
    in_weights = np.ones_like(in_values, dtype=float)

    below_count = float(np.sum(values < lower_threshold)) if lower_threshold is not None else 0.0
    above_count = float(np.sum(values > upper_threshold)) if upper_threshold is not None else 0.0

    if lower_threshold is not None and below_count > 0.0:
        left_limits = (float(lower_threshold),)
        left_weights = (below_count,)
    else:
        left_limits = ()
        left_weights = ()

    if upper_threshold is not None and above_count > 0.0:
        right_limits = (float(upper_threshold),)
        right_weights = (above_count,)
    else:
        right_limits = ()
        right_weights = ()

    return CensoredWeibullData(
        in_values=tuple(float(v) for v in in_values),
        in_weights=tuple(float(w) for w in in_weights),
        left_limits=left_limits,
        left_weights=left_weights,
        right_limits=right_limits,
        right_weights=right_weights,
    )



def _build_buoy_summary(
    total_weight: float,
    *,
    left_count: float = 0.0,
    right_count: float = 0.0,
) -> dict[str, float]:
    total_weight = float(total_weight)
    left_count = float(left_count)
    right_count = float(right_count)
    in_count = max(total_weight - left_count - right_count, 0.0)
    censored = left_count + right_count
    return {
        "total_observations": total_weight,
        "censored_ratio": censored / total_weight if total_weight > 0.0 else 0.0,
        "below_ratio": left_count / total_weight if total_weight > 0.0 else 0.0,
        "in_ratio": in_count / total_weight if total_weight > 0.0 else 0.0,
        "valid_count": in_count,
    }


def _compute_buoy_censoring_summary(
    speeds: np.ndarray,
    *,
    lower: float | None,
    upper: float | None,
) -> dict[str, float | None]:
    total = float(speeds.size)
    if total <= 0.0:
        return {
            "total_count": 0.0,
            "below_count": 0.0,
            "above_count": 0.0,
            "in_count": 0.0,
            "censored_ratio": None,
            "below_ratio": None,
            "in_ratio": None,
        }

    below = float(np.sum(speeds < lower)) if lower is not None else 0.0
    above = float(np.sum(speeds > upper)) if upper is not None else 0.0
    in_count = max(total - below - above, 0.0)
    censored = below + above

    return {
        "total_count": total,
        "below_count": below,
        "above_count": above,
        "in_count": in_count,
        "censored_ratio": censored / total if total > 0.0 else None,
        "below_ratio": below / total if total > 0.0 else None,
        "in_ratio": in_count / total if total > 0.0 else None,
    }


def _weibull_statistics(shape: float | None, scale: float | None, *, probabilities: Sequence[float]) -> tuple[float | None, list[float | None]]:
    if shape is None or scale is None or shape <= 0.0 or scale <= 0.0:
        return None, [None for _ in probabilities]
    mean_value = scale * math.gamma(1.0 + 1.0 / shape)
    quantiles: list[float | None] = []
    for probability in probabilities:
        if probability <= 0.0:
            quantiles.append(0.0)
        elif probability >= 1.0:
            quantiles.append(None)
        else:
            quantiles.append(scale * (-math.log(1.0 - probability)) ** (1.0 / shape))
    return mean_value, quantiles


def _compute_buoy_model_metrics(
    speeds: np.ndarray,
    *,
    height: HeightCorrection,
    power_curve: PowerCurve,
    air_density: float,
    tail_surrogate: float,
    min_in_range: float,
    km_criteria: KaplanMeierSelectionCriteria,
    dataset: CensoredWeibullData | None = None,
) -> dict[str, object]:
    if dataset is None:
        dataset = _build_uncensored_dataset(speeds)
    if dataset is None:
        return {
            "method": "weibull",
            "mean_speed_model": None,
            "p50_model": None,
            "p90_model": None,
            "p99_model": None,
            "power_density_model": None,
            "power_density_notes": (),
            "power_density_air_density": air_density,
            "method_notes": (),
            "selection_reasons": (),
            "power_curve_estimate": None,
            "weibull": None,
            "kaplan_meier": None,
        }

    summary_row = _build_buoy_summary(
        dataset.total_weight,
        left_count=dataset.left_count,
        right_count=dataset.right_count,
    )
    method, power_density, power_curve_estimate, weibull, km_result, selection_reasons, method_notes = compute_power_distribution(
        data=dataset,
        summary_row=summary_row,
        power_curve=power_curve,
        air_density=air_density,
        tail_surrogate=tail_surrogate,
        min_in_range=min_in_range,
        km_criteria=km_criteria,
        height=height,
    )

    values = np.asarray(dataset.in_values, dtype=float)
    values = values[np.isfinite(values)]
    probabilities = (0.5, 0.9, 0.99)

    if method == "weibull" and weibull.success and weibull.shape and weibull.scale:
        scale = float(weibull.scale) * height.speed_scale
        shape = float(weibull.shape)
        mean_model, quantiles = _weibull_statistics(shape, scale, probabilities=probabilities)
    else:
        mean_model = float(np.mean(values)) if values.size > 0 else None
        quantiles = [
            float(np.quantile(values, prob)) if values.size > 0 else None for prob in probabilities
        ]
        if method == "kaplan_meier" and km_result is not None:
            temp_quantiles: list[float | None] = []
            for prob in probabilities:
                quantile_value = km_result.quantile(prob)
                if quantile_value is None and values.size > 0:
                    quantile_value = float(np.quantile(values, prob))
                temp_quantiles.append(float(quantile_value) if quantile_value is not None else None)
            quantiles = temp_quantiles

    model_payload: dict[str, object] = {
        "method": method,
        "mean_speed_model": mean_model,
        "p50_model": quantiles[0],
        "p90_model": quantiles[1],
        "p99_model": quantiles[2],
        "power_density_model": power_density.estimate_w_per_m2,
        "power_density_notes": tuple(power_density.notes),
        "power_density_air_density": power_density.air_density,
        "method_notes": tuple(method_notes),
        "selection_reasons": tuple(selection_reasons),
        "power_curve_estimate": {
            "mean_power_kw": power_curve_estimate.mean_power_kw,
            "capacity_factor": power_curve_estimate.capacity_factor,
            "air_density": power_curve_estimate.air_density,
            "notes": tuple(power_curve_estimate.notes),
        },
    }

    if weibull is not None:
        model_payload["weibull"] = {
            "success": weibull.success,
            "reliable": weibull.reliable,
            "shape": weibull.shape,
            "scale": weibull.scale,
            "in_weight": weibull.in_count,
            "left_weight": weibull.left_count,
            "right_weight": weibull.right_count,
        }
    if km_result is not None:
        model_payload["kaplan_meier"] = km_result.to_mapping()

    return model_payload

def _build_buoy_records(
    speeds: np.ndarray,
    *,
    lower_threshold: float | None,
    upper_threshold: float | None,
    apply_thresholds: bool,
) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    for idx, raw_value in enumerate(np.asarray(speeds, dtype=float)):
        if not math.isfinite(raw_value):
            continue
        speed = float(raw_value)
        prob_below = prob_in = prob_above = 0.0
        label = "in"
        if apply_thresholds and lower_threshold is not None and speed < float(lower_threshold):
            prob_below = 1.0
            label = "below"
        elif apply_thresholds and upper_threshold is not None and speed > float(upper_threshold):
            prob_above = 1.0
            label = "above"
        else:
            prob_in = 1.0
        records.append(
            {
                "timestamp": float(idx),
                "pred_wind_speed": speed,
                "prob_range_below": prob_below,
                "prob_range_in": prob_in,
                "prob_range_above": prob_above,
                "range_flag": label,
                "range_flag_confident": True,
            }
        )
    return tuple(records)


def _bootstrap_buoy_model_metrics(
    speeds: np.ndarray,
    *,
    air_density: float,
    replicates: int,
    confidence: float,
    seed: int | None,
    height: HeightCorrection,
    power_curve: PowerCurve,
    tail_surrogate: float,
    min_in_range: float,
    km_criteria: KaplanMeierSelectionCriteria,
    sample_estimates: Mapping[str, object],
    model_estimates: Mapping[str, object],
    lower_threshold: float | None = None,
    upper_threshold: float | None = None,
    resampling_mode: str = "iid",
    block_length: int = 1,
    apply_thresholds: bool = False,
) -> dict[str, object] | None:
    if replicates <= 0 or speeds.size == 0:
        return None
    if not (0.0 < confidence < 1.0):
        raise ValueError("Bootstrap confidence level must lie inside (0, 1).")

    finite_speeds = np.asarray(speeds, dtype=float)
    finite_speeds = finite_speeds[np.isfinite(finite_speeds)]
    if finite_speeds.size == 0:
        return None

    effective_lower = float(lower_threshold) if apply_thresholds and lower_threshold is not None else float(np.min(finite_speeds))
    effective_upper = float(upper_threshold) if apply_thresholds and upper_threshold is not None else float(np.max(finite_speeds))

    records = _build_buoy_records(
        speeds,
        lower_threshold=effective_lower if apply_thresholds else None,
        upper_threshold=effective_upper if apply_thresholds else None,
        apply_thresholds=apply_thresholds,
    )
    if not records:
        return None

    node_id = "buoy_resource"
    effective_block_length = max(1, int(block_length))
    jackknife_limit = max(1, min(len(records), 500))

    config_lower = effective_lower
    config_upper = effective_upper
    tail_value = float(tail_surrogate) if tail_surrogate is not None else None

    config = StratifiedBootstrapConfig(
        replicas=int(replicates),
        confidence_level=float(confidence),
        random_seed=int(seed) if seed is not None else None,
        apply_rmse_noise=False,
        rmse_mode="none",
        ci_method="bca",
        jackknife_max_samples=jackknife_limit,
        label_strategy="fixed",
        resampling_mode=resampling_mode,
        block_length=effective_block_length,
        node_block_lengths={node_id: effective_block_length},
        air_density=float(air_density),
        lower_threshold=config_lower,
        upper_threshold=config_upper,
        min_confidence=0.5,
        min_in_range_weight=float(min_in_range),
        tail_surrogate=tail_value,
        noise_truncation_multiplier=4.0,
        power_curve=power_curve,
        km_criteria=km_criteria,
    )

    result = compute_stratified_bootstrap_uncertainty(
        NodeBootstrapInput(node_id=node_id, records=records, height=height),
        config=config,
        rmse_provider=GlobalRmseProvider(),
    )

    summary = _bootstrap_result_to_summary(result)
    metrics_map = summary.get("metrics", {})
    bootstrap_means = summary.get("bootstrap_means", {})

    def _prepare_interval(name: str, default: object | None) -> dict[str, object | None]:
        payload: dict[str, object | None] = {
            "estimate": _to_float(default),
            "lower": None,
            "upper": None,
            "confidence": float(confidence),
            "replicates": int(replicates),
            "notes": [],
        }
        entry = metrics_map.get(name)
        if isinstance(entry, Mapping):
            estimate_value = entry.get("estimate")
            if estimate_value is None:
                estimate_value = entry.get("bootstrap_estimate")
            payload["estimate"] = _to_float(estimate_value) if estimate_value is not None else payload["estimate"]
            payload["lower"] = _to_float(entry.get("lower"))
            payload["upper"] = _to_float(entry.get("upper"))
            payload["confidence"] = _to_float(entry.get("confidence")) or float(confidence)
            replicates_value = entry.get("replicates")
            payload["replicates"] = int(replicates_value) if replicates_value is not None else int(replicates)
            payload["notes"] = list(entry.get("notes") or ())
            bootstrap_estimate = entry.get("bootstrap_estimate")
            if bootstrap_estimate is not None and math.isfinite(bootstrap_estimate):
                payload["bootstrap_estimate"] = float(bootstrap_estimate)
        mean_value = bootstrap_means.get(name)
        if mean_value is not None and math.isfinite(mean_value):
            payload["bootstrap_estimate"] = float(mean_value)
        return payload

    sample_intervals = {
        "mean_speed": _prepare_interval("mean_speed", sample_estimates.get("mean_speed")),
        "p50": _prepare_interval("p50", sample_estimates.get("p50")),
        "p90": _prepare_interval("p90", sample_estimates.get("p90")),
        "p99": _prepare_interval("p99", sample_estimates.get("p99")),
    }

    model_intervals = {
        "mean_speed_model": _prepare_interval("mean_speed", model_estimates.get("mean_speed_model")),
        "p50_model": _prepare_interval("p50", model_estimates.get("p50_model")),
        "p90_model": _prepare_interval("p90", model_estimates.get("p90_model")),
        "p99_model": _prepare_interval("p99", model_estimates.get("p99_model")),
        "power_density_model": _prepare_interval("power_density", model_estimates.get("power_density_model")),
    }

    payload: dict[str, object] = {
        "replicates": int(replicates),
        "confidence": float(confidence),
        "seed": int(seed) if seed is not None else None,
        "sample": sample_intervals,
        "model": model_intervals,
    }
    if summary.get("notes"):
        payload["notes"] = summary["notes"]
    diagnostics = summary.get("power_diagnostics")
    if diagnostics is not None:
        payload["power_diagnostics"] = diagnostics

    return payload


def _load_bootstrap_intervals(path: Path, node_id: str) -> dict[str, object] | None:
    if not path.exists():
        return None

    metric_keys = ("mean_speed", "p50", "p90", "p99", "power_density")

    def _to_int_value(value: object | None) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return None

    def _record_to_metrics(record: Mapping[str, object]) -> dict[str, object]:
        metrics: dict[str, object] = {}
        metrics_section = record.get("metrics")
        if isinstance(metrics_section, Mapping):
            for key in metric_keys:
                entry = metrics_section.get(key)
                if isinstance(entry, Mapping):
                    metrics[key] = {
                        "estimate": _to_float(entry.get("estimate")),
                        "lower": _to_float(entry.get("lower")),
                        "upper": _to_float(entry.get("upper")),
                        "bootstrap_estimate": _to_float(
                            entry.get("bootstrap_estimate") or entry.get("mean") or entry.get("bootstrap_mean")
                        ),
                        "replicates": _to_int_value(entry.get("replicates")),
                        "confidence": _to_float(entry.get("confidence")),
                    }
                else:
                    metrics[key] = {
                        "estimate": None,
                        "lower": None,
                        "upper": None,
                        "bootstrap_estimate": None,
                        "replicates": None,
                        "confidence": None,
                    }
        else:
            for key in metric_keys:
                estimate = record.get(f"{key}_estimate")
                lower = record.get(f"{key}_lower")
                upper = record.get(f"{key}_upper")
                bootstrap_estimate = record.get(f"{key}_bootstrap_estimate")
                replicates = record.get(f"{key}_replicates")
                metrics[key] = {
                    "estimate": _to_float(estimate),
                    "lower": _to_float(lower),
                    "upper": _to_float(upper),
                    "bootstrap_estimate": _to_float(bootstrap_estimate),
                    "replicates": _to_int_value(replicates),
                    "confidence": _to_float(record.get(f"{key}_confidence")),
                }

        rmse_record = record.get("rmse_record")
        if isinstance(rmse_record, Mapping):
            metrics["rmse_value"] = _to_float(rmse_record.get("value"))
            metrics["rmse_source"] = rmse_record.get("source")
        else:
            metrics["rmse_value"] = _to_float(record.get("rmse_value"))
            metrics["rmse_source"] = record.get("rmse_source")

        diagnostics = record.get("power_diagnostics")
        if isinstance(diagnostics, Mapping):
            metrics["power_method"] = diagnostics.get("method")
            metrics["power_notes"] = " | ".join(diagnostics.get("method_notes", [])) if diagnostics.get("method_notes") else None
            metrics["selection_reasons"] = " | ".join(diagnostics.get("selection_reasons", []))
        else:
            metrics["power_method"] = record.get("power_method")
            metrics["power_notes"] = record.get("power_method_notes")
            metrics["selection_reasons"] = record.get("power_selection_reasons")
        return metrics

    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            frame = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(f"Failed to read bootstrap summary: {path}") from exc

        subset = frame.loc[frame["node_id"] == node_id]
        if subset.empty:
            return None
        row = subset.iloc[0].to_dict()
        return _record_to_metrics(row)

    records: Iterable[Mapping[str, object]]
    if suffix == ".jsonl":
        parsed: list[Mapping[str, object]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                data = json.loads(payload)
                if isinstance(data, Mapping):
                    parsed.append(data)
        records = parsed
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records = [item for item in payload if isinstance(item, Mapping)]
        elif isinstance(payload, Mapping):
            records = [value for value in payload.values() if isinstance(value, Mapping)]
        else:  # pragma: no cover - defensive guard
            records = []
    else:
        raise RuntimeError(f"Unsupported bootstrap summary format: {path.suffix}")

    for record in records:
        if str(record.get("node_id")) != node_id:
            continue
        return _record_to_metrics(record)
    return None


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    frame = pd.DataFrame(list(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    ann_label = args.ann_label
    buoy_label = args.buoy_label

    output_dir = _resolve_with_root(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "resource_comparison.json"
    csv_path = output_dir / "resource_metrics.csv"
    if args.reuse_summary:
        if not json_path.exists():
            raise FileNotFoundError(
                f"Existing summary not found at {json_path}. Run without --reuse-summary to generate it first."
            )
        if csv_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Artefacts already exist under {output_dir}. Pass --overwrite to replace them."
            )
        summary = _load_json(json_path)
        rows = _build_rows_from_summary(summary)
        _write_csv(csv_path, rows)
        print(f"Reused existing summary at {json_path}")
        print(f"Tabular metrics written to {csv_path}")
        return

    matched_path = _resolve_with_root(args.matched_dataset)
    if not matched_path.exists():
        raise FileNotFoundError(f"Matched dataset not found: {matched_path}")

    if not args.overwrite and (json_path.exists() or csv_path.exists()):
        raise FileExistsError(
            f"Artefacts already exist under {output_dir}. Pass --overwrite to replace them."
        )

    frame = pd.read_parquet(matched_path)
    sample_count = int(len(frame))

    thresholds = load_range_thresholds(args.range_thresholds)
    power_curve = _load_power_curve(_resolve_with_root(args.power_curve_config), args.power_curve_key)
    height_defaults = _load_height_defaults(_resolve_with_root(args.height_config))
    height = _resolve_height_correction(height_defaults, power_curve=power_curve)
    tail_surrogate = args.right_tail_surrogate if args.right_tail_surrogate is not None else thresholds.upper
    km_criteria = load_kaplan_meier_selection_criteria(args.km_criteria_config)
    buoy_height_config = _load_buoy_height_config(_resolve_with_root(args.paired_summary))
    buoy_alignment_height, buoy_alignment_metadata = _build_buoy_height_alignment(height, buoy_height_config)
    buoy_height_for_models = buoy_alignment_height if buoy_alignment_height is not None else IDENTITY_HEIGHT
    buoy_speed_scale = buoy_alignment_height.speed_scale if buoy_alignment_height is not None else 1.0

    raw_speeds = frame["wind_speed"].to_numpy(dtype=float)
    raw_speeds = raw_speeds[np.isfinite(raw_speeds)]

    ann_metrics, censoring_summary = _compute_ann_metrics(
        frame,
        thresholds=thresholds,
        height=height,
        min_confidence=args.min_confidence,
        min_in_range=args.min_in_range,
        air_density=args.air_density,
        power_curve=power_curve,
        tail_surrogate=tail_surrogate,
        km_criteria=km_criteria,
    )
    bootstrap_metadata_path = _resolve_with_root(args.bootstrap_metadata)
    metadata: dict[str, object]
    node_block_lengths: dict[str, int]
    try:
        metadata, node_block_lengths = _load_ann_bootstrap_metadata(bootstrap_metadata_path)
    except FileNotFoundError:
        ann_replicates = args.ann_paired_bootstrap_replicates
        if ann_replicates is None or ann_replicates > 0:
            raise
        metadata = {}
        node_block_lengths = {}
    ann_resampling_mode = str(metadata.get("resampling_mode", "iid"))
    ann_base_block_length = int(metadata.get("block_length", 1))
    node_block_length = int(node_block_lengths.get(args.node_id, ann_base_block_length))
    buoy_block_config = _load_buoy_block_config(_resolve_with_root(args.buoy_block_config))
    buoy_resampling_mode = str(
        args.buoy_resampling_mode or buoy_block_config.get("resampling_mode", ann_resampling_mode)
    )
    if buoy_resampling_mode not in {"iid", "moving_block", "stationary"}:
        buoy_resampling_mode = ann_resampling_mode

    buoy_block_lengths = _load_block_lengths_csv(_resolve_with_root(args.buoy_block_lengths_csv))
    buoy_block_length_value = buoy_block_config.get("suggested_block_length", buoy_block_config.get("block_length"))
    if args.buoy_block_length is not None:
        buoy_block_length_value = args.buoy_block_length
    elif args.node_id in buoy_block_lengths:
        buoy_block_length_value = buoy_block_lengths[args.node_id]

    try:
        buoy_block_length = int(buoy_block_length_value) if buoy_block_length_value is not None else node_block_length
    except (TypeError, ValueError):
        buoy_block_length = node_block_length
    if args.buoy_max_block_length is not None and args.buoy_max_block_length > 0:
        buoy_block_length = min(buoy_block_length, int(args.buoy_max_block_length))
    if buoy_block_length <= 0:
        buoy_block_length = node_block_length

    ann_paired_bootstrap = _compute_ann_paired_bootstrap(
        frame=frame,
        node_id=args.node_id,
        height=height,
        thresholds=thresholds,
        power_curve=power_curve,
        air_density=args.air_density,
        min_confidence=args.min_confidence,
        min_in_range=args.min_in_range,
        tail_surrogate=tail_surrogate,
        km_criteria=km_criteria,
        metadata=metadata,
        node_block_lengths=node_block_lengths,
        replicates_override=args.ann_paired_bootstrap_replicates,
        confidence_override=args.ann_paired_bootstrap_confidence,
        seed_override=args.ann_paired_bootstrap_seed,
    )
    ann_paired_ci = ann_paired_bootstrap["metrics"] if ann_paired_bootstrap is not None else None

    bootstrap_path = _resolve_with_root(args.bootstrap_summary)
    ann_global_raw = _load_bootstrap_intervals(bootstrap_path, args.node_id)
    if ann_global_raw is None:
        raise RuntimeError(
            "Bootstrap summary for ANN node not found. Please execute "
            f"'python scripts/generate_bootstrap_uncertainty.py --dataset {matched_path}' "
            f"(or the equivalent CLI pipeline) so that {bootstrap_path} includes node '{args.node_id}'."
        )
    ann_global_summary = _normalize_ann_global_bootstrap(ann_global_raw, metadata=metadata)
    ann_global_ci = ann_global_summary["metrics"]

    ann_metrics["mean_speed_model"] = ann_metrics.get("mean_speed")
    ann_metrics["p50_model"] = ann_metrics.get("p50")
    ann_metrics["p90_model"] = ann_metrics.get("p90")
    ann_metrics["p99_model"] = ann_metrics.get("p99")

    ann_global_metrics = _ci_map_to_metrics(ann_global_ci)
    ann_global_metrics["mean_speed_model"] = ann_global_metrics.get("mean_speed")
    ann_global_metrics["p50_model"] = ann_global_metrics.get("p50")
    ann_global_metrics["p90_model"] = ann_global_metrics.get("p90")
    ann_global_metrics["p99_model"] = ann_global_metrics.get("p99")

    buoy_metrics = _compute_buoy_metrics(frame, air_density=args.air_density, speed_scale=buoy_speed_scale)
    tail_surrogate_buoy = tail_surrogate
    buoy_model = _compute_buoy_model_metrics(
        raw_speeds,
        height=buoy_height_for_models,
        power_curve=power_curve,
        air_density=args.air_density,
        tail_surrogate=tail_surrogate_buoy,
        min_in_range=args.min_in_range,
        km_criteria=km_criteria,
    )
    buoy_bootstrap = _bootstrap_buoy_model_metrics(
        raw_speeds,
        air_density=args.air_density,
        replicates=args.buoy_bootstrap_replicates,
        confidence=args.buoy_bootstrap_confidence,
        seed=args.buoy_bootstrap_seed,
        height=buoy_height_for_models,
        power_curve=power_curve,
        tail_surrogate=tail_surrogate_buoy,
        min_in_range=args.min_in_range,
        km_criteria=km_criteria,
        sample_estimates=buoy_metrics,
        model_estimates=buoy_model,
        lower_threshold=thresholds.lower,
        upper_threshold=thresholds.upper,
        resampling_mode=buoy_resampling_mode,
        block_length=buoy_block_length,
        apply_thresholds=False,
    )

    buoy_full_series = load_buoy_timeseries(
        _resolve_with_root(args.buoy_dataset),
        sentinel_config=BuoySentinelConfig(),
        height_correction=buoy_height_config,
    )
    buoy_full_speeds = buoy_full_series.dataframe["wind_speed"].to_numpy(dtype=float)
    buoy_full_speeds = buoy_full_speeds[np.isfinite(buoy_full_speeds)]
    buoy_global_frame = pd.DataFrame({"wind_speed": buoy_full_speeds})
    buoy_global_metrics = _compute_buoy_metrics(
        buoy_global_frame,
        air_density=args.air_density,
        speed_scale=buoy_speed_scale,
    )
    buoy_global_model = _compute_buoy_model_metrics(
        buoy_full_speeds,
        height=buoy_height_for_models,
        power_curve=power_curve,
        air_density=args.air_density,
        tail_surrogate=tail_surrogate_buoy,
        min_in_range=args.min_in_range,
        km_criteria=km_criteria,
    )
    buoy_global_bootstrap = _bootstrap_buoy_model_metrics(
        buoy_full_speeds,
        air_density=args.air_density,
        replicates=args.buoy_bootstrap_replicates,
        confidence=args.buoy_bootstrap_confidence,
        seed=args.buoy_bootstrap_seed,
        height=buoy_height_for_models,
        power_curve=power_curve,
        tail_surrogate=tail_surrogate_buoy,
        min_in_range=args.min_in_range,
        km_criteria=km_criteria,
        sample_estimates=buoy_global_metrics,
        model_estimates=buoy_global_model,
        lower_threshold=thresholds.lower,
        upper_threshold=thresholds.upper,
        resampling_mode=buoy_resampling_mode,
        block_length=buoy_block_length,
        apply_thresholds=False,
    )

    buoy_global_censoring = _compute_buoy_censoring_summary(
        buoy_full_speeds,
        lower=thresholds.lower,
        upper=thresholds.upper,
    )

    censored_dataset = _build_censored_dataset_from_thresholds(
        buoy_full_speeds,
        lower_threshold=thresholds.lower,
        upper_threshold=thresholds.upper,
    )
    if censored_dataset is None:
        buoy_global_censored_speeds = np.array([], dtype=float)
    else:
        buoy_global_censored_speeds = np.asarray(censored_dataset.in_values, dtype=float)

    buoy_global_censored_frame = pd.DataFrame({"wind_speed": buoy_global_censored_speeds})
    buoy_global_censored_metrics = _compute_buoy_metrics(
        buoy_global_censored_frame,
        air_density=args.air_density,
        speed_scale=buoy_speed_scale,
    )
    buoy_global_censored_model = _compute_buoy_model_metrics(
        buoy_full_speeds,
        height=buoy_height_for_models,
        power_curve=power_curve,
        air_density=args.air_density,
        tail_surrogate=tail_surrogate_buoy,
        min_in_range=args.min_in_range,
        km_criteria=km_criteria,
        dataset=censored_dataset,
    )
    buoy_global_censored_bootstrap = _bootstrap_buoy_model_metrics(
        buoy_full_speeds,
        air_density=args.air_density,
        replicates=args.buoy_bootstrap_replicates,
        confidence=args.buoy_bootstrap_confidence,
        seed=args.buoy_bootstrap_seed,
        height=buoy_height_for_models,
        power_curve=power_curve,
        tail_surrogate=tail_surrogate_buoy,
        min_in_range=args.min_in_range,
        km_criteria=km_criteria,
        sample_estimates=buoy_global_censored_metrics,
        model_estimates=buoy_global_censored_model,
        lower_threshold=thresholds.lower,
        upper_threshold=thresholds.upper,
        resampling_mode=buoy_resampling_mode,
        block_length=buoy_block_length,
        apply_thresholds=True,
    )


    ann_paired_metrics_for_diff = dict(ann_metrics)
    ann_paired_metrics_for_diff["power_density_model"] = ann_metrics.get("power_density_model")
    ann_global_metrics_for_diff = dict(ann_global_metrics)

    buoy_model_for_diff = dict(buoy_model)
    buoy_global_model_for_diff = dict(buoy_global_model)
    buoy_global_censored_model_for_diff = dict(buoy_global_censored_model)

    differences = {
        "paired": _compute_differences(ann_paired_metrics_for_diff, buoy_model_for_diff),
        "global": _compute_differences(ann_global_metrics_for_diff, buoy_global_model_for_diff),
        "global_censored": _compute_differences(ann_global_metrics_for_diff, buoy_global_censored_model_for_diff),
    }

    summary = {
        "node_id": args.node_id,
        "matched_dataset": str(matched_path),
        "sample_count_paired": sample_count,
        "air_density": args.air_density,
        "range_thresholds": {"lower": thresholds.lower, "upper": thresholds.upper},
        "ann": {
            "paired": ann_metrics,
            "global": ann_global_metrics,
        },
        "ann_bootstrap": {
            "paired": ann_paired_bootstrap,
            "global": ann_global_summary,
        },
        "buoy": {
            "paired": {
                "sample": buoy_metrics,
                "model": buoy_model,
            },
            "global": {
                "sample": buoy_global_metrics,
                "model": buoy_global_model,
            },
            "global_censored": {
                "sample": buoy_global_censored_metrics,
                "model": buoy_global_censored_model,
                "censoring": buoy_global_censoring,
                "sample_count": int(buoy_global_censored_speeds.size),
            },
        },
        "buoy_bootstrap": {
            "paired": buoy_bootstrap,
            "global": buoy_global_bootstrap,
            "global_censored": buoy_global_censored_bootstrap,
        },
        "differences": differences,
        "censoring_summary_paired": censoring_summary,
        "censoring_summary_buoy_global": buoy_global_censoring,
        "comparison_config": str(_resolve_with_root(args.config)) if args.config is not None else None,
        "buoy_resampling_config": {
            "mode": buoy_resampling_mode,
            "block_length": buoy_block_length,
            "source": str(_resolve_with_root(args.buoy_block_config)),
            "block_lengths_csv": str(_resolve_with_root(args.buoy_block_lengths_csv)),
            "max_block_length": args.buoy_max_block_length,
            "block_length_override": args.buoy_block_length,
            "resampling_mode_override": args.buoy_resampling_mode,
        },
        "power_curve": {
            "name": power_curve.name,
            "reference_air_density": power_curve.reference_air_density,
            "hub_height_m": power_curve.hub_height_m,
            "rated_power_kw": power_curve.rated_power_kw,
        },
    }
    if buoy_alignment_metadata is not None:
        summary["buoy_height_alignment"] = buoy_alignment_metadata

    _write_json(json_path, summary)

    ann_global_ci_model = _prepare_model_ci(ann_global_ci)
    ann_paired_ci_model = _prepare_model_ci(ann_paired_ci)
    buoy_paired_sample_ci = buoy_bootstrap["sample"] if buoy_bootstrap is not None else None
    buoy_paired_model_ci = _prepare_model_ci(buoy_bootstrap["model"]) if buoy_bootstrap is not None else None
    buoy_global_sample_ci = buoy_global_bootstrap["sample"] if buoy_global_bootstrap is not None else None
    buoy_global_model_ci = _prepare_model_ci(buoy_global_bootstrap["model"]) if buoy_global_bootstrap is not None else None
    buoy_global_censored_sample_ci = (
        buoy_global_censored_bootstrap["sample"] if buoy_global_censored_bootstrap is not None else None
    )
    buoy_global_censored_model_ci = (
        _prepare_model_ci(buoy_global_censored_bootstrap["model"])
        if buoy_global_censored_bootstrap is not None
        else None
    )

    csv_rows = [
        _build_row(
            dataset=ann_label,
            scope="paired",
            sample_metrics=ann_metrics,
            sample_ci=ann_paired_ci,
            model_metrics=ann_metrics,
            model_ci=ann_paired_ci_model,
            censoring=censoring_summary,
        ),
        _build_row(
            dataset=ann_label,
            scope="global",
            sample_metrics=ann_global_metrics,
            sample_ci=ann_global_ci,
            model_metrics=ann_global_metrics,
            model_ci=ann_global_ci_model,
            censoring=None,
        ),
        _build_row(
            dataset=buoy_label,
            scope="paired",
            sample_metrics=buoy_metrics,
            sample_ci=buoy_paired_sample_ci,
            model_metrics=buoy_model,
            model_ci=buoy_paired_model_ci,
            censoring=None,
        ),
        _build_row(
            dataset=buoy_label,
            scope="global",
            sample_metrics=buoy_global_metrics,
            sample_ci=buoy_global_sample_ci,
            model_metrics=buoy_global_model,
            model_ci=buoy_global_model_ci,
            censoring=None,
        ),
        _build_row(
            dataset=buoy_label,
            scope="global_censored",
            sample_metrics=buoy_global_censored_metrics,
            sample_ci=buoy_global_censored_sample_ci,
            model_metrics=buoy_global_censored_model,
            model_ci=buoy_global_censored_model_ci,
            censoring=buoy_global_censoring,
        ),
    ]
    _write_csv(csv_path, csv_rows)

    table_path = output_dir / "resource_metrics_table.md"
    _write_markdown_table(
        table_path,
        csv_rows,
        sample_count=sample_count,
        main_report="docs/empirical_metrics_summary.md",
        ann_label=ann_label,
        buoy_label=buoy_label,
    )
    bias_svg_path = output_dir / "resource_bias.svg"
    _write_bias_svg(
        bias_svg_path,
        differences=differences.get("paired"),
        paired_ann=_find_row(csv_rows, ann_label, "paired"),
        paired_buoy=_find_row(csv_rows, buoy_label, "paired"),
        sample_count=sample_count,
    )

    print(f"Resource comparison written to {json_path}")
    print(f"Tabular metrics written to {csv_path}")
    print(f"Markdown summary written to {table_path}")
    print(f"Bias chart written to {bias_svg_path}")


if __name__ == "__main__":
    main()
