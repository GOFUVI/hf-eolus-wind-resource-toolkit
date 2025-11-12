#!/usr/bin/env python3
"""Generate seasonal and interannual diagnostics for ANN wind-speed data."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from hf_wind_resource.io import resolve_catalog_asset
from hf_wind_resource.stats import SeasonalAnalysisResult, compute_seasonal_analysis

DEFAULT_IMAGE = "duckdb/duckdb:latest"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("artifacts/seasonal_analysis")
DEFAULT_REPORT_PATH = Path("artifacts/seasonal_analysis/seasonal_variation_summary.md")
DEFAULT_STAC_CONFIG = Path("config/stac_catalogs.json")
DEFAULT_STAC_DATASET = "sar_range_final_pivots_joined"
HEIGHT_METADATA_CANDIDATES: Sequence[Path] = (
    Path("artifacts/power_estimates/metadata.json"),
    Path("artifacts/power_estimates/power_estimates_summary.csv"),
)

def _resolve_with_root(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _run_duckdb(sql: str, *, workdir: Path, image: str = DEFAULT_IMAGE) -> str:
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
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - exercised via CLI failures
        raise RuntimeError(f"DuckDB command failed: {exc.stderr}") from exc
    return completed.stdout


def _fetch_observations(
    dataset: Path,
    *,
    workdir: Path,
    image: str = DEFAULT_IMAGE,
    engine: str = "docker",
) -> pd.DataFrame:
    sql = textwrap.dedent(
        f"""
        SELECT
          timestamp,
          node_id,
          pred_wind_speed,
          pred_range_label
        FROM read_parquet('{{path}}');
        """
    )

    if engine == "python":
        try:
            import duckdb  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError("DuckDB Python package not available; cannot use execution-mode=python.") from exc

        dataset_path = (workdir / dataset).resolve()
        query = sql.format(path=dataset_path.as_posix())
        connection = duckdb.connect(database=":memory:")
        frame = connection.execute(query).fetch_df()  # type: ignore[attr-defined]
        connection.close()
        return frame

    query = sql.format(path=dataset.as_posix())
    raw_csv = _run_duckdb(query, workdir=workdir, image=image)
    if not raw_csv.strip():
        return pd.DataFrame(columns=["timestamp", "node_id", "pred_wind_speed", "pred_range_label"])
    frame = pd.read_csv(io.StringIO(raw_csv), parse_dates=["timestamp"])
    return frame


def _height_metadata_from_json(path: Path) -> dict[str, object] | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    height = payload.get("height_correction")
    if isinstance(height, Mapping):
        result: dict[str, object] = {}
        for key, value in height.items():
            if value is None:
                continue
            if isinstance(value, float) and pd.isna(value):
                continue
            result[key] = value
        return result
    return None


def _height_metadata_from_csv(path: Path) -> dict[str, object] | None:
    usecols = [
        "height_method",
        "height_source_m",
        "height_target_m",
        "height_speed_scale",
        "height_power_law_alpha",
        "height_roughness_length_m",
    ]
    frame = pd.read_csv(path, usecols=usecols)
    if frame.empty:
        return None
    series = frame.iloc[0]
    return {
        "method": None if pd.isna(series["height_method"]) else series["height_method"],
        "source_height_m": None if pd.isna(series["height_source_m"]) else float(series["height_source_m"]),
        "target_height_m": None if pd.isna(series["height_target_m"]) else float(series["height_target_m"]),
        "speed_scale": None if pd.isna(series["height_speed_scale"]) else float(series["height_speed_scale"]),
        "power_law_alpha": None
        if pd.isna(series["height_power_law_alpha"])
        else float(series["height_power_law_alpha"]),
        "roughness_length_m": None
        if pd.isna(series["height_roughness_length_m"])
        else float(series["height_roughness_length_m"]),
    }


def _load_height_metadata() -> tuple[dict[str, object] | None, Path | None]:
    for candidate in HEIGHT_METADATA_CANDIDATES:
        resolved = _resolve_with_root(candidate)
        if not resolved.exists():
            continue
        if resolved.suffix == ".json":
            meta = _height_metadata_from_json(resolved)
        else:
            meta = _height_metadata_from_csv(resolved)
        if meta:
            return meta, resolved
    return None, None


def _analysis_to_tables(result: SeasonalAnalysisResult) -> SeasonalTables:
    seasonal_df = pd.DataFrame([asdict(item) for item in result.per_season])
    annual_df = pd.DataFrame([asdict(item) for item in result.per_year])
    summary_df = pd.DataFrame([asdict(item) for item in result.variation])
    return seasonal_df, annual_df, summary_df


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def _build_markdown_table(rows: Iterable[Mapping[str, object]], columns: Sequence[tuple[str, str]]) -> str:
    rows = list(rows)
    if not rows:
        return "*(no data)*"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append("" if value is None else str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator] + body)


def _summarise_variation(summary_df: pd.DataFrame) -> dict[str, object]:
    payload: dict[str, object] = {}
    payload["total_nodes"] = int(len(summary_df))
    if not summary_df.empty:
        amp_series = summary_df["seasonal_amplitude"].dropna()
        payload["median_seasonal_amplitude"] = float(amp_series.median()) if not amp_series.empty else None
        trend_series = summary_df["annual_trend_slope"].dropna()
        payload["median_annual_trend"] = float(trend_series.median()) if not trend_series.empty else None
    else:
        payload["median_seasonal_amplitude"] = None
        payload["median_annual_trend"] = None

    top_amplitudes = (
        summary_df.dropna(subset=["seasonal_amplitude"])
        .sort_values(by="seasonal_amplitude", ascending=False)
        .head(5)
    )
    payload["top_amplitudes"] = top_amplitudes[["node_id", "seasonal_amplitude", "strongest_season", "weakest_season", "seasonal_coverage", "annual_trend_slope"]].to_dict(orient="records")

    positive_trends = (
        summary_df.dropna(subset=["annual_trend_slope"])
        .sort_values(by="annual_trend_slope", ascending=False)
        .head(5)
    )
    negative_trends = (
        summary_df.dropna(subset=["annual_trend_slope"])
        .sort_values(by="annual_trend_slope", ascending=True)
        .head(5)
    )
    payload["strongest_positive_trends"] = positive_trends[["node_id", "annual_trend_slope", "annual_samples", "trend_note"]].to_dict(orient="records")
    payload["strongest_negative_trends"] = negative_trends[["node_id", "annual_trend_slope", "annual_samples", "trend_note"]].to_dict(orient="records")

    incomplete = summary_df[summary_df["seasonal_coverage"] < 4]
    payload["seasonal_gaps"] = incomplete[["node_id", "seasonal_coverage"]].to_dict(orient="records")
    return payload


def _format_height_lines(height_meta: dict[str, object] | None) -> list[str]:
    if not height_meta:
        return [
            "- Height metadata: not available.",
            "- Analysis uses ANN predictions at 10 m without additional vertical correction.",
        ]

    method = height_meta.get("method") or height_meta.get("height_method")
    source_raw = height_meta.get("source_height_m") or height_meta.get("height_source_m")
    target_raw = height_meta.get("target_height_m") or height_meta.get("height_target_m")
    scale_raw = height_meta.get("speed_scale") or height_meta.get("height_speed_scale")
    roughness_raw = height_meta.get("roughness_length_m")
    alpha_raw = height_meta.get("power_law_alpha")

    source = float(source_raw) if isinstance(source_raw, (int, float)) else None
    target = float(target_raw) if isinstance(target_raw, (int, float)) else None
    scale = float(scale_raw) if isinstance(scale_raw, (int, float)) else None
    roughness = float(roughness_raw) if isinstance(roughness_raw, (int, float)) else None
    alpha = float(alpha_raw) if isinstance(alpha_raw, (int, float)) else None

    parts = []
    if method:
        parts.append(f"method={method}")
    if source is not None and target is not None:
        parts.append(f"{source:.1f}→{target:.1f} m")
    if scale is not None:
        parts.append(f"speed scale {scale:.3f}")
    if roughness is not None:
        parts.append(f"z0={roughness:.5f} m")
    if alpha is not None:
        parts.append(f"alpha={alpha:.3f}")
    meta_line = "- Height metadata: " + ", ".join(parts) + "." if parts else "- Height metadata: available but fields were empty."

    analysis_line = "- Analysis uses ANN predictions at 10 m; no additional scaling applied within this script."
    return [meta_line, analysis_line]


def _write_markdown(
    report_path: Path,
    dataset: Path,
    output_dir: Path,
    summary_df: pd.DataFrame,
    payload: dict[str, object],
    height_lines: Iterable[str],
    height_source: Path | None,
) -> None:
    lines: list[str] = []
    lines.append("# Seasonal and Interannual Variations")
    lines.append("")
    lines.append(f"- Dataset: `{dataset}`")
    lines.append(f"- Output directory: `{output_dir}`")
    lines.append(f"- Nodes analysed: {payload.get('total_nodes', 0)}")
    if height_source:
        lines.append(f"- Height metadata source: `{height_source}`")
    for entry in height_lines:
        lines.append(entry)
    lines.append("")

    top_rows = payload.get("top_amplitudes") or []
    lines.append("## Top seasonal amplitudes")
    lines.append("")
    lines.append(
        _build_markdown_table(
            top_rows,
            (
                ("node_id", "node_id"),
                ("seasonal_amplitude", "amplitude (m/s)"),
                ("strongest_season", "strongest"),
                ("weakest_season", "weakest"),
                ("seasonal_coverage", "#seasons"),
                ("annual_trend_slope", "trend (m/s/yr)"),
            ),
        )
    )
    lines.append("")

    lines.append("## Strongest annual trends")
    lines.append("")
    positive = payload.get("strongest_positive_trends") or []
    negative = payload.get("strongest_negative_trends") or []
    lines.append("### Increasing trends")
    lines.append(
        _build_markdown_table(
            positive,
            (
                ("node_id", "node_id"),
                ("annual_trend_slope", "slope (m/s/yr)"),
                ("annual_samples", "years"),
                ("trend_note", "note"),
            ),
        )
    )
    lines.append("")
    lines.append("### Decreasing trends")
    lines.append(
        _build_markdown_table(
            negative,
            (
                ("node_id", "node_id"),
                ("annual_trend_slope", "slope (m/s/yr)"),
                ("annual_samples", "years"),
                ("trend_note", "note"),
            ),
        )
    )
    lines.append("")

    lines.append("## Seasonal coverage gaps")
    lines.append("")
    gaps = payload.get("seasonal_gaps") or []
    if gaps:
        lines.append(
            _build_markdown_table(
                gaps,
                (
                    ("node_id", "node_id"),
                    ("seasonal_coverage", "seasons available"),
                ),
            )
        )
    else:
        lines.append("All nodes cover the four meteorological seasons.")
    lines.append("")

    lines.append("_Report generated by `scripts/generate_seasonal_variations.py`._")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse seasonal wind-speed variability per node.")
    parser.add_argument("--dataset", type=Path, default=None, help="Override the GeoParquet dataset path.")
    parser.add_argument(
        "--stac-config",
        type=Path,
        default=DEFAULT_STAC_CONFIG,
        help="Path to the STAC catalog index (used when --dataset is not provided).",
    )
    parser.add_argument(
        "--stac-dataset",
        default=DEFAULT_STAC_DATASET,
        help="Dataset key inside the STAC catalog index to resolve the GeoParquet snapshot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV/JSON outputs will be stored.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown report path summarising key findings.",
    )
    parser.add_argument(
        "--docker-image",
        default=DEFAULT_IMAGE,
        help="Docker image that provides the DuckDB CLI. Defaults to duckdb/duckdb:latest.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("docker", "python"),
        default="docker",
        help=(
            "Backend used to query the GeoParquet snapshot. "
            "`docker` executes the DuckDB CLI inside the specified container (default, recommended). "
            "`python` uses the duckdb Python package and is suited for controlled test environments."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset is not None:
        dataset_path = _resolve_with_root(args.dataset)
    else:
        dataset_path = resolve_catalog_asset(
            args.stac_dataset,
            config_path=_resolve_with_root(args.stac_config),
            root=REPO_ROOT,
        ).require_local_path()

    try:
        dataset_rel = dataset_path.relative_to(REPO_ROOT)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "Dataset must reside within the repository root so it can be mounted inside the DuckDB container."
        ) from exc

    output_dir = _resolve_with_root(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = _resolve_with_root(args.report)

    frame = _fetch_observations(
        dataset_rel,
        workdir=REPO_ROOT,
        image=args.docker_image,
        engine=args.execution_mode,
    )
    result = compute_seasonal_analysis(frame)
    seasonal_df, annual_df, summary_df = _analysis_to_tables(result)

    seasonal_path = output_dir / "seasonal_slices.csv"
    annual_path = output_dir / "annual_slices.csv"
    summary_path = output_dir / "variation_summary.csv"
    seasonal_df.to_csv(seasonal_path, index=False)
    annual_df.to_csv(annual_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    height_meta, height_source = _load_height_metadata()
    payload = _summarise_variation(summary_df)
    payload["height_metadata"] = height_meta
    payload["execution_mode"] = args.execution_mode
    payload["dataset"] = str(dataset_rel)
    try:
        payload["seasonal_csv"] = str(seasonal_path.relative_to(REPO_ROOT))
        payload["annual_csv"] = str(annual_path.relative_to(REPO_ROOT))
        payload["summary_csv"] = str(summary_path.relative_to(REPO_ROOT))
    except ValueError:
        payload["seasonal_csv"] = str(seasonal_path)
        payload["annual_csv"] = str(annual_path)
        payload["summary_csv"] = str(summary_path)

    overview_path = output_dir / "seasonal_analysis_summary.json"
    overview_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    height_lines = _format_height_lines(height_meta)
    try:
        output_dir_rel = output_dir.relative_to(REPO_ROOT)
    except ValueError:
        output_dir_rel = output_dir

    if height_source is not None:
        try:
            height_source_rel = height_source.relative_to(REPO_ROOT)
        except ValueError:
            height_source_rel = height_source
    else:
        height_source_rel = None

    _write_markdown(
        report_path,
        dataset=dataset_rel,
        output_dir=output_dir_rel,
        summary_df=summary_df,
        payload=payload,
        height_lines=height_lines,
        height_source=height_source_rel,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
