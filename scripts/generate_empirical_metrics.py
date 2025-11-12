#!/usr/bin/env python3
"""Generate empirical wind-speed metrics and validation dashboards.

This helper orchestrates the DuckDB aggregations required to compute
per-node descriptive statistics, merges the output with the taxonomy, and
produces both tabular artefacts and lightweight visual dashboards. It is
intended to be re-runnable: delete the destination directory if a fresh
run is needed.
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hf_wind_resource.stats import (
    BiasThresholds,
    assemble_per_node_metrics,
    load_taxonomy_records,
)
from hf_wind_resource.io import resolve_catalog_asset


DEFAULT_IMAGE = "duckdb/duckdb:latest"
REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL_COUNTS_FILENAME = "per_node_label_counts.csv"
VALID_METRICS_FILENAME = "per_node_valid_metrics.csv"
CDF_FILENAME = "per_node_cdf.csv"
SUMMARY_FILENAME = "per_node_summary.csv"
DASHBOARD_FILENAME = "validation_dashboard.html"

_CDF_POINTS = 100
_PERCENTILE_LIST = "[" + ",".join(
    ("0" if i == 0 else "1" if i == _CDF_POINTS else format(i / _CDF_POINTS, ".2f").rstrip("0").rstrip(".") )
    for i in range(_CDF_POINTS + 1)
) + "]"
REPORT_FILENAME = "docs/empirical_metrics_summary.md"
DEFAULT_STAC_CONFIG = Path("config/stac_catalogs.json")
DEFAULT_STAC_DATASET = "sar_range_final_pivots_joined"


def _resolve_with_root(path: Path) -> Path:
    """Resolve *path* relative to the repository root when not absolute."""

    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _run_duckdb(
    sql: str,
    *,
    workdir: Path,
    image: str = DEFAULT_IMAGE,
    dataset_path: Path | None = None,
    mode: str = "plain",
) -> str:
    """Execute *sql* returning the CSV output, preferring Docker but falling back to a local duckdb binary."""

    docker_cli = shutil.which("docker")
    if docker_cli:
        command: list[str] = [
            docker_cli,
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
            result = subprocess.run(command, text=True, capture_output=True, check=True)
            return result.stdout
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

    local_cli = shutil.which("duckdb")
    if local_cli:
        command = [
            local_cli,
            "-csv",
            "-header",
            "-cmd",
            sql,
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=True, cwd=workdir)
        return result.stdout

    try:
        import duckdb  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("DuckDB executable not available and python 'duckdb' module missing.") from exc

    con = duckdb.connect(database=":memory:")
    try:
        try:
            stmt = con.execute(sql)
        except duckdb.NotImplementedException:
            if mode == "cdf" and dataset_path is not None:
                return _run_duckdb_cdf_python(dataset_path)
            raise
        columns = [meta[0] for meta in stmt.description]
        rows = stmt.fetchall()
    finally:
        con.close()

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(columns)
    writer.writerows(rows)
    return buffer.getvalue()


def _run_duckdb_cdf_python(dataset_path: Path) -> str:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("duckdb python module is required for CDF fallback.") from exc

    dataset_full = _resolve_with_root(dataset_path).as_posix()
    sql = textwrap.dedent(
        f"""
        WITH normalized AS (
            SELECT
              node_id,
              pred_wind_speed,
              CASE
                WHEN pred_range_label IS NULL THEN 'uncertain'
                WHEN lower(pred_range_label) IN ('in', 'inside', 'within', 'in_range') THEN 'in'
                WHEN lower(pred_range_label) IN ('below', 'under', 'below_range', 'left') THEN 'below'
                WHEN lower(pred_range_label) IN ('above', 'over', 'upper', 'right') THEN 'above'
                ELSE 'uncertain'
              END AS label
            FROM read_parquet('{dataset_full}')
        )
        SELECT
            node_id,
            QUANTILE(pred_wind_speed, {_PERCENTILE_LIST}) AS quantiles
        FROM normalized
        WHERE label = 'in' AND pred_wind_speed IS NOT NULL
        GROUP BY node_id
        ORDER BY node_id;
        """
    )
    con = duckdb.connect(database=":memory:")
    try:
        stmt = con.execute(sql)
        rows = stmt.fetchall()
    finally:
        con.close()

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["node_id", "percentile", "wind_speed"])
    for node_id, values in rows:
        if values is None:
            continue
        for idx, value in enumerate(values):
            percentile = idx / _CDF_POINTS
            writer.writerow([node_id, percentile, value])
    return buffer.getvalue()


def _label_counts_sql(dataset: Path) -> str:
    return textwrap.dedent(
        f"""
        WITH normalized AS (
            SELECT
              node_id,
              CASE
                WHEN pred_range_label IS NULL THEN 'uncertain'
                WHEN lower(pred_range_label) IN ('in', 'inside', 'within', 'in_range') THEN 'in'
                WHEN lower(pred_range_label) IN ('below', 'under', 'below_range', 'left') THEN 'below'
                WHEN lower(pred_range_label) IN ('above', 'over', 'upper', 'right') THEN 'above'
                ELSE 'uncertain'
              END AS label
            FROM read_parquet('{dataset.as_posix()}')
        )
        SELECT
            node_id,
            COUNT(*) AS total_observations,
            SUM(CASE WHEN label = 'in' THEN 1 ELSE 0 END) AS in_count,
            SUM(CASE WHEN label = 'below' THEN 1 ELSE 0 END) AS below_count,
            SUM(CASE WHEN label = 'above' THEN 1 ELSE 0 END) AS above_count,
            SUM(CASE WHEN label = 'uncertain' THEN 1 ELSE 0 END) AS uncertain_count
        FROM normalized
        GROUP BY node_id
        ORDER BY node_id;
        """
    )


def _valid_metrics_sql(dataset: Path) -> str:
    return textwrap.dedent(
        f"""
        WITH normalized AS (
            SELECT
              node_id,
              pred_wind_speed,
              CASE
                WHEN pred_range_label IS NULL THEN 'uncertain'
                WHEN lower(pred_range_label) IN ('in', 'inside', 'within', 'in_range') THEN 'in'
                WHEN lower(pred_range_label) IN ('below', 'under', 'below_range', 'left') THEN 'below'
                WHEN lower(pred_range_label) IN ('above', 'over', 'upper', 'right') THEN 'above'
                ELSE 'uncertain'
              END AS label
            FROM read_parquet('{dataset.as_posix()}')
        )
        SELECT
            node_id,
            COUNT(*) AS valid_count,
            AVG(pred_wind_speed) AS mean_speed,
            STDDEV_SAMP(pred_wind_speed) AS std_speed,
            MIN(pred_wind_speed) AS min_speed,
            MAX(pred_wind_speed) AS max_speed,
            QUANTILE(pred_wind_speed, 0.5) AS p50,
            QUANTILE(pred_wind_speed, 0.9) AS p90,
            QUANTILE(pred_wind_speed, 0.99) AS p99
        FROM normalized
        WHERE label = 'in' AND pred_wind_speed IS NOT NULL
        GROUP BY node_id
        ORDER BY node_id;
        """
    )


def _cdf_sql(dataset: Path) -> str:
    return textwrap.dedent(
        f"""
        WITH normalized AS (
            SELECT
              node_id,
              pred_wind_speed,
              CASE
                WHEN pred_range_label IS NULL THEN 'uncertain'
                WHEN lower(pred_range_label) IN ('in', 'inside', 'within', 'in_range') THEN 'in'
                WHEN lower(pred_range_label) IN ('below', 'under', 'below_range', 'left') THEN 'below'
                WHEN lower(pred_range_label) IN ('above', 'over', 'upper', 'right') THEN 'above'
                ELSE 'uncertain'
              END AS label
            FROM read_parquet('{dataset.as_posix()}')
        ),
        quantiles AS (
            SELECT
              node_id,
              QUANTILE(pred_wind_speed, {_PERCENTILE_LIST}) AS values
            FROM normalized
            WHERE label = 'in' AND pred_wind_speed IS NOT NULL
            GROUP BY node_id
        ),
        expanded AS (
            SELECT
              q.node_id,
              CAST(ord - 1 AS DOUBLE) / {_CDF_POINTS} AS percentile,
              val AS wind_speed
            FROM quantiles q,
                 UNNEST(values) WITH ORDINALITY AS t(val, ord)
        )
        SELECT node_id, percentile, wind_speed
        FROM expanded
        WHERE wind_speed IS NOT NULL
        ORDER BY node_id, percentile;
        """
    )


def _read_csv_dicts(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
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


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = ["node_id"]
    extra_fields = sorted({key for row in rows for key in row.keys() if key != "node_id"})
    fieldnames.extend(extra_fields)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _generate_dashboard(
    summary: Sequence[Mapping[str, object]],
    cdf: Sequence[Mapping[str, object]],
    output_path: Path,
    *,
    top_n: int = 4,
) -> None:
    """Render a compact dashboard as static HTML with inline SVG charts."""

    if not summary:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_summary = sorted(
        summary,
        key=lambda row: float(row.get("mean_speed") or float("-inf")),
        reverse=True,
    )
    top_nodes = [row.get("node_id") for row in sorted_summary[:top_n] if row.get("node_id")]

    max_mean = max(float(row.get("mean_speed") or 0.0) for row in sorted_summary)
    max_censored = max(float(row.get("censored_ratio") or 0.0) for row in sorted_summary)

    bar_width = 12
    chart_padding = 40
    bar_chart_width = chart_padding * 2 + bar_width * len(sorted_summary)
    bar_chart_height = 320

    bar_rects = []
    for idx, row in enumerate(sorted_summary):
        node_id = str(row.get("node_id"))
        mean_speed = float(row.get("mean_speed") or 0.0)
        censored_ratio = float(row.get("censored_ratio") or 0.0)

        x = chart_padding + idx * bar_width
        mean_height = 0 if max_mean == 0 else int((mean_speed / max_mean) * (bar_chart_height - chart_padding))
        censored_height = 0 if max_censored == 0 else int((censored_ratio / max_censored) * (bar_chart_height - chart_padding))

        bar_rects.append(
            "".join(
                [
                    f'<rect x="{x}" y="{bar_chart_height - mean_height}" width="{bar_width - 2}" '
                    f'height="{mean_height}" fill="#4C72B0">',
                    f'<title>{node_id}: mean {mean_speed:.2f} m/s</title>',
                    "</rect>",
                ]
            )
        )
        bar_rects.append(
            "".join(
                [
                    f'<rect x="{x}" y="{bar_chart_height * 2 - chart_padding - censored_height}" '
                    f'width="{bar_width - 2}" height="{censored_height}" fill="#55A868">',
                    f'<title>{node_id}: censored ratio {censored_ratio:.2%}</title>',
                    "</rect>",
                ]
            )
        )

    cdf_data: dict[str, list[tuple[float, float]]] = {}
    for row in cdf:
        node_id = str(row.get("node_id"))
        if node_id not in top_nodes:
            continue
        percentile = float(row.get("percentile") or 0.0)
        wind_speed = float(row.get("wind_speed") or 0.0)
        cdf_data.setdefault(node_id, []).append((wind_speed, percentile))

    cdf_width = bar_chart_width
    cdf_height = 320
    max_speed = max((max(values)[0] for values in cdf_data.values()), default=0.0)
    svg_paths = []
    color_palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
    for idx, (node_id, pairs) in enumerate(cdf_data.items()):
        if not pairs:
            continue
        pairs_sorted = sorted(pairs)
        points = []
        for wind_speed, percentile in pairs_sorted:
            x = chart_padding + (wind_speed / max_speed) * (cdf_width - chart_padding * 2) if max_speed else chart_padding
            y = cdf_height - chart_padding - percentile * (cdf_height - chart_padding * 2)
            points.append(f"{x:.2f},{y:.2f}")
        color = color_palette[idx % len(color_palette)]
        svg_paths.append(
            "".join(
                [
                    f'<polyline fill="none" stroke="{color}" stroke-width="2" points="',
                    " ".join(points),
                    '">',
                    f'<title>{node_id} empirical CDF</title>',
                    "</polyline>",
                ]
            )
        )

    html = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
      <meta charset=\"utf-8\" />
      <title>Empirical metrics dashboard</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ font-size: 1.6rem; }}
        section {{ margin-bottom: 32px; }}
        svg {{ background-color: #fafafa; border: 1px solid #ccc; }}
      </style>
    </head>
    <body>
      <h1>Empirical metrics dashboard</h1>
      <section>
        <h2>Mean speed and censored ratio per node</h2>
        <p>Blue bars show the mean in-range wind speed (m/s); green bars show the censored share.</p>
        <svg width="{bar_chart_width}" height="{bar_chart_height * 2}" viewBox="0 0 {bar_chart_width} {bar_chart_height * 2}">
          {''.join(bar_rects)}
        </svg>
      </section>
      <section>
        <h2>Empirical CDF for top nodes</h2>
        <svg width="{cdf_width}" height="{cdf_height}" viewBox="0 0 {cdf_width} {cdf_height}">
          {''.join(svg_paths)}
        </svg>
        <p>Top nodes: {', '.join(str(node) for node in top_nodes)}</p>
      </section>
    </body>
    </html>
    """

    output_path.write_text(html, encoding="utf-8")


def _write_report(
    summary: Sequence[Mapping[str, object]],
    *,
    dataset: Path,
    output_path: Path,
    dashboard_path: Path,
) -> None:
    """Persist a Markdown summary covering key findings."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_nodes = len(summary)
    flagged = [str(row.get("node_id")) for row in summary if row.get("any_bias")]
    top_means = sorted(
        summary,
        key=lambda row: float(row.get("mean_speed") or float("-inf")),
        reverse=True,
    )[:5]

    table_lines = [
        "| node_id | mean_speed | p90 | p99 | censored_ratio | bias_notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in top_means:
        mean_speed = float(row.get("mean_speed") or 0.0)
        p90 = float(row.get("p90") or 0.0)
        p99 = float(row.get("p99") or 0.0)
        censored = float(row.get("censored_ratio") or 0.0)
        bias_notes = str(row.get("bias_notes") or "")
        table_lines.append(
            f"| {row.get('node_id')} | {mean_speed:.3f} | {p90:.3f} | {p99:.3f} | {censored:.3f} | {bias_notes} |"
        )

    lines = [
        "# Empirical Metrics Summary",
        "",
        f"- Dataset: `{dataset}`",
        f"- Nodes processed: {total_nodes}",
        f"- Dashboard: `{dashboard_path}`",
        "",
        "## Top nodes by mean in-range wind speed",
        "",
        *table_lines,
        "",
        "## Bias monitoring",
        "",
        "- Nodes flagged for potential bias: "
        + (", ".join(flagged) if flagged else "None detected"),
        "",
        "The bias column aggregates the following checks: high censoring ratios, low in-range share, "
        "taxonomy entries marked as low coverage, sparse coverage bands, and nodes with insufficient valid samples.",
        "",
        "_Report generated by `scripts/generate_empirical_metrics.py`._",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate empirical metrics and dashboards from the SAR range-aware dataset.",
    )
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
        "--taxonomy",
        type=Path,
        default=Path("config/node_taxonomy.json"),
        help="Path to the node taxonomy JSON exported previously.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/empirical_metrics"),
        help="Directory where CSV outputs and dashboards will be written.",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default=DEFAULT_IMAGE,
        help="Docker image providing the DuckDB CLI.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV artefacts if they exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset is not None:
        dataset_path = _resolve_with_root(args.dataset)
    else:
        stac_config_path = _resolve_with_root(args.stac_config)
        dataset_path = resolve_catalog_asset(
            args.stac_dataset,
            config_path=stac_config_path,
            root=REPO_ROOT,
        ).require_local_path()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir = _resolve_with_root(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_counts_path = output_dir / LABEL_COUNTS_FILENAME
    valid_metrics_path = output_dir / VALID_METRICS_FILENAME
    cdf_path = output_dir / CDF_FILENAME

    try:
        dataset_rel = dataset_path.relative_to(REPO_ROOT)
    except ValueError as exc:  # pragma: no cover - defensive, repo layout is stable
        raise RuntimeError(
            "Dataset and output directory must reside within the repository root "
            f"({REPO_ROOT})."
        ) from exc

    if args.overwrite or not label_counts_path.exists():
        csv_output = _run_duckdb(
            _label_counts_sql(dataset_rel),
            workdir=REPO_ROOT,
            image=args.docker_image,
            dataset_path=dataset_rel,
        )
        label_counts_path.write_text(csv_output, encoding="utf-8")
    if args.overwrite or not valid_metrics_path.exists():
        csv_output = _run_duckdb(
            _valid_metrics_sql(dataset_rel),
            workdir=REPO_ROOT,
            image=args.docker_image,
            dataset_path=dataset_rel,
        )
        valid_metrics_path.write_text(csv_output, encoding="utf-8")
    if args.overwrite or not cdf_path.exists():
        csv_output = _run_duckdb(
            _cdf_sql(dataset_rel),
            workdir=REPO_ROOT,
            image=args.docker_image,
            dataset_path=dataset_rel,
            mode="cdf",
        )
        cdf_path.write_text(csv_output, encoding="utf-8")

    label_counts = _read_csv_dicts(label_counts_path)
    valid_metrics = _read_csv_dicts(valid_metrics_path)
    cdf = _read_csv_dicts(cdf_path)
    taxonomy_path = _resolve_with_root(args.taxonomy)
    taxonomy = load_taxonomy_records(taxonomy_path)

    summary = assemble_per_node_metrics(valid_metrics, label_counts, taxonomy, thresholds=BiasThresholds())

    summary_path = output_dir / SUMMARY_FILENAME
    _write_csv(summary_path, summary)

    dashboard_path = output_dir / DASHBOARD_FILENAME
    _generate_dashboard(summary, cdf, dashboard_path)

    try:
        dashboard_rel = dashboard_path.relative_to(REPO_ROOT)
    except ValueError:
        dashboard_rel = dashboard_path

    report_path = REPO_ROOT / REPORT_FILENAME
    _write_report(summary, dataset=dataset_rel, output_path=report_path, dashboard_path=dashboard_rel)

    print(f"Wrote summary tables to {summary_path}")
    if dashboard_path.exists():
        print(f"Dashboard saved to {dashboard_path}")
    print(f"Markdown report updated at {report_path}")


if __name__ == "__main__":
    main()
