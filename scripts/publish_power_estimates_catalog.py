#!/usr/bin/env python3
"""Publish wind-resource estimates as a GeoParquet + STAC catalog.

The command fuses the per-node power summaries with ANN node geometries,
exports a GeoParquet table, and materialises a STAC collection/item plus a
provenance manifest under ``use_case/catalogs/`` using the version tag supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hf_wind_resource.io.stac import ResolvedStacAsset, resolve_catalog_asset  # noqa: E402

DEFAULT_VERSION_TAG = "sar-range-final-20251018"
DEFAULT_STAC_DATASET = "sar_range_final_pivots_joined"
DEFAULT_STAC_CONFIG = REPO_ROOT / "config" / "stac_catalogs.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "use_case" / "catalogs"
DEFAULT_COLLECTION_NAME = "sar_range_final_power_estimates"
DEFAULT_IMAGE = "duckdb/duckdb:latest"
TABLE_EXTENSION = "https://stac-extensions.github.io/table/v1.2.0/schema.json"
SCIENTIFIC_EXTENSION = "https://stac-extensions.github.io/scientific/v1.0.0/schema.json"
TASK5_DOI = "10.5281/zenodo.17594220"
TASK5_CITATION = (
    "Herrera Cortijo, J. L., Fernández-Baladrón, A., Rosón, G., Gil Coto, M., Dubert, J., & Varela Benvenuto, R. "
    "(2025). HF-EOLUS. Task 5. Wind Resource Estimation Results. Zenodo. https://doi.org/10.5281/zenodo.17594220"
)

COLUMN_DESCRIPTIONS: Dict[str, str] = {
    "node_id": "Spatial grid node identifier aligning with SAR range-aware ANN outputs.",
    "geometry": "Node footprint polygon encoded as WKB (CRS84).",
    "method": "Estimator used to derive power density (weibull or kaplan_meier).",
    "power_density_w_m2": "Estimated wind power density at hub height (W/m²).",
    "capacity_factor": "Expected capacity factor for the reference turbine (0-1).",
    "turbine_mean_power_kw": "Expected mean power for the reference turbine (kW).",
    "air_density": "Air density assumed for the power curve (kg/m³).",
    "power_curve_name": "Human-friendly name of the reference power curve.",
    "power_density_method": "Estimator name used when computing power density.",
    "power_density_notes": "Notes describing how the power density estimate was obtained.",
    "power_curve_notes": "Notes describing curve assumptions (cut-in/cut-out, rated speed).",
    "weibull_reliable": "True when the Weibull fit passed reliability thresholds.",
    "weibull_shape": "Shape parameter of the fitted Weibull distribution.",
    "weibull_scale": "Scale parameter of the fitted Weibull distribution.",
    "kaplan_meier_tail_probability": "Probability mass assigned to censored right tail (Kaplan–Meier).",
    "height_method": "Vertical extrapolation method applied to ANN wind speeds.",
    "height_source_m": "Height (m) of the ANN wind speeds.",
    "height_target_m": "Target height (m) for resource metrics.",
    "height_speed_scale": "Multiplicative factor applied to wind speeds during vertical extrapolation.",
    "mean_speed_m_s": "Mean ANN wind speed within the regression range (m/s).",
    "std_speed_m_s": "Standard deviation of ANN wind speeds within the regression range (m/s).",
    "min_speed_m_s": "Minimum ANN wind speed within the regression range (m/s).",
    "max_speed_m_s": "Maximum ANN wind speed within the regression range (m/s).",
    "speed_p50_m_s": "Median ANN wind speed within the regression range (m/s).",
    "speed_p90_m_s": "P90 ANN wind speed within the regression range (m/s).",
    "speed_p99_m_s": "P99 ANN wind speed within the regression range (m/s).",
    "valid_record_count": "Number of ANN records contributing in-range observations.",
    "total_observation_count": "Total ANN records for the node.",
    "taxonomy_observation_count": "Observation count registered in node taxonomy metadata.",
    "censored_ratio": "Share of samples classified outside the regression range.",
    "below_ratio": "Share of samples classified below the regression range.",
    "above_ratio": "Share of samples classified above the regression range.",
    "in_ratio": "Share of samples classified within the regression range.",
    "uncertain_ratio": "Share of samples with low classifier confidence.",
    "low_coverage": "Node flagged with low coverage according to taxonomy rules.",
    "coverage_band": "Coverage band assigned during empirical QA.",
    "continuity_band": "Continuity band assigned during empirical QA.",
    "coverage_bias": "True when coverage anomalies were detected.",
    "censoring_bias": "True when censoring ratios exceeded thresholds.",
    "any_bias": "True when any bias flag was raised for the node.",
    "sample_bias": "True when sample-size bias triggered for the node.",
    "bias_notes": "Notes summarising empirical bias diagnostics.",
    "parametric_selection_metric": "Information criterion (AIC/BIC) used to select the preferred parametric model.",
    "parametric_preferred_model": "Parametric model (weibull/lognormal/gamma) with the lowest selected metric.",
    "parametric_preferred_metric_value": "Metric value attained by the preferred parametric model.",
    "parametric_notes": "Notes documenting how parametric candidates were compared for the node.",
    "weibull_aic": "AIC computed from the censored Weibull log-likelihood.",
    "weibull_bic": "BIC computed from the censored Weibull log-likelihood.",
    "weibull_ks_statistic": "Weighted Kolmogorov–Smirnov statistic for the Weibull candidate.",
    "weibull_ks_pvalue": "Approximate p-value of the Weibull KS statistic.",
    "weibull_parametric_success": "Whether the Weibull candidate was considered valid for comparison metrics.",
    "weibull_parametric_notes": "Notes produced while evaluating the Weibull candidate.",
    "lognormal_log_likelihood": "Censored log-likelihood produced by the log-normal candidate.",
    "lognormal_aic": "AIC derived from the log-normal log-likelihood.",
    "lognormal_bic": "BIC derived from the log-normal log-likelihood.",
    "lognormal_ks_statistic": "Weighted KS statistic for the log-normal candidate.",
    "lognormal_ks_pvalue": "Approximate p-value of the log-normal KS statistic.",
    "lognormal_parametric_success": "Whether the log-normal candidate yielded a valid comparison fit.",
    "lognormal_parametric_notes": "Notes emitted while evaluating the log-normal candidate.",
    "lognormal_mu": "Mean (mu) of the underlying normal distribution used in the log-normal candidate.",
    "lognormal_sigma": "Standard deviation (sigma) of the underlying normal distribution used in the log-normal candidate.",
    "gamma_log_likelihood": "Censored log-likelihood produced by the gamma candidate.",
    "gamma_aic": "AIC derived from the gamma log-likelihood.",
    "gamma_bic": "BIC derived from the gamma log-likelihood.",
    "gamma_ks_statistic": "Weighted KS statistic for the gamma candidate.",
    "gamma_ks_pvalue": "Approximate p-value of the gamma KS statistic.",
    "gamma_parametric_success": "Whether the gamma candidate was evaluated successfully.",
    "gamma_parametric_notes": "Notes emitted while evaluating the gamma candidate (e.g., SciPy availability).",
    "gamma_shape": "Shape parameter estimated for the gamma candidate.",
    "gamma_scale": "Scale parameter estimated for the gamma candidate.",
}


class CommandError(RuntimeError):
    """Raised when DuckDB execution fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-tag", default=DEFAULT_VERSION_TAG, help="Version tag recorded in outputs/manifest.")
    parser.add_argument(
        "--stac-config", type=Path, default=DEFAULT_STAC_CONFIG, help="Path to STAC catalog index JSON."
    )
    parser.add_argument(
        "--stac-dataset",
        default=DEFAULT_STAC_DATASET,
        help="Dataset key within the STAC index resolving the ANN inference GeoParquet.",
    )
    parser.add_argument(
        "--power-summary",
        type=Path,
        default=REPO_ROOT / "artifacts" / "power_estimates" / "power_estimates_summary.csv",
        help="Per-node power metrics generated by scripts/generate_power_estimates.py.",
    )
    parser.add_argument(
        "--empirical-summary",
        type=Path,
        default=REPO_ROOT / "artifacts" / "empirical_metrics" / "per_node_summary.csv",
        help="Empirical QA metrics (coverage, censoring ratios, taxonomy flags).",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=REPO_ROOT / "artifacts" / "power_estimates" / "metadata.json",
        help="Metadata emitted alongside the power estimates (parameters, configuration).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for published collections (defaults to use_case/catalogs/).",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Directory name for the published collection under --output-root.",
    )
    parser.add_argument(
        "--engine",
        choices=("docker", "python"),
        default="docker",
        help="Execution engine for DuckDB (docker uses duckdb/duckdb:latest).",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image for DuckDB when --engine=docker.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing collection directory if it already exists.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def run_duckdb(sql: str, *, engine: str, image: str) -> None:
    if engine == "docker":
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{REPO_ROOT}:/workspace",
            "-w",
            "/workspace",
            image,
            "duckdb",
            "-cmd",
            sql.replace("\n", " "),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise CommandError(f"DuckDB (docker) failed:\n{result.stderr.strip()}")
        return

    try:
        import duckdb  # type: ignore
    except ImportError as exc:  # pragma: no cover - fallback when duckdb missing
        raise CommandError("DuckDB Python module not installed. Install `duckdb` or use --engine docker.") from exc

    with duckdb.connect(database=":memory:") as conn:
        conn.execute(sql)


def read_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV file has no header: {path}") from exc


def select_columns(prefix: str, columns: Iterable[str], *, exclude: Sequence[str]) -> List[str]:
    return [f"{prefix}.{name}" for name in columns if name not in exclude]


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(
    *,
    version_tag: str,
    metadata: Mapping[str, object],
    resolved_asset: ResolvedStacAsset,
    metadata_json: Path,
    power_summary: Path,
    empirical_summary: Path,
    output_parquet: Path,
) -> Mapping[str, object]:
    commit = (
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or "unknown"
    )

    generated_at = datetime.now(timezone.utc).isoformat()

    dataset_path = resolved_asset.require_local_path()

    inputs = {
        "sar_range_final_pivots_joined": {
            "collection": repo_relative(resolved_asset.collection_path),
            "item": repo_relative(resolved_asset.item_path),
            "asset": repo_relative(resolved_asset.path or dataset_path),
            "sha256": sha256sum(dataset_path),
        },
        "power_estimates_summary_csv": {
            "path": repo_relative(power_summary),
            "sha256": sha256sum(power_summary),
        },
        "empirical_metrics_summary_csv": {
            "path": repo_relative(empirical_summary),
            "sha256": sha256sum(empirical_summary),
        },
    }

    outputs = {
        "power_estimates_nodes": {
            "path": repo_relative(output_parquet),
            "sha256": sha256sum(output_parquet),
        },
    }

    parameters = {
        "air_density": metadata.get("air_density"),
        "min_confidence": metadata.get("min_confidence"),
        "min_in_range": metadata.get("min_in_range"),
        "right_tail_surrogate": metadata.get("right_tail_surrogate"),
        "range_thresholds": metadata.get("range_thresholds"),
        "height_correction": metadata.get("height_correction"),
        "power_curve_key": metadata.get("power_curve_key"),
        "power_curve": metadata.get("power_curve"),
        "kaplan_meier_criteria": metadata.get("km_criteria"),
        "engine": metadata.get("engine"),
        "docker_image": metadata.get("docker_image"),
        "summary_source": metadata.get("summary_source"),
    }

    notes = {
        "dataset_reference_path": repo_relative(dataset_path),
        "metadata_source": repo_relative(metadata_json),
    }

    return {
        "version": version_tag,
        "generated_at": generated_at,
        "code_commit": commit,
        "inputs": inputs,
        "outputs": outputs,
        "parameters": parameters,
        "notes": notes,
    }


def build_table_columns(info_csv: Path) -> List[Mapping[str, object]]:
    columns: List[Mapping[str, object]] = []
    with info_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row["column_name"]
            col_type = row["column_type"].lower()
            columns.append(
                {
                    "name": name,
                    "type": col_type,
                    "description": COLUMN_DESCRIPTIONS.get(name, ""),
                }
            )
    return columns


def build_stac_payloads(
    *,
    version_tag: str,
    resolved_item_path: Path,
    manifest: Mapping[str, object],
    table_columns: List[Mapping[str, object]],
    row_count: int,
) -> Mapping[str, object]:
    item_payload = load_json(resolved_item_path)

    geometry = item_payload["geometry"]
    bbox = item_payload["bbox"]
    start_dt = item_payload["properties"]["start_datetime"]
    end_dt = item_payload["properties"]["end_datetime"]

    collection = {
        "type": "Collection",
        "stac_version": "1.1.0",
        "stac_extensions": [TABLE_EXTENSION, SCIENTIFIC_EXTENSION],
        "id": "SAR_RANGE_FINAL_POWER_ESTIMATES",
        "title": "HF-EOLUS SAR Range Final Wind Resource Estimates",
        "description": (
            "Derived wind resource indicators (power density, capacity factor, Weibull/Kaplan–Meier diagnostics) "
            "computed from the SAR range-aware ANN inference snapshot for VILA/PRIO nodes."
        ),
        "keywords": [
            "wind resource",
            "offshore wind",
            "HF radar",
            "Weibull distribution",
            "Kaplan-Meier",
            "GeoParquet",
        ],
        "license": "GPL-3.0",
        "sci:doi": TASK5_DOI,
        "sci:citation": TASK5_CITATION,
        "providers": [
            {
                "name": "HF-EOLUS Project",
                "roles": ["producer"],
                "description": "HF-EOLUS task 5 wind resource publication pipeline.",
            },
            {
                "name": "Grupo de Oceanografía Física, Universidade de Vigo (GOFUV)",
                "roles": ["processor"],
                "description": "Processed HF radar ANN outputs into wind resource indicators.",
            },
        ],
        "extent": {
            "spatial": {"bbox": [bbox]},
            "temporal": {"interval": [[start_dt, end_dt]]},
        },
        "links": [
            {"rel": "self", "href": "collection.json", "type": "application/json"},
            {"rel": "root", "href": "../../catalog.json", "type": "application/json"},
            {"rel": "parent", "href": "../catalog.json", "type": "application/json"},
            {"rel": "item", "href": "items/power_estimates_nodes.json", "type": "application/json"},
        ],
    }

    item = {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": [TABLE_EXTENSION, SCIENTIFIC_EXTENSION],
        "id": "power_estimates_nodes",
        "geometry": geometry,
        "bbox": bbox,
        "properties": {
            "title": "Per-node wind resource indicators",
            "description": (
                "GeoParquet table storing power density, capacity factor, Weibull/Kaplan–Meier diagnostics, "
                "and empirical QA summaries per HF radar node."
            ),
            "start_datetime": start_dt,
            "end_datetime": end_dt,
            "created": manifest["generated_at"],
            "table:row_count": row_count,
            "table:primary_geometry": "geometry",
            "hf_eolus:version": version_tag,
            "hf_eolus:code_commit": manifest["code_commit"],
            "sci:doi": TASK5_DOI,
            "sci:citation": TASK5_CITATION,
        },
        "links": [
            {"rel": "self", "href": "../items/power_estimates_nodes.json", "type": "application/json"},
            {"rel": "root", "href": "../collection.json", "type": "application/json"},
            {"rel": "parent", "href": "../collection.json", "type": "application/json"},
            {"rel": "collection", "href": "../collection.json", "type": "application/json"},
            {"rel": "derived_from", "href": "../../sar_range_final_pivots_joined/items/data.json", "type": "application/json"},
        ],
        "assets": {
            "power_estimates_nodes": {
                "href": "../assets/power_estimates_nodes.parquet",
                "type": "application/x-parquet",
                "roles": ["data"],
                "title": "Per-node wind resource GeoParquet",
                "description": "Power density, turbine metrics, and QA diagnostics for each HF radar node.",
                "table:columns": table_columns,
            },
            "manifest": {
                "href": "../manifest.json",
                "type": "application/json",
                "roles": ["metadata"],
                "title": "Run manifest",
                "description": "Provenance manifest with code commit, input hashes, and processing parameters.",
            },
        },
    }

    return collection, item


def update_root_catalog(collection_dir: Path) -> None:
    root_catalog = REPO_ROOT / "catalogs" / "catalog.json"
    if not root_catalog.exists():
        return

    payload = json.loads(root_catalog.read_text(encoding="utf-8"))
    links = payload.setdefault("links", [])
    rel_href = f"./{collection_dir.name}/collection.json"
    if any(link.get("href") == rel_href for link in links):
        root_catalog.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return

    links.append(
        {
            "rel": "child",
            "href": rel_href,
            "type": "application/json",
            "title": "HF-EOLUS SAR range final wind-resource indicators",
        }
    )
    root_catalog.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def count_nodes(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def main() -> None:
    args = parse_args()

    ensure_exists(args.power_summary, label="power summary")
    ensure_exists(args.empirical_summary, label="empirical summary")
    ensure_exists(args.metadata_json, label="metadata json")

    resolved_asset = resolve_catalog_asset(
        args.stac_dataset,
        config_path=args.stac_config,
        root=REPO_ROOT,
    )
    dataset_path = resolved_asset.require_local_path()

    power_columns = read_header(args.power_summary)
    empirical_columns = read_header(args.empirical_summary)

    select_clause = (
        ["g.node_id", "g.geometry"]
        + select_columns("p", power_columns, exclude=("node_id",))
        + select_columns("e", empirical_columns, exclude=("node_id",))
    )

    collection_dir = (args.output_root / args.collection_name).resolve()
    assets_dir = collection_dir / "assets"
    items_dir = collection_dir / "items"

    if collection_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Collection directory {collection_dir} already exists. Use --overwrite to regenerate."
            )
        shutil.rmtree(collection_dir)

    assets_dir.mkdir(parents=True, exist_ok=True)
    items_dir.mkdir(parents=True, exist_ok=True)

    output_parquet = assets_dir / "power_estimates_nodes.parquet"
    table_info_csv = collection_dir / "_table_info.csv"

    sql = f"""
    INSTALL spatial;
    LOAD spatial;
    CREATE OR REPLACE TEMP TABLE geometries AS
        SELECT node_id, MIN(geometry) AS geometry
        FROM read_parquet('{repo_relative(dataset_path)}')
        GROUP BY node_id;
    CREATE OR REPLACE TEMP TABLE power AS
        SELECT * FROM read_csv_auto('{repo_relative(args.power_summary)}');
    CREATE OR REPLACE TEMP TABLE empirical AS
        SELECT * FROM read_csv_auto('{repo_relative(args.empirical_summary)}');
    COPY (
        SELECT {', '.join(select_clause)}
        FROM power p
        INNER JOIN geometries g USING (node_id)
        LEFT JOIN empirical e USING (node_id)
        ORDER BY node_id
    ) TO '{repo_relative(output_parquet)}' (FORMAT 'parquet');
    CREATE OR REPLACE TEMP VIEW published AS
        SELECT * FROM read_parquet('{repo_relative(output_parquet)}');
    COPY (
        SELECT name AS column_name, type AS column_type
        FROM pragma_table_info('published')
        ORDER BY cid
    ) TO '{repo_relative(table_info_csv)}' (FORMAT 'csv', HEADER);
    """
    run_duckdb(sql, engine=args.engine, image=args.image)

    if not table_info_csv.exists():
        raise RuntimeError(
            "DuckDB execution finished but table information file was not created. "
            "Inspect the DuckDB logs for errors."
        )

    table_columns = build_table_columns(table_info_csv)
    table_info_csv.unlink(missing_ok=True)

    metadata = load_json(args.metadata_json)
    manifest = build_manifest(
        version_tag=args.version_tag,
        metadata=metadata,
        resolved_asset=resolved_asset,
        metadata_json=args.metadata_json,
        power_summary=args.power_summary,
        empirical_summary=args.empirical_summary,
        output_parquet=output_parquet,
    )

    manifest_path = collection_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    row_count = count_nodes(args.power_summary)
    collection_payload, item_payload = build_stac_payloads(
        version_tag=args.version_tag,
        resolved_item_path=resolved_asset.item_path,
        manifest=manifest,
        table_columns=table_columns,
        row_count=row_count,
    )

    (collection_dir / "collection.json").write_text(json.dumps(collection_payload, indent=2) + "\n", encoding="utf-8")
    (items_dir / "power_estimates_nodes.json").write_text(json.dumps(item_payload, indent=2) + "\n", encoding="utf-8")

    update_root_catalog(collection_dir)

    print(f"Published catalog at {repo_relative(collection_dir)}")


if __name__ == "__main__":
    main()
