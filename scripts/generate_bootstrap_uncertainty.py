#!/usr/bin/env python3
"""Generate bootstrap confidence intervals aligned with the power pipeline."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import os
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableSet, Sequence, Tuple

import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hf_wind_resource.io import resolve_catalog_asset
from hf_wind_resource.stats import (
    BootstrapConfidenceInterval,
    BootstrapUncertaintyResult,
    GlobalRmseProvider,
    GlobalRmseRecord,
    HeightCorrection,
    NodeBootstrapInput,
    StratifiedBootstrapConfig,
    compute_stratified_bootstrap_uncertainty,
    load_kaplan_meier_selection_criteria,
)
from hf_wind_resource.stats.power import PowerCurve


DEFAULT_IMAGE = "duckdb/duckdb:latest"
ENGINE_CHOICES = {"docker", "python"}
DEFAULT_STAC_CONFIG = Path("config/stac_catalogs.json")
DEFAULT_STAC_DATASET = "sar_range_final_pivots_joined"
DEFAULT_OUTPUT_DIR = Path("artifacts/bootstrap_uncertainty")
DEFAULT_NODE_SUMMARY = Path("artifacts/power_estimates/node_summary/node_summary.csv")
DEFAULT_POWER_CURVE_CONFIG = Path("config/power_curves.json")
DEFAULT_POWER_CURVE_KEY = "reference_offshore_6mw"
DEFAULT_BLOCK_LENGTHS = Path("artifacts/bootstrap_uncertainty/block_bootstrap_diagnostics.csv")
INPUT_FILENAME = "bootstrap_inputs.jsonl"
PARTIAL_RESULTS_FILENAME = "bootstrap_results.jsonl"

logger = logging.getLogger("generate_bootstrap_uncertainty")


def _resolve_dataset(path: Path | None, *, catalog: Path, dataset: str) -> Path:
    if path is not None:
        return path if path.is_absolute() else (Path.cwd() / path).resolve()
    asset = resolve_catalog_asset(dataset, config_path=catalog)
    return asset.require_local_path()


def _prepare_output_directory(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)


def _build_rmse_provider(value: float | None, source: str | None) -> GlobalRmseProvider | None:
    if value is None:
        return None

    note = "User-provided RMSE for bootstrap uncertainty."

    def loader() -> Sequence[GlobalRmseRecord]:
        now = datetime.now(timezone.utc)
        record = GlobalRmseRecord(
            version="custom",
            value=float(value),
            unit="m/s",
            effective_from=now,
            effective_until=None,
            source=source or "user_provided",
            computed_at=now,
            notes=(note,),
        )
        return (record,)

    return GlobalRmseProvider(loader=loader)


def _build_sql(dataset: Path, *, max_nodes: int | None = None, dataset_kind: str = "ann") -> str:
    limit_clause = ""
    if max_nodes is not None:
        limit_clause = f"\nLIMIT {max_nodes}"

    dataset_literal = dataset.as_posix()

    if dataset_kind == "uncensored":
        select_clause = textwrap.dedent(
            """
            SELECT
                node_id,
                LIST(
                    struct_pack(
                        "timestamp" := timestamp,
                        "pred_wind_speed" := wind_speed,
                        "prob_range_below" := 0.0,
                        "prob_range_in" := 1.0,
                        "prob_range_above" := 0.0,
                        "range_flag" := 'in_range',
                        "range_flag_confident" := TRUE
                    )
                    ORDER BY timestamp
                ) AS records
            FROM read_parquet('{dataset}')
            GROUP BY node_id
            ORDER BY node_id{limit};
            """
        )
        return select_clause.format(dataset=dataset_literal, limit=limit_clause)

    return textwrap.dedent(
        f"""
        SELECT
            node_id,
            LIST(
                struct_pack(
                    "timestamp" := timestamp,
                    "pred_wind_speed" := pred_wind_speed,
                    "prob_range_below" := prob_range_below,
                    "prob_range_in" := prob_range_in,
                    "prob_range_above" := prob_range_above,
                    "range_flag" := range_flag,
                    "range_flag_confident" := range_flag_confident
                )
                ORDER BY timestamp
            ) AS records
        FROM read_parquet('{dataset_literal}')
        GROUP BY node_id
        ORDER BY node_id{limit_clause};
        """
    )


def _run_duckdb_export(sql: str, *, workdir: Path, image: str, destination: Path, engine: str) -> None:
    if engine == "docker":
        relative_path = destination.relative_to(workdir)
        command: List[str] = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{workdir}:/workspace",
            "-w",
            "/workspace",
            image,
            "duckdb",
            "-cmd",
            f"COPY ({sql}) TO '{relative_path.as_posix()}' (FORMAT JSON);",
        ]
        subprocess.run(command, check=True)
        return

    if engine == "python":
        try:
            import duckdb  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("duckdb Python module not available; install duckdb or use --engine docker") from exc

        con = duckdb.connect()
        try:
            cleaned_sql = sql.strip()
            if cleaned_sql.endswith(";"):
                cleaned_sql = cleaned_sql[:-1]
            con.execute(
                f"COPY ({cleaned_sql}) TO '{destination.as_posix()}' (FORMAT JSON);"
            )
        finally:
            con.close()
        return

    raise ValueError(f"Unsupported engine: {engine}")


def _load_power_curve(path: Path, key: str) -> PowerCurve:
    resolved = path if path.is_absolute() else (Path.cwd() / path)
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if key not in data:
        raise KeyError(f"Power-curve key '{key}' not present in {resolved}")
    entry = data[key]
    return PowerCurve(
        name=str(entry.get("name", key)),
        speeds=tuple(float(x) for x in entry["speeds"]),
        power_kw=tuple(float(x) for x in entry["power_kw"]),
        reference_air_density=float(entry.get("reference_air_density", 1.225)),
        hub_height_m=float(entry["hub_height_m"]) if entry.get("hub_height_m") is not None else None,
        notes=tuple(str(note) for note in entry.get("notes", ())),
    )


def _load_height_corrections(node_summary_path: Path) -> Dict[str, HeightCorrection]:
    resolved = node_summary_path if node_summary_path.is_absolute() else (Path.cwd() / node_summary_path)
    corrections: Dict[str, HeightCorrection] = {}
    if not resolved.exists():
        return corrections

    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            node_id = row.get("node_id")
            if not node_id:
                continue
            method = (row.get("height_method") or "none").strip().lower()
            try:
                source = float(row.get("height_source_m")) if row.get("height_source_m") else 10.0
                target = float(row.get("height_target_m")) if row.get("height_target_m") else source
                speed_scale = float(row.get("height_speed_scale")) if row.get("height_speed_scale") else 1.0
            except ValueError:
                continue
            alpha_value = row.get("height_power_law_alpha")
            roughness_value = row.get("height_roughness_length_m")
            alpha = float(alpha_value) if alpha_value not in (None, "") else None
            roughness = float(roughness_value) if roughness_value not in (None, "") else None
            corrections[node_id] = HeightCorrection(
                method=method,
                source_height_m=source,
                target_height_m=target,
                speed_scale=speed_scale,
                power_law_alpha=alpha,
                roughness_length_m=roughness,
            )

    return corrections


def _load_block_lengths(path: Path | None) -> Dict[str, int]:
    if path is None:
        return {}
    resolved = path if path.is_absolute() else (Path.cwd() / path)
    if not resolved.exists():
        logger.warning("Block length mapping not found at %s", resolved)
        return {}

    mapping: Dict[str, int] = {}
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            node_id = row.get("node_id")
            length_value = row.get("suggested_block_length")
            if not node_id or not length_value:
                continue
            try:
                length = int(float(length_value))
            except ValueError:
                continue
            mapping[node_id] = max(1, length)
    return mapping


def _load_inputs_from_json(
    path: Path,
    *,
    height_corrections: Mapping[str, HeightCorrection],
    default_height: HeightCorrection,
) -> List[NodeBootstrapInput]:
    inputs: List[NodeBootstrapInput] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            node_id = str(record["node_id"])
            raw_records = tuple(record.get("records") or [])
            height = height_corrections.get(node_id, default_height)
            inputs.append(
                NodeBootstrapInput(
                    node_id=node_id,
                    records=raw_records,
                    height=height,
                )
            )
    return inputs


def _summarise_result(result: BootstrapUncertaintyResult) -> dict[str, object]:
    return _summarise_results([result])[0]


def _compute_node_summary(
    item: NodeBootstrapInput,
    config: StratifiedBootstrapConfig,
    rmse_provider: GlobalRmseProvider | None,
) -> dict[str, object]:
    result = compute_stratified_bootstrap_uncertainty(
        item,
        config=config,
        rmse_provider=rmse_provider,
    )
    return _summarise_result(result)


def _load_partial_rows(path: Path) -> Tuple[List[dict[str, object]], MutableSet[str]]:
    if not path.exists():
        return [], set()
    rows: List[dict[str, object]] = []
    processed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            node_id = str(row.get("node_id"))
            if node_id:
                processed.add(node_id)
    return rows, processed


def _append_partial_row(path: Path, row: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row))
        handle.write("\n")


def _process_nodes(
    inputs: Iterable[NodeBootstrapInput],
    *,
    config: StratifiedBootstrapConfig,
    rmse_provider: GlobalRmseProvider | None,
    partial_path: Path,
    resume: bool,
    progress_interval: int,
    workers: int,
) -> tuple[List[dict[str, object]], int]:
    if resume:
        rows, processed_ids = _load_partial_rows(partial_path)
    else:
        if partial_path.exists():
            partial_path.unlink()
        rows, processed_ids = [], set()

    inputs_list = list(inputs)
    total = len(inputs_list)
    processed_count = len(processed_ids)
    pending_items = [item for item in inputs_list if item.node_id not in processed_ids]

    effective_workers = workers
    if effective_workers <= 0:
        effective_workers = os.cpu_count() or 1
    if pending_items:
        effective_workers = max(1, min(effective_workers, len(pending_items)))
    else:
        effective_workers = max(1, effective_workers)

    if rmse_provider is not None and effective_workers > 1:
        logger.info("RMSE override detected; forcing sequential execution to keep the custom provider in-process.")
        effective_workers = 1

    logger.info(
        "Starting bootstrap evaluation for %d nodes (resume=%s, already processed=%d)",
        total,
        resume,
        processed_count,
    )

    if not pending_items:
        return rows, effective_workers

    if effective_workers == 1:
        for item in pending_items:
            row = _compute_node_summary(item, config, rmse_provider)
            _append_partial_row(partial_path, row)
            rows.append(row)
            processed_ids.add(item.node_id)
            processed_count += 1

            if progress_interval > 0 and (
                processed_count % progress_interval == 0 or processed_count == total
            ):
                logger.info(
                    "Processed %d/%d nodes (latest=%s, method=%s)",
                    processed_count,
                    total,
                    item.node_id,
                    row.get("power_method"),
                )
        return rows, effective_workers

    logger.info("Using %d parallel workers", effective_workers)

    with concurrent.futures.ProcessPoolExecutor(max_workers=effective_workers) as executor:
        future_to_item = {
            executor.submit(_compute_node_summary, item, config, rmse_provider): item for item in pending_items
        }
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                row = future.result()
            except Exception as exc:
                logger.exception("Bootstrap evaluation failed for node %s", item.node_id)
                raise

            _append_partial_row(partial_path, row)
            rows.append(row)
            processed_ids.add(item.node_id)
            processed_count += 1

            if progress_interval > 0 and (
                processed_count % progress_interval == 0 or processed_count == total
            ):
                logger.info(
                    "Processed %d/%d nodes (latest=%s, method=%s)",
                    processed_count,
                    total,
                    item.node_id,
                    row.get("power_method"),
                )

    return rows, effective_workers


def _summarise_results(results: Iterable[BootstrapUncertaintyResult]) -> List[dict[str, object]]:
    rows: List[dict[str, object]] = []
    for result in results:
        metrics = result.metrics
        bootstrap_means = result.bootstrap_means
        power_diag = result.power_diagnostics
        row: dict[str, object] = {
            "node_id": result.node_id,
            "rmse_version": result.rmse_record.version,
            "rmse_value": result.rmse_record.value,
            "rmse_unit": result.rmse_record.unit,
            "rmse_source": result.rmse_record.source,
            "total_samples": result.total_samples,
            "label_count_below": result.label_counts.get("below", 0.0),
            "label_count_in": result.label_counts.get("in", 0.0),
            "label_count_above": result.label_counts.get("above", 0.0),
            "label_count_uncertain": result.label_counts.get("uncertain", 0.0),
            "label_ratio_below": result.label_proportions.get("below", 0.0),
            "label_ratio_in": result.label_proportions.get("in", 0.0),
            "label_ratio_above": result.label_proportions.get("above", 0.0),
            "label_ratio_uncertain": result.label_proportions.get("uncertain", 0.0),
            "notes": " | ".join(result.notes),
        }

        _inject_metric(row, metrics, bootstrap_means, "mean_speed", prefix="mean_speed")
        _inject_metric(row, metrics, bootstrap_means, "p50", prefix="p50")
        _inject_metric(row, metrics, bootstrap_means, "p90", prefix="p90")
        _inject_metric(row, metrics, bootstrap_means, "p99", prefix="p99")
        _inject_metric(row, metrics, bootstrap_means, "power_density", prefix="power_density")

        if power_diag is not None:
            row["power_method"] = power_diag.method
            row["power_selection_reasons"] = " | ".join(power_diag.selection_reasons)
            row["power_method_notes"] = " | ".join(power_diag.method_notes)
            for method, count in sorted(power_diag.replicate_method_counts.items()):
                row[f"replicate_method_{method}"] = count
        else:
            row["power_method"] = None
            row["power_selection_reasons"] = ""
            row["power_method_notes"] = ""

        rows.append(row)
    return rows


def _inject_metric(
    row: dict[str, object],
    metrics: Mapping[str, BootstrapConfidenceInterval],
    bootstrap_means: Mapping[str, float | None],
    name: str,
    *,
    prefix: str,
) -> None:
    interval = metrics.get(name)
    if interval is None:
        row[f"{prefix}_estimate"] = None
        row[f"{prefix}_lower"] = None
        row[f"{prefix}_upper"] = None
        row[f"{prefix}_replicates"] = 0
        row[f"{prefix}_bootstrap_estimate"] = bootstrap_means.get(name)
        return

    row[f"{prefix}_estimate"] = interval.estimate
    row[f"{prefix}_lower"] = interval.lower
    row[f"{prefix}_upper"] = interval.upper
    row[f"{prefix}_replicates"] = interval.replicates
    row[f"{prefix}_bootstrap_estimate"] = bootstrap_means.get(name)


def _write_summary_csv(rows: Iterable[dict[str, object]], destination: Path) -> None:
    rows = _deduplicate_rows(rows)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _deduplicate_rows(rows: Iterable[dict[str, object]]) -> List[dict[str, object]]:
    collected = list(rows)
    if not collected:
        return []
    latest: dict[str, dict[str, object]] = {}
    for row in collected:
        node_id = str(row.get("node_id"))
        latest[node_id] = row
    return sorted(latest.values(), key=lambda row: row.get("node_id"))


def _write_metadata(
    destination: Path,
    *,
    config: StratifiedBootstrapConfig,
    dataset: Path,
    args: argparse.Namespace,
    partial_path: Path,
    processed_nodes: int,
    workers: int,
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset.as_posix(),
        "replicas": config.replicas,
        "confidence_level": config.confidence_level,
        "apply_rmse_noise": config.apply_rmse_noise,
        "min_confidence": config.min_confidence,
        "min_in_range_weight": config.min_in_range_weight,
        "tail_surrogate": config.tail_surrogate,
        "random_seed": config.random_seed,
        "air_density": config.air_density,
        "rmse_mode": config.rmse_mode,
        "resampling_mode": config.resampling_mode,
        "block_length": config.block_length,
        "block_lengths_csv": (args.block_lengths_csv.as_posix() if args.block_lengths_csv else None),
        "max_block_length": args.max_block_length,
        "node_block_lengths_loaded": len(config.node_block_lengths or {}),
        "label_strategy": config.label_strategy,
        "ci_method": config.ci_method,
        "jackknife_max_samples": config.jackknife_max_samples,
        "power_curve": config.power_curve.to_mapping(),
        "km_criteria": config.km_criteria.__dict__,
        "docker_image": args.image,
        "max_nodes": args.max_nodes,
        "resume": args.resume,
        "progress_interval": args.progress_interval,
        "partial_results_path": partial_path.as_posix(),
        "processed_nodes": processed_nodes,
        "workers_requested": args.workers,
        "workers_used": workers,
    }
    if args.rmse is not None:
        payload["rmse_override"] = float(args.rmse)
    if args.rmse_source is not None:
        payload["rmse_override_source"] = args.rmse_source
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Path to the GeoParquet dataset", default=None)
    parser.add_argument("--stac-config", type=Path, default=DEFAULT_STAC_CONFIG, help="Path to the STAC catalog JSON")
    parser.add_argument("--stac-dataset", default=DEFAULT_STAC_DATASET, help="Dataset key in the STAC config")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output artefacts")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="DuckDB container image")
    parser.add_argument("--engine", choices=sorted(ENGINE_CHOICES), default="docker", help="Execution engine for DuckDB access")
    parser.add_argument("--node-summary", type=Path, default=DEFAULT_NODE_SUMMARY, help="CSV with node height metadata")
    parser.add_argument("--power-curve-config", type=Path, default=DEFAULT_POWER_CURVE_CONFIG, help="JSON file with power curves")
    parser.add_argument("--power-curve-key", default=DEFAULT_POWER_CURVE_KEY, help="Key selecting the power curve")
    parser.add_argument("--km-criteria-config", type=Path, default=None, help="Optional JSON with Kaplan–Meier selection criteria")
    parser.add_argument("--air-density", type=float, default=1.225, help="Air density (kg/m³) used for power calculations")
    parser.add_argument("--right-tail-surrogate", type=float, default=None, help="Optional surrogate wind speed for right-censored mass")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum confidence threshold for range labels")
    parser.add_argument("--min-in-range", type=float, default=500.0, help="Minimum in-range weight expected for Weibull fits")
    parser.add_argument(
        "--rmse",
        type=float,
        default=None,
        help="Override the RMSE value (m/s) used for noise and reporting.",
    )
    parser.add_argument(
        "--rmse-source",
        default=None,
        help="Optional source label recorded when --rmse is provided.",
    )
    parser.add_argument("--replicas", type=int, default=500, help="Number of bootstrap replicas")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level for the intervals")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the bootstrap")
    parser.add_argument("--no-rmse-noise", action="store_true", help="Disable RMSE perturbations")
    parser.add_argument(
        "--rmse-mode",
        choices=("velocity", "power", "none"),
        default="velocity",
        help="How to propagate RMSE noise: perturb velocities, power metric, or disable.",
    )
    parser.add_argument(
        "--resampling-mode",
        choices=("iid", "moving_block", "stationary"),
        default="iid",
        help="Bootstrap resampling strategy (iid, moving_block, stationary).",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=4,
        help="Default block length used for moving/stationary bootstrap modes.",
    )
    parser.add_argument(
        "--block-lengths-csv",
        type=Path,
        default=DEFAULT_BLOCK_LENGTHS,
        help="Optional CSV with per-node block length recommendations.",
    )
    parser.add_argument(
        "--max-block-length",
        type=int,
        default=None,
        help="Maximum block length when applying recommendations from the CSV.",
    )
    parser.add_argument(
        "--label-strategy",
        choices=("fixed", "label_resample"),
        default="fixed",
        help="How to handle range-label uncertainty during resampling.",
    )
    parser.add_argument(
        "--ci-method",
        choices=("percentile", "bca", "percentile_t"),
        default="percentile",
        help="Confidence interval method (percentile, BCa, percentile-t).",
    )
    parser.add_argument(
        "--jackknife-max-samples",
        type=int,
        default=200,
        help="Maximum sample size for jackknife-based corrections (BCa/percentile-t).",
    )
    parser.add_argument("--max-nodes", type=int, default=None, help="Process at most this number of nodes (debug)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing partial results if available")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Log progress every N processed nodes (0 disables periodic progress logs)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel worker processes (0 selects cpu_count).",
    )
    parser.add_argument(
        "--dataset-kind",
        choices=("ann", "uncensored"),
        default="ann",
        help="Schema of the input dataset: 'ann' for inference outputs, 'uncensored' for generic wind series.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    dataset = _resolve_dataset(args.dataset, catalog=args.stac_config, dataset=args.stac_dataset)
    output_dir = args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir)
    output_dir = output_dir.resolve()
    _prepare_output_directory(output_dir)

    power_curve = _load_power_curve(args.power_curve_config, args.power_curve_key)
    km_criteria = load_kaplan_meier_selection_criteria(args.km_criteria_config)
    rmse_provider = _build_rmse_provider(args.rmse, args.rmse_source)

    default_height = HeightCorrection(method="none", source_height_m=10.0, target_height_m=10.0, speed_scale=1.0)
    height_corrections = _load_height_corrections(args.node_summary)
    node_block_lengths = _load_block_lengths(args.block_lengths_csv)
    if args.max_block_length is not None and args.max_block_length > 0:
        node_block_lengths = {
            key: min(value, args.max_block_length)
            for key, value in node_block_lengths.items()
        }

    config = StratifiedBootstrapConfig(
        replicas=args.replicas,
        confidence_level=args.confidence,
        random_seed=args.seed,
        apply_rmse_noise=(not args.no_rmse_noise) and args.rmse_mode != "none",
        rmse_mode=args.rmse_mode,
        ci_method=args.ci_method,
        jackknife_max_samples=args.jackknife_max_samples,
        resampling_mode=args.resampling_mode,
        block_length=args.block_length,
        node_block_lengths=node_block_lengths,
        label_strategy=args.label_strategy,
        air_density=args.air_density,
        min_confidence=args.min_confidence,
        min_in_range_weight=args.min_in_range,
        tail_surrogate=args.right_tail_surrogate,
        power_curve=power_curve,
        km_criteria=km_criteria,
    )

    sql = _build_sql(dataset, max_nodes=args.max_nodes, dataset_kind=args.dataset_kind)
    inputs_path = output_dir / INPUT_FILENAME
    _run_duckdb_export(sql, workdir=output_dir, image=args.image, destination=inputs_path, engine=args.engine)

    inputs = _load_inputs_from_json(inputs_path, height_corrections=height_corrections, default_height=default_height)
    partial_path = output_dir / PARTIAL_RESULTS_FILENAME
    rows, used_workers = _process_nodes(
        inputs,
        config=config,
        rmse_provider=rmse_provider,
        partial_path=partial_path,
        resume=args.resume,
        progress_interval=max(0, args.progress_interval),
        workers=args.workers,
    )

    rows = _deduplicate_rows(rows)

    summary_path = output_dir / "bootstrap_summary.csv"
    _write_summary_csv(rows, summary_path)

    metadata_path = output_dir / "bootstrap_metadata.json"
    _write_metadata(
        metadata_path,
        config=config,
        dataset=dataset,
        args=args,
        partial_path=partial_path,
        processed_nodes=len(rows),
        workers=used_workers,
    )


if __name__ == "__main__":
    main()
