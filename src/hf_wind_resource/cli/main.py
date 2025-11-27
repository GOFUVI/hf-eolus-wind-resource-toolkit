"""Command-line orchestrator for the HF wind-resource toolkit.

This module exposes a unified CLI that wires together the standalone
helpers under ``scripts/`` so the full wind-resource pipeline can run
end-to-end (or partially) through a single entry point. The implementation
relies on the schema documented in ``docs/sar_range_final_schema.md`` and
follows the modular boundaries described in ``docs/python_architecture.md``:
I/O resolution via :mod:`hf_wind_resource.io`, preprocessing filters that
respect the ANN range-classification fields, statistical routines living in
``hf_wind_resource.stats``, and reporting/export stages under ``scripts/``.

Two sub-commands are provided:

``compute_resource``
    Executes the production pipeline stages (empirical metrics, power
    estimation, node summary, bootstrap uncertainty, geospatial exports,
    and optional STAC publication). Callers can filter the workflow to a
    subset of nodes, pick individual stages, and tweak bootstrap options
    without invoking each script manually.

``validate_buoy``
    Reuses the shared dataset resolution and node-filtering logic to align
    ANN predictions with the Vilano buoy reference. The helper prepares the
    matched time series and generates direction/error diagnostics in a
    single invocation.

The CLI intentionally shells out to the existing ``scripts/`` entry points
instead of duplicating their logic so that artefact generation remains
consistent with the project documentation and test harness.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping, Sequence

from hf_wind_resource.io import resolve_catalog_asset

StageBuilder = Callable[["PipelineStage", "PipelineContext", argparse.Namespace], list[str]]

_DEFAULT_PREPARE_IMAGE = "wind-resource-tests"


def _is_python_command(command: Sequence[str]) -> bool:
    if not command:
        return False
    return Path(command[0]).name.lower().startswith("python")


def _capture_python_command(
    command: Sequence[str],
    *,
    repo_root: Path,
    logger: logging.Logger,
) -> subprocess.CompletedProcess[str]:
    printable = " ".join(shlex.quote(arg) for arg in command)
    logger.info("Running: %s", printable)
    result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result


def _run_python_with_backend(
    command: Sequence[str],
    *,
    repo_root: Path,
    logger: logging.Logger,
    backend: str,
) -> None:
    if backend == "docker":
        _run_python_in_docker(command, repo_root=repo_root, logger=logger)
        return
    if backend == "host":
        result = _capture_python_command(command, repo_root=repo_root, logger=logger)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                command,
                result.stdout,
                result.stderr,
            )
        return
    if backend != "auto":
        raise ValueError(f"Unknown python backend: {backend!r}")

    # auto backend: try host first, then fallback to docker on missing deps
    result = _capture_python_command(command, repo_root=repo_root, logger=logger)
    if result.returncode == 0:
        return

    stderr_text = result.stderr or ""
    if "ModuleNotFoundError" in stderr_text or "ImportError" in stderr_text:
        logger.warning(
            "Python dependency missing in host interpreter, retrying inside '%s'.",
            _DEFAULT_PREPARE_IMAGE,
        )
        _run_python_in_docker(command, repo_root=repo_root, logger=logger)
        return

    raise subprocess.CalledProcessError(
        result.returncode,
        command,
        result.stdout,
        stderr_text,
    )


def _run_python_in_docker(
    command: Sequence[str],
    *,
    repo_root: Path,
    logger: logging.Logger,
) -> None:
    forwarded = list(command)
    for index, value in enumerate(forwarded):
        if value == "--engine" and index + 1 < len(forwarded) and forwarded[index + 1] == "docker":
            forwarded[index + 1] = "python"
        elif value.startswith("--engine=") and value.endswith("docker"):
            forwarded[index] = value.replace("docker", "python")

    docker_command: list[str] = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{repo_root}:/workspace",
        "-w",
        "/workspace",
        "--entrypoint",
        "python",
        _DEFAULT_PREPARE_IMAGE,
    ]
    docker_command.extend(forwarded[1:])
    _run_subprocess(docker_command, cwd=repo_root, logger=logger)


@dataclass(frozen=True)
class PipelineStage:
    """Declarative definition for a pipeline step."""

    name: str
    description: str
    builder: StageBuilder
    supports_filtered_dataset: bool = True

    def build_command(self, context: "PipelineContext", args: argparse.Namespace) -> list[str]:
        """Delegate to the stage builder."""

        return self.builder(self, context, args)


@dataclass
class PipelineContext:
    """Runtime parameters shared across pipeline stages."""

    repo_root: Path
    dataset_path: Path
    original_dataset_path: Path
    filtered_dataset_path: Path | None
    engine: str
    docker_image: str
    overwrite: bool
    max_nodes: int | None
    disable_buoy_outputs: bool
    python_backend: str

    def dataset_for_stage(self, stage: PipelineStage) -> Path:
        """Return the dataset path visible to the given stage."""

        if self.filtered_dataset_path and stage.supports_filtered_dataset:
            return self.filtered_dataset_path
        return self.dataset_path

    def to_cli_path(self, path: Path) -> str:
        """Return a repository-relative string when possible."""

        try:
            return str(path.resolve().relative_to(self.repo_root))
        except ValueError:
            return str(path)


def _build_logger(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")
    return logging.getLogger("hf_wind_resource.cli")


def _resolve_repo_root() -> Path:
    # main.py sits under repo_root / "src" / "hf_wind_resource" / "cli"
    return Path(__file__).resolve().parents[3]


def _resolve_dataset(
    repo_root: Path,
    *,
    direct_path: Path | None,
    stac_config: Path,
    stac_dataset: str,
) -> Path:
    if direct_path is not None:
        return (direct_path if direct_path.is_absolute() else (Path.cwd() / direct_path)).resolve()

    resolved = resolve_catalog_asset(
        stac_dataset,
        config_path=stac_config,
        root=repo_root,
    )
    return resolved.require_local_path()


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return str(path.resolve())


def _run_subprocess(command: Sequence[str], *, cwd: Path, logger: logging.Logger) -> None:
    printable = " ".join(shlex.quote(arg) for arg in command)
    logger.info("Running: %s", printable)
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        stderr_text = exc.stderr or ""
        if "ModuleNotFoundError" in stderr_text or "ImportError" in stderr_text:
            logger.error(
                "Python dependency missing while executing %s.\n"
                "Please install the required scientific packages (e.g. numpy, pandas, pyarrow, matplotlib) "
                "in the active environment and rerun the CLI.",
                printable,
            )
        raise


def _require_repo_relative(path: Path, repo_root: Path, *, label: str) -> str:
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError as exc:  # pragma: no cover - defensive guardrail
        raise RuntimeError(
            f"{label} must reside within the repository tree ({repo_root}). "
            "Mount paths outside the repository are not accessible to the DuckDB container."
        ) from exc


def _materialise_filtered_dataset(
    *,
    dataset: Path,
    nodes: Sequence[str],
    repo_root: Path,
    image: str,
    engine: str,
    logger: logging.Logger,
) -> Path:
    if not nodes:
        raise ValueError("Node filter cannot be empty when creating a filtered dataset.")

    node_list = tuple(dict.fromkeys(node.strip() for node in nodes if node.strip()))
    if not node_list:
        raise ValueError("Node filter yielded no valid identifiers.")

    dataset_rel = _require_repo_relative(dataset, repo_root, label="Dataset path")

    hash_input = ",".join(node_list).encode("utf-8")
    digest = sha256(hash_input).hexdigest()[:12]
    output_dir = repo_root / "artifacts" / "tmp_cli_filters"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"filtered_{digest}.parquet"

    if output_path.exists():
        logger.debug("Reusing cached filtered dataset at %s", output_path)
        return output_path

    node_literals = ", ".join(f"'{node}'" for node in node_list)
    sql = (
        "COPY ("
        f"SELECT * FROM read_parquet('{dataset_rel}') WHERE node_id IN ({node_literals})"
        ") TO '{output}' (FORMAT PARQUET);"
    ).format(output=_require_repo_relative(output_path, repo_root, label="Filtered dataset path"))

    logger.info("Materialising filtered dataset with %d nodes → %s", len(node_list), output_path)

    if engine == "python":
        try:
            import duckdb  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("duckdb Python module is required when --engine=python") from exc
        con = duckdb.connect()
        try:
            con.execute(sql)
        finally:
            con.close()
    else:
        command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{repo_root}:/workspace",
            "-w",
            "/workspace",
            image,
            "duckdb",
            "-cmd",
            sql,
        ]
        _run_subprocess(command, cwd=repo_root, logger=logger)

    if not output_path.exists():  # pragma: no cover - defensive
        raise RuntimeError(f"Filtered dataset not produced at {output_path}")
    return output_path


def _build_compute_stages() -> list[PipelineStage]:
    return [
        PipelineStage(
            name="empirical",
            description="Generate empirical metrics (scripts/generate_empirical_metrics.py)",
            builder=_build_empirical_command,
        ),
        PipelineStage(
            name="power",
            description="Compute Weibull/Kaplan-Meier power estimates (scripts/generate_power_estimates.py)",
            builder=_build_power_command,
        ),
        PipelineStage(
            name="node_summary",
            description="Assemble per-node summary table (scripts/generate_node_summary_table.py)",
            builder=_build_node_summary_command,
        ),
        PipelineStage(
            name="bootstrap",
            description="Run bootstrap uncertainty analysis (scripts/generate_bootstrap_uncertainty.py)",
            builder=_build_bootstrap_command,
        ),
        PipelineStage(
            name="geospatial",
            description="Produce GeoParquet/GeoJSON/visualisations (scripts/generate_geospatial_products.py)",
            builder=_build_geospatial_command,
        ),
        PipelineStage(
            name="publish",
            description="Publish STAC deliverable (scripts/publish_power_estimates_catalog.py)",
            builder=_build_publish_command,
            supports_filtered_dataset=False,
        ),
    ]


def _select_stages(
    stages: list[PipelineStage],
    *,
    include: Sequence[str] | None,
    skip: Sequence[str] | None,
) -> list[PipelineStage]:
    name_map: Mapping[str, PipelineStage] = {stage.name: stage for stage in stages}

    if include:
        selected: list[PipelineStage] = []
        for name in include:
            try:
                selected.append(name_map[name])
            except KeyError as exc:
                raise ValueError(f"Unknown stage {name!r}; available stages: {', '.join(name_map)}") from exc
    else:
        selected = list(stages)

    if skip:
        skip_set = set(skip)
        selected = [stage for stage in selected if stage.name not in skip_set]

    if not selected:
        raise ValueError("No pipeline stages selected after applying filters.")
    return selected


def _build_empirical_command(stage: PipelineStage, context: PipelineContext, args: argparse.Namespace) -> list[str]:
    command: list[str] = [
        sys.executable,
        "scripts/generate_empirical_metrics.py",
        "--dataset",
        context.to_cli_path(context.dataset_for_stage(stage)),
        "--docker-image",
        context.docker_image,
    ]
    if context.overwrite:
        command.append("--overwrite")
    return command


def _build_power_command(stage: PipelineStage, context: PipelineContext, args: argparse.Namespace) -> list[str]:
    command: list[str] = [
        sys.executable,
        "scripts/generate_power_estimates.py",
        "--dataset",
        context.to_cli_path(context.dataset_for_stage(stage)),
        "--image",
        context.docker_image,
        "--engine",
        context.engine,
    ]
    if context.max_nodes is not None:
        command.extend(["--max-nodes", str(context.max_nodes)])
    if args.power_curve_key:
        command.extend(["--power-curve-key", args.power_curve_key])
    if args.air_density is not None:
        command.extend(["--air-density", str(args.air_density)])
    if args.right_tail_surrogate is not None:
        command.extend(["--right-tail-surrogate", str(args.right_tail_surrogate)])
    return command


def _build_node_summary_command(stage: PipelineStage, context: PipelineContext, args: argparse.Namespace) -> list[str]:
    command: list[str] = [
        sys.executable,
        "scripts/generate_node_summary_table.py",
        "--image",
        context.docker_image,
        "--engine",
        context.engine,
    ]
    dataset = context.dataset_for_stage(stage)
    if dataset != context.original_dataset_path:
        command.extend(["--dataset", context.to_cli_path(dataset)])
    if context.overwrite:
        command.append("--overwrite")
    if context.overwrite:
        command.append("--overwrite")
    return command


def _build_bootstrap_command(stage: PipelineStage, context: PipelineContext, args: argparse.Namespace) -> list[str]:
    command: list[str] = [
        sys.executable,
        "scripts/generate_bootstrap_uncertainty.py",
        "--dataset",
        str(context.dataset_for_stage(stage)),
        "--image",
        context.docker_image,
        "--engine",
        context.engine,
        "--replicas",
        str(args.bootstrap_replicas),
        "--confidence",
        str(args.bootstrap_confidence),
    ]
    if args.bootstrap_seed is not None:
        command.extend(["--seed", str(args.bootstrap_seed)])
    if args.bootstrap_workers is not None:
        command.extend(["--workers", str(args.bootstrap_workers)])
    if args.bootstrap_resume:
        command.append("--resume")
    if args.bootstrap_disable_rmse:
        command.append("--no-rmse-noise")
    if context.max_nodes is not None:
        command.extend(["--max-nodes", str(context.max_nodes)])
    return command


def _build_geospatial_command(stage: PipelineStage, context: PipelineContext, args: argparse.Namespace) -> list[str]:
    command: list[str] = [
        sys.executable,
        "scripts/generate_geospatial_products.py",
        "--dataset",
        context.to_cli_path(context.dataset_for_stage(stage)),
        "--image",
        context.docker_image,
        "--engine",
        context.engine,
    ]
    if args.geospatial_config is not None:
        command.extend(["--config", str(args.geospatial_config)])
    if context.overwrite or args.geospatial_overwrite:
        command.append("--overwrite")
    if args.geospatial_max_roses is not None:
        command.extend(["--max-wind-roses", str(args.geospatial_max_roses)])
    if context.disable_buoy_outputs or args.disable_buoy_rose:
        command.append("--disable-buoy-rose")
    if args.buoy_dataset is not None:
        command.extend(["--buoy-dataset", str(args.buoy_dataset)])
    if args.buoy_node_id is not None:
        command.extend(["--buoy-node-id", args.buoy_node_id])
    return command


def _build_publish_command(stage: PipelineStage, context: PipelineContext, args: argparse.Namespace) -> list[str]:
    command: list[str] = [
        sys.executable,
        "scripts/publish_power_estimates_catalog.py",
        "--engine",
        context.engine,
        "--image",
        context.docker_image,
    ]
    if args.publish_version_tag:
        command.extend(["--version-tag", args.publish_version_tag])
    if context.overwrite or args.publish_overwrite:
        command.append("--overwrite")
    return command


def _handle_compute_resource(args: argparse.Namespace) -> int:
    logger = _build_logger(args.verbose)
    repo_root = _resolve_repo_root()

    dataset_path = _resolve_dataset(
        repo_root,
        direct_path=args.dataset,
        stac_config=args.stac_config,
        stac_dataset=args.stac_dataset,
    )

    nodes: list[str] = []
    if args.nodes:
        nodes.extend(args.nodes)
    if args.nodes_from:
        text = Path(args.nodes_from).read_text(encoding="utf-8")
        nodes.extend(line.strip() for line in text.splitlines())
    nodes = [node for node in nodes if node]

    filtered_dataset: Path | None = None
    if nodes:
        filtered_dataset = _materialise_filtered_dataset(
            dataset=dataset_path,
            nodes=nodes,
            repo_root=repo_root,
            image=args.image,
            engine=args.engine,
            logger=logger,
        )

    context = PipelineContext(
        repo_root=repo_root,
        dataset_path=dataset_path,
        original_dataset_path=dataset_path,
        filtered_dataset_path=filtered_dataset,
        engine=args.engine,
        docker_image=args.image,
        overwrite=args.overwrite,
        max_nodes=args.max_nodes,
        disable_buoy_outputs=args.disable_buoy_outputs,
        python_backend=args.python_backend,
    )

    stages = _select_stages(
        list(_COMPUTE_STAGE_REGISTRY_LIST),
        include=args.stages,
        skip=args.skip_stages,
    )

    logger.info("Selected stages: %s", ", ".join(stage.name for stage in stages))
    try:
        for stage in stages:
            command = stage.build_command(context, args)
            if _is_python_command(command):
                _run_python_with_backend(
                    command,
                    repo_root=repo_root,
                    logger=logger,
                    backend=context.python_backend,
                )
            else:
                _run_subprocess(command, cwd=repo_root, logger=logger)
    finally:
        if filtered_dataset and not args.keep_filtered_dataset:
            try:
                filtered_dataset.unlink()
                logger.debug("Removed temporary dataset %s", filtered_dataset)
            except FileNotFoundError:
                pass

    return 0


def _handle_uncensored_resource(args: argparse.Namespace) -> int:
    logger = _build_logger(args.verbose)
    repo_root = _resolve_repo_root()

    command: list[str] = [
        sys.executable,
        "scripts/generate_uncensored_resource.py",
        "--config",
        str(args.config),
    ]
    if args.dataset is not None:
        command.extend(["--dataset", str(args.dataset)])
    if args.stac_config is not None:
        command.extend(["--stac-config", str(args.stac_config)])
    if args.stac_dataset is not None:
        command.extend(["--stac-dataset", args.stac_dataset])
    if args.asset_key is not None:
        command.extend(["--asset-key", args.asset_key])
    if args.output_dir is not None:
        command.extend(["--output-dir", str(args.output_dir)])
    if args.rmse is not None:
        command.extend(["--rmse", str(args.rmse)])
    if args.rmse_source is not None:
        command.extend(["--rmse-source", args.rmse_source])
    if args.speed_column is not None:
        command.extend(["--speed-column", args.speed_column])
    if args.node_column is not None:
        command.extend(["--node-column", args.node_column])
    if args.timestamp_column is not None:
        command.extend(["--timestamp-column", args.timestamp_column])

    _run_python_with_backend(
        command,
        repo_root=repo_root,
        logger=logger,
        backend=args.python_backend,
    )
    return 0


def _handle_validate_buoy(args: argparse.Namespace) -> int:
    logger = _build_logger(args.verbose)
    repo_root = _resolve_repo_root()
    dataset_path = _resolve_dataset(
        repo_root,
        direct_path=args.ann_dataset,
        stac_config=args.stac_config,
        stac_dataset=args.stac_dataset,
    )
    try:
        dataset_cli_path = dataset_path.relative_to(repo_root)
    except ValueError:
        dataset_cli_path = dataset_path

    command_prepare: list[str] = [
        sys.executable,
        "scripts/prepare_buoy_timeseries.py",
        "--buoy-dataset",
        str(args.buoy_dataset),
        "--ann-dataset",
        str(dataset_cli_path),
        "--node-id",
        args.node_id,
        "--output-parquet",
        str(args.output_parquet),
        "--output-summary",
        str(args.output_summary),
    ]
    if args.tolerance_minutes is not None:
        command_prepare.extend(["--tolerance-minutes", str(args.tolerance_minutes)])
    if args.nearest_matching:
        command_prepare.append("--nearest-matching")
    if args.disable_height_correction:
        command_prepare.append("--disable-height-correction")
    if args.buoy_height_method is not None:
        command_prepare.extend(["--height-method", args.buoy_height_method])
    if args.buoy_measurement_height is not None:
        command_prepare.extend(["--measurement-height-m", str(args.buoy_measurement_height)])
    if args.buoy_target_height is not None:
        command_prepare.extend(["--target-height-m", str(args.buoy_target_height)])
    if args.buoy_power_law_alpha is not None:
        command_prepare.extend(["--power-law-alpha", str(args.buoy_power_law_alpha)])
    if args.buoy_roughness_length is not None:
        command_prepare.extend(["--roughness-length-m", str(args.buoy_roughness_length)])
    if args.ann_kind is not None:
        command_prepare.extend(["--ann-kind", args.ann_kind])

    _run_python_with_backend(
        command_prepare,
        repo_root=repo_root,
        logger=logger,
        backend=args.python_backend,
    )

    command_direction: list[str] = [
        sys.executable,
        "scripts/generate_direction_comparison.py",
        "--matched-dataset",
        str(args.output_parquet),
        "--output-dir",
        str(args.direction_output),
    ]
    if args.scatter_sample_limit is not None:
        command_direction.extend(["--scatter-sample-limit", str(args.scatter_sample_limit)])

    _run_python_with_backend(
        command_direction,
        repo_root=repo_root,
        logger=logger,
        backend=args.python_backend,
    )

    matched_cli_path = _relative_to_repo(args.output_parquet, repo_root)
    resource_output_dir = _relative_to_repo(args.resource_output_dir, repo_root)

    command_resource: list[str] = [
        sys.executable,
        "scripts/generate_buoy_resource_comparison.py",
        "--matched-dataset",
        matched_cli_path,
        "--node-id",
        args.node_id,
        "--output-dir",
        resource_output_dir,
        "--buoy-dataset",
        _relative_to_repo(args.buoy_dataset, repo_root),
        "--paired-summary",
        _relative_to_repo(args.output_summary, repo_root),
        "--air-density",
        str(args.resource_air_density),
        "--min-confidence",
        str(args.resource_min_confidence),
        "--min-in-range",
        str(args.resource_min_in_range),
    ]
    if args.resource_power_curve_config is not None:
        command_resource.extend(
            ["--power-curve-config", _relative_to_repo(args.resource_power_curve_config, repo_root)]
        )
    if args.resource_power_curve_key is not None:
        command_resource.extend(["--power-curve-key", args.resource_power_curve_key])
    if args.resource_height_config is not None:
        command_resource.extend(["--height-config", _relative_to_repo(args.resource_height_config, repo_root)])
    if args.resource_range_thresholds is not None:
        command_resource.extend(
            ["--range-thresholds", _relative_to_repo(args.resource_range_thresholds, repo_root)]
        )
    if args.resource_bootstrap_summary is not None:
        command_resource.extend(
            ["--bootstrap-summary", _relative_to_repo(args.resource_bootstrap_summary, repo_root)]
        )
    if args.resource_bootstrap_metadata is not None:
        command_resource.extend(
            ["--bootstrap-metadata", _relative_to_repo(args.resource_bootstrap_metadata, repo_root)]
        )
    if args.resource_ann_label is not None:
        command_resource.extend(["--ann-label", args.resource_ann_label])
    if args.resource_buoy_label is not None:
        command_resource.extend(["--buoy-label", args.resource_buoy_label])
    if args.resource_right_tail_surrogate is not None:
        command_resource.extend(["--right-tail-surrogate", str(args.resource_right_tail_surrogate)])
    if args.resource_km_criteria_config is not None:
        command_resource.extend(
            ["--km-criteria-config", _relative_to_repo(args.resource_km_criteria_config, repo_root)]
        )
    if args.resource_buoy_bootstrap_replicates is not None:
        command_resource.extend(["--buoy-bootstrap-replicates", str(args.resource_buoy_bootstrap_replicates)])
    if args.resource_buoy_bootstrap_confidence is not None:
        command_resource.extend(["--buoy-bootstrap-confidence", str(args.resource_buoy_bootstrap_confidence)])
    if args.resource_buoy_bootstrap_seed is not None:
        command_resource.extend(["--buoy-bootstrap-seed", str(args.resource_buoy_bootstrap_seed)])
    if args.resource_buoy_resampling_mode is not None:
        command_resource.extend(["--buoy-resampling-mode", args.resource_buoy_resampling_mode])
    if args.resource_buoy_block_lengths_csv is not None:
        command_resource.extend(
            ["--buoy-block-lengths-csv", _relative_to_repo(args.resource_buoy_block_lengths_csv, repo_root)]
        )
    if args.resource_buoy_block_length is not None:
        command_resource.extend(["--buoy-block-length", str(args.resource_buoy_block_length)])
    if args.resource_buoy_max_block_length is not None:
        command_resource.extend(["--buoy-max-block-length", str(args.resource_buoy_max_block_length)])
    if args.ann_paired_bootstrap_replicates is not None:
        command_resource.extend(["--ann-paired-bootstrap-replicates", str(args.ann_paired_bootstrap_replicates)])
    if args.ann_paired_bootstrap_confidence is not None:
        command_resource.extend(["--ann-paired-bootstrap-confidence", str(args.ann_paired_bootstrap_confidence)])
    if args.ann_paired_bootstrap_seed is not None:
        command_resource.extend(["--ann-paired-bootstrap-seed", str(args.ann_paired_bootstrap_seed)])
    if args.resource_overwrite:
        command_resource.append("--overwrite")

    _run_python_with_backend(
        command_resource,
        repo_root=repo_root,
        logger=logger,
        backend=args.python_backend,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hf_wind_resource")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compute = subparsers.add_parser(
        "compute_resource",
        help="Run the wind-resource pipeline (empirical metrics, power, bootstrap, geospatial exports).",
    )
    compute.add_argument("--dataset", type=Path, help="Direct path to the ANN GeoParquet asset.")
    compute.add_argument(
        "--stac-config",
        type=Path,
        default=Path("config/stac_catalogs.json"),
        help="Path to the STAC catalog index JSON (used when --dataset is omitted).",
    )
    compute.add_argument(
        "--stac-dataset",
        default="sar_range_final_pivots_joined",
        help="Dataset key inside the STAC catalog (ignored when --dataset is provided).",
    )
    compute.add_argument("--nodes", nargs="+", help="List of node identifiers to process.")
    compute.add_argument(
        "--nodes-from",
        type=Path,
        help="Path to a text file with one node identifier per line to process.",
    )
    compute.add_argument(
        "--keep-filtered-dataset",
        action="store_true",
        help="Do not delete the temporary Parquet dataset generated for --nodes filtering.",
    )
    compute.add_argument(
        "--engine",
        choices=("docker", "python"),
        default="docker",
        help="Execution engine for DuckDB-backed stages.",
    )
    compute.add_argument(
        "--python-backend",
        choices=("auto", "host", "docker"),
        default="auto",
        help="Interpreter used for Python helpers (auto tries host first, then docker fallback).",
    )
    compute.add_argument(
        "--image",
        default="duckdb/duckdb:latest",
        help="Docker image providing DuckDB when --engine=docker.",
    )
    compute.add_argument(
        "--stages",
        nargs="+",
        choices=[stage.name for stage in _COMPUTE_STAGE_REGISTRY.values()],
        help="Run only the specified stages (empirical, power, node_summary, bootstrap, geospatial, publish).",
    )
    compute.add_argument(
        "--skip-stages",
        nargs="+",
        choices=[stage.name for stage in _COMPUTE_STAGE_REGISTRY.values()],
        help="Skip specific stages from the default pipeline.",
    )
    compute.add_argument(
        "--overwrite",
        action="store_true",
        help="Propagate overwrite flags to stages that support them.",
    )
    compute.add_argument(
        "--max-nodes",
        type=int,
        help="Limit processing to the first N nodes (passed to power/bootstrap helpers).",
    )
    compute.add_argument("--power-curve-key", help="Override the power curve key for power estimation.")
    compute.add_argument(
        "--air-density",
        type=float,
        help="Override the air-density assumption for power estimation (kg/m^3).",
    )
    compute.add_argument(
        "--right-tail-surrogate",
        type=float,
        help="Surrogate speed assigned to right-censored mass during power estimation.",
    )
    compute.add_argument(
        "--bootstrap-replicas",
        type=int,
        default=500,
        help="Number of bootstrap replicas for uncertainty estimation.",
    )
    compute.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap intervals.",
    )
    compute.add_argument(
        "--bootstrap-seed",
        type=int,
        help="Random seed forwarded to the bootstrap helper.",
    )
    compute.add_argument(
        "--bootstrap-workers",
        type=int,
        help="Number of worker processes for bootstrap computation.",
    )
    compute.add_argument(
        "--bootstrap-resume",
        action="store_true",
        help="Resume bootstrap runs when partial results exist.",
    )
    compute.add_argument(
        "--bootstrap-disable-rmse",
        action="store_true",
        help="Disable RMSE perturbations during bootstrap.",
    )
    compute.add_argument(
        "--geospatial-config",
        type=Path,
        help="Alternate configuration JSON for geospatial products.",
    )
    compute.add_argument(
        "--geospatial-max-roses",
        type=int,
        help="Cap the number of wind roses generated by the geospatial helper.",
    )
    compute.add_argument(
        "--geospatial-overwrite",
        action="store_true",
        help="Force the geospatial script to overwrite existing artefacts.",
    )
    compute.add_argument(
        "--disable-buoy-outputs",
        action="store_true",
        help="Skip buoy-derived visualisations when running geospatial products.",
    )
    compute.add_argument(
        "--disable-buoy-rose",
        action="store_true",
        help="Forward --disable-buoy-rose to scripts/generate_geospatial_products.py explicitly.",
    )
    compute.add_argument("--buoy-dataset", type=Path, help="Override the buoy dataset path for geospatial outputs.")
    compute.add_argument("--buoy-node-id", help="Node identifier that represents the buoy in the ANN dataset.")
    compute.add_argument(
        "--publish-version-tag",
        help="Version tag recorded when publishing the STAC collection.",
    )
    compute.add_argument(
        "--publish-overwrite",
        action="store_true",
        help="Allow replacing an existing published collection when running the publish stage.",
    )
    compute.add_argument("--verbose", action="store_true", help="Increase logging verbosity.")

    compute.set_defaults(func=_handle_compute_resource)

    uncensored = subparsers.add_parser(
        "compute_uncensored_resource",
        help="Estimate uncensored (buoy-like) wind-resource metrics with a user-provided RMSE.",
    )
    uncensored.add_argument(
        "--config",
        type=Path,
        default=Path("config/uncensored_resource.json"),
        help="JSON configuration file for uncensored resource estimation.",
    )
    uncensored.add_argument("--dataset", type=Path, help="Direct path to the uncensored GeoParquet dataset.")
    uncensored.add_argument(
        "--stac-config",
        type=Path,
        default=Path("config/stac_catalogs.json"),
        help="Path to the STAC catalog index JSON (used when --dataset is omitted).",
    )
    uncensored.add_argument(
        "--stac-dataset",
        help="Dataset key inside the STAC catalog (ignored when --dataset is provided).",
    )
    uncensored.add_argument("--asset-key", help="Asset key inside the STAC item (default: data).")
    uncensored.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory for resource outputs (overrides the configuration).",
    )
    uncensored.add_argument(
        "--rmse",
        type=float,
        help="Override the RMSE value (m/s) declared in the configuration.",
    )
    uncensored.add_argument(
        "--rmse-source",
        help="Optional source string recorded next to the RMSE value.",
    )
    uncensored.add_argument(
        "--speed-column",
        help="Wind-speed column name when different from the configuration.",
    )
    uncensored.add_argument(
        "--node-column",
        help="Node identifier column name when different from the configuration.",
    )
    uncensored.add_argument(
        "--timestamp-column",
        help="Timestamp column name when different from the configuration.",
    )
    uncensored.add_argument(
        "--python-backend",
        choices=["auto", "host", "docker"],
        default="auto",
        help="Python execution backend (host by default with Docker fallback).",
    )
    uncensored.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    uncensored.set_defaults(func=_handle_uncensored_resource)

    validate = subparsers.add_parser(
        "validate_buoy",
        help="Prepare the buoy validation dataset and generate comparison diagnostics.",
    )
    validate.add_argument(
        "--buoy-dataset",
        type=Path,
        default=Path("use_case/catalogs/pde_vilano_buoy/assets/Vilano.parquet"),
        help="Path to the buoy GeoParquet asset.",
    )
    validate.add_argument(
        "--ann-dataset",
        type=Path,
        help="Direct path to the ANN GeoParquet snapshot (defaults to STAC resolution).",
    )
    validate.add_argument(
        "--ann-kind",
        choices=("ann", "uncensored"),
        default="ann",
        help=(
            "Interpretation of --ann-dataset: 'ann' for standard inference outputs, "
            "'uncensored' for generic wind series without pred_* columns (e.g., interpolated models)."
        ),
    )
    validate.add_argument(
        "--stac-config",
        type=Path,
        default=Path("config/stac_catalogs.json"),
        help="STAC catalog index used when --ann-dataset is omitted.",
    )
    validate.add_argument(
        "--stac-dataset",
        default="sar_range_final_pivots_joined",
        help="Dataset key inside the STAC index.",
    )
    validate.add_argument(
        "--python-backend",
        choices=("auto", "host", "docker"),
        default="auto",
        help="Interpreter used for Python helpers (auto tries host first, then docker fallback).",
    )
    validate.add_argument(
        "--node-id",
        default="Vilano_buoy",
        help="Node identifier to extract from the ANN dataset.",
    )
    validate.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("artifacts/processed/vilano_buoy_synced.parquet"),
        help="Where the matched ANN/buoy series will be written.",
    )
    validate.add_argument(
        "--output-summary",
        type=Path,
        default=Path("artifacts/processed/vilano_buoy_summary.json"),
        help="Where the ingestion summary JSON will be stored.",
    )
    validate.add_argument(
        "--direction-output",
        type=Path,
        default=Path("artifacts/direction_comparison"),
        help="Directory for the direction comparison artefacts.",
    )
    validate.add_argument(
        "--tolerance-minutes",
        type=float,
        default=30.0,
        help="Maximum timestamp tolerance (minutes) when matching ANN and buoy records.",
    )
    validate.add_argument(
        "--nearest-matching",
        action="store_true",
        help="Enable nearest-neighbour matching when timestamps do not align exactly.",
    )
    validate.add_argument(
        "--disable-height-correction",
        action="store_true",
        help="Skip the vertical wind-speed correction for the buoy series.",
    )
    validate.add_argument(
        "--buoy-height-method",
        choices=("power_law", "log_profile"),
        help="Override the height-correction profile applied to the buoy (defaults to config/buoy_height.json).",
    )
    validate.add_argument(
        "--buoy-measurement-height",
        type=float,
        help="Override the buoy anemometer height in metres.",
    )
    validate.add_argument(
        "--buoy-target-height",
        type=float,
        help="Override the target comparison height in metres.",
    )
    validate.add_argument(
        "--buoy-power-law-alpha",
        type=float,
        help="Override the neutral power-law exponent when --buoy-height-method=power_law.",
    )
    validate.add_argument(
        "--buoy-roughness-length",
        type=float,
        help="Override the logarithmic roughness length (metres) when --buoy-height-method=log_profile.",
    )
    validate.add_argument(
        "--scatter-sample-limit",
        type=int,
        help="Limit the number of scatter-plot points in direction diagnostics.",
    )
    validate.add_argument(
        "--resource-output-dir",
        type=Path,
        default=Path("artifacts/buoy_validation"),
        help="Directory that will receive the ANN vs. buoy resource comparison artefacts.",
    )
    validate.add_argument(
        "--resource-power-curve-config",
        type=Path,
        default=Path("config/power_curves.json"),
        help="JSON file describing power curves for the resource comparison.",
    )
    validate.add_argument(
        "--resource-power-curve-key",
        default="reference_offshore_6mw",
        help="Key selecting the power curve used in the resource comparison.",
    )
    validate.add_argument(
        "--resource-height-config",
        type=Path,
        default=Path("config/power_height.json"),
        help="Height-correction configuration applied to ANN winds during the comparison.",
    )
    validate.add_argument(
        "--resource-range-thresholds",
        type=Path,
        default=Path("config/range_thresholds.json"),
        help="Range-threshold configuration forwarded to the resource comparison script.",
    )
    validate.add_argument(
        "--resource-bootstrap-summary",
        type=Path,
        default=Path("artifacts/bootstrap_velocity_block/bootstrap_summary.csv"),
        help="Existing bootstrap summary CSV used to retrieve confidence intervals for the ANN node.",
    )
    validate.add_argument(
        "--resource-bootstrap-metadata",
        type=Path,
        default=Path("artifacts/bootstrap_velocity_block/bootstrap_metadata.json"),
        help="Metadata JSON describing the ANN bootstrap run used for confidence intervals.",
    )
    validate.add_argument(
        "--resource-ann-label",
        default="ANN",
        help="Dataset label used for ANN rows in the resource comparison outputs.",
    )
    validate.add_argument(
        "--resource-buoy-label",
        default="Buoy",
        help="Dataset label used for buoy rows in the resource comparison outputs.",
    )
    validate.add_argument(
        "--resource-right-tail-surrogate",
        type=float,
        help="Override the surrogate speed assigned to right-censored probability mass.",
    )
    validate.add_argument(
        "--resource-air-density",
        type=float,
        default=1.225,
        help="Air-density assumption propagated to the resource comparison script.",
    )
    validate.add_argument(
        "--resource-min-confidence",
        type=float,
        default=0.5,
        help="Minimum ANN range-flag confidence treated as deterministic when computing resource metrics.",
    )
    validate.add_argument(
        "--resource-min-in-range",
        type=float,
        default=500.0,
        help="Minimum in-range weight required to accept Weibull fits in the resource comparison.",
    )
    validate.add_argument(
        "--resource-km-criteria-config",
        type=Path,
        default=None,
        help="Optional Kaplan–Meier criteria override for the resource comparison.",
    )
    validate.add_argument(
        "--resource-overwrite",
        action="store_true",
        help="Allow the resource comparison stage to overwrite existing artefacts.",
    )
    validate.add_argument(
        "--resource-buoy-bootstrap-replicates",
        type=int,
        default=50,
        help="Number of bootstrap replicates for buoy statistics in the resource comparison (default: 50).",
    )
    validate.add_argument(
        "--resource-buoy-bootstrap-confidence",
        type=float,
        default=0.95,
        help="Confidence level for buoy bootstrap intervals (default: 0.95).",
    )
    validate.add_argument(
        "--resource-buoy-bootstrap-seed",
        type=int,
        help="Random seed used when bootstrapping buoy statistics.",
    )
    validate.add_argument(
        "--resource-buoy-resampling-mode",
        choices=("iid", "moving_block", "stationary"),
        help="Resampling mode for buoy bootstrap (defaults to ANN mode when omitted).",
    )
    validate.add_argument(
        "--resource-buoy-block-lengths-csv",
        type=Path,
        default=Path("artifacts/buoy_block_diagnostics/block_bootstrap_diagnostics.csv"),
        help="CSV with per-node block-length recommendations for the buoy (optional).",
    )
    validate.add_argument(
        "--resource-buoy-block-length",
        type=int,
        help="Explicit block length for buoy bootstrap (overrides CSV/config when set).",
    )
    validate.add_argument(
        "--resource-buoy-max-block-length",
        type=int,
        help="Cap the buoy block length when using recommendations (optional).",
    )
    validate.add_argument(
        "--ann-paired-bootstrap-replicates",
        type=int,
        help="Override the number of bootstrap replicates for the ANN paired subset (defaults to metadata).",
    )
    validate.add_argument(
        "--ann-paired-bootstrap-confidence",
        type=float,
        help="Override the confidence level for the ANN paired bootstrap (defaults to metadata).",
    )
    validate.add_argument(
        "--ann-paired-bootstrap-seed",
        type=int,
        help="Random seed used when bootstrapping the ANN paired subset.",
    )
    validate.add_argument("--verbose", action="store_true", help="Increase logging verbosity.")

    validate.set_defaults(func=_handle_validate_buoy)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


_COMPUTE_STAGE_REGISTRY_LIST = _build_compute_stages()
_COMPUTE_STAGE_REGISTRY: Mapping[str, PipelineStage] = {stage.name: stage for stage in _COMPUTE_STAGE_REGISTRY_LIST}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
