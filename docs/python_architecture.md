# Python Package Architecture

This document outlines the Python package structure that will power the marine wind-resource workflows driven by the SAR range-aware ANN inference snapshot described in `docs/sar_range_final_schema.md`. The architecture keeps a strict separation between I/O, preprocessing, statistical modelling, reporting outputs, and command-line entry points so that each layer can evolve independently while remaining testable offline.

## Repository layout

All Python source code will live under a `src/` layout to keep the namespace clean and to support editable installs:

```text
src/
  hf_wind_resource/
    __init__.py
    io/
      __init__.py
      stac.py
      geoparquet.py
      duckdb_client.py
      schema_registry.py
    preprocessing/
      __init__.py
      filters.py
      censoring.py
      node_alignment.py
      validation.py
    qa/
      __init__.py
      range_quality.py
    stats/
      __init__.py
      empirical.py
      weibull.py
      rmse.py
      bootstrapping.py
      uncertainty.py
    outputs/
      __init__.py
      tables.py
      charts.py
      exporters.py
      catalog_writer.py
    cli/
      __init__.py
      main.py
      commands/
        __init__.py
        ingest.py
        compute_resource.py
        report.py
```

Key principles:
- `hf_wind_resource.io` centralises data access. `stac.py` discovers catalog items using the dataset index declared in `config/stac_catalogs.json`, `geoparquet.py` loads the HF-radar inference tables, `duckdb_client.py` wraps containerised DuckDB usage, and `schema_registry.py` mirrors the column definitions from `docs/sar_range_final_schema.md` to guard against drift.
- `hf_wind_resource.preprocessing` handles dataset preparation. `filters.py` applies range-label filters based on `prob_range_*` thresholds, `censoring.py` converts `pred_range_label` into left/right-censored observations, `node_alignment.py` aligns multiple sources (e.g. buoy references) on timestamps, and `validation.py` runs QA checks such as ensuring `(timestamp, node_id)` uniqueness.
- `hf_wind_resource.qa` consolidates range-focused quality assurance. `range_quality.py` consumes per-node censoring summaries (from `preprocessing.censoring`) and temporal coverage metrics to apply configurable thresholds, flag nodes with excessive censoring, low density, or large gaps, and surface a `parametric_reliable` indicator before triggering censored Weibull fitting. The module reads project defaults from `config/range_quality_thresholds.json` and exposes helpers used by reporting scripts.

The first concrete preprocessing feature lives in `preprocessing/node_alignment.py`. It exposes dataclasses such as `TemporalNormalizationConfig`, `TemporalNormalizationResult`, `CadenceStats`, and `TemporalQualityFlags` plus the helper `normalise_temporal_records`. The routine rounds timestamps to the nominal 30 minute cadence, removes `(node_id, timestamp)` duplicates, quantifies missing windows, and records coverage/gap diagnostics through `GapReport`. Downstream statistics modules can therefore consume a cleaned dataframe together with per-node temporal summaries and structured anomaly notes.
- `hf_wind_resource.stats` concentrates the statistical core. `empirical.py` computes histograms, empirical CDFs, and summary metrics on in-range samples; `weibull.py` implements the censored maximum-likelihood fit described in `docs/weibull_censoring_methodology.md`; `kaplan_meier.py` provides a weighted Kaplan–Meier fallback that activates on heavy censoring, exposing selection diagnostics plus left/right tail accounting for reporting layers; `parametric.py` compares Weibull, log-normal, and gamma candidates (censored log-likelihoods, AIC/BIC, weighted KS) and records the preferred model per node; `rmse.py` publishes global and per-node RMSE diagnostics; `bootstrapping.py` implements the stratified bootstrap with RMSE-informed noise, label-resampling, and jackknife-assisted BCa/percentile-t intervals documented in `docs/bootstrap_uncertainty_methodology.md`, yielding confidence intervals for mean speed, percentiles, and power density that comply with the findings in `audits/bootstrap-methodology.md`; `seasonal.py` quantifies seasonal and interannual variability (amplitude, seasonal coverage, trend diagnostics) for reuse in reporting layers. Configuration-driven metadata (e.g. RMSE catalogues, per-node taxonomy, censoring thresholds, taxonomy band thresholds, classifier confidence cut-offs, height-extrapolation defaults, non-parametric triggers) lives in `config/global_rmse.json`, `config/node_taxonomy.json`, `config/range_thresholds.json`, `config/taxonomy_bands.json`, `config/power_height.json`, and `config/kaplan_meier_thresholds.json`, which the module family loads to keep runtime logic decoupled from documentation formats.
- `hf_wind_resource.outputs` formats deliverables. `tables.py` synthesises per-node summaries (mean speed, P50/P90/P99, power density), `charts.py` generates diagnostic plots, `exporters.py` writes GeoParquet/CSV/JSON artefacts, and `catalog_writer.py` updates STAC metadata when publishing new versions.
- Bootstrap artefacts are produced via `scripts/generate_bootstrap_uncertainty.py`, which queries the ANN GeoParquet snapshots with containerised DuckDB, instantiates `StratifiedBootstrapConfig`, and emits CSV/JSON reports under `artifacts/bootstrap_uncertainty/` documenting the confidence intervals and RMSE provenance.
- `hf_wind_resource.stats.power` exposes reusable helpers to derive mean power density and expected turbine output from either censored Weibull parameters or the Kaplan–Meier estimator, accounting for air-density assumptions and reference power-curve metadata.
- `scripts/generate_power_estimates.py` orchestrates the power derivation workflow end-to-end: it pulls per-node samples via Dockerised DuckDB, aplica la selección Weibull/Kaplan–Meier e integra la curva de potencia configurada. Además de los resúmenes por nodo (`power_estimates_summary.csv` + JSON/diagnósticos), emite agregados estacionales (`seasonal_power_summary.csv`) y series mensuales (`monthly_power_timeseries.csv`) en `artifacts/power_estimates/`, listos para análisis ARIMA. La metodología detallada (densidad `½ρ·E[v³]`, curvas por altura, criterios de censura) está recogida en `docs/seasonal_analysis_methodology.md`.
- `scripts/generate_seasonal_variations.py` extracts `timestamp`, `node_id`, `pred_wind_speed`, and `pred_range_label` via containerised DuckDB (or the embedded DuckDB engine for tests), feeds them into `hf_wind_resource.stats.seasonal`, and emits CSV/JSON artefacts under `artifacts/seasonal_analysis/`, including the Markdown digest `artifacts/seasonal_analysis/seasonal_variation_summary.md` and the overview JSON with height-correction context. The derivation is documented in `docs/seasonal_analysis_methodology.md`.
- `hf_wind_resource.cli` exposes Click-based commands. `commands/ingest.py` stages new catalog snapshots, `commands/compute_resource.py` orchestrates preprocessing + statistics for selected nodes, and `commands/report.py` materialises documentation-ready outputs (PDF/Markdown/HTML) using results from `outputs`.

Each module will communicate through typed dataclasses defined in `hf_wind_resource.io.schema_registry` to embed field names, units, and censoring semantics lifted directly from the ANN inference schema.

## Offline-first dependency set

The codebase must operate without public internet access. The minimal dependency footprint is:

- Core scientific stack: `numpy`, `pandas`, `pyarrow`, `duckdb`, `scipy`, `statsmodels`, `matplotlib` (for charts), `seaborn` (optional styling).
- Geospatial helpers: `shapely`, `pyproj` (only when spatial reprojection is required; keep optional).
- CLI tooling: `click`, `rich` (for coloured terminal progress when available).
- Packaging/testing: `setuptools`, `wheel`, `pytest` (tests run inside Docker).

Guidelines:
- Vendor wheels in `vendor/wheels/` and install with `pip install --no-index --find-links vendor/wheels -r requirements.txt`.
- Maintain a lock file (`requirements.lock`) generated in an environment with the same Python minor version used in production (target `python>=3.11,<3.12`).
- Keep optional extras for heavyweight features (e.g. plotting) under `[project.optional-dependencies]` in `pyproject.toml` so core workflows can run with a strictly minimal stack.

## Virtual environments and Docker

Two execution environments will be supported:

1. **Local virtual environment**
   - Create with `python3 -m venv .venv`.
   - Activate (`source .venv/bin/activate`) and install dependencies offline using the vendor wheels.
   - Store reproducible commands in `scripts/bootstrap_venv.sh`, which verifies Python version, installs wheels, and runs smoke tests (`python -m hf_wind_resource.cli --help`).

2. **Docker image**
   - Base image: `python:3.11-slim`.
   - Copy `vendor/wheels/` into the image and install using `pip install --no-index`.
   - Include DuckDB CLI via `apt-get install duckdb-cli` or by vendoring the static binary; never rely on network package mirrors at build time.
   - Expose entrypoint `hf-wind-resource` (wrapper around `python -m hf_wind_resource.cli.main`).
   - Document build/run commands in `docker/README.md`, e.g.:
     ```bash
     docker build -f docker/Dockerfile -t wind-resource:latest .
     docker run --rm -v "$PWD/catalogs":/workspace/catalogs wind-resource compute-resource --nodes all
     ```
   - After running containerised tests, delete images/containers you created to keep the sandbox clean (see project-wide instructions).

## Data versioning policies

- Treat `use_case/catalogs/` as the *raw* data namespace for the HF-EOLUS case. Each catalogue snapshot resides in a subfolder named `sar_range_final_pivots_joined/<version-tag>/` when refreshed. The current export remains under `.../assets/data.parquet`.
- Capture every refresh in `use_case/catalogs/CHANGELOG.md`, recording the STAC collection `id`, original S3 prefix, snapshot timestamp, and checksum of the GeoParquet asset.
- Downstream derived artefacts (node summaries, statistical tables, plots) live in `outputs/<dataset_version>/`. Use semantic versioning aligned with the ANN inference version (`sar-range-final` → `sar-range-final-<YYYYMMDD>`).
- When recomputing metrics, include the inference RMSE metadata (see `docs/hf_dev_plan.md`) in the output metadata so uncertainty remains traceable.
- Adopt commit-level tagging for reproducibility: each run should emit a `manifest.json` capturing code commit SHA, dataset version, and parameter choices (e.g. censoring thresholds, bootstrap iterations).

## Packaging guidelines

- Use a single `pyproject.toml` with `setuptools` backend and a `src` layout to avoid namespace collisions.
- Expose console scripts under `[project.scripts]` (e.g. `hf-wind-resource = "hf_wind_resource.cli.main:app"`).
- Keep runtime configuration lightweight: prefer `yaml` configs stored in `configs/` and validated through `pydantic` models located in `hf_wind_resource.io.schema_registry`.
- Enforce consistent linting/formatting with `ruff` and `black` (vendor wheels as needed); document their usage in `scripts/lint.sh`.
- Supply type hints throughout; integrate `mypy` in CI/docker test recipes once the stub infrastructure is ready.
- Provide module-level docstrings referencing the relevant sections in `docs/sar_range_final_schema.md`, especially when functions assume specific column names or censoring semantics.

By adhering to this architecture, the toolkit remains maintainable, reproducible, and aligned with the SAR range-aware data characteristics while respecting the offline constraints of the HF-EOLUS project.

## Testing Strategy

### Objectives and Coverage Targets
- Guarantee deterministic, offline-friendly validation of the IO layer before higher-level modules depend on it.
- Provide fast unit coverage (target ≥90% for `hf_wind_resource.io`, ≥80% package-wide once modules exist) and representative integration scenarios using the synthetic fixtures.
- Surface regressions in range-awareness (classifier labels vs. regression bounds) and temporal continuity before statistical routines execute.
- Keep all test orchestration reproducible inside Docker so no external services or package mirrors are required.

### Unit Test Matrix
| Module / Area | Key behaviours to verify | Data dependencies | Tooling notes |
| --- | --- | --- | --- |
| `io.TimeWindow`, `NodeFilter`, `IOFilters` | Boundary handling (inclusive windows, include/exclude precedence, `require_in_range` + `min_confidence` validation) | In-memory synthetic rows constructed in test | Pure `pytest`, no fixtures required |
| `io.PerformanceBudget`, `CachePolicy`, `ChunkMetadata` | Serialization/validation helpers, default fallbacks, estimation heuristics | None | Use parametrised tests to cover edge values |
| `io.GapReport` / `describe_gap_expectations` | Gap registration, merge semantics, cadence initialisation | `tests/fixtures/sar_range_final_synthetic.parquet` (30/60 min cadence) | Validate aggregated descriptors and severity tagging |
| `io.AnomalySink` protocols | Contract enforcement through lightweight stubs (e.g. raising if missing `emit`) | Synthetic stub classes | `typing.Protocol` compliance checked via `mypy --strict` once configured |
| `io.iter_in_memory` (once implemented) | Chunk slicing fidelity, filter pushdown, chunk size guarantees, respect for `PerformanceBudget` | `sar_range_final_synthetic.parquet` filtered by node/time and label combinations | Add property-based tests (`hypothesis`) to stress chunk sizing with random budgets |
| Future `io.schema_registry` | Column/type alignment with `docs/sar_range_final_schema.md`, EWMA of schema versions | Snapshot of schema JSON exported from documentation scripts | Ensure changes fail tests before docs drift |
| `preprocessing.node_alignment` | Temporal normalisation, duplicate removal, gap logging, coverage flags | `tests/fixtures/sar_range_final_synthetic.parquet` + inline frames with synthetic gaps | `pytest` cases assert deduplication, gap detection, rounding strategies, and insufficient coverage notes |
| `stats` (future) | Empirical/parametric metrics, bootstrap reproducibility with fixed seeds | Fixture-driven Series/DataFrames with known statistics | Validate against analytical expectations |
| `outputs`/`cli` (future) | Formatting, STAC manifest generation, CLI argument wiring | Golden files (JSON/CSV) committed under `tests/fixtures/expected/` | CLI invoked via `pytest` `CliRunner` once `click` is wired |

### Integration Test Layers
- **DuckDB-backed IO reader**: spin up `duckdb/duckdb:latest` via `docker run --rm -v "$PWD":/workspace duckdb/duckdb:latest` in the test harness to execute the planned `ChunkedDatasetReader` implementation against `tests/fixtures/sar_range_final_synthetic.parquet`. Validate chunk planning, filter pushdown, anomaly emission when `prob_range_*` sums deviate, and gap detection for injected >90 minute gaps.
- **IO → preprocessing bridge**: with the temporal normalisation helpers available, the next step is to feed streamed chunks into the preprocessing layer to assert that censored samples align with the IO layer's `require_in_range` semantics and that gap logging remains stable. These tests should use the same synthetic Parquet plus targeted edits introducing classifier/regression mismatches.
- **End-to-end smoke**: orchestrate the CLI (`compute-resource`) inside Docker wiring visible fixtures, ensuring manifests and outputs land under a temporary directory and respect offline dependencies. Target runtime <5 minutes and reuse the test Dockerfile to keep environment parity with CI.

### Data and Fixture Management
Comprehensive download and regeneration steps live in [`docs/data_access.md`](docs/data_access.md); the notes below focus on how the architecture consumes those datasets.

- Keep `tests/fixtures/sar_range_final_synthetic.*` as the canonical dataset for unit/integration tests. Extend it via the existing generator when new edge cases appear (e.g. additional nodes, extreme RMSE scenarios). Document every change in `tests/fixtures/README.md`.
- Reference the production GeoParquet (`use_case/catalogs/sar_range_final_pivots_joined/assets/data.parquet`) only in optional regression tests guarded by `pytest` markers (e.g. `@pytest.mark.requires_dataset`) so CI can skip them when the file is unavailable.
- Store golden outputs for CLI/statistical tests under `tests/fixtures/expected/` and validate them with strict byte comparisons to detect accidental formatting drift.
- Keep `tests/test_fixture_smoke.py` as a gating smoke suite that loads the synthetic Parquet via DuckDB and validates range-probability invariants so the Docker image fails fast if fixtures or dependencies go missing.
- Configuration artefacts under `config/` are regenerated via scripts rather than manual edits. `config/stac_catalogs.json` lists the STAC collections and default item/asset pairs that the IO layer resolves to discover datasets; each entry now preserves the DOI of the sample asset so the toolkit can cite the ANN release (`sar_range_final_pivots_joined`, DOI 10.5281/zenodo.17131227) and the optional buoy dataset (`pde_vilano_buoy`, DOI 10.5281/zenodo.17098037). `config/global_rmse.json` tracks the time-versioned RMSE catalogue derived from buoy comparisons—the bundled record is explicitly marked as the HF-EOLUS Vilano template so users know they must regenerate it for other references. `config/node_taxonomy.json` records observation counts, cadence-gap statistics, and the derived `low_coverage` flag (driven by the project defaults stored in `config/low_coverage_rules.json`) and is likewise labelled as the HF-EOLUS template built from `use_case/catalogs/sar_range_final_pivots_joined`. `config/taxonomy_bands.json` stores the coverage and continuity band thresholds (defaults tuned to the HF-EOLUS SAR range final archive), `config/range_thresholds.json` defines both the ANN’s valid wind-speed range and the classifier confidence cut-off (`range_flag_threshold = 0.5` per `hf-wind-inversion/hf_eolus/sar/fine_tuning_l2sp/script_args.json`) along with notes reminding maintainers to adjust them when the ANN changes, `config/range_quality_thresholds.json` holds the QA guard-rails applied before parametric fitting (censoring ratios, minimum in-range support, temporal density), and `config/time_series.json` keeps the continuity heuristics used by the seasonal/ARIMA helpers (maximum gap of six months, minimum segments of 36 months by default). Use `python3 scripts/update_node_taxonomy.py --output config/node_taxonomy.json` (Docker required) to regenerate the taxonomy; pass `--dataset` if a specific GeoParquet path should override the STAC-resolved asset. Update `config/global_rmse.json` by appending the latest metrics once a new validation run is confirmed, adjust `config/low_coverage_rules.json` (or the CLI flags) to change the baseline coverage thresholds, and revise the range/continuity JSON whenever the ANN or observation cadence is recalibrated.

### Tooling and Execution
- Primary runner: `scripts/run_tests.sh` builds `docker/tests/Dockerfile` (based on `python:3.11-slim`) and executes `pytest` inside the resulting container. By default the build installs dependencies from PyPI (internet required) and falls back to an offline cache when wheel files exist under `vendor/wheels/`. The script forwards arbitrary pytest arguments, offers `--build-only`, and provides two toggles: set `ALLOW_EMPTY_WHEELS=1` to skip dependency installation entirely (smoke builds only) or `--no-online-install`/`ALLOW_ONLINE_INSTALL=0` to enforce offline mode using the cached wheels.
- Enforce static analysis (`ruff`, `mypy`) as separate stages within the same container to catch protocol drift early; treat them as part of the test matrix once configuration files land in the repo.
- Cache DuckDB binaries inside the Docker image or mount the official container as a service container in CI. For local development, prefer `docker run --rm -v` invocations to comply with the project instruction of using the DockerHub image rather than system binaries.

### Reporting and Future Enhancements
- Track coverage with `pytest-cov`, exporting `coverage.xml` for future CI integration. Highlight module-level coverage in `docs/python_architecture.md` as features ship.
- Add `pytest` markers (`io`, `preprocessing`, `integration`, `slow`) to enable selective execution (`pytest -m io` for fast checks before commits).
- Document any newly required fixtures or Docker services directly in this section to keep architectural and testing guidance co-located.
