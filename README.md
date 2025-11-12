# HF EOLUS Wind Resource Toolkit

Toolkit for estimating offshore wind-resource metrics from HF-radar ANN inference and optional in-situ buoy references. The repository packages the Python modules, Dockerized helpers, and STAC-aware configuration needed to run the workflow end-to-end against any catalogue shaped like the HF-EOLUS inputs.

## Quick Start
1. **Prepare the environment**. Install Python 3.11+, Docker, and the `wind-resource-tests` image (see `docker/README.md`). The CLI expects to run inside the repo root with `PYTHONPATH=src`.
2. **Download the input catalogues**. Place the ANN STAC snapshot under `use_case/catalogs/sar_range_final_pivots_joined/` and, if you want to run the buoy validation example, place the buoy STAC snapshot under `use_case/catalogs/pde_vilano_buoy/`. Zenodo DOIs for the public demo datasets: ANN [10.5281/zenodo.17131227](https://doi.org/10.5281/zenodo.17131227) and buoy [10.5281/zenodo.17098037](https://doi.org/10.5281/zenodo.17098037).
3. **Run the pipeline**. Launch:
   ```bash
   PYTHONPATH=src python3 -m hf_wind_resource.cli.main compute_resource \
     --engine docker \
     --image duckdb/duckdb:latest \
     --overwrite
   ```
   This executes empirical metrics, parametric power estimation, bootstrap uncertainty, geospatial exports, and STAC publication using the defaults in `config/`.
4. **Inspect artefacts**. Review `artifacts/power_estimates/` (per-node CSV/JSON summaries), `artifacts/bootstrap_uncertainty/` (intervals), and `artifacts/power_estimates/geospatial/` (maps + GeoParquet). Use `scripts/publish_power_estimates_catalog.py` to refresh the STAC collection once outputs are verified.

> **Publishing note:** when mirroring this repository into a public toolkit, omit any internal planning or audit folders that are present only for coordination purposes.

## Documentation map
- [`docs/sar_range_final_schema.md`](docs/sar_range_final_schema.md): Schema handbook for the ANN GeoParquet (column definitions, QA semantics, DuckDB recipes, and adaptation notes for new catalogues).
- [`docs/python_architecture.md`](docs/python_architecture.md): Repository layout, regeneration workflow for configuration artefacts, testing approach, and offline/Docker execution guidelines.
- [`docs/data_access.md`](docs/data_access.md): Acquisition and reproducibility guide covering Zenodo downloads, local snapshots, subsetting, and fixture management.
- [`docs/empirical_metrics_methodology.md`](docs/empirical_metrics_methodology.md), [`docs/nonparametric_distribution_methodology.md`](docs/nonparametric_distribution_methodology.md), [`docs/weibull_censoring_methodology.md`](docs/weibull_censoring_methodology.md), [`docs/power_estimation_methodology.md`](docs/power_estimation_methodology.md), [`docs/bootstrap_uncertainty_methodology.md`](docs/bootstrap_uncertainty_methodology.md), and [`docs/block_bootstrap_assessment.md`](docs/block_bootstrap_assessment.md): Methodology stack for empirical statistics, censored parametric fits, Kaplan–Meier fallbacks, bootstrap uncertainty, and their validation.
- [`docs/resource_pipeline_guide.md`](docs/resource_pipeline_guide.md): Operational runbook for the CLI, listing inputs, configuration surfaces, and expected outputs per stage.
- [`docs/geospatial_products.md`](docs/geospatial_products.md) & [`docs/node_summary_table.md`](docs/node_summary_table.md): Instructions for building the publishable node summary table and the GeoParquet/figure bundle consumed by dashboards.
- [`docs/catalog_publication.md`](docs/catalog_publication.md): Publishing checklist for STAC-ready catalogues and manifests.
- [`docs/buoy_height_configuration.md`](docs/buoy_height_configuration.md) & [`docs/direction_comparison.md`](docs/direction_comparison.md): Configuration and diagnostics for buoy-based validation, including vertical corrections and angular statistics.
- [`docs/temporal_normalisation_summary.md`](docs/temporal_normalisation_summary.md) & [`docs/seasonal_analysis_methodology.md`](docs/seasonal_analysis_methodology.md): Coverage and seasonal-variability references.
- [`use_case/docs/vilano_comparison.md`](use_case/docs/vilano_comparison.md) and [`docs/sar_inference_on_vilano_10m_all.md`](docs/sar_inference_on_vilano_10m_all.md): Case-study artefacts retained only as examples; consult them when reproducing the Vilano dataset, not when documenting the generic toolkit.

## Known Limitations
- Coverage quality is entirely driven by the taxonomy flags derived from `config/node_taxonomy.json`; treat nodes labelled `low_coverage` or `any_bias = true` as advisory filters before using the outputs in site-selection work.
- Resource metrics currently assume neutral log-profile extrapolation from 10 m to the configured hub height, fixed air density (1.225 kg/m^3 by default), and no atmospheric-stability corrections.
- Power-density and turbine results are theoretical (no wakes, availability, or turbine-specific controls). Bootstrap intervals rely on the RMSE-aware settings from `docs/bootstrap_uncertainty_methodology.md`; adjust `--bootstrap-*` knobs when propagating different uncertainty sources.
## Published Outputs
- `use_case/catalogs/sar_range_final_power_estimates/` packages the public wind-resource deliverable derived from the SAR range-aware ANN snapshot. It contains a GeoParquet table with per-node power density, turbine metrics, and QA flags, a STAC collection/item mirroring the input catalogue structure, and a `manifest.json` capturing the code commit, input hashes, and processing parameters. Consumers can resolve the dataset through the STAC item `items/power_estimates_nodes.json` and retrieve asset checksums from the manifest for reproducibility.
- Run `python3 scripts/publish_power_estimates_catalog.py` (optionally with `--overwrite`) to regenerate the GeoParquet, manifest, and STAC catalog automatically once `scripts/generate_power_estimates.py` and the empirical metrics pipeline have produced their artefacts. The helper resolves node geometries from the STAC input, so no manual DuckDB commands are required.
- `artifacts/power_estimates/node_summary/` hosts the per-node diagnostic table (`node_summary.parquet` + `node_summary.csv`) and the accompanying `node_summary_metadata.json` that links every column back to the definitions in `docs/sar_range_final_schema.md`. This dataset merges empirical label ratios, Weibull diagnostics, Kaplan-Meier percentiles, bias flags, and height-extrapolation parameters for each mesh node.
- `artifacts/power_estimates/geospatial/` aggregates reporting-ready artefacts: `node_resource_map.parquet` (GeoParquet), `node_resource_map.geojson`, individual map panels (`power_density_map.svg` and `uncertainty_map.svg`, the latter showing the bootstrap interval width for power density), and `wind_rose_panels.svg` with representative ANN roses. When a buoy dataset is configured (e.g., the Vilano example release), the helper also emits `buoy_wind_rose_panels.svg` with the observed-versus-ANN comparison. Each run records provenance in `metadata.json` so the visual outputs stay traceable to the ANN snapshot and node summary inputs.
- Run `python3 scripts/generate_node_summary_table.py` (use `--overwrite` for refreshes) after `generate_empirical_metrics.py`, `generate_power_estimates.py`, and `generate_nonparametric_distributions.py` have emitted their artefacts. The script executes DuckDB inside `duckdb/duckdb:latest`, materialises both CSV and Parquet outputs, and writes the metadata mapping automatically.

## Configuration
- `config/global_rmse.json` stores the catalogue of RMSE records ingested by `hf_wind_resource.stats.GlobalRmseProvider`. Append a new entry whenever a buoy comparison is recomputed (the bundled example uses the Vilano buoy), recording at least `version`, `value`, `unit`, `effective_from`, `source`, and `computed_at`. The default record is flagged as the HF-EOLUS Vilano template and embeds the ANN and buoy DOIs (10.5281/zenodo.17131227 and 10.5281/zenodo.17098037) so downstream artefacts can cite the public dataset; replace or add entries when validating against other references.
- `config/node_taxonomy.json` captures per-node observation totals and cadence-gap statistics consumed by the same provider. The shipped file is the HF-EOLUS taxonomy template derived from `use_case/catalogs/sar_range_final_pivots_joined` and references the ANN DOI so the sample case remains reproducible. Regenerate it whenever the GeoParquet snapshot or its coverage changes:

```
python3 scripts/update_node_taxonomy.py \
  --output config/node_taxonomy.json \
  --schema-version 2025-10-19
```

The helper resolves the ANN GeoParquet via `config/stac_catalogs.json` and invokes the `duckdb/duckdb:latest` container, so Python 3 and Docker (with access to the daemon socket) must be available. Pass `--dataset <path/to/snapshot.parquet>` only when you need to override the STAC catalogue entry; otherwise the defaults apply and a new ISO-8601 timestamp is embedded under `generated_at`.

Each taxonomy record now exposes a `low_coverage` boolean that turns `true`
when a node has fewer than the configured number of valid observations and its
longest gap exceeds the configured multi-year threshold (defaults: 1,000
observations, 730 days). Override the defaults with
`--low-coverage-min-observations` and `--low-coverage-min-gap-days` when running
the script; the resulting JSON persists the thresholds under
`low_coverage_rules`, making it straightforward for reporting code and
documentation to reuse the audit logic consistently.
- `config/low_coverage_rules.json` stores the default thresholds applied by the
  taxonomy generator. Edit this file to change the project-wide baseline (CLI
  flags still allow ad-hoc overrides during regeneration).
- `config/geospatial_products.json` controls the geospatial publication helper: besides output paths it exposes a `style` block (figure size, padding, stroke, gradients, annotations) and a `power_uncertainty` block pointing to the bootstrap summary used in the uncertainty map.

- `config/range_thresholds.json` defines the lower/upper wind-speed bounds (in m/s) that delimit the valid regression range of the ANN. `hf_wind_resource.preprocessing.censoring.load_range_thresholds` consumes this file when partitioning samples, computing per-node proportions of censored data, and reporting discrepancies. Update the JSON whenever the ANN is retrained with different physical limits or when running sensitivity analyses.
- `config/power_height.json` centralises the default vertical extrapolation applied by `scripts/generate_power_estimates.py`, including the source/target heights, the extrapolation method (log-law/power-law), and parameters such as `z0` or the power-law exponent. CLI flags can override individual fields on demand.
- `config/range_quality_thresholds.json` centralises the QA thresholds for censored proportions, temporal coverage, and minimum in-range support. `hf_wind_resource.qa.load_range_qa_thresholds` reads this configuration to flag nodes that should bypass parametric Weibull fitting or require manual inspection.
- `config/stac_catalogs.json` enumerates the STAC collections and default item/asset pairs that the IO layer resolves at runtime. The index currently exposes two template entries: `sar_range_final_pivots_joined` (ANN dataset, DOI [10.5281/zenodo.17131227](https://doi.org/10.5281/zenodo.17131227)) and `pde_vilano_buoy` (buoy dataset, DOI [10.5281/zenodo.17098037](https://doi.org/10.5281/zenodo.17098037)). Extend or override those entries to wire your own catalogues while keeping the example DOIs for reference.

Maintainers should keep these configuration files aligned with the documentation; they are the single source of truth for the runtime code.

## Python package overview
- `hf_wind_resource.io`: Contracts for chunked readers, anomaly sinks, caching interfaces, and time-window helpers aligned with the SAR range-aware snapshot.
- `hf_wind_resource.preprocessing`: Temporal normalisation routines that deduplicate `(node_id, timestamp)` pairs, compute cadence statistics, and emit gap logs for downstream QA. It also provides ingestion helpers for buoy datasets, translating sentinel markers into missing values and synchronising the cleaned series with the ANN node selected by the caller. The range-censoring helpers in this package partition samples by label, derive per-node proportions, and validate them against the configurable thresholds stored in `config/range_thresholds.json`.
- `hf_wind_resource.stats`: Global RMSE registry backed by `config/global_rmse.json` (versioned RMSE records) and `config/node_taxonomy.json` (per-node metadata extracted from the catalog documentation), exposing effective-date lookups, refresh hooks, and taxonomy-aware limitation messages.

## Requirements
Exploration, QA, and the downstream resource-estimation analyses rely on [DuckDB](https://duckdb.org/) to query the GeoParquet snapshots in place. Install DuckDB locally (`pip install duckdb` or use your OS packages) or run the official image. To inspect the snapshot registered in `config/stac_catalogs.json`, resolve the path and launch DuckDB with a single command:

```
duckdb_path=$(python3 - <<'PY'
from hf_wind_resource.io import resolve_ann_asset
print(resolve_ann_asset(root=".", config_path="config/stac_catalogs.json").require_local_path())
PY
)
docker run --rm -v "$(pwd)":/workspace -w /workspace duckdb/duckdb:latest \
  duckdb -cmd "SELECT COUNT(*) FROM read_parquet('${duckdb_path}');"
```

The documentation references similar commands when computing row counts, per-node coverage, gap statistics, and buoy comparisons; ensure DuckDB is available to reproduce them.

To regenerate the temporal normalisation report after refreshing the ANN snapshot, run:

```
docker run --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  --entrypoint python \
  wind-resource-tests \
  scripts/generate_temporal_summary.py \
  --output docs/temporal_normalisation_summary.md
```

This command reuses the test container (which already bundles `pandas`, `duckdb`, `pyarrow`) to load the GeoParquet resolved via `config/stac_catalogs.json`, apply the normalisation routine, and write the Markdown summary.

To synchronise any buoy time series with the ANN predictions, run:

```
docker run --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  --entrypoint python \
  wind-resource-tests \
  scripts/prepare_buoy_timeseries.py \
    --buoy-dataset use_case/catalogs/<buoy_catalog>/assets/<file>.parquet \
    --stac-dataset sar_range_final_pivots_joined \
    --node-id <target_node_id> \
    --output-parquet artifacts/processed/<buoy>_synced.parquet \
    --output-summary artifacts/processed/<buoy>_summary.json
```

The ANN GeoParquet snapshot is resolved via the STAC index declared in `config/stac_catalogs.json`; specify `--ann-dataset` when a manual override is required. By default the script keeps only exact timestamp matches between the buoy and ANN series; add `--nearest-matching` to accept nearest-neighbour matches within the tolerance window. Fine-tune the tolerance with `--tolerance-minutes`, point to another buoy by changing `--buoy-dataset` and `--node-id`, or adjust the geometry metadata with `--geometry-column` / `--geometry-crs`. Sample values for the Vilano release appear in the “Example dataset” section.

The vertical wind-speed correction applied during the synchronisation step is fully configurable:

- The default parameters live in [`config/buoy_height.json`](config/buoy_height.json) (3 m sensor height, 10 m target, neutral logarithmic profile with roughness length 0.0002 m). Edit this file to change the baseline behaviour for all runs.
- CLI flags can override individual values at runtime (`--height-method`, `--measurement-height-m`, `--target-height-m`, `--power-law-alpha`, `--roughness-length-m`).
- Pass `--disable-height-correction` to skip the adjustment entirely and retain the raw buoy wind speeds.

To generate the angular-discrepancy indicators and visuals, run:

```
docker run --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  --entrypoint python \
  wind-resource-tests \
  scripts/generate_direction_comparison.py \
    --matched-dataset artifacts/processed/<buoy>_synced.parquet \
    --output-dir artifacts/direction_comparison
```

The command emits `direction_metrics_summary.csv`, `direction_quality.json`, and `direction_errors.parquet` alongside SVG plots (`absolute_error_histogram.svg`, `direction_scatter.svg`, `coverage_comparison.svg`) that capture the ANN-versus-buoy discrepancies.

To audit censored proportions, temporal density, and continuity before launching parametric resource estimation, run:

```
python3 scripts/generate_range_quality_report.py \
  --output-dir artifacts/range_quality
```

The helper resolves the ANN dataset via `config/stac_catalogs.json`, executes the necessary DuckDB aggregations inside the `duckdb/duckdb:latest` container, and evaluates the thresholds defined in `config/range_quality_thresholds.json`. It writes three artefacts under the chosen directory: `range_quality_summary.csv` (per-node metrics), `range_quality_summary.json` (serialised records), and `range_quality_summary.log` (human-readable flag summary with parametric reliability markers).
When Docker is not available on the host, append `--engine python` to run the queries through the local DuckDB Python module (make sure the `duckdb` package is installed in that environment).

To derive theoretical power-density estimates and turbine outputs, use:

```
docker run --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  --entrypoint python \
  wind-resource-tests \
  scripts/generate_power_estimates.py \
    --output artifacts/power_estimates \
    --power-curve-key reference_offshore_6mw \
    --engine python
```

The command resolves the ANN GeoParquet via `config/stac_catalogs.json`, fits censored Weibull models where possible, falls back to the Kaplan–Meier estimator under heavy censoring, and integrates the default 6 MW offshore power curve declared in `config/power_curves.json`. Outputs include `power_estimates_summary.csv`, per-node JSON diagnostics under `artifacts/power_estimates/nodes/`, and `metadata.json` capturing the air-density and surrogate assumptions. Adjust `--air-density`, `--right-tail-surrogate`, or select another curve key to explore alternative operating conditions.

The CLI now benchmarks the censored Weibull fit against log-normal and gamma
alternatives, computing censored log-likelihoods, AIC/BIC, and weighted KS
statistics for each candidate. Use the new flags to control the diagnostics:

- `--parametric-selection-metric {aic,bic}` decides which information criterion
  selects the preferred model (`bic` by default).
- `--parametric-min-in-weight` and `--parametric-ks-min-weight` enforce the
  minimum in-range support for running the alternative fits or the KS test.
- `--disable-gamma-fit` skips the gamma candidate when SciPy is unavailable in
  the execution environment.

The resulting columns (`parametric_preferred_model`, `lognormal_mu`,
`gamma_parametric_notes`, etc.) propagate to the per-node CSV/JSON outputs and
the STAC publication so dashboards can explain why a specific parametric model
was preferred.
Starting with the current release, the CLI also generates:

- `artifacts/power_estimates/seasonal_power_summary.csv` with per-node seasonal estimates (DJF/MAM/JJA/SON tied to the corresponding meteorological year).
- `artifacts/power_estimates/monthly_power_timeseries.csv` with a per-node monthly series (ready for ARIMA/ETS analysis) that includes censoring counts and the automatic Weibull/Kaplan–Meier selection metadata.

When running directly on the host (with the Docker CLI available), you may omit `--engine python` to let the script call the official `duckdb/duckdb:latest` image instead.

To build the consolidated geospatial deliverables and companion visualisations, execute:

```
python3 scripts/generate_geospatial_products.py --overwrite
```

The helper reads `artifacts/power_estimates/node_summary/node_summary.csv`, resolves the ANN GeoParquet through `config/stac_catalogs.json`, and emits a GeoParquet + GeoJSON pair together with dedicated SVG maps (`power_density_map.svg`, `uncertainty_map.svg` showing the bootstrap interval width for power density) and a compact set of wind roses (top power, top uncertainty, and a low-coverage exemplar). When a buoy dataset is provided, the script additionally produces `buoy_wind_rose_panels.svg` with the observed-versus-ANN comparison for that node. Outputs land in `artifacts/power_estimates/geospatial/` and include a `metadata.json` file capturing inputs, selected nodes, and timestamps for reproducibility.

Configuration lives in `config/geospatial_products.json`. Override any entry (node summary path, STAC dataset/config, output directory, artifact filenames, buoy settings, engine/image, etc.) or supply an alternate file via `--config`. CLI flags still take precedence when provided.

To characterise seasonal and interannual variability before publishing resource reports, execute:

```
python3 scripts/generate_seasonal_variations.py \
  --output-dir artifacts/seasonal_analysis
```

The helper selects `timestamp`, `node_id`, `pred_wind_speed`, and `pred_range_label` from the ANN GeoParquet via the `duckdb/duckdb:latest` container, runs the aggregations in `hf_wind_resource.stats.seasonal`, and materialises:

- `artifacts/seasonal_analysis/seasonal_slices.csv` (per-node seasonal metrics with label ratios),
- `artifacts/seasonal_analysis/annual_slices.csv` (annual means, quantiles, and sample counts),
- `artifacts/seasonal_analysis/variation_summary.csv` (amplitudes, seasonal coverage, annual trend slopes),
- `artifacts/seasonal_analysis/seasonal_analysis_summary.json` (machine-readable snapshot with top amplitudes/trends and the `height_*` context pulled from `artifacts/power_estimates/metadata.json`), and
- `artifacts/seasonal_analysis/seasonal_variation_summary.md` (Markdown digest highlighting the largest seasonal spreads and strongest annual trends, explicitly noting whether the analysis uses the native 10 m series or height-corrected values).

When Docker cannot be invoked from the host shell (e.g., inside the test harness), supply `--execution-mode python` to run the same DuckDB query with the embedded `duckdb` Python package.

See `docs/seasonal_analysis_methodology.md` for a detailed statistical and mathematical description of the seasonal workflow.

For exploratory time-series modelling (ARIMA/SARIMA/ETS) on the monthly series:

```
python3 scripts/generate_time_series_models.py \
  --forecast-steps 12 \
  --output artifacts/power_estimates/time_series \
  --max-gap-months 6 \
  --min-observations 36
```

The script expects `artifacts/power_estimates/monthly_power_timeseries.csv`. Default values (`max_gap_months`, `min_segment_months`) come from `config/time_series.json` and can be overridden with the CLI flags shown above. It segments each node into continuous sub-periods (gaps shorter than `--max-gap-months` are linearly interpolated; longer gaps start a new segment) and fits SARIMA and ETS models per segment. Every node/segment combination produces a row in `time_series_summary.csv` and a JSON file under `nodes/<node_id>.json` with orders, parameters, Ljung–Box diagnostics, and forecasts. Run the command inside the `wind-resource-tests` image to ensure the required dependencies (`statsmodels`) are available.

> **Dependency note:** to let the `wind-resource-tests` container execute the ARIMA/ETS models offline, place the compatible wheels for `statsmodels`, `scipy`, and `patsy` (versions pinned in `docker/tests/requirements-tests.lock`) under `vendor/wheels/`. They will be installed automatically during the image build with `pip --no-index`.


## Command-line interface
The consolidated CLI defined in `hf_wind_resource.cli` orchestrates the individual helpers under `scripts/` so the full pipeline can be executed without chaining shell commands manually. Run it through the Python module entry point to keep the `src/` tree on `PYTHONPATH`, for example:

```bash
PYTHONPATH=src python3 -m hf_wind_resource.cli.main compute_resource \
  --engine docker \
  --image duckdb/duckdb:latest
```

By default the command runs the empirical metrics, power estimation, node summary, bootstrap, and geospatial stages. Use `--stages power bootstrap` to select a subset, or `--skip-stages bootstrap` to skip costly stages. The `--nodes` option filters the ANN GeoParquet before launching any stage, allowing fast iterations on a handful of nodes; pass `--keep-filtered-dataset` to inspect the temporary Parquet snapshot left under `artifacts/tmp_cli_filters/`.

Bootstrap options such as `--bootstrap-replicas`, `--bootstrap-confidence`, `--bootstrap-workers`, `--bootstrap-resume`, and `--bootstrap-disable-rmse` are forwarded to `scripts/generate_bootstrap_uncertainty.py`. Geospatial exports honour `--geospatial-config`, `--geospatial-max-roses`, and `--disable-buoy-outputs`, while publication parameters can be adjusted through `--publish-version-tag` and `--publish-overwrite`.

### Buoy validation helper
`validate_buoy` wraps `scripts/prepare_buoy_timeseries.py`, the angular diagnostics, and the optional resource-bias comparison. Invoke it with the node identifier and buoy dataset that correspond to your catalogue:

```bash
PYTHONPATH=src python3 -m hf_wind_resource.cli.main validate_buoy \
  --node-id <target_node_id> \
  --buoy-dataset use_case/catalogs/<buoy_catalog>/assets/<file>.parquet \
  --resource-overwrite
```

The command honours the same STAC configuration used elsewhere, accepts absolute or repository-relative paths, and writes all artefacts under `artifacts/direction_comparison/` and `artifacts/buoy_validation/`. Adjust the height-correction parameters via `config/buoy_height.json` or the CLI overrides documented in [`docs/buoy_height_configuration.md`](docs/buoy_height_configuration.md).

To keep the toolkit generic, `run_pipeline.sh` skips the buoy validation unless the environment variable `BUOY_VALIDATION_CONFIG` points to a JSON file that lists the desired overrides. The helper `scripts/run_buoy_validation_from_config.py` expands that JSON into CLI switches, so the Vilano example can be reproduced via:

```bash
BUOY_VALIDATION_CONFIG=use_case/config/vilano_buoy_validation.json ./run_pipeline.sh
```

A neutral template lives at `config/buoy_comparison.json`; copy it next to your case inputs (for example `use_case/config/buoy_comparison.json`) and replace the placeholder paths before launching `validate_buoy`.

## Data availability
The toolkit expects STAC catalogues that describe the ANN inference dataset and any optional buoy benchmarks. The repository ships read-only copies of the demo catalogues under `use_case/catalogs/` so Dockerised commands have something to resolve by default. When reproducing the public example, download the Zenodo packages [10.5281/zenodo.17131227](https://doi.org/10.5281/zenodo.17131227) (ANN) and [10.5281/zenodo.17098037](https://doi.org/10.5281/zenodo.17098037) (buoy) and unpack them following [`docs/data_access.md`](docs/data_access.md).

For other deployments, replace the STAC entries in `config/stac_catalogs.json` with your catalogues, refresh the manifests and changelog under `use_case/catalogs/`, and verify the checksums listed in `use_case/catalogs/CHANGELOG.md`.

## Input data overview
The default dataset key (`sar_range_final_pivots_joined`) points to a GeoParquet whose schema is documented in [`docs/sar_range_final_schema.md`](docs/sar_range_final_schema.md). Each record bundles aggregated HF radar metrics for VILA/PRIO, neural-network predictions (`pred_*` columns), classifier outputs (`prob_range_*`, `pred_range_label`, `range_flag_*`), QA helpers, and the WKB geometry of the node in CRS84. Samples labelled outside the valid range should be treated as censored observations in downstream statistics; the exact thresholds and classifier confidence cut-offs are declared in `config/range_thresholds.json`.

| Column | Description |
| --- | --- |
| `pred_wind_speed` | Model prediction for wind speed (m/s). |
| `pred_wind_direction` | Model prediction for wind direction (degrees clockwise from geographic north). |
| `pred_cos_wind_dir` | Cosine component of the predicted wind direction unit vector (unitless). |
| `pred_sin_wind_dir` | Sine component of the predicted wind direction unit vector (unitless). |
| `prob_range_below` | Posterior probability that wind speed lies below the calibrated operating range (0-1). |
| `prob_range_in` | Posterior probability that wind speed lies within the calibrated operating range (0-1). |
| `prob_range_above` | Posterior probability that wind speed lies above the calibrated operating range (0-1). |
| `pred_range_label` | Discrete range class selected by the classifier (below/in/above). |
| `pred_range_confidence` | Confidence associated with the predicted range label (0-1). |
| `pred_speed_range_label` | Range class inferred deterministically from the predicted wind speed (below/in/above). |
| `range_near_lower_margin` | Flag indicating the predicted wind speed lies near the lower reliability margin. |
| `range_near_upper_margin` | Flag indicating the predicted wind speed lies near the upper reliability margin. |
| `range_near_any_margin` | Flag indicating the predicted wind speed lies near any reliability margin. |
| `range_flag` | Consolidated range-awareness flag (below/in/above/uncertain). |
| `range_flag_confident` | True when the range-awareness flag exceeds the configured confidence threshold (default 0.5; see `config/range_thresholds.json`). |
| `range_prediction_consistent` | Indicates whether the range-awareness flag matches the class derived from predicted wind speed. |

Per-node coverage, cadence gaps, taxonomy bands, and buoy co-location notes are summarised in `docs/sar_range_final_schema.md` and `docs/temporal_normalisation_summary.md`. Recompute those reports after ingesting a new ANN snapshot so the QA context remains aligned with the data you distribute.

## Example dataset: Vilano buoy
The repository retains the Vilano case study as a demonstrator of how to run the toolkit end-to-end:

- Download the ANN and buoy catalogues from the Zenodo DOIs listed above and keep their layout under `use_case/catalogs/`.
- Run the quick-start command or execute `docker run --rm -v "$(pwd)":/workspace -w /workspace --entrypoint python wind-resource-tests scripts/generate_buoy_resource_comparison.py --node-id Vilano_buoy --buoy-dataset use_case/catalogs/pde_vilano_buoy/assets/Vilano.parquet` to regenerate the comparison artefacts.
- Inspect `artifacts/buoy_validation/` (`resource_metrics_table.md`, `resource_bias.svg`, etc.) and `artifacts/direction_comparison/` for ready-to-use visuals. These outputs serve purely as examples; replace them with your own buoy benchmarks when adapting the toolkit to a different region.

## Private development files
Internal planning notes and audit reports live in dedicated directories within this working copy. Exclude them when preparing the public-facing toolkit so the published repository contains only reusable code, configuration, and documentation.



## Acknowledgements

This work has been funded by the HF-EOLUS project (TED2021-129551B-I00), financed by MICIU/AEI /10.13039/501100011033 and by the European Union NextGenerationEU/PRTR - BDNS 598843 - Component 17 - Investment I3. Members of the Marine Research Centre (CIM) of the University of Vigo have participated in the development of this repository.



## Disclaimer

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

## License

This repository, together with the STAC outputs published under `use_case/catalogs/`, is distributed under the GNU General Public License v3.0. See [`LICENSE`](LICENSE) for the full terms and mirror that reference in any downstream catalog metadata to guarantee traceability.



---
<p align="center">
  <a href="https://next-generation-eu.europa.eu/">
    <img src="logos/EN_Funded_by_the_European_Union_RGB_POS.png" alt="Funded by the European Union" height="80"/>
  </a>
  <a href="https://planderecuperacion.gob.es/">
    <img src="logos/LOGO%20COLOR.png" alt="Logo Color" height="80"/>
  </a>
  <a href="https://www.aei.gob.es/">
    <img src="logos/logo_aei.png" alt="AEI Logo" height="80"/>
  </a>
  <a href="https://www.ciencia.gob.es/">
    <img src="logos/MCIU_header.svg" alt="MCIU Header" height="80"/>
  </a>
  <a href="https://cim.uvigo.gal">
    <img src="logos/Logotipo_CIM_original.png" alt="CIM logo" height="80"/>
  </a>
  <a href="https://www.iim.csic.es/">
    <img src="logos/IIM.svg" alt="IIM logo" height="80"/>
  </a>

  
</p>