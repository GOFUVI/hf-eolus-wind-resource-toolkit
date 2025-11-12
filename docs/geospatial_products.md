# Geospatial Products Helper

## Purpose
`scripts/generate_geospatial_products.py` publishes the map-ready artefacts derived from the ANN wind-resource workflow. It reads the per-node summary table, resolves the STAC metadata to locate the ANN GeoParquet, and emits cartographic outputs (maps, wind-rose panels, metadata) intended for reports and dashboards.

## Requirements
- Python 3.
- Docker CLI with access to `duckdb/duckdb:latest` (unless using the embedded DuckDB Python engine).
- The repository artefacts referenced below must already exist:
  - `artifacts/power_estimates/node_summary/node_summary.csv` (from `scripts/generate_node_summary_table.py`).
  - STAC catalogue entries under `config/stac_catalogs.json` pointing at the ANN GeoParquet snapshot.
  - Optional: the buoy dataset declared in `config/geospatial_products.json` (defaults to the Vilano example under `use_case/catalogs/pde_vilano_buoy/`) for comparison panels.

## Configuration
Execution is controlled via `config/geospatial_products.json` (override with `--config PATH`). The JSON exposes:

- `node_summary`: path to the per-node summary CSV.
- `taxonomy`: path to the taxonomy JSON (`config/node_taxonomy.json`) used to propagate low-coverage and band flags into the exported artefacts.
- `output_dir`: base directory for generated artefacts.
- `stac`: keys `config_path` and `dataset` used to resolve the ANN GeoParquet when `--dataset` is omitted.
- `ann_dataset`: optional direct path to bypass STAC resolution.
- `outputs`: filenames/subdirectories for each artefact (GeoParquet, GeoJSON, maps, wind-rose panels, histograms, metadata).
- `style`: aesthetic options:
  - `figure_scale`, plus the nested `figure` block, adjust canvas dimensions and margins.
  - `node_padding` keeps node markers away from the frame edges.
  - `low_coverage_stroke` and `low_coverage_stroke_width` configure the highlighted outline for sparse nodes.
  - `power_gradient` / `uncertainty_gradient` define colour ramps; `colorbar.width` shifts/resizes scale bars.
  - `label_note`, `power_subtitle`, `uncertainty_subtitle`, `uncertainty_note` tweak explanatory captions.
  - `reference_outline`, `reference_outline_width`, `reference_note` style the buoy comparison node when present.
  - `missing_marker` customises the hollow markers used when a node lacks the requested estimator (e.g., KM spread).
- `power_uncertainty`: points the tool at the bootstrap summary (defaults to `artifacts/bootstrap_velocity_block/bootstrap_summary.csv`), exposes field overrides for the confidence-interval columns, and optionally accepts a companion metadata file (e.g. `bootstrap_metadata.json`) so replica counts, confidence level, and resampling settings are recorded alongside the map outputs.

CLI flags (e.g. `--node-summary`, `--output-dir`, `--stac-config`, `--engine`, `--buoy-dataset`) take precedence over configuration values. Pass `--disable-buoy-rose` to omit the buoy comparison even if configured.

## Usage
```
python3 scripts/generate_geospatial_products.py --overwrite
```
Run inside Docker (matching the project test image):
```
docker run --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  --entrypoint python \
  wind-resource-tests \
  scripts/generate_geospatial_products.py --overwrite --engine python
```
Common flags:
- `--config config/geospatial_products.json` – alternate configuration file.
- `--dataset PATH/ann.parquet` – bypass STAC and point at a different ANN snapshot.
- `--engine python` – use the local DuckDB Python package instead of Docker.
- `--max-wind-roses 6` – adjust the number of representative ANN roses.
- `--buoy-dataset use_case/catalogs/<buoy_catalog>/assets/<file>.parquet` – override buoy source.

## Outputs
All artefacts are written under `output_dir` (defaults to `artifacts/power_estimates/geospatial/`). Notable files:

- `node_resource_map.parquet` / `.geojson` – per-node GeoParquet/GeoJSON with metrics and geometry.
- `power_density_map.svg` – thematic map of power density.
- `uncertainty_map.svg` – thematic map of the bootstrap confidence-interval width of power density (W/m²); hollow markers highlight nodes publicados vía Weibull (sin intervalo bootstrap).
- `wind_rose_panels.svg` – representative ANN wind-rose panels for key nodes; panels linked to sparse nodes append a "(low coverage)" suffix and include a taxonomy note.
- `buoy_wind_rose_panels.svg` – buoy vs. ANN roses when reference data is available.
- `wind_rose_histogram.csv` & `buoy_wind_rose_histogram.csv` – sector histograms used to build the roses.
- `metadata.json` – provenance summary (inputs, configuration source, selected nodes, ancillary statistics).
- `metadata.json` now records `taxonomy_source` and enumerates `low_coverage_nodes` so downstream tooling can filter or annotate sparse nodes consistently.

## Notes
- The script honours low-coverage flags from the taxonomy, accenting those nodes with a bright outline.
- Missing KM estimates appear as hollow markers; the legend documents this behaviour.
- When a buoy dataset is available, the ANN node and buoy comparison receive a dedicated outline and legend entry so readers can distinguish observed vs. inferred roses.
- Regeneration is idempotent: rerun with `--overwrite` to refresh artefacts after updating inputs or configuration.
