# Data Access and Reproducibility

This guide enumerates the artefacts required by the wind-resource toolkit, the
locations where authoritative copies are published, and the workflows used to
refresh or subset the datasets. It complements the schema reference in
`docs/sar_range_final_schema.md` and the architectural guidance in
`docs/python_architecture.md`.

## Primary Input Datasets

- **ANN inference (SAR range-final)**  
  STAC catalogue: `use_case/catalogs/sar_range_final_pivots_joined/collection.json`  
  Default asset: `use_case/catalogs/sar_range_final_pivots_joined/assets/data.parquet`  
  DOI package: [10.5281/zenodo.17464583](https://doi.org/10.5281/zenodo.17464583)

- **Sample buoy benchmark (Vilano)**  
  STAC catalogue: `use_case/catalogs/pde_vilano_buoy/collection.json`  
  Default asset: `use_case/catalogs/pde_vilano_buoy/assets/Vilano.parquet`  
  DOI package: [10.5281/zenodo.17098037](https://doi.org/10.5281/zenodo.17098037)

Both catalogues are indexed in `config/stac_catalogs.json` so that CLI helpers
and scripts can resolve them automatically. Replace the buoy entry when working
with a different benchmark.

## Download and Synchronisation Paths

### Local snapshots

Retrieve the latest archives from their respective Zenodo records (see links
above) and unpack them into `use_case/catalogs/`, preserving the catalogue layout
described by each release. The repository keeps read-only copies of the
snapshots under `use_case/catalogs/`. Downstream tooling expects the following relative
paths to exist (update the second entry if you rely on another buoy dataset):

- `use_case/catalogs/sar_range_final_pivots_joined/assets/data.parquet`
- `use_case/catalogs/pde_vilano_buoy/assets/Vilano.parquet`

Do not rewrite these files manually. When a new release is ingested from Zenodo or regenerated locally, refresh the folder with the verified artefacts (checking hashes against the Zenodo metadata) and update `use_case/catalogs/catalog.json` plus `use_case/catalogs/CHANGELOG.md` accordingly.

## Regenerating the ANN inference dataset

The ANN inference GeoParquet originates from the
[`hf-wind-inversion`](https://github.com/GaliciaAI/hf-wind-inversion) project.
Consult that repository’s documentation for the official generation workflow.
Once a new release is produced and published (e.g. via Zenodo), place the
updated `data.parquet` under `use_case/catalogs/sar_range_final_pivots_joined/assets/`,
verify its checksum, and refresh the STAC metadata and changelog in this
repository.

## Working with subsets and fixtures

### Filtering nodes via DuckDB

Use the official DuckDB container to extract a small Parquet for exploratory
analysis or rapid iteration:

```bash
docker run --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  duckdb/duckdb:latest \
  duckdb -cmd "COPY (
    SELECT *
    FROM read_parquet('catalogs/sar_range_final_pivots_joined/assets/data.parquet')
    WHERE node_id IN ('<NODE_A>', '<NODE_B>')
  ) TO 'artifacts/tmp_cli_filters/example_subset.parquet' (FORMAT PARQUET);"
```

All intermediate files must remain inside the repository tree so they are
visible to subsequent Docker runs.

### CLI-driven subsetting

The unified orchestrator accepts node filters and honours the same repository
constraints:

```bash
PYTHONPATH=src python3 -m hf_wind_resource.cli.main compute_resource \
  --nodes <NODE_A> <NODE_B> \
  --keep-filtered-dataset \
  --overwrite
```

The CLI materialises a filtered Parquet under `artifacts/tmp_cli_filters/` and
propagates it through the pipeline stages.

### Synthetic fixtures

Automated tests use deterministic fixtures located in `tests/fixtures/`. The
dataset covers range-label combinations, cadence variations, and regression
edge cases. Regenerate them with:

```bash
python3 tests/fixtures/build_sar_range_final_synthetic.py
```

The script produces CSV/JSON summaries and calls the DuckDB container to refresh
the Parquet representation unless `--skip-parquet` is provided. Document any
fixture changes in `tests/fixtures/README.md`.

## Versioning, manifests, and checksum validation

- `use_case/catalogs/CHANGELOG.md` records the publication history of input and output
  catalogues, listing the STAC version, DOI, and manifest hash for each
  snapshot.
- `use_case/catalogs/sar_range_final_power_estimates/manifest.json` encapsulates the
  exact inputs and configuration used to derive the published power estimates.
- Run `shasum -a 256 <file>` (macOS/Linux) or `certutil -hashfile <file> SHA256`
  (Windows) to verify local files against the hashes logged in the manifest.
- Whenever a new dataset is introduced, update `use_case/catalogs/catalog.json` so the
  STAC root exposes the fresh collection and cite the change in the changelog.

Following these steps keeps the toolkit reproducible across environments while
documenting every dependency required to recompute wind-resource outputs.
