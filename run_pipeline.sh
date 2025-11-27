#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MAX_NODES="${MAX_NODES:-}"
BOOTSTRAP_REPLICAS="${BOOTSTRAP_REPLICAS:-500}"
BOOTSTRAP_CONFIDENCE="${BOOTSTRAP_CONFIDENCE:-0.95}"

MAX_NODE_ARGS=()
if [[ -n "${MAX_NODES}" ]]; then
  echo ">> Limiting compute-intensive stages to the first ${MAX_NODES} node(s)"
  MAX_NODE_ARGS=(--max-nodes "${MAX_NODES}")
fi

echo "[1/8] Updating node taxonomy"
PYTHONPATH=src python3 scripts/update_node_taxonomy.py \
  --output config/node_taxonomy.json \
  --stac-config config/stac_catalogs.json \
  --stac-dataset sar_range_final_pivots_joined

echo "[2/8] Generating range-quality diagnostics"
docker run --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  --entrypoint python \
  wind-resource-tests \
  scripts/generate_range_quality_report.py \
    --stac-config config/stac_catalogs.json \
    --stac-dataset sar_range_final_pivots_joined \
    --output-dir artifacts/range_quality \
    --threshold-config config/range_quality_thresholds.json \
    --image duckdb/duckdb:latest \
    --engine python

echo "[3/8] Computing empirical metrics"
docker run --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  --entrypoint python \
  wind-resource-tests \
  scripts/generate_empirical_metrics.py \
    --stac-config config/stac_catalogs.json \
    --stac-dataset sar_range_final_pivots_joined \
    --taxonomy config/node_taxonomy.json \
    --output-dir artifacts/empirical_metrics \
    --docker-image duckdb/duckdb:latest \
    --overwrite

echo "[4/8] Building Kaplan-Meier non-parametric distributions"
bash scripts/run_nonparametric_distributions.sh \
  --stac-config config/stac_catalogs.json \
  --stac-dataset sar_range_final_pivots_joined \
  --summary artifacts/empirical_metrics/per_node_summary.csv \
  --output artifacts/nonparametric_distributions \
  --image duckdb/duckdb:latest \
  "${MAX_NODE_ARGS[@]}"

echo "[5/8] Running the core resource pipeline"
docker run --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  --entrypoint python \
  wind-resource-tests \
  -m hf_wind_resource.cli.main compute_resource \
    --stac-config config/stac_catalogs.json \
    --stac-dataset sar_range_final_pivots_joined \
    --engine python \
    --image duckdb/duckdb:latest \
    --skip-stages empirical publish \
    --overwrite \
    --bootstrap-replicas "${BOOTSTRAP_REPLICAS}" \
    --bootstrap-confidence "${BOOTSTRAP_CONFIDENCE}" \
    --python-backend auto \
    "${MAX_NODE_ARGS[@]}"
if [ -d artifacts/bootstrap_uncertainty ]; then
  # Mirror bootstrap artefacts into the legacy directory expected by downstream tools.
  rm -rf artifacts/bootstrap_velocity_block
  mkdir -p artifacts/bootstrap_velocity_block
  cp -R artifacts/bootstrap_uncertainty/. artifacts/bootstrap_velocity_block/
fi

echo "[5b] Publishing STAC catalog"
PYTHONPATH=src python3 scripts/publish_power_estimates_catalog.py \
  --engine docker \
  --image duckdb/duckdb:latest \
  --overwrite

echo "[6/8] Generating seasonal diagnostics"
docker run --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  --entrypoint python \
  wind-resource-tests \
  scripts/generate_seasonal_variations.py \
    --stac-config config/stac_catalogs.json \
    --stac-dataset sar_range_final_pivots_joined \
    --output-dir artifacts/seasonal_analysis \
    --report artifacts/seasonal_analysis/seasonal_variation_summary.md \
    --docker-image duckdb/duckdb:latest \
    --execution-mode python

echo "[7/8] Fitting monthly time-series models"
docker run --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  --entrypoint python \
  wind-resource-tests \
  scripts/generate_time_series_models.py \
    --input artifacts/power_estimates/monthly_power_timeseries.csv \
    --output artifacts/power_estimates/time_series \
    --forecast-steps 12 \
    --seasonal-periods 12 \
    --max-gap-months 6 \
    --min-observations 36 \
    "${MAX_NODE_ARGS[@]}"

if [[ -n "${BUOY_VALIDATION_CONFIG:-}" ]]; then
  echo "[8/8a] Computing buoy block diagnostics"
  docker run --rm \
    -v "$(pwd)":/workspace \
    -w /workspace \
    --entrypoint python \
    wind-resource-tests \
    scripts/evaluate_bootstrap_dependence.py \
      --dataset use_case/catalogs/pde_vilano_buoy/assets/Vilano.parquet \
      --output-dir artifacts/buoy_block_diagnostics \
      --lags 1 2 3 \
      --dataset-kind uncensored \
      --node-column none \
      --node-id PdE_Vilano
fi

if [[ -n "${BUOY_VALIDATION_CONFIG:-}" ]]; then
  echo "[8/8b] Validating ANN winds against the configured buoy reference"
  PYTHONPATH=src python3 scripts/run_buoy_validation_from_config.py \
    --config "${BUOY_VALIDATION_CONFIG}"
else
  echo "[8/8] Skipping buoy validation (set BUOY_VALIDATION_CONFIG to enable this step)"
fi
