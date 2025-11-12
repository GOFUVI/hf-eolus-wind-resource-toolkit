# Block Bootstrap Assessment

This note evaluates whether the ANN inference records exhibit temporal
dependence strong enough to justify block or stationary bootstrap resampling.
The analysis supports the audit recommendations captured in
`audits/bootstrap-methodology.md`.

## Methodology

The script `scripts/evaluate_bootstrap_dependence.py` reads the canonical
GeoParquet dataset (`use_case/catalogs/sar_range_final_pivots_joined/assets/data.parquet`)
and, for every `node_id`:

- computes autocorrelations for lags 1–3 using DuckDB window functions;
- derives the average time separation between consecutive observations;
- estimates an effective sample size under an AR(1) approximation;
- proposes a moving-block length defined as `round(n / n_eff)` constrained to
  be at least one sample.

The script produces two artefacts under
`artifacts/bootstrap_uncertainty/`:

- `block_bootstrap_diagnostics.csv`: node-level metrics including lagged
  autocorrelations and suggested block length;
- `block_bootstrap_summary.json`: aggregated statistics for easy inspection.

All numbers below were obtained on 2025-10-25 using the full dataset hosted in
the repository.

## Findings

- 56 nodes contain valid samples. The mean temporal gap between observations is
  approximately 6.7 hours, with a wide standard deviation (≈ 128 hours), which
  reflects the irregular radar coverage.
- First-lag autocorrelations are high: 93 % of the nodes exceed 0.3, 66 % exceed
  0.5, and 23 % exceed 0.7. The median `acf_lag_1` is 0.58, while the maximum is
  0.82 (node `VILA_PRIO46`).
- The resulting suggested moving-block lengths range from 1 to 10 samples with
  a median of 4 and an average of 4.34. Nodes near the radar range limits tend
  to require the largest blocks (8–10 samples) due to extremely persistent
  sequences.

These metrics are recorded verbatim in the CSV artefact. The JSON summary shows
the same threshold proportions and descriptive statistics for reproducibility.

## Recommendations

- **Adopt block bootstrap for uncertainty estimates.** The prevalence of strong
  serial correlation implies that the current i.i.d. bootstrap understates the
  variance. Moving-block or stationary bootstrap variants should be made
  available for nodes exhibiting `acf_lag_1 ≥ 0.5` (≈ two-thirds of the domain).
  The CLI supports this through `--resampling-mode moving_block` (or
  `stationary`) combined with `--block-length` / `--block-lengths-csv`.
- **Default block length.** A block length of 6 samples (close to the upper
  quartile) offers a conservative option while remaining computationally
  tractable. Users may provide a custom length when processing specific nodes,
  for instance using the per-node suggestions listed in
  `block_bootstrap_diagnostics.csv`.
- **Graceful fallback.** For nodes with weak autocorrelation (`acf_lag_1 < 0.3`)
  the traditional stratified i.i.d. bootstrap remains adequate, so the CLI
  should continue to expose that option.

The script can be re-run after catalog updates to refresh the diagnostics:

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -w /workspace \
  --entrypoint python \
  wind-resource-tests \
  scripts/evaluate_bootstrap_dependence.py \
  --output-dir artifacts/bootstrap_uncertainty
```

This command regenerates both the CSV and JSON artefacts, enabling periodic
monitoring of temporal dependence as new ANN inference batches become available.
