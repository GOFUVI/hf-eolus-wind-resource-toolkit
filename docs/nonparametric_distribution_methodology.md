# Non-parametric Distribution Methodology

The ANN range-aware outputs described in `docs/sar_range_final_schema.md` exhibit
substantial censoring at specific nodes (either because most samples fall below
the lower operating limit or because high winds exceed the calibrated band).
To compare wind-speed distributions under these conditions we complement the
censored Weibull fit with a weighted Kaplan–Meier estimator implemented in
`hf_wind_resource.stats.kaplan_meier`.

## Selection criteria

The non-parametric fallback is activated on a per-node basis when the censoring
pattern meets the project-defined triggers listed in
`config/kaplan_meier_thresholds.json` (also accessible via
`KaplanMeierSelectionCriteria`). Deployments can tune the minimum sample
requirement, maximum share of in-range observations, and censoring ratios in
that configuration file. By default the project uses minimums of 200 samples in
total, 0.20 for the combined censoring ratio, 0.15 for the left-censored share,
0.55 as the maximum in-range share, and 150 for the uncensored weight
threshold.

Each activation records the reasons alongside the resulting estimator so
dashboards and reports can explain why the parametric model was bypassed.

## Estimator outline

1. `build_censored_data_from_records` aggregates ANN posteriors into weighted
   uncensored speeds plus left/right censoring counts at the configured
   thresholds (`config/range_thresholds.json`).
2. `run_weighted_kaplan_meier` applies the Kaplan–Meier product-limit estimator:
   - Left-censored weights collapse into a single jump at the configured lower
     censoring limit (default 5.7 m/s).
   - Uncensored weights contribute standard events sorted by predicted wind
     speed.
   - Right-censored weights decrease the risk set while contributing to the
     residual survival probability reported as the “right-tail” mass (upper
     limit defaults to 17.8 m/s).
3. `evaluate_step_cdf` turns the step function into evenly spaced CDF samples so
   that charting layers can combine multiple nodes.

The script `scripts/generate_nonparametric_distributions.py` orchestrates the
process: it reads `artifacts/empirical_metrics/per_node_summary.csv` to locate
triggered nodes, fetches the relevant ANN columns through the official DuckDB
container, emits per-node JSON artefacts, and writes SVG comparisons grouped by
`coverage_band` and `continuity_band`. Pass `--criteria-config` to inject an
alternative JSON file when project-specific thresholds are required.

### Dockerised execution

For environments without the Python dependencies installed locally, use the
wrapper `scripts/run_nonparametric_distributions.sh`, which builds the image
`wind-resource-nonparametric` (based on `docker/nonparametric-runner/Dockerfile`)
and executes the helper inside it. Example:

```bash
scripts/run_nonparametric_distributions.sh \
    --dataset use_case/catalogs/sar_range_final_pivots_joined/assets/data.parquet \
    --summary artifacts/empirical_metrics/per_node_summary.csv
```

The script auto-detects the Docker socket (`/var/run/docker.sock` or
`$HOME/.docker/run/docker.sock`). To override the location, export
`DOCKER_SOCKET=/custom/path/docker.sock`. Use `FORCE_REBUILD=1` to refresh the
image after dependency updates, or `ALLOW_ONLINE_INSTALL=0` to force offline
wheels during the build. The wrapper also propagates the host workspace path via
`HOST_WORKSPACE_PATH` so that inner `docker run` invocations mount the correct
project directory; the default is the directory where the script is invoked.

## Limitations

- Left-censored mass is assigned to the lower operating threshold because the
  dataset does not expose individual censoring limits. The Kaplan–Meier curve
  therefore represents a conservative bound: actual winds could be lower than
  the reported jump.
- The estimator handles right-censoring but cannot recover quantiles beyond the
  upper threshold when the right-tail probability remains > 0. The accompanying
  summary files flag such cases so downstream consumers can fall back to
  scenario analysis or empirical percentiles.
- Nodes with extremely sparse support (total observations < 200) remain outside
  the fallback to avoid over-interpreting handfuls of samples. Extend the
  selection criteria cautiously if future datasets improve coverage.

Despite these bounds, the Kaplan–Meier outputs provide a robust comparison tool
for heavily censored nodes, enabling analysts to highlight differences across
taxonomy bands without assuming a specific parametric family.
