# Power Estimation Methodology

This document describes how the wind-resource pipeline derives theoretical power
indicators from the HF radar ANN inference snapshot documented in
`docs/sar_range_final_schema.md`. The goal is to obtain comparable metrics across
nodes that quantify:

- **Mean power density** (W/m²), expressing the kinetic energy flux of the wind
  irrespective of any turbine model; and
- **Expected turbine production** (kW and capacity factor) using a configurable
  reference power curve representative of an offshore machine.

Both products are designed to be reproducible offline and do not attempt to
forecast the performance of a specific wind farm. They should be reported as
theoretical estimates conditioned on the ANN inversion and the assumptions
outlined below.

## Statistical inputs

- **Censored Weibull fit:** When a node has sufficient in-range support
  (`min_in_range` in the script, default 500 observations) and the optimisation
  converges, we use the censored maximum-likelihood parameters described in
  `docs/weibull_censoring_methodology.md`. The third raw moment of the Weibull
  distribution yields a closed-form expression for mean power density:

  \[
  \mathbb{E}[P_\text{density}] = \tfrac{1}{2}\,\rho\,\lambda^3 \,\Gamma\!\left(1 + \frac{3}{k}\right)
  \]

  where \(k\) and \(\lambda\) are the shape and scale parameters, \(\rho\) is the
  air density, and \(\Gamma\) denotes the gamma function.

- **Kaplan–Meier fallback:** Nodes with heavy censoring trigger the non-parametric
  estimator described in `docs/nonparametric_distribution_methodology.md`. Since
  the right tail remains undefined when samples are right-censored, we approximate
  the missing mass at the ANN upper threshold (configurable through the
  `--right-tail-surrogate` argument). This produces a **lower-bound** power
  density because any true winds above the surrogate would increase the flux.

The selection thresholds default to `config/kaplan_meier_thresholds.json` and can
be overridden per run.

## Parametric quality assessment

The CLI now benchmarks the censored Weibull fit against additional parametric
families to document how sensitive each node is to the chosen model.

- **Log-normal candidate:** parameters are obtained via weighted moments on the
  logarithm of the in-range observations. The censored log-likelihood is then
  recomputed using left/right probabilities so that censored records still
  contribute to the fit.
- **Gamma candidate:** uses weighted moment matching (mean/variance) over the
  in-range subset and evaluates the censored log-likelihood and CDF via
  `scipy.special.gammainc`. Pass `--disable-gamma-fit` when SciPy is not
  available inside the execution environment.
- **Diagnostics:** each candidate records its log-likelihood, AIC, BIC, and a
  weighted Kolmogorov–Smirnov statistic computed on the in-range observations.
  The KS p-value is approximate because the inputs carry fractional weights, but
  it provides a quick sanity check when comparing nodes.
- **Selection:** `--parametric-selection-metric` controls whether AIC or BIC
  decides the preferred model. The default (`bic`) favours parsimonious fits.
  The CLI records this choice in `parametric_preferred_model` together with the
  metric value and free-form notes summarising any skips or warnings.
- **Sample guards:** `--parametric-min-in-weight` enforces the minimum in-range
  weight required before evaluating alternative fits (defaults to 200), while
  `--parametric-ks-min-weight` controls the minimum support needed for KS
  diagnostics (defaults to 75). Nodes that do not meet the thresholds record the
  skip reason in the summary so dashboards can filter them out explicitly.

The resulting columns (`weibull_aic`, `lognormal_mu`, `gamma_parametric_notes`,
etc.) are included in `artifacts/power_estimates/power_estimates_summary.csv`,
the per-node JSON payloads, and the STAC publication so downstream consumers can
justify why a given parametric family was preferred.

## Reference power curve

- The default curve (`config/power_curves.json`, key `reference_offshore_6mw`)
  approximates a contemporary **6 MW offshore turbine**: cut-in at 3 m/s, rated
  power reached near 13 m/s, cut-out at 25 m/s. The curve is synthetic but
  conservatively shaped to avoid overestimation.
- Power values are tabulated in kilowatts for an air density of **1.225 kg/m³**
  at sea level and a hub height of 110 m. The script scales the curve linearly
  by the ratio between the requested air density and the reference value; this
  is an industry-standard first-order correction, but analysts should document
  deviations when site-specific densities are available.
- Expected turbine output and capacity factor are computed by integrating the
  power curve against the selected wind distribution (Weibull or Kaplan–Meier).
  For the Weibull route we numerically integrate the PDF against the tabulated
  curve; for Kaplan–Meier we sum the discrete support points plus the surrogate
  tail. Both methods treat the results as **theoretical** (no wake losses,
  availability, or control strategies are modelled).

## Outputs and reproducibility

`scripts/generate_power_estimates.py` produces:

- `artifacts/power_estimates/power_estimates_summary.csv` with per-node metrics,
  diagnostic flags, and the notes backing each estimate;
- `artifacts/power_estimates/nodes/<node_id>.json` with the detailed inputs,
  Weibull/Kaplan–Meier diagnostics, and turbine assumptions;
- `artifacts/power_estimates/metadata.json` summarising configuration choices,
  timestamps, and the reference curve used.

By default the script launches the official `duckdb/duckdb:latest` container to
run SQL against the GeoParquet archive. When executing inside the project’s
`wind-resource-tests` image (which lacks the Docker CLI), pass `--engine python`
so the local DuckDB module handles the queries instead.

Height-extrapolation defaults are stored in `config/power_height.json` (method,
source/target heights, log-law roughness or power-law exponent). CLI flags such
as `--height-method`, `--target-height-m`, or `--power-law-alpha` override the
file on demand, and the resulting metadata is embedded in the summary and per-node
JSON artefacts for traceability.

All metrics should be quoted alongside the air-density assumption, the reference
curve identity, and the chosen surrogate speed for right-censored mass so the
theoretical nature of the figures remains explicit. Analysts are encouraged to
contrast the Weibull and Kaplan–Meier outputs, especially for sparsely sampled
nodes, before drawing operational conclusions.
