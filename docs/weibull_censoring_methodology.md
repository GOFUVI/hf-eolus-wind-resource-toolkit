# Weibull Censoring Methodology

The parametric wind-resource stage fits Weibull distributions to the ANN-derived HF radar dataset described in `docs/sar_range_final_schema.md`. This document summarises how range-awareness metadata (labels and probabilities) feeds the censored maximum-likelihood estimation implemented in `hf_wind_resource.stats.weibull`.

## Inputs from the range-aware ANN

- **Speed proxy**: `pred_wind_speed` (m/s) is only trusted when the classifier agrees that the sample lies inside the calibrated operating band.
- **Classifier posteriors**: `prob_range_below`, `prob_range_in`, and `prob_range_above` form a simplex (see schema doc). They quantify how likely the sample is to fall below the lower threshold, within the reliable operating band, or above the upper threshold respectively.
- **Consolidated label**: `range_flag` combines classifier, deterministic speed checks, and safety margins. The boolean `range_flag_confident` signals that the winning label cleared the ANN confidence threshold from `config/range_thresholds.json`.

## Preparing censored samples

`build_censored_data_from_records` ingests raw ANN outputs and returns a `CensoredWeibullData` structure ready for likelihood evaluation. The routine receives the lower/upper limits of the ANN’s operating band (typically supplied by `config/range_thresholds.json`) so that the process remains data-driven:

1. **Confidence-aware labelling**  
   - When `range_flag_confident` is true and the associated posterior exceeds a configurable `min_confidence` (default 0.5), the observation is treated as a hard label:
     - `range_flag == "in"` → uncensored sample with weight 1 and value `pred_wind_speed`.
     - `range_flag == "below"` → left-censored observation at the configured lower threshold.
     - `range_flag == "above"` → right-censored observation at the configured upper threshold.
   - Otherwise, the posterior probabilities are used as fractional weights: `prob_range_in` scales the uncensored contribution, while `prob_range_below`/`prob_range_above` accumulate into censored counts. This preserves information from uncertain classifications without over-trusting extrapolated speeds.

2. **Weighted aggregation**  
   Identical censoring thresholds are coalesced by summing their weights, so a node with 200 below-range samples contributes a single left-censored entry with weight 200 instead of 200 duplicated rows.

3. **Validation**  
   Non-finite or non-positive speeds are dropped. Probabilities are re-normalised when floating-point drift causes their sum to deviate from unity.

The resulting dataset encodes three weighted collections: uncensored speeds, left-censored limits at the supplied lower threshold, and right-censored limits at the supplied upper threshold.

## Maximum-likelihood optimisation

`fit_censored_weibull` performs gradient-ascent MLE on the Weibull shape (`k`) and scale (`λ`) parameters:

- The log-likelihood combines contributions from:
  - **Uncensored samples**: standard Weibull density terms.
  - **Left-censored weights**: `log(F(lower_threshold))` where `F` is the Weibull CDF.
  - **Right-censored weights**: `log(1 - F(upper_threshold))` (survival function).
- Analytic gradients are available via `compute_censored_weibull_log_likelihood(..., compute_gradients=True)` and power an adaptive line-search optimiser constrained to realistic parameter ranges (shape 0.3–12, scale 0.5–40 m/s).
- The initial guess derives from the uncensored weighted mean/variance (method-of-moments). When no reliable uncensored samples are available, defaults `(k=2, λ=9)` provide a neutral starting point.

## Reliability threshold

The fit is only attempted when the total weight of uncensored observations (`in_count`) exceeds a configurable minimum (`min_in_count`, default 500). Nodes below that threshold return a `WeibullFitResult` flagged as `reliable=False`, signalling that downstream products should fall back to empirical metrics.

## Diagnostics and outputs

`WeibullFitResult` exposes:

- `shape`, `scale`, and the maximised `log_likelihood`.
- `diagnostics`: iterations performed, gradient norm at termination, final step size, and a human-readable message covering convergence, failure of the line search, or early exit due to insufficient support.
- `in_count`, `left_count`, and `right_count` for transparent accounting of the data backing the fit.
- `success` vs `reliable` flags — a fit may converge numerically yet be marked unreliable when it violates the sample threshold.

These artefacts allow reporting layers to combine parametric estimates with empirical summaries, quantify uncertainty, and document censoring behaviour per node in future dashboards.

## Non-parametric fallback

When censoring exceeds the thresholds described in
`docs/nonparametric_distribution_methodology.md`, the reporting pipeline switches
to a weighted Kaplan–Meier estimator that highlights left/right censoring mass
without enforcing a Weibull shape. The fallback artefacts include the reasons
for activating the non-parametric path so analysts can contrast both views in
dashboards and reports.
The activation thresholds are configurable via
`config/kaplan_meier_thresholds.json` to accommodate alternative censoring
policies.
