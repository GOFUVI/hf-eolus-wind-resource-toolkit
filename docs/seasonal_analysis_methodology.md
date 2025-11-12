# Seasonal and Interannual Variability Methodology

This note formalises the procedure implemented in `hf_wind_resource.stats.seasonal`
and orchestrated by `scripts/generate_seasonal_variations.py`. The goal is to
quantify seasonal variability and long-term trends in the ANN-derived wind-speed
series while respecting the censoring semantics of the HF-radar inversion.

## Data inputs

The analysis consumes four columns from the ANN GeoParquet snapshot
`sar_range_final_pivots_joined`: the UTC timestamp `t`, node identifier `n`,
predicted 10 m wind speed `v` (`pred_wind_speed`), and raw range label `ℓ`
(`pred_range_label`). Timestamps are normalised to UTC using
`pandas.to_datetime(..., utc=True)`.

### Label normalisation

Each raw label is mapped to the canonical set
\{`below`, `in`, `above`, `uncertain`\} via

```
L = f(ℓ) = 
  ┌ below      if ℓ ∈ {below, below_range, under, left}
  ├ in         if ℓ ∈ {in, inside, within, in_range}
  ├ above      if ℓ ∈ {above, over, upper, right}
  └ uncertain  otherwise or when ℓ is missing
```

All downstream ratios and sample extractions refer to this canonical label
`L`.

## Seasonal aggregation

Let the meteorological season function `S(month)` be defined as:

```
S(m) =  ┌ DJF if m ∈ {12, 1, 2}
        ├ MAM if m ∈ {3, 4, 5}
        ├ JJA if m ∈ {6, 7, 8}
        └ SON if m ∈ {9, 10, 11}
```

For each node `n` and season `s`, the script considers the subset

```
𝒟ₙₛ = { vᵢ | node_idᵢ = n, S(month(tᵢ)) = s, Lᵢ = in, vᵢ is finite }.
```

The following statistics are computed whenever `𝒟ₙₛ` is non-empty:

- Seasonal sample count `Nₙₛ = |𝒟ₙₛ|`.
- Seasonal mean
  `μₙₛ = (1/Nₙₛ) ∑_{v∈𝒟ₙₛ} v`.
- Seasonal population standard deviation
  `σₙₛ = sqrt( (1/Nₙₛ) ∑_{v∈𝒟ₙₛ} (v - μₙₛ)² )`.
- Percentiles `P₅₀`, `P₉₀`, and `P₉₉` using the linear interpolation method
  (`pandas.Series.quantile` with defaults).

### Censoring ratios

In parallel, the canonical label counts per `(n, s)` are accumulated:

```
Cₙₛ(label) = |{ i | node_idᵢ = n, S(month(tᵢ)) = s, Lᵢ = label }|.
```

The total seasonal count is `Tₙₛ = ∑_{label} Cₙₛ(label)` and ratios are
reported as `Cₙₛ(label) / Tₙₛ` (or `None` when `Tₙₛ = 0`). The combined
censoring ratio equals `ratio(below) + ratio(above)`.

All seasonal rows are emitted in chronological order of the seasons
(DJF, MAM, JJA, SON) with string node identifiers.

## Interannual aggregation

Annual groups use calendar years. For each node `n` and year `y`:

```
𝒜ₙᵧ = { vᵢ | node_idᵢ = n, year(tᵢ) = y, Lᵢ = in, vᵢ is finite }.
```

From `𝒜ₙᵧ` we compute `Nₙᵧ`, `μₙᵧ`, `σₙᵧ`, and the same percentiles as for
seasons. Annual rows are sorted by `(node_id, year)`.

## Node-level summaries

Let `𝒮ₙ = { μₙₛ | μₙₛ is defined }` and `𝒴ₙ = { (y, μₙᵧ) }`.

- **Seasonal amplitude**:
  `Aₙ = max(𝒮ₙ) - min(𝒮ₙ)` when at least one seasonal mean exists.
- **Seasonal mean standard deviation**:
  `STDₙ = std(𝒮ₙ)` (population definition).
- **Seasonal coverage**:
  number of seasons with `μₙₛ` available (0–4).
- **Strongest / weakest seasons**:
  argmax/argmin of `μₙₛ` when available.

### Annual trend slope

If at least two annual means are available, the script fits an ordinary
least-squares line to `(y, μₙᵧ)` ignoring `NaN` values:

```
μₙᵧ ≈ αₙ + βₙ · y,
```

where `βₙ` is the reported trend (m/s per year). When fewer than two valid
annual means exist, `βₙ` is set to `None` and an explanatory note is recorded
(`"Only one year available."` or `"No annual samples available."`).

All summaries include the trend note, the number of annual samples, and the
unit string `"m/s per year"` for clarity.

## Output artefacts

Running `scripts/generate_seasonal_variations.py` yields:

- `seasonal_slices.csv`: per-node, per-season metrics and label ratios.
- `annual_slices.csv`: per-node, per-year statistics.
- `variation_summary.csv`: node-level amplitudes, coverage, and trend slopes.
- `seasonal_analysis_summary.json`: machine-readable overview with medians,
  top amplitudes, strongest positive/negative trends, seasonal gaps, and the
  `height_*` metadata read from `artifacts/power_estimates/metadata.json`
  (when present).
- `seasonal_variation_summary.md`: a quick-look Markdown digest placed in
  `artifacts/seasonal_analysis/`.

## Execution modes

By default the script resolves the STAC dataset using `config/stac_catalogs.json`
and executes the DuckDB query inside the `duckdb/duckdb:latest` container
(`--execution-mode docker`). For controlled environments (e.g., the test image)
the query can be executed via the embedded DuckDB Python package
(`--execution-mode python`). Both modes yield identical statistics; the choice
affects only how the GeoParquet is read.

## Reproducibility considerations

- Seasonal and annual statistics only use samples labelled `in` and with
  finite predicted wind speed, preserving the ANN-defined operating range.
- Seasonal ratios expose the proportion of left- and right-censored samples
  so downstream reporting can qualify the reliability of seasonal means.
- Trends are simple linear fits and should be interpreted as exploratory;
  uncertainty bounds are not yet reported. When stricter inferences are
  required, the seasonal module can be extended to propagate bootstrap
  variance using the same `hf_wind_resource.stats.bootstrapping` primitives.

The methodology is fully deterministic once the snapshot and configuration
files are fixed, enabling regeneration through the same script.

## Extension to power-density and turbine aggregates

`scripts/generate_power_estimates.py` reuses the preceding pipeline to derive energy-oriented metrics per node and period. For each subset (monthly or seasonal) it performs:

1. **Censored-data construction**
   Node observations are filtered with `build_censored_data_from_records`, applying the posterior probabilities `prob_range_*` and the configured confidence threshold (`min_confidence`). This yields a `CensoredWeibullData` object containing the in-range, left-censored, and right-censored weights.

2. **Estimator selection**
   A censored Weibull fit (`fit_censored_weibull`) is attempted first. If the fit is unreliable or fails the censoring criteria defined in `KaplanMeierSelectionCriteria`, the pipeline falls back to the Kaplan–Meier estimator (`run_weighted_kaplan_meier`). The summary consumed by `evaluate_kaplan_meier_selection` is computed from the weighted probabilities

   \
   \text{total} = \sum_i 1,\quad\n   \text{below\_ratio} = \frac{\sum_i w_i^{\text{below}}}{\text{total}},\quad\n   \text{in\_ratio} = \frac{\sum_i w_i^{\text{in}}}{\text{total}},\quad\n   \text{censored\_ratio} = 1 - \text{in\_ratio}\n   \]

   where \(w_i^{\text{below}}, w_i^{\text{in}}, w_i^{\text{above}}\) are treated deterministically when the classifier is confident (probability ≥ `min_confidence`) and otherwise correspond to the full posterior probabilities.

3. **Power computation**
   - **Power density (W/m²):** For the Weibull fit we evaluate \(½·ρ·(k^{−3/k})·Γ(1+3/k)·λ³\), where \(k\) and \(λ\) are the fitted parameters and the height-correction factor (`speed_scale`) is applied. For the Kaplan–Meier estimator we integrate the survival function using \(½·ρ·\sum_j p_j·(s_j·\text{scale})^3\), adding an optional right-tail surrogate when censored mass remains.
   - **Reference power curve:** The helpers `estimate_power_curve_from_*` interpolate the tabulated power curve, adjust it for air-density differences, and provide the expected turbine power (kW) together with the capacity factor.

4. **Metadata**
   Each row records `period_type` (`monthly` or `seasonal`), timestamps (`period_start`), censored weights, the selected method, the applied height correction (`height_*`), and explanatory notes about Kaplan–Meier activation.

The generated artefacts are:

- `artifacts/power_estimates/seasonal_power_summary.csv`: seasonal aggregates per node (DJF/MAM/JJA/SON) and meteorological year.
- `artifacts/power_estimates/monthly_power_timeseries.csv`: monthly time series per node (`year`, `month`, `period_start`) prepared for ARIMA/ETS analysis.

## Preparation for ARIMA/ETS models

The monthly time series are published per node with:

- Average power (`turbine_mean_power_kw`) and capacity factor.
- Power density (`power_density_w_m2`).
- Censoring weights (`uncensored`, `left`, `right`) that assess reliability.

Before fitting ARIMA/ETS, the series are segmented into continuous sub-periods: gaps shorter than `max_gap_months` are filled via linear interpolation (`gap_filled=True` in the artefacts), whereas gaps of `max_gap_months` or more create a new segment. This prevents the models from learning unrealistic trends over data-free intervals and lets each segment reflect the observational support available. Default thresholds reside in `config/time_series.json`.

Recommendations:

1. **Check continuity:** remove or explicitly flag long sequences without data before modelling.
2. **Difference and de-seasonalise when necessary:** apply seasonal differencing (lag 12) if residual trends persist after preprocessing.
3. **Propagate uncertainty:** combine forecasts with bootstrap intervals (`artifacts/bootstrap_uncertainty/`) or use the censoring weights to down-weight segments with limited support.

## Time-series modelling (ARIMA / ETS)

For each continuous segment \(\{y_t\}_{t=1}^T\) of the monthly turbine-power series we fit a seasonal ARIMA model and an additive ETS model. The process is as follows.

### Segmentation and preprocessing

1. Load the parameters `max_gap_months` and `min_segment_months` from `config/time_series.json` (defaults are 6 and 36 respectively). CLI flags `--max-gap-months` and `--min-observations` override these values when required.
2. Reindex the series on a regular monthly grid (`freq='MS'`). Gaps shorter than `max_gap_months` are filled via linear interpolation; larger gaps create independent subseries. The boolean `gap_filled` marks whether the surviving segment contains interpolated values.
3. Discard segments that do not reach `min_segment_months` valid observations, ensuring stable parameter estimation.

### Seasonal ARIMA model

The seasonal ARIMA explored by `fit_sarima_auto` satisfies
\[
\Phi_P(B^s) \phi_p(B) (1-B)^d (1-B^s)^D y_t = \Theta_Q(B^s) \theta_q(B) \varepsilon_t,\quad \varepsilon_t \sim \text{WN}(0,\sigma^2),
\]
where \(B\) is the backshift operator, \(s = 12\) is the seasonal period, and \((p,d,q)\) and \((P,D,Q)\) are the non-seasonal and seasonal orders. The grid search covers \(p,q \in \{0,1\}\) and \(P,Q \in \{0,1\}\) while \(d, D\) are chosen heuristically (testing first-differences and seasonal differences for variance reductions). For each candidate we maximise the Gaussian log-likelihood \(L\) and evaluate:

- Akaike information criterion: \(\text{AIC} = -2 \log L + 2k\).
- Small-sample adjustment: \(\text{AIC}_c = \text{AIC} + \frac{2k(k+1)}{T - k - 1}\), with \(k\) parameters and \(T\) observations.
- Bayesian information criterion: \(\text{BIC} = -2 \log L + k \log T\).

The model with smallest \(\text{AIC}_c\) is retained. Residual diagnostics include:

- Autocorrelation (ACF) and partial autocorrelation (PACF) up to \(\min\{24, \lfloor (T-1)/2 \rfloor\}\) lags (statsmodels constrains PACF to half the sample length).
- Ljung–Box \(Q\)-test at lag \(\min\{12, T-1\}\); the resulting \(p\)-value is reported in the artefacts.

### Additive ETS (Holt–Winters)

We use the additive Holt–Winters formulation
\[
\begin{aligned}
\ell_t &= \alpha (y_t - s_{t-s}) + (1-\alpha)(\ell_{t-1} + b_{t-1}),\\
b_t &= \beta (\ell_t - \ell_{t-1}) + (1-\beta)b_{t-1},\\
s_t &= \gamma (y_t - \ell_{t-1} - b_{t-1}) + (1-\gamma)s_{t-s},
\end{aligned}
\]
with level \(\ell_t\), trend \(b_t\), seasonal component \(s_t\), and monthly seasonal period \(s=12\). Parameters \(\alpha,\beta,\gamma\) are obtained by maximising the Gaussian likelihood. We record \(\text{AIC}\), \(\text{AIC}_c\), \(\text{BIC}\), the Ljung–Box \(p\)-value, and the forecast horizon.

### Artefacts and interpretation

- Each node produces `artifacts/power_estimates/time_series/nodes/<node_id>.json` containing all segments, their time spans, whether interpolation was applied, the selected model (ARIMA or ETS), and the forecast vectors of length `forecast_steps`.
- `time_series_summary.csv` consolidates per-segment diagnostics (orders, information criteria, Ljung–Box statistic, number of observations, `max_gap_months`, `min_segment_months`).
- Segments that fail to converge (e.g. statsmodels raises `ConvergenceWarning`) are skipped; the warning is printed to the console so the analyst can adjust orders or segment thresholds if required.

Residual analysis should guide any further refinement: significant Ljung–Box statistics suggest increasing the AR/MA orders or shortening the segment, and large `gap_filled` segments may call for caution when interpreting forecasts since part of the signal was interpolated.

 > **Dependencies:** The script `scripts/generate_time_series_models.py` requires `statsmodels` (bundled in the test Docker image). When the library is missing, the affected nodes are skipped automatically and a warning is displayed.
 