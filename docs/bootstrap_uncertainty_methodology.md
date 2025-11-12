# Bootstrap Uncertainty Methodology

This note formalises the bootstrap machinery implemented in
`src/hf_wind_resource/stats/bootstrapping.py` and executed through
`scripts/generate_bootstrap_uncertainty.py`. The goal is to quantify sampling
uncertainty of node-wise wind-resource functionals while propagating the global
root-mean-square error (RMSE) delivered by the ANN inversion. The exposition is
organised as follows: notation (§1), resampling scheme (§2), interval
construction (§3), RMSE propagation options (§4), practical usage (§5), and the
block-bootstrap readiness checklist (§6).

## 1. Notation

Let a node expose a finite set of ANN inference records
\(\mathcal{D} = \{(v_i, \mathbf{p}_i, f_i, c_i)\}_{i=1}^n\), where:

- \(v_i \in \mathbb{R}_{\ge 0}\) is the predicted wind speed at the source height.
- \(\mathbf{p}_i = (p_{i}^{\text{below}}, p_{i}^{\text{in}}, p_{i}^{\text{above}})\) are
  posterior probabilities of the ANN range classifier (normalised so the sum is
  one).
- \(f_i \in \{\text{below}, \text{in}, \text{above}\}\) is the discrete flag and
  \(c_i \in \{0,1\}\) encodes its confidence.

Define the canonical statistic of interest as
\(T(\mathcal{D}) = (\mu, q_{0.5}, q_{0.9}, q_{0.99}, P)\), comprising mean wind
speed, empirical quantiles, and mean power density obtained by executing the
deterministic pipeline (Kaplan–Meier/Weibull selection, height adjustment, power
curve evaluation).

## 2. Stratified bootstrap

The algorithm builds i.i.d. replicates \(T^{(b)}\) of the statistic using three
layers:

1. **Stratification**. Records are assigned to the canonical labels
   \(L_i \in \{\text{below}, \text{in}, \text{above}, \text{uncertain}\}\)
   according to the probabilities \(\mathbf{p}_i\) and the confidence flag
   \(c_i\). The aggregate counts
   \(N_\ell = \sum_{i=1}^n \mathbb{1}\{L_i=\ell\}\) are retained as descriptive
   diagnostics. This explicit stratification fulfils the first recommendation in
   `audits/bootstrap-methodology.md` by locking the observed mixture inside every
   bootstrap replicate instead of merely preserving it in expectation.

2. **Resampling**. For every bootstrap replicate \(b = 1,\dots,B\) we sample with
   replacement *within each stratum*, drawing exactly \(N_\ell\) observations
   from bucket \(\ell\). The concatenated sample keeps the proportion of
   in-range, below-range, above-range, and uncertain records identical to the
   original dataset. When `--resampling-mode` switches to `moving_block` or
   `stationary`, a block bootstrap is performed on the time-ordered records and
   the label strategy (fixed or resampled) is re-applied afterwards so that the
   final replica honours the stratified counts.

3. **Evaluation**. Each \(\mathcal{D}^{(b)}\) is processed by the same functional
   \(T\) used on the original dataset (including height correction and conditional
   Kaplan–Meier/Weibull selection). The resulting vector \(T^{(b)}\) is stored.

The collection \(\{T^{(b)}\}_{b=1}^B\) provides consistent estimates of the
sampling distribution of \(T(\mathcal{D})\) under mild regularity conditions
[Efron & Tibshirani (1993), Ch. 6] and feeds the interval estimators described
in §3.

### 2.1 Label uncertainty strategies

By default the CLI uses `--label-strategy fixed`, which keeps the empirical
range assignments unchanged and performs stratified sampling within the four
buckets. Passing `--label-strategy label_resample` activates a Monte Carlo stage
in which every bootstrap replicate redraws the label of each record according to
its posterior probabilities \(\mathbf{p}_i\). Samples that exceed the confidence
threshold (`--min-confidence`) remain in the deterministic bucket, whereas
low-confidence cases become discrete draws (\(\text{below}\),
\(\text{in}\), \(\text{above}\)) with unit weights. This reproduces the
censoring variability highlighted in `audits/bootstrap-methodology.md` and
typically widens the confidence intervals because each replicate captures a
different censoring pattern.

### 2.2 Bias correction

Finite-sample effects (non-linear transformations, clamping, etc.) introduce
bias between the bootstrap mean \(\bar{T}_k = B^{-1}\sum_b T_k^{(b)}\) and the
deterministic estimator \(T_k(\mathcal{D})\). Following the “centred bootstrap”
prescription [Efron & Tibshirani (1993), §10.1], we subtract the empirical bias
from the replicates before computing intervals:
\[
\tilde{T}_k^{(b)} = T_k^{(b)} - \bigl(\bar{T}_k - T_k(\mathcal{D})\bigr).
\]
The transformation keeps the shape (variance, skewness) of the bootstrap
distribution but re-centres it exactly at \(T_k(\mathcal{D})\), ensuring the
intervals are anchored to the deterministic value.

## 3. Interval construction

Bootstrap replicates are summarised into confidence intervals via the selector
`StratifiedBootstrapConfig.ci_method`, which implements the audit guidance in
`audits/bootstrap-methodology.md` about favouring bias-aware intervals:

- **`percentile`** *(default)*. Uses the empirical quantiles of the
  re-centred draws (§2.2). Replicates contaminated by NaN or infinities are
  discarded prior to calculating the interval and the bootstrap mean.
- **`bca`**. Applies the bias-corrected and accelerated transformation from
  Efron (1987). We build delete-one jackknife replicas within each stratum (up
  to `jackknife_max_samples`) so that the acceleration factor respects the
  censoring structure. If the jackknife cannot be computed (e.g. large nodes or
  degenerate statistics), the code records a diagnostic note and transparently
  falls back to the percentile interval.
- **`percentile_t`**. Uses the same stratified jackknife to estimate the
  Studentised distribution of each metric. When the jackknife variance is zero
  or insufficient samples are available, the routine again degrades to the
  percentile interval with an explicit note in the metadata.

Selecting BCa or percentile-t requires additional computation because each
jackknife delete-one dataset must be re-evaluated by the full deterministic
pipeline. The cap imposed by `jackknife_max_samples` keeps this tractable while
still enabling second-order corrections on moderately sized nodes. Regardless
of the chosen method, ensure the number of bootstrap replicates \(B\) is large
enough (≳500) so that quantiles are stable, as emphasised in the audit memo.

## 4. RMSE propagation

The global RMSE \(\sigma_\text{ANN}\) summarises the mean-squared deviation
between ANN predictions and ground truth at the reference height. The CLI offers
three propagation modes:

1. **`velocity`** (default). Speeds are perturbed prior to resampling:
   \(v_i \mapsto v_i + \varepsilon_i\) with
   \(\varepsilon_i \sim \mathcal{N}(0, \sigma_\text{ANN}^2)\). The perturbed
   values are clamped to the operational range `[lower_threshold, upper_threshold]`. The
   resulting bias on the power density (due to the cubic moment) is corrected by
   subtracting the analytical term \(1.5\,\rho\,(\sigma_\text{ANN}\,s)^2\,\bar{v}\),
   where \(s\) denotes the height scaling factor and \(\bar{v}\) the bootstrap
   mean. Residual bias is removed by re-centring the replicates as described in
   §2.2.

2. **`power`**. Speeds are left untouched. Each replicate draws a Gaussian term
   \(\delta^{(b)} \sim \mathcal{N}(0, \sigma_P^2)\) that is added exclusively to
   the power density: \(P^{(b)} \gets \max(P^{(b)} + \delta^{(b)}, 0)\). The
   standard deviation \(\sigma_P\) is derived via error propagation, using the
   derivative \(\frac{\partial P}{\partial v} = 1.5\,\rho\,v^2\), and is scaled by
   the effective sample size (total weight) to reflect that the RMSE is reported
   per observation. This implements the audit recommendation on RMSE
   propagation for power-density metrics.

3. **`none`**. Disables RMSE perturbations entirely; the bootstrap captures only
   sampling variability.

In all cases, negative power densities produced by the perturbations are clamped
to zero to maintain physical interpretability.

## 5. Ejecución y artefactos

```
python3 scripts/generate_bootstrap_uncertainty.py \
  --replicas 1000 \
  --confidence 0.95 \
  --seed 20251022 \
  --power-curve-config config/power_curves.json \
  --power-curve-key reference_offshore_6mw \
  --min-confidence 0.5 \
  --min-in-range 500 \
  --rmse-mode velocity \
  --resampling-mode iid \
  --workers 4 \
  --progress-interval 10 \
  --resume
```

The command above processes all nodes, computing \(B = 1000\) replicates, logs
progress every ten nodes, and supports resuming (`--resume`) thanks to the
incremental file `artifacts/bootstrap_uncertainty/bootstrap_results.jsonl`. The
flag `--label-strategy label_resample` can be added to inject label resampling
for the range classifier when the censoring probabilities carry substantial
uncertainty. To enable the moving-block or stationary bootstrap, replace
`--resampling-mode` with `moving_block` (or `stationary`) and provide suitable
block lengths, e.g.
`--block-length 6 --block-lengths-csv artifacts/bootstrap_uncertainty/block_bootstrap_diagnostics.csv`.
The summary is written to `bootstrap_summary.csv`, and metadata (including RMSE
mode, worker count, and timestamp) is stored in `bootstrap_metadata.json`.

These artefacts feed downstream reporting layers (node-level tables, uncertainty
maps) and contain sufficient provenance (RMSE version, power curve, Kaplan–Meier
criteria, etc.) to reproduce the analysis after catalogue updates.

## 6. Block-bootstrap readiness check

Temporal dependence was assessed with the companion script
`scripts/evaluate_bootstrap_dependence.py`, which queries the full ANN GeoParquet
snapshot and reports lagged autocorrelations for every node. The resulting
artefacts live in `artifacts/bootstrap_uncertainty/`:

- `block_bootstrap_diagnostics.csv`: node-level autocorrelations, effective
  sample sizes and suggested moving-block lengths derived from the AR(1)
  approximation.
- `block_bootstrap_summary.json`: aggregated thresholds and descriptive
  statistics (share of nodes with `acf_lag_1 ≥ 0.3/0.5/0.7`, mean and median
  block length, etc.).

On the current dataset (2025-10-25) 93 % of the nodes exhibit
`acf_lag_1 ≥ 0.3`, 66 % exceed `acf_lag_1 ≥ 0.5`, and the median suggested block
length is four samples (maximum ten). Consequently, a moving-block bootstrap is
recommended whenever the stratified i.i.d. resampler is engaged for ANN
uncertainty. Section [Block Bootstrap Assessment](block_bootstrap_assessment.md)
documents the procedure and should be consulted before altering the default
strategy.

## Referencias

- B. Efron & R. Tibshirani (1993). *An Introduction to the Bootstrap*. Chapman
  & Hall. Chapters 6 and 10.
- P. Hall (1992). *The Bootstrap and Edgeworth Expansion*. Springer.
