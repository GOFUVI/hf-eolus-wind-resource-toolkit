# Buoy Height Configuration

The vertical wind-speed correction applied when comparing ANN predictions with buoy observations is controlled entirely through configuration. This document summarises the available options and the physical assumptions behind them.

## Default parameters

The file [`config/buoy_height.json`](../config/buoy_height.json) stores the baseline settings:

```json
{
  "method": "log_profile",
  "measurement_height_m": 3.0,
  "target_height_m": 10.0,
  "power_law_alpha": 0.11,
  "roughness_length_m": 0.0002,
  "notes": "Default Puertos del Estado REDEXT buoy configuration using a neutral logarithmic profile (anemometer at 3 m above MSL extrapolated to 10 m)."
}
```

- `measurement_height_m` is the physical anemometer height above mean sea level (3 m in the bundled sample configuration; change it to match your buoy).
- `target_height_m` is the reference height used throughout the ANN pipeline (10 m).
- `method` selects the vertical profile model (defaults to `log_profile`):
  - `log_profile` applies the logarithmic profile \( u(z)=\frac{u_*}{\kappa}\ln(z/z_0) \), parameterised through the roughness length.
  - `power_law` applies a neutral-stability power law of the form \( u(z)=u(z_r)(z/z_r)^{\alpha} \).
- `power_law_alpha` is the exponent \\( \alpha \\) used when `method = "power_law"` (0.11 is typical for neutral offshore boundary layers).
- `roughness_length_m` is the surface roughness length `z₀` used when `method = "log_profile"` (0.0002 m for open sea conditions).

Edit this JSON to change the defaults for all executions. Invalid configuration (e.g. negative heights, unsupported methods) raises a descriptive error when the preprocessing module loads the file.

## CLI overrides

`scripts/prepare_buoy_timeseries.py` exposes optional flags that override individual parameters without modifying the JSON:

```
  --height-method {power_law,log_profile}
  --measurement-height-m FLOAT
  --target-height-m FLOAT
  --power-law-alpha FLOAT
  --roughness-length-m FLOAT
  --disable-height-correction
```

The script hydrates unspecified flags from `config/buoy_height.json`, ensuring a consistent baseline. Passing `--disable-height-correction` skips the adjustment altogether, keeping the buoy wind speeds at their original measurement height.

## Outputs and traceability

When the correction is applied:

- The processed DataFrame retains the raw measurements in the column `wind_speed_original_height`.
- The matched ANN/buoy dataset written by the CLI also includes this column for downstream diagnostics.
- Summaries produced with `--output-summary` embed the method, heights, scale factor, parameter values, and the underlying hypotheses (neutral power law or logarithmic profile over homogeneous sea surface roughness).

Refer to [`vilano_comparison.md`](./vilano_comparison.md) for the methodological context and to [`README.md`](../README.md) for a quick usage recap.
