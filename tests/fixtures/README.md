# Synthetic SAR Range Fixtures

This folder hosts deterministic fixtures that exercise the range-awareness
logic of the HF radar ANN outputs. They are small enough to load entirely in
unit tests while still covering all range label combinations (`below`, `in`,
`above`) and the interaction between the classifier and deterministic labels
derived from `pred_wind_speed`.

- `sar_range_final_synthetic.csv`: canonical fixture generated with the
  script. Values are produced with the pseudo-random seed `20241019`, so the
  layout and numeric values remain reproducible.
- `sar_range_final_synthetic.parquet`: Parquet representation where `geometry`
  is stored as a WKB BLOB, matching the real GeoParquet snapshot. The generator
  invokes DuckDB via the official Docker image to materialise this file.
- `sar_range_final_synthetic_summary.json`: coverage metadata highlighting how
  many samples exist per range label and per `(pred_range_label,
  pred_speed_range_label)` combination.
- `build_sar_range_final_synthetic.py`: generator that materialises the CSV
  and summary files. It keeps cadence variations (`30 min`, `60 min`, and
  irregular offsets) and emits limit cases right at the inversion bounds
  (5.7 m/s and 17.8 m/s).
- `stats_synthetic_datasets.json`: deterministic statistical fixtures (Weibull
  reference, fully censored series and RMSE stress records) generated with the
  pseudo-random seeds documented in `tests/stats/test_power_pipeline_integration.py`.
  They exercise the full power-distribution pipeline and the bootstrap module.

Regeneration steps:
1. `python3 tests/fixtures/build_sar_range_final_synthetic.py`
   - The script regenerates the CSV, updates the JSON summary and, by default,
     calls the DuckDB Docker image to refresh the Parquet representation with
     binary geometry.
2. If Docker is not available, use `--skip-parquet` when running the script and
   execute the logged DuckDB command manually once Docker access is restored.

The fixtures intentionally keep geometry values as WKB hex strings to mirror
the GeoParquet encoding documented in `docs/sar_range_final_schema.md` while
remaining editable in version control.
