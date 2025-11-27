# SAR Range Final Pivots Joined Schema

## Dataset overview
- Source: HF radar observations from VILA and PRIO stations processed with the SAR range-aware ANN inversion pipeline (see `docs/hf_dev_plan.md` and Zenodo DOI 10.5281/zenodo.17464583).
- Table name: `sar_range_final_pivots_joined` (STAC table extension).
- Coverage: 1,158,085 rows spanning 2011-09-30T07:30:00Z to 2023-05-10T07:30:00Z, bounding box [-10.1651, 43.0180, -8.1672, 44.4621] in CRS84.
- Row count validated on 2025-10-18 with DuckDB (`duckdb/duckdb:latest`):
  `docker run --rm -v "$(pwd)":/workspace -w /workspace duckdb/duckdb:latest duckdb -cmd "SELECT COUNT(*) FROM read_parquet('use_case/catalogs/sar_range_final_pivots_joined/assets/data.parquet');"`
- Temporal cadence: nominal 30-minute steps per node (gaps possible when either radar fails quality controls).
- Spatial support: each record represents a mesh node (≈20 km spacing) with geometry provided as WKB polygons.

## Access paths and storage
- STAC catalog: `use_case/catalogs/sar_range_final_pivots_joined/collection.json` → `items/data.json` → `assets/data.parquet`.
- Local asset snapshot: `use_case/catalogs/sar_range_final_pivots_joined/assets/data.parquet` (GeoParquet, 355 MiB, timestamped 2024-10-18 07:44 local).

## Table design and keys
- Logical primary key: `(timestamp, node_id)`; geometry is constant per node and stored redundantly for spatial joins.
- Primary geometry: `geometry` (WKB, CRS84) as advertised in `table:primary_geometry` within the STAC item.
- Relationships: single-table dataset. References to other products are implicit via the shared `node_id` used throughout HF-EOLUS catalogs.
- Bragg peak handling: radar statistics are pivoted into paired columns with suffix `_0` (negative Bragg peak, `pos_bragg=0`) and `_1` (positive Bragg peak, `pos_bragg=1`).

## Column reference
The following tables transcribe the STAC column metadata, adding unit summaries for convenience.

### Core columns
| Column | Type | Units | Description |
| --- | --- | --- | --- |
| `geometry` | binary | WKB | Cell geometry encoded as WKB in CRS84 (lon/lat). |
| `node_id` | string | unitless | Spatial grid node identifier (unitless). |
| `prio_bearing` | number | degrees | PRIO: bearing from station to cell centroid, degrees clockwise from geographic north. |
| `prio_dist_km` | number | km | PRIO: great-circle distance from station to cell centroid, in kilometres. |
| `timestamp` | datetime | UTC | Observation timestamp (UTC). |
| `vila_bearing` | number | degrees | VILA: bearing from station to cell centroid, degrees clockwise from geographic north. |
| `vila_dist_km` | number | km | VILA: great-circle distance from station to cell centroid, in kilometres. |

### Predicted wind outputs
| Column | Type | Units | Description |
| --- | --- | --- | --- |
| `pred_cos_wind_dir` | number | unitless | Cosine component of the predicted wind direction unit vector (unitless). |
| `pred_range_confidence` | number | unitless | Confidence associated with the predicted range label (0-1). |
| `pred_range_label` | string | unitless | Discrete range class selected by the classifier (below/in/above). |
| `pred_sin_wind_dir` | number | unitless | Sine component of the predicted wind direction unit vector (unitless). |
| `pred_speed_range_label` | string | unitless | Range class inferred deterministically from the predicted wind speed (below/in/above). |
| `pred_wind_direction` | number | degrees | Model prediction for wind direction (degrees clockwise from geographic north). |
| `pred_wind_speed` | number | m/s | Model prediction for wind speed (m/s). |

### Range probabilities
| Column | Type | Units | Description |
| --- | --- | --- | --- |
| `prob_range_above` | number | probability | Posterior probability that wind speed lies above the calibrated operating range (0-1). |
| `prob_range_below` | number | probability | Posterior probability that wind speed lies below the calibrated operating range (0-1). |
| `prob_range_in` | number | probability | Posterior probability that wind speed lies within the calibrated operating range (0-1). |

### Range flags
| Column | Type | Units | Description |
| --- | --- | --- | --- |
| `range_flag` | string | unitless | Consolidated range-awareness flag (below/in/above/uncertain). |
| `range_flag_confident` | boolean | unitless | True when the range-awareness flag exceeds the configured confidence threshold (default 0.5 from `config/range_thresholds.json`). |
| `range_near_any_margin` | boolean | unitless | Flag indicating the predicted wind speed lies near any reliability margin. |
| `range_near_lower_margin` | boolean | unitless | Flag indicating the predicted wind speed lies near the lower reliability margin. |
| `range_near_upper_margin` | boolean | unitless | Flag indicating the predicted wind speed lies near the upper reliability margin. |
| `range_prediction_consistent` | boolean | unitless | Indicates whether the range-awareness flag matches the class derived from predicted wind speed. |

### VILA aggregated radar metrics
| Column | Type | Units | Description |
| --- | --- | --- | --- |
| `vila_aggregated__n_0` | integer | unitless | VILA: count of contributing observations (negative Bragg peak, unitless). |
| `vila_aggregated__n_1` | integer | unitless | VILA: count of contributing observations (positive Bragg peak, unitless). |
| `vila_aggregated__pwr_mad_0` | number | dB | VILA: median absolute deviation of power (negative Bragg peak), in dB. |
| `vila_aggregated__pwr_mad_1` | number | dB | VILA: median absolute deviation of power (positive Bragg peak), in dB. |
| `vila_aggregated__pwr_max_0` | number | dB | VILA: maximum signal power (negative Bragg peak), in dB. |
| `vila_aggregated__pwr_max_1` | number | dB | VILA: maximum signal power (positive Bragg peak), in dB. |
| `vila_aggregated__pwr_mean_0` | number | dB | VILA: mean signal power (negative Bragg peak), in dB. |
| `vila_aggregated__pwr_mean_1` | number | dB | VILA: mean signal power (positive Bragg peak), in dB. |
| `vila_aggregated__pwr_median_0` | number | dB | VILA: median signal power (negative Bragg peak), in dB. |
| `vila_aggregated__pwr_median_1` | number | dB | VILA: median signal power (positive Bragg peak), in dB. |
| `vila_aggregated__pwr_min_0` | number | dB | VILA: minimum signal power (negative Bragg peak), in dB. |
| `vila_aggregated__pwr_min_1` | number | dB | VILA: minimum signal power (positive Bragg peak), in dB. |
| `vila_aggregated__pwr_n_0` | integer | unitless | VILA: sample count used for power statistics (negative Bragg peak, unitless). |
| `vila_aggregated__pwr_n_1` | integer | unitless | VILA: sample count used for power statistics (positive Bragg peak, unitless). |
| `vila_aggregated__pwr_stddev_0` | number | dB | VILA: standard deviation of signal power (negative Bragg peak), in dB. |
| `vila_aggregated__pwr_stddev_1` | number | dB | VILA: standard deviation of signal power (positive Bragg peak), in dB. |
| `vila_aggregated__velo_mad_0` | number | cm/s | VILA: MAD of radial velocity (negative Bragg peak), in cm/s. |
| `vila_aggregated__velo_mad_1` | number | cm/s | VILA: MAD of radial velocity (positive Bragg peak), in cm/s. |
| `vila_aggregated__velo_max_0` | number | cm/s | VILA: maximum radial velocity (negative Bragg peak), in cm/s. |
| `vila_aggregated__velo_max_1` | number | cm/s | VILA: maximum radial velocity (positive Bragg peak), in cm/s. |
| `vila_aggregated__velo_mean_0` | number | cm/s | VILA: mean radial velocity (negative Bragg peak), in cm/s. |
| `vila_aggregated__velo_mean_1` | number | cm/s | VILA: mean radial velocity (positive Bragg peak), in cm/s. |
| `vila_aggregated__velo_median_0` | number | cm/s | VILA: median radial velocity (negative Bragg peak), in cm/s. |
| `vila_aggregated__velo_median_1` | number | cm/s | VILA: median radial velocity (positive Bragg peak), in cm/s. |
| `vila_aggregated__velo_min_0` | number | cm/s | VILA: minimum radial velocity (negative Bragg peak), in cm/s. |
| `vila_aggregated__velo_min_1` | number | cm/s | VILA: minimum radial velocity (positive Bragg peak), in cm/s. |
| `vila_aggregated__velo_n_0` | integer | unitless | VILA: sample count used for radial velocity statistics (negative Bragg peak, unitless). |
| `vila_aggregated__velo_n_1` | integer | unitless | VILA: sample count used for radial velocity statistics (positive Bragg peak, unitless). |
| `vila_aggregated__velo_stddev_0` | number | cm/s | VILA: standard deviation of radial velocity (negative Bragg peak), in cm/s. |
| `vila_aggregated__velo_stddev_1` | number | cm/s | VILA: standard deviation of radial velocity (positive Bragg peak), in cm/s. |

### VILA maintenance metadata
| Column | Type | Units | Description |
| --- | --- | --- | --- |
| `vila_bearing` | number | degrees | VILA: bearing from station to cell centroid, degrees clockwise from geographic north. |
| `vila_dist_km` | number | km | VILA: great-circle distance from station to cell centroid, in kilometres. |
| `vila_maintenance_interval_id` | string | unitless | VILA: identifier of the maintenance interval covering the observation. |
| `vila_maintenance_type` | string | unitless | VILA: maintenance category reported by station logs (e.g., calibration, outage). |
| `vila_maintenance_start` | string | ISO8601 | VILA: timestamp marking the start of the active maintenance interval (ISO8601 string). |
| `vila_hours_since_last_calibration` | number | hours | VILA: elapsed hours since the station's last calibration event. |

### PRIO aggregated radar metrics
| Column | Type | Units | Description |
| --- | --- | --- | --- |
| `prio_aggregated__n_0` | integer | unitless | PRIO: count of contributing observations (negative Bragg peak, unitless). |
| `prio_aggregated__n_1` | integer | unitless | PRIO: count of contributing observations (positive Bragg peak, unitless). |
| `prio_aggregated__pwr_mad_0` | number | dB | PRIO: median absolute deviation of power (negative Bragg peak), in dB. |
| `prio_aggregated__pwr_mad_1` | number | dB | PRIO: median absolute deviation of power (positive Bragg peak), in dB. |
| `prio_aggregated__pwr_max_0` | number | dB | PRIO: maximum signal power (negative Bragg peak), in dB. |
| `prio_aggregated__pwr_max_1` | number | dB | PRIO: maximum signal power (positive Bragg peak), in dB. |
| `prio_aggregated__pwr_mean_0` | number | dB | PRIO: mean signal power (negative Bragg peak), in dB. |
| `prio_aggregated__pwr_mean_1` | number | dB | PRIO: mean signal power (positive Bragg peak), in dB. |
| `prio_aggregated__pwr_median_0` | number | dB | PRIO: median signal power (negative Bragg peak), in dB. |
| `prio_aggregated__pwr_median_1` | number | dB | PRIO: median signal power (positive Bragg peak), in dB. |
| `prio_aggregated__pwr_min_0` | number | dB | PRIO: minimum signal power (negative Bragg peak), in dB. |
| `prio_aggregated__pwr_min_1` | number | dB | PRIO: minimum signal power (positive Bragg peak), in dB. |
| `prio_aggregated__pwr_n_0` | integer | unitless | PRIO: sample count used for power statistics (negative Bragg peak, unitless). |
| `prio_aggregated__pwr_n_1` | integer | unitless | PRIO: sample count used for power statistics (positive Bragg peak, unitless). |
| `prio_aggregated__pwr_stddev_0` | number | dB | PRIO: standard deviation of signal power (negative Bragg peak), in dB. |
| `prio_aggregated__pwr_stddev_1` | number | dB | PRIO: standard deviation of signal power (positive Bragg peak), in dB. |
| `prio_aggregated__velo_mad_0` | number | cm/s | PRIO: MAD of radial velocity (negative Bragg peak), in cm/s. |
| `prio_aggregated__velo_mad_1` | number | cm/s | PRIO: MAD of radial velocity (positive Bragg peak), in cm/s. |
| `prio_aggregated__velo_max_0` | number | cm/s | PRIO: maximum radial velocity (negative Bragg peak), in cm/s. |
| `prio_aggregated__velo_max_1` | number | cm/s | PRIO: maximum radial velocity (positive Bragg peak), in cm/s. |
| `prio_aggregated__velo_mean_0` | number | cm/s | PRIO: mean radial velocity (negative Bragg peak), in cm/s. |
| `prio_aggregated__velo_mean_1` | number | cm/s | PRIO: mean radial velocity (positive Bragg peak), in cm/s. |
| `prio_aggregated__velo_median_0` | number | cm/s | PRIO: median radial velocity (negative Bragg peak), in cm/s. |
| `prio_aggregated__velo_median_1` | number | cm/s | PRIO: median radial velocity (positive Bragg peak), in cm/s. |
| `prio_aggregated__velo_min_0` | number | cm/s | PRIO: minimum radial velocity (negative Bragg peak), in cm/s. |
| `prio_aggregated__velo_min_1` | number | cm/s | PRIO: minimum radial velocity (positive Bragg peak), in cm/s. |
| `prio_aggregated__velo_n_0` | integer | unitless | PRIO: sample count used for radial velocity statistics (negative Bragg peak, unitless). |
| `prio_aggregated__velo_n_1` | integer | unitless | PRIO: sample count used for radial velocity statistics (positive Bragg peak, unitless). |
| `prio_aggregated__velo_stddev_0` | number | cm/s | PRIO: standard deviation of radial velocity (negative Bragg peak), in cm/s. |
| `prio_aggregated__velo_stddev_1` | number | cm/s | PRIO: standard deviation of radial velocity (positive Bragg peak), in cm/s. |

### PRIO maintenance metadata
| Column | Type | Units | Description |
| --- | --- | --- | --- |
| `prio_bearing` | number | degrees | PRIO: bearing from station to cell centroid, degrees clockwise from geographic north. |
| `prio_dist_km` | number | km | PRIO: great-circle distance from station to cell centroid, in kilometres. |
| `prio_maintenance_interval_id` | string | unitless | PRIO: identifier of the maintenance interval covering the observation. |
| `prio_maintenance_type` | string | unitless | PRIO: maintenance category reported by station logs (e.g., calibration, outage). |
| `prio_maintenance_start` | string | ISO8601 | PRIO: timestamp marking the start of the active maintenance interval (ISO8601 string). |
| `prio_hours_since_last_calibration` | number | hours | PRIO: elapsed hours since the station's last calibration event. |

## Range classes and censoring thresholds
- Valid inversion speeds lie between approximately 5.7 m/s and 17.8 m/s, as defined by the HF radar operating band (see `docs/hf_dev_plan.md` and `ann_training_report.md`).
- Classifier outputs:
  - `pred_range_label` is the argmax class among the posterior probabilities.
  - `prob_range_below`, `prob_range_in`, and `prob_range_above` form a simplex; their sum should equal 1 within floating-point tolerance.
  - `pred_speed_range_label` deterministically maps `pred_wind_speed` against the lower/upper thresholds; divergences with `pred_range_label` indicate mismatches between the regression head and classifier.
- Flags and margins:
  - `range_flag` combines classifier, deterministic label, and margin checks into one categorical flag (`below`, `in`, `above`, `uncertain`).
- `range_flag_confident` is raised when the flag derives from a probability exceeding the ANN confidence threshold (`range_flag_threshold = 0.5`, recorded in `config/range_thresholds.json` from the SAR fine-tuning script arguments at `hf-wind-inversion/hf_eolus/sar/fine_tuning_l2sp/script_args.json`).
  - `range_near_lower_margin`, `range_near_upper_margin`, and `range_near_any_margin` monitor buffer zones around the censoring thresholds; consult the ANN configuration constants when quantifying the exact width of these margins.
- Censoring guidance: treat `below` samples as left-censored at 5.7 m/s and `above` samples as right-censored at 17.8 m/s when fitting statistical models for wind resource estimation.

## Operational notes for consumers
- Radar statistics are provided in centimetres per second (radial velocity) and decibels (signal power); convert to SI wind speed (m/s) after merging with the inferred wind vectors when required.
- Distances to each station (`*_dist_km`) and bearings (`*_bearing`) support geometry-aware QA/QC. Geometry is encoded as WKB; load with libraries such as `pyarrow + shapely` or `geopandas`.
- Because the dataset is a pivot join, each row carries the best-available information from both radars; missing values may persist if one station lacked coverage during aggregation.
- If a manual refresh is ever required, regenerate the STAC metadata alongside the updated Parquet asset to capture schema or threshold changes introduced during ANN retraining.

## Metadata cross-checks
- Temporal extent (`start_datetime`, `end_datetime`) and spatial bounds (`extent.spatial.bbox`) were cross-checked against `use_case/catalogs/sar_range_final_pivots_joined/collection.json` and `items/data.json` to ensure consistency with the reported values in this document.
- Column definitions in this guide were programmatically extracted from the STAC `table:columns` metadata to avoid transcription errors.
- Geometry uniqueness verified on 2025-10-18: DuckDB reported zero nodes with more than one distinct geometry via
  `docker run --rm -v "$(pwd)":/workspace -w /workspace duckdb/duckdb:latest duckdb -cmd "SELECT node_id, COUNT(*) AS n_rows, COUNT(DISTINCT geometry) AS n_geoms FROM read_parquet('use_case/catalogs/sar_range_final_pivots_joined/assets/data.parquet') GROUP BY 1 HAVING COUNT(DISTINCT geometry) > 1;"`

## Node sampling density
- Counts per node were extracted with DuckDB on 2025-10-18 using
  `docker run --rm -v "$(pwd)":/workspace -w /workspace duckdb/duckdb:latest duckdb -cmd "SELECT node_id, COUNT(*) AS n_rows FROM read_parquet('use_case/catalogs/sar_range_final_pivots_joined/assets/data.parquet') GROUP BY 1 ORDER BY node_id;"`
- The snapshot contains 56 nodes. Minimum coverage is 307 samples (`VILA_PRIO15`), the median is 20 897 samples, and the maximum is 55 037 samples (`VILA_PRIO46`). Export the full per-node list with the query above whenever you need the exact figures for a refreshed dataset.
## Node availability summary
Run the following DuckDB query to inspect per-node observation counts for any ANN snapshot registered in `config/stac_catalogs.json`:

```sql
COPY (
  SELECT node_id, COUNT(*) AS observations
  FROM read_parquet('use_case/catalogs/sar_range_final_pivots_joined/assets/data.parquet')
  GROUP BY node_id
  ORDER BY node_id
) TO 'artifacts/tmp_cli_filters/node_counts.csv' (FORMAT CSV, HEADER);
```

Overlay these counts with the taxonomy bands stored in `config/node_taxonomy.json` (or regenerate them with `scripts/update_node_taxonomy.py`) to flag sparse nodes before fitting resource distributions. Publish the resulting CSV/JSON artefacts alongside the schema so downstream users understand the coverage available for each node.

## Temporal continuity checks
- Launch DuckDB in Docker (as shown below) to compute cadence adherence and gap statistics:
  ```bash
  docker run --rm -v \"$(pwd)\":/workspace -w /workspace duckdb/duckdb:latest duckdb -cmd \"WITH diffs AS (
    SELECT node_id,
           CAST(EXTRACT(EPOCH FROM timestamp - LAG(timestamp) OVER (PARTITION BY node_id ORDER BY timestamp)) AS BIGINT) AS dt
    FROM read_parquet('use_case/catalogs/sar_range_final_pivots_joined/assets/data.parquet')
  )
  SELECT node_id,
         COUNT(*) FILTER (WHERE dt IS NULL OR dt = 1800) AS on_cadence,
         COUNT(*) FILTER (WHERE dt > 1800 AND dt <= 7200) AS short_gaps,
         COUNT(*) FILTER (WHERE dt > 7200) AS long_gaps,
         MAX(dt) AS max_dt
  FROM diffs
  GROUP BY node_id
  ORDER BY node_id;\"\n  ```
- Use the resulting CSV to identify nodes dominated by long gaps (>2 hours) and to document the maximum hiatus per node. Integrate these findings into `config/node_taxonomy.json` so downstream tooling inherits the continuity warnings automatically.
- Aggregations (e.g. share of on-cadence transitions across all nodes) should be recomputed whenever a new ANN snapshot is ingested; record the results and query used in the changelog for reproducibility.
## Buoy co-location reference
- The sample catalogue includes node `Vilano_buoy`, which coincides with the Puertos del Estado oceanographic buoy distributed under `use_case/catalogs/pde_vilano_buoy/`. Treat it as a template for wiring other buoy benchmarks into the workflow.
- Align ANN and buoy series on `timestamp` (UTC) after removing sentinels (e.g. `wind_speed <= -9000`). Keep both datasets on a common cadence (30 minutes in the ANN release) before computing skill metrics or angular discrepancies.
- When documenting a new buoy comparison, record the DuckDB queries used to inspect temporal coverage, the QA filters applied, and the cadence assumptions. Store the cleaned buoy dataset under `artifacts/processed/` and cite it in `docs/data_access.md` so future users can reproduce the validation.
- Global RMSE values propagated through `config/global_rmse.json` still originate from the Vilano analysis packaged with this repository; regenerate the file when a new buoy benchmark becomes available.
