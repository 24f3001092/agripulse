# AgriPulse — Regional Agricultural Intelligence & Forecasting MVP

A working end-to-end data pipeline, prototype ML forecasting, regional
analytical segmentation, and Streamlit dashboard — all powered by real
datasets, real Pandera data-quality contracts, PySpark/Spark SQL
transformations, Delta Lake structuring, and an Airflow orchestration DAG.

---

## 1. Purpose

Demonstrate a complete data-engineering + data-science workflow:

- Ingest external weather, market, and geospatial data.
- Clean, validate, and structure it through a Bronze / Silver / Gold layer
  architecture.
- Produce machine-readable ML outputs (forecast results, model report,
  segmentation) that the dashboard and Airflow DAG consume.

## 2. Architecture

```
External Data (Open-Meteo weather API / static market & geo CSVs)
    |
    v
Bronze    data/bronze/       raw JSON/CSV as-is + ingestion metadata
    |
    v
Silver    data/silver/*.parquet   validated, typed, de-duplicated (Pandera)
    |
    v
Gold      data/gold/              PySpark + Spark SQL joins + Delta Lake
    |
    v
ML        src/ml/                 forecast (LinearRegression) + rule-based segments
    |
    v
Outputs   forecast_results.parquet / region_segments.parquet / model_report.json
    |
    v
Dashboard app.py (Streamlit)     user-facing five-tab analytical app
```

Orchestrated by: `run_pipeline.py` (local) or `dags/agripulse_dag.py` (Airflow)

## 3. Data sources

| Source | Module | Description |
|--------|--------|-------------|
| Weather | `src/ingestion/weather_api.py` | Live pull from Open-Meteo Forecast API (daily temp, precip, humidity, wind for 10 configured agricultural regions) |
| Market | `data/bronze/us_ag_exports_raw.csv` | 2011 US agricultural exports by state (corn / wheat / cotton / dairy in $M) — public dataset used as a static proxy for a market-intelligence feed |
| Geospatial | `data/bronze/us_state_geo_raw.csv` | US state capitals with latitude/longitude — regional coordinate anchors |
| Fixture fallback | `src/ingestion/weather_fixture_generator.py` | Schema-identical synthetic weather data for sandboxes where Open-Meteo is unreachable |

## 4. Bronze / Silver / Gold layers

### Bronze (`data/bronze/`)

Raw JSON (weather per region) and CSV (ag exports, geospatial) as delivered by
the source APIs/files. Ingestion metadata (`*.meta.json`) is written alongside
every weather pull with status, timestamps, and row counts.

### Silver (`data/silver/`)

Three validated Parquet files produced by `src/transform/bronze_to_silver.py`:

| File | Rows | Validation contract |
|------|------|---------------------|
| `weather.parquet` | ~370 | `weather_schema` (temp ranges, humidity 0-100, etc.) |
| `ag_exports.parquet` | ~50 | `ag_exports_schema` (2-char state code, non-negative values) |
| `geo.parquet` | ~50 | `geo_schema` (lat/lon in range, non-nullable name) |

Failing rows are written separately to `data/silver/_rejects/` for inspection;
the pipeline then fails the stage with an exit code rather than silently
dropping bad data.

### Gold (`data/gold/`)

Produced by `src/transform/silver_to_gold.py` using **PySpark + Spark SQL**:

- **`region_daily_features/`** — partitioned by region; 16 columns including
  weather (temp_avg_c, precipitation_mm, humidity_pct, windspeed_max_kmh),
  exports (total_exports_musd, corn/wheat/cotton/dairy_musd), and coordinates.
- **`region_summary/`** — one row per region with aggregated weather + max
  export values. Written as Delta Lake (falls back to partitioned Parquet if
  the Delta JVM connector cannot be resolved).

## 5. Data-quality monitoring (`src/quality/monitor.py`)

Checks run after the Silver layer is produced:

| Check | Logic |
|-------|-------|
| Freshness | latest bronze weather meta timestamp within 24 hours |
| Row-count | minimum row thresholds: weather ≥ 300, ag_exports ≥ 45, geo ≥ 45 |

Outputs a structured JSON report: `catalog/health_report.json` with status PASS/FAIL
and a list of every issue detected.

## 6. ML models

### Demand / export forecast (`src/ml/demand_forecast_model.py`)

- **Algorithm**: scikit-learn `LinearRegression` on four weather features
  (avg_temp_c, total_precip_mm, avg_humidity_pct, avg_windspeed) predicting
  `total_exports_musd`.
- **Dataset**: 10 monitored US agricultural regions — explicitly too small for
  a meaningful train/test split. The module trains on all data and reports
  **in-sample metrics only** (MAE, R², MAPE) with an honest warning in the
  model report.
- **Outputs**:
  - `data/gold/forecast_results.parquet` — region_name, actual_exports_musd,
    predicted_exports_musd, difference_musd, prediction_error_pct
  - `catalog/model_report.json` — features used, evaluation methodology,
    coefficients, limitations, small-sample warning

### Regional segmentation (`src/ml/marketing_segmentation.py`)

**Method**: rule-based (transparent, deterministic) segmentation using observed
Gold-layer features — chosen over KMeans because 10 regions is too small for
stable clustering.

Rules applied (computed from observed data terciles/medians):
- **export_tier**: High-Export / Mid-Export / Emerging-Export (by terciles)
- **dominant_crop**: corn-centric / wheat-centric / cotton-centric / diversified
  (requires ≥ 40% share of the crop basket)

Output: `data/gold/region_segments.parquet` with columns region_name, segment,
segment_reason (citing the actual numbers used), export_tier, dominant_crop.

**Labels are analytical groupings only — not validated commercial/customer
classifications.**

## 7. Unified ML pipeline (`src/ml/prediction_pipeline.py`)

Coordinates forecast + segmentation in a single stage:

```bash
python src/ml/prediction_pipeline.py
```

Produces all ML outputs plus `catalog/prediction_run.json` (execution metadata).

## 8. Dashboard (`app.py` — Streamlit)

```bash
python -m streamlit run app.py
```

Five tabs, every figure sourced from generated repository data:

| Tab | Reads | Content |
|-----|-------|---------|
| Overview | region_summary, health_report, model_report | Total regions, total exports, avg temp, data quality status, model status, export bar chart |
| Regional Intelligence | region_daily_features | Region selector; export profile bar chart, daily temperature line chart, precipitation, humidity, weather detail table |
| Forecast | forecast_results.parquet, model_report.json | Actual vs predicted bar chart, forecast table, metrics, methodology, limitations |
| Marketing Segments | region_segments.parquet | Segment distribution chart, regions per segment, per-segment region table, assignment reasons |
| Data Health | health_report.json, bronze meta, silver/gold artifact checks | Health status, issues list, freshness/row-count checks, ingestion metadata, artifact presence |

### Deployment artifact fallback

App paths are resolved relative to the repository (`pathlib`), so the dashboard
works from any working directory. Each artifact is read from its **locally
generated** path first (`data/gold/…`, `data/silver/…`, `catalog/…`). When that
output was not generated (for example on **Streamlit Community Cloud**, where
the generated files are gitignored), the app automatically falls back to the
tracked deployment copies under:

```
data/demo/
```

The `data/demo/` files are the **real** pipeline outputs (forecast results,
segments, health/model reports, and the silver/gold tables the dashboard draws
from), committed so the deployed app has data without running PySpark. The
Data Health tab labels each artifact's source — `live generated artifact`
versus `tracked data/demo artifact` — so a deployed dashboard never pretends it
ran the pipeline on the hosting machine.

## 9. Airflow DAG (`dags/agripulse_dag.py`)

```text
ingest_weather ─┐
ingest_market   ├──> bronze_to_silver ──> silver_to_gold ──> monitor ──> prediction_pipeline
ingest_geo     ─┘
```

- Schedule: daily at 04:00 UTC
- `PROJECT_ROOT` should point to the checked-out repo on the worker
- `PYTHON_BIN` defaults to `python3`; adjust for your Airflow host
- Silver→Gold requires valid `JAVA_HOME`; on Windows workers also set
  `HADOOP_HOME` to `hadoop-utils/` with `hadoop-utils/bin` on `PATH`

### Airflow platform note

Apache Airflow only supports Linux/macOS (or WSL2 / containers on Windows).
Importing `airflow` on plain Windows fails inside Airflow's own library
(e.g. Airflow 3.x calls `os.register_at_fork`, which does not exist on
Windows), so run the DAG on a Linux Airflow host/WSL2 — not on this repo's
Windows sandbox.

The DAG is covered by a dependency-light unit test
(`tests/test_dag.py`) that stubs the exact Airflow API surface it uses, so
the DAG's structure, task graph, and module paths are validated on any
platform without an Airflow install:

```
pytest tests/test_dag.py -v
```

## 10. Project structure

```
agripulse/
├── app.py                         Streamlit dashboard
├── run_pipeline.py                local orchestrator (all stages)
├── requirements.txt
├── README.md
│
├── catalog/
│   ├── data_catalog.json          dataset metadata / schema contracts
│   ├── regions.json               monitored regions and coordinates
│   ├── health_report.json         freshness + row-count check report
│   ├── model_report.json          forecast model features/metrics/limitations
│   └── prediction_run.json        last prediction pipeline execution metadata
│
├── data/
│   ├── bronze/                    raw weather JSON, ag exports CSV, geo CSV, *.meta.json
│   ├── silver/                    validated weather/exports/geo Parquet, _rejects/
│   ├── gold/
│   │   ├── region_daily_features/ Delta/Parquet partitioned by region
│   │   ├── region_summary/        Delta/Parquet one row per region
│   │   ├── forecast_results.parquet   actual vs predicted exports
│   │   └── region_segments.parquet    analytical segment assignments
│   └── demo/                      tracked real pipeline outputs for deployments
│       ├── gold/                  region_daily_features / region_summary /
│       │                          forecast_results / region_segments parquet
│       ├── silver/                weather / ag_exports / geo parquet
│       ├── catalog/               health_report.json, model_report.json
│       └── bronze/                latest_weather.meta.json
│
├── src/
│   ├── ingestion/
│   │   ├── weather_api.py         live Open-Meteo weather ingestion
│   │   └── weather_fixture_generator.py   sandbox fixture fallback
│   ├── transform/
│   │   ├── bronze_to_silver.py    Pandas + Pandera cleaning/validation
│   │   └── silver_to_gold.py      PySpark + Spark SQL + Delta Lake
│   ├── quality/
│   │   ├── schemas.py             Pandera data contracts
│   │   └── monitor.py             freshness + row-count monitoring
│   └── ml/
│       ├── demand_forecast_model.py      prototype export forecast
│       ├── marketing_segmentation.py     rule-based region segments
│       └── prediction_pipeline.py        unified ML orchestrator
│
├── dags/
│   └── agripulse_dag.py          Airflow DAG
│
├── tests/
│   ├── test_schemas.py            Pandera contract tests (6 tests)
│   ├── test_forecasting.py        forecast model + output tests (7 tests)
│   └── test_segmentation.py       segmentation tests (5 tests)
│
├── hadoop-utils/                  Windows Hadoop native libraries for PySpark
│   └── bin/
│       ├── winutils.exe
│       └── hadoop.dll
│
└── venv/                          Python virtual environment (not committed)
```

## 11. Setup

```bash
# clone the repo
cd agripulse

# create / activate virtual environment (use Python 3.11+)
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# on Windows only: ensure JAVA_HOME is set and hadoop-utils/bin is on PATH
# (run_pipeline.py handles this automatically on this machine)
```

### Requirements

| Package | Used for |
|---------|----------|
| pandas, pyarrow | Silver cleaning, Gold reading, parquet IO |
| pandera | Data-quality contract enforcement |
| pyspark, delta-spark | Gold layer: PySpark + Spark SQL + Delta Lake |
| scikit-learn | prototype LinearRegression forecast |
| streamlit, plotly | dashboard |
| pytest | test runner |
| requests | weather_api.py live ingestion |
| apache-airflow | DAG orchestration (only on Airflow worker) |

## 12. Running the pipeline

### Local (recommended for demo)

```bash
# full run — re-ingests weather from live API; if the API is unreachable
# (network-restricted sandbox), automatically falls back to the project's
# schema-identical fixture generator.
python run_pipeline.py

# skip re-ingestion — reuse existing bronze data (gold/regeneration runs)
python run_pipeline.py --skip-ingestion
```

### Individual stages (for development / debugging)

```bash
python src/ingestion/weather_api.py --config catalog/regions.json --out data/bronze
python src/ingestion/weather_fixture_generator.py --config catalog/regions.json --out data/bronze
python src/transform/bronze_to_silver.py
python src/transform/silver_to_gold.py
python src/quality/monitor.py
python src/ml/prediction_pipeline.py
```

### Dashboard

```bash
python -m streamlit run app.py
```

The app runs with local pipeline artifacts when present and falls back to the
tracked `data/demo/` deployment artifacts otherwise.

## 13. Running the tests

```bash
pytest tests/ -v
```

Expected output: 23 tests pass (6 schema, 7 forecasting, 5 segmentation, 5 DAG).

## 14. Limitations

1. **Small sample size**: the model trains on 10 regions — metrics reported are
   in-sample only and must not be interpreted as validated out-of-sample
   accuracy. The `model_report.json` and dashboard both state this explicitly.
2. **Static market data**: the ag exports dataset is a single annual
   cross-section (2011); there is no multi-year history, so the model cannot
   demonstrate temporal forecast skill.
3. **Sandbox weather**: when Open-Meteo is unreachable (network-restricted
   environment), the fixture generator produces schema-identical synthetic data
   using seeded seasonal baselines. The data is realistic for a demo but is
   not live observational data.
4. **Delta Lake**: `delta-spark` requires Maven Central access at runtime to
   fetch its JVM connector. On machines without Maven Central access, the
   pipeline automatically falls back to partitioned Parquet with a logged
   warning. On Databricks or machines with normal internet, Delta works with
   zero code changes.
5. **Windows + PySpark**: Spark on Windows requires `winutils.exe` + `hadoop.dll`
   from a Hadoop distribution, plus a valid `JAVA_HOME`. The repo bundles
   `hadoop-utils/bin` and `run_pipeline.py` sets the environment automatically;
   the Airflow DAG assumes the worker's own environment is correctly configured.
6. **Monitoring freshness**: the freshness check uses a 24-hour window. If using
   `--skip-ingestion` with stale bronze data (>24 hours old), the monitoring
   stage will fail — re-run the fixture generator or a live ingestion first.
7. **Airflow on Windows**: Airflow itself cannot import on native Windows
   (upstream limitation, not this repo's DAG). Deploy the DAG to a Linux
   Airflow host / WSL2, or validate it with `pytest tests/test_dag.py`, which
   runs anywhere.

## 15. Deployment (Streamlit Community Cloud)

The dashboard runs on Streamlit Community Cloud with `app.py` as the entry
point (`python -m streamlit run app.py`). The deployed instance has no PySpark
pipeline and no locally generated Gold layer, so it reads the **tracked real
pipeline artifacts committed under `data/demo/`**:

- Local mode: artifacts are read from `data/gold/`, `data/silver/`, and
  `catalog/` as generated by `python run_pipeline.py --skip-ingestion`.
- Deployed mode: when those generated files are absent, `app.py` falls back to
  the identical files under `data/demo/` (forecast results, segments, health
  and model reports, silver tables, and the latest weather ingestion metadata).

The Data Health tab labels every artifact's source (`live generated artifact` vs
`tracked data/demo artifact`), so the deployed app never claims a live pipeline
run it did not perform. Update the demo bundle by re-running the pipeline
locally and refreshing the files under `data/demo/`.

## 16. Demo Flow

A 5-minute guided walkthrough of the MVP:

1. **Run the pipeline** — materialize all artifact layers from code:

   ```bash
   python run_pipeline.py --skip-ingestion
   ```

   (On a fresh checkout with network access, use `python run_pipeline.py` for a
   live weather pull; the pipeline falls back to fixtures when the API is
   unreachable.)

2. **Open the dashboard**

   ```bash
   python -m streamlit run app.py
   ```

3. **Select a region** — go to **Regional Intelligence** and pick a state from
   the dropdown to load its weather and export profile.

4. **Inspect weather & agriculture indicators** — read the three metric cards
   (total exports, average temperature, total precipitation), compare the crop
   profile bar chart with the daily temperature line chart, and inspect the
   daily detail table.

5. **Review the forecast** — open **Forecast** to see actual vs predicted
   exports per region, the in-sample R²/MAE, and the limitations panel (always
   visible so the prototype's scope is explicit).

6. **Review segmentation** — open **Marketing Segments** to see how regions
   are grouped into `tier > dominant crop` segments, the reason behind each
   assignment, and the regions in each group.

7. **Inspect data health** — open **Data Health** to confirm the overall
   status (`PASS`), review the checks performed (freshness + row counts), the
   latest weather ingestion, and artifact presence across Bronze/Silver/Gold.

Optional setup step: verify tests first with `pytest tests/ -v` (23 tests).
