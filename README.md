# AgriPulse — External Data Pipeline for Marketing Segmentation & Sales Forecasting

A working data-engineering pipeline modeled directly on Syngenta's AMEA Data & AI
Commercial IT internship: ingest external weather, market, and geospatial data,
clean and validate it, structure it in a Delta Lake-style Gold layer, and hand it
off to a downstream ML model — the same Bronze → Silver → Gold shape used in
production Databricks environments.

## Architecture

```
                 ┌────────────────────┐
                 │   External Sources  │
                 │  Weather API        │
                 │  Market CSV (real)  │
                 │  Geospatial CSV     │
                 └─────────┬───────────┘
                           │  Python + Requests
                           ▼
                 ┌────────────────────┐
                 │   BRONZE (raw)      │  data/bronze/
                 │  JSON / CSV as-is   │  + ingestion metadata (data catalog)
                 └─────────┬───────────┘
                           │  Pandas + Pandera
                           ▼
                 ┌────────────────────┐
                 │   SILVER (clean)    │  data/silver/*.parquet
                 │  Validated, typed,  │  rejects logged separately
                 │  standardized       │
                 └─────────┬───────────┘
                           │  PySpark + Spark SQL
                           ▼
                 ┌────────────────────┐
                 │   GOLD (feature)    │  data/gold/region_daily_features (Delta)
                 │  Joined, aggregated │  data/gold/region_summary (Delta)
                 │  ML-ready           │
                 └─────────┬───────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │  ML / Forecasting   │  src/ml/demand_forecast_model.py
                 └────────────────────┘

     Orchestrated daily by dags/agripulse_dag.py (Airflow)
     Monitored by src/quality/monitor.py (freshness + row-count checks)
```

## Tech stack (matches the JD's "Technical Skills" list)

| Category | Tools used here |
|---|---|
| Programming & Data Processing | Python, Pandas, PySpark, Spark SQL |
| Data Engineering & Cloud | Delta Lake (`delta-spark`), designed for Databricks + ADF |
| Data Integration | REST APIs (Open-Meteo), external CSV sources |
| Version Control | Git |
| Orchestration | Apache Airflow DAG |
| Data Quality | Pandera schema contracts, pytest, pandas-native monitoring |

## Real data sources used

1. **Weather** — [Open-Meteo Forecast API](https://open-meteo.com) (free, no key). Live ingestion module: `src/ingestion/weather_api.py`.
2. **Market intelligence** — [2011 US Agricultural Exports by state](https://github.com/plotly/datasets/blob/master/2011_us_ag_exports.csv), a real public dataset (corn/wheat/cotton/dairy export values per state).
3. **Geospatial** — [US state capitals with lat/lon](https://github.com/jasperdebie/VisInfo/blob/master/us-state-capitals.csv), used to anchor each region's coordinates.

All three are cataloged in `catalog/data_catalog.json` with source URL, refresh cadence, owner, schema, and known limitations — exactly the "data catalogue entries for onboarded datasets" the JD asks for.

## ⚠️ One honest caveat: Delta Lake & live weather calls in a sandbox

This project was built and tested inside a network-restricted sandbox that only
allows outbound traffic to package registries (PyPI, npm, GitHub) — not to
Maven Central (`repo1.maven.org`) or general APIs like `api.open-meteo.com`.
Two consequences, both handled explicitly rather than faked:

- **Weather data**: `weather_api.py` is the real, live ingestion module and
  will pull live data the moment it's run somewhere with normal internet
  access. Inside this sandbox it can't reach the API (confirmed via a direct
  403 from the egress proxy), so `weather_fixture_generator.py` generates
  schema-identical synthetic data so the rest of the pipeline could be built
  and tested end-to-end. Swap one script for the other — nothing downstream
  changes, because both produce the same JSON schema.

- **Delta Lake**: `delta-spark` resolves its JVM jar from Maven Central at
  runtime — there's no pip-installable jar. `src/transform/silver_to_gold.py`
  tries the real Delta path first, and **automatically falls back to
  partitioned Parquet** if the jar can't be resolved, logging a clear warning
  either way. On Databricks Community Edition or any machine with normal
  internet access, the Delta path works with zero code changes — `delta_available`
  will simply come back `True`.

This is exactly the kind of environment constraint you'd document and route
around on a real team, so it's built that way instead of hidden.

## Setup

```bash
pip install -r requirements.txt
```

## Running the pipeline

```bash
# Full run (re-ingests weather; market/geo are static demo CSVs)
python run_pipeline.py

# Reuse existing bronze data (skip re-ingestion)
python run_pipeline.py --skip-ingestion
```

Or run each stage individually:

```bash
python src/ingestion/weather_api.py --config catalog/regions.json --out data/bronze   # real API
python src/ingestion/weather_fixture_generator.py --config catalog/regions.json --out data/bronze  # sandbox fallback
python src/transform/bronze_to_silver.py
python src/transform/silver_to_gold.py
python src/quality/monitor.py
python src/ml/demand_forecast_model.py
```

## Tests

```bash
pytest tests/ -v
```

## Project layout

```
agripulse/
├── src/
│   ├── ingestion/       # Bronze-layer pull scripts (weather API + fixture fallback)
│   ├── transform/       # Bronze->Silver (pandas/pandera), Silver->Gold (PySpark/SQL/Delta)
│   ├── quality/         # Pandera schemas + freshness/row-count monitoring
│   ├── ml/              # Downstream model consuming the Gold feature table
│   └── orchestration/   # (reserved for future custom orchestration helpers)
├── dags/                # Airflow DAG
├── catalog/             # regions.json config, data_catalog.json, health_report.json
├── data/                # bronze/silver/gold layers (gitignored in real deployment)
├── tests/               # pytest suite for data-quality contracts
├── run_pipeline.py      # single-command local orchestrator
└── requirements.txt
```

## What this demonstrates for the Syngenta Data Engineering Intern role

- Sourcing and onboarding external datasets (weather, market, geospatial) via REST APIs and files
- Python/Pandas ingestion and cleaning pipelines
- PySpark + Spark SQL transformations and joins across multiple sources
- Delta Lake structuring for downstream ML (with a documented, honest fallback path)
- Data quality validation with explicit pass/fail contracts (Pandera) and rejected-row logging
- Basic pipeline monitoring/alerting (freshness + row-count anomaly checks)
- Data catalog documentation for every onboarded dataset
- Airflow DAG for daily orchestration
- Git-tracked, incrementally committed history (see `git log`)
