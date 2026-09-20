# AgriPulse India — Agricultural Intelligence, Market Analytics & Business Scenario Platform

A working end-to-end data pipeline, prototype ML analytics, and a **public-use
Streamlit product** with ten pages — built for the Indian agricultural context:
real weather ingestion, real geography (State/UT > District > Mandi/APMC), real
Pandera data-quality contracts, PySpark/Spark SQL transformations, Delta Lake
structuring, and an Airflow orchestration DAG.

The dashboard is a public, deployment-ready product: it reads only
repository-generated artifacts (never hard-coded numbers, no live API required
to open), labels every figure's provenance, exposes an explainable
district/segment view, a user-driven **scenario planner**, and an honest data
source & limitations page.

> **Honesty over data-fabrication.** Wholesale mandi prices/arrivals shown on the
> dashboard are **real AGMARKNET records** (ingested via the India Data Portal's
> structured resource, see `src/ingestion/india_mandi.py`), labelled with source,
> last-updated and data period. A **live e-NAM feed is not yet wired in** — this
> repository never invents market prices or arrivals. Market-price-forecast
> outputs remain **gated** until a live feed is integrated. See §3 and §14.

---

## 1. Purpose

Demonstrate a complete data-engineering + data-science workflow for Indian
agriculture:

- Ingest external weather, geospatial, agricultural-production and AGMARKNET
  mandi data for 10 monitored districts.
- Clean, validate, and structure it through a Bronze / Silver / Gold layer
  architecture with India-specific schema contracts.
- Produce machine-readable ML outputs (ML status, weather-exposure segments,
  India crop-production forecast + model report, prediction-run metadata) that
  the dashboard and Airflow DAG consume.

## 2. Architecture

```
External Data (Open-Meteo live weather API; GeoIndia reference)
    |
    v
Bronze    data/india/bronze/       raw JSON/CSV as-is + ingestion metadata
    |
    v
Silver    data/india/silver/*.parquet   validated, typed (Pandera + GeoIndia contract)
    |
    v
Gold      data/india/gold/              PySpark + Spark SQL joins + Delta Lake
    |
    v
ML        src/ml/                       market-gated forecast + India crop-production
                                        prototype + rule-based weather segments
    |
    v
Outputs   ml_status.json / region_segments.parquet / model_report.json /
          india_forecasts.parquet / india_model_report.json
    |
    v
Dashboard app.py (Streamlit)     ten-page public product + src/products/
```

Orchestrated by: `run_pipeline.py` (local) or `dags/agripulse_dag.py` (Airflow).
The active country profile lives in `catalog/pipeline_config.json`
(`"market_source": "not_configured"`), switchable at runtime with the
`AGRIPULSE_COUNTRY` environment variable. All data trees live under `data/india/`.

## 3. Data sources

| Source | Module | Description |
|--------|--------|-------------|
| Weather | `src/ingestion/weather_api.py` | Live pull from Open-Meteo Forecast + Historical API (daily temp, precip, humidity, wind) for the 10 configured districts |
| IMD Weather (official) | `src/ingestion/imd_weather.py` | Official India Meteorological Department API (`api.imd.gov.in`): current weather, 7-day city forecast, district-wise rainfall & warnings. Credentials via env (`AGRIPULSE_IMD_API_KEY`, `AGRIPULSE_IMD_TOKEN`); reports `not_configured` without them |
| Agricultural production (official) | `src/ingestion/india_agriculture.py` | DE&S / MoAFW district-wise, season-wise **crop Area/Production/Yield** statistics since 1997 (State > District > Crop > Season, GODL-India). Source: `data.gov.in` OGD catalog; acquired via India Data Portal CKAN or a local file (`AGRIPULSE_AGRICULTURE_FILE`). Missing values are preserved as null — never fabricated |
| Geography | `catalog/india_regions.json` | India location catalog: State/UT > District > Mandi/APMC with ISO 3166-2:IN state codes, LGD (Local Government Directory) district codes, and approximate coordinate anchors |
| Commodities | `catalog/india/commodities.json` | 16 Indian agricultural commodities (metadata only — no fabricated prices) |
| Fixture fallback | `src/ingestion/weather_fixture_generator.py` | Schema-identical synthetic weather for sandboxes where Open-Meteo is unreachable |
| IMD fixture fallback | `src/ingestion/india_weather_fixture_generator.py` | Clearly-labelled synthetic IMD-schema fixtures for sandboxes without approved IMD credentials; **never presented as live data** |
| Market-data (official) | `src/ingestion/india_mandi.py` | AGMARKNET daily APMC wholesale **prices & arrivals** (Min/Modal/Max, ₹/Quintal etc. plus arrival in Metric Tonnes/Bundle/Nos), acquired via the India Data Portal structured resource (GODL-India). Real, attributed records; missing values stay null; no traded quantity is synthesized; live e-NAM is not scraped (no public documented API). Bounded real subset for the monitored districts is committed under `data/india/demo/` |
| Market-price forecast (planned) | `src/ingestion/india_market_source.py` | Extension point for a live e-NAM feed; **not configured**, never fabricates a forecast |

## 4. Bronze / Silver / Gold layers

### Bronze (`data/india/bronze/`)

Raw weather JSON per district plus ingestion metadata (`*.meta.json` with
source, timestamps, status, row counts). IMD pulls land in
`data/india/bronze/india_weather/` (per-pull payload + `.meta.json` provenance:
source, endpoint, retrieved_at, location, date). Agricultural-production pulls
land in `data/india/bronze/india_agriculture/` (verbatim raw copy under `raw/`,
normalized parquet + `.meta.json` provenance: source, source_url, ingested_at,
row_count, units).

### Silver (`data/india/silver/`)

Validated Parquet files produced by `src/transform/bronze_to_silver.py`:

| File | Validation contract |
|------|---------------------|
| `weather.parquet` | `weather_schema` (temp ranges, humidity 0–100, etc.) |
| `geo.parquet` | `geo_india_schema` (2-char state code, lat 6–37, lon 68–98) |
| `india_agriculture.parquet` | `india_agriculture_schema` (State/District/Crop/Season/Year + non-negative Area/Production/Yield; production & yield nullable) |
| `india_mandi.parquet` | `mandi_price_schema` (non-null date/state/district/apmc/commodity/variety; non-negative prices & arrival; min ≤ modal ≤ max when present; non-empty commodity/apmc) |
| `market_status.json` | written when the live e-NAM feed is not configured (honest marker) |
| `agriculture_status.json` | written when no agriculture Bronze exists (honest marker) |
| `mandi_status.json` | written when no mandi Bronze exists (honest marker) |

Failing rows are written to `data/india/silver/_rejects/`; the stage then fails
with a non-zero exit code rather than silently dropping bad data.

### Gold (`data/india/gold/`)

Produced by `src/transform/silver_to_gold.py` using **PySpark + Spark SQL**
(the climate exposure layer), plus the integrated India agricultural Gold layer
by `src/transform/india_silver_to_gold.py` (pandas, consistent with the mandi
summary); the latter builds **Location + Crop + Date/Year** features from the
three India Silver tables:

- **`region_daily_features/`** — partitioned by district; weather columns,
  full India geography ride-along (`state_ut`, `state_code`, `district`,
  `mandi_apmc`, `latitude`, `longitude`). Market columns are added **only** when
  a real market loader is configured — nothing is fabricated.
- **`region_summary/`** — one row per district with aggregated weather.
- **`region_segments.parquet`** — weather-exposure segments (see §6).
- **`india_region_features/`** — region x date grid: weather block (temperature,
  rainfall, humidity, wind) outer-joined with the observed APMC market block
  (median modal price within **Rs./Quintal**, total arrival within **Metric
  Tonnes**, market/commodity counts). Both windows stay visible with honest nulls.
- **`india_crop_summary/`** — one row per `(state, district, crop)`: APY block
  (latest + all-time area/production/yield, latest year/season), observed market
  block (latest min/modal/max, 7-day average and change, arrivals) and observed
  weather block (district means). Crop→commodity joins are **exact
  name-normalized** (no fuzzy matching); unmatched crops report
  `market_status=no_records`. `weather_warning` is an explicit null column with
  the manifest recording `weather_warning_status=not_ingested` (the IMD warning
  loader is not configured — no advisory values are fabricated).
- **`india_crop_year.parquet`** — long-form `(state, district, crop, season,
  year_start)` Area-Production-Yield grain (the Date/Year axis).
- **`india_market_summary/`** — per market/commodity/variety price & arrival
  summary (reuses `src/transform/mandi_to_gold.py` as the single source of
  truth).

Each India Gold output carries a `manifest.json`, and `data/india/catalog/
gold_lineage.json` records per-output `source / input / output / schema /
generated_at`; the static lineage is also documented in
`catalog/india/data_catalog.json`. Written as Delta Lake with automatic
partitioned-Parquet fallback.

## 5. Data-quality monitoring (`src/quality/monitor.py`)

| Check | Logic |
|-------|-------|
| Freshness | latest bronze weather meta timestamp within 24 hours |
| Row-count | min row thresholds derived from the active region count (10 districts) |
| Market source | informational: reports `not_configured` without failing the run |
| Staleness | gold `manifest.json` `generated_at` within 24 hours; the frozen AGMARKNET publication gap is informational |
| Duplicates | trade-day identity (`india_mandi`), season-aware APY key, `(region_name, date)` and `(state, district, crop)` identities |
| Missing geography | empty state/district; crop rows with no monitored district (weather/market blocks only exist for monitored districts) |
| Impossible values | temperatures outside [-60, 60] °C, humidity outside [0, 100], negative prices/arrivals/area/production/yield |
| Missing prices | AGMARKNET rows without a modal price (informational — the source allows nulls; counts are surfaced, never hidden) |
| Missing weather | weather-window region rows without temperature/humidity |

Outputs `data/india/catalog/health_report.json` with status PASS/FAIL, every
detected issue, and a structured `india.artifact_checks` section (PASS / INFO /
FAIL per check with observed metrics) that the Data Health page renders.

## 6. ML analytics

### Forecast (`src/ml/demand_forecast_model.py`) — *market-gated*

The forecast target is the **market price signal** (`₹ / quintal`). Because no
authorized market feed is wired in yet, the model **does not generate forecast
outputs**; `run_forecast()` writes `data/india/catalog/ml_status.json` with
`model_status: "not_generated"` and a list of the upstream market sources
required before it can run. The dashboard's Forecast page reflects this
honestly. When a real loader is configured, training runs a scikit-learn
`LinearRegression` on weather features and reports **in-sample metrics only**
(MAE, R², MAPE) for the 10-district sample.

### Regional segmentation (`src/ml/marketing_segmentation.py`)

Rule-based (transparent, deterministic) weather-exposure segmentation using
observed Gold-layer weather:
- **segment / segment_reason** — e.g. *High-Rainfall Risk*, *Water-Limited*,
  *Heat-Stress Baseline* — each reason cites the observed drivers.
- **exposure_tier** — High / Medium / Low exposure.
- **dominant_factor** — the single weather variable with the largest deviation
  from the monitored-cohort median.

Output: `data/india/gold/region_segments.parquet` (`region_name`, `segment`,
`segment_reason`, `exposure_tier`, `dominant_factor`).

**Labels are analytical groupings only — not validated commercial/customer
classifications.**

### Unified ML pipeline (`src/ml/prediction_pipeline.py`)

Coordinates forecast + segmentation, writes `data/india/catalog/prediction_run.json`.

### India crop-production forecast (`src/ml/india_forecast.py`) — *prototype*

A one-year-ahead (2023-2024) production forecast built exclusively on the real
DE&S / MoAFW **Area-Production-Yield** panel (26 crop years 1997-98..2022-23,
737 districts, 115 crops). The **FIRST AUDIT** confirmed that mandi
prices/arrivals only have a 61-day window and the weather window is a single
recent month disjoint from the agriculture years — so the only defensible
target is **production (tonnes)**; price/arrival forecasting is explicitly NOT
attempted and never fabricated.

- **Model selection is honest**: a depth-limited `HistGradientBoostingRegressor`
  is compared against last-year persistence under a **strict temporal split**
  (train years precede validation years precede test years — no random shuffle,
  no leakage). Model selection uses validation years only; the current panel
  selected **naive persistence** (GBM did not beat it on validation MAE, and
  that underperformance is disclosed in the report).
- Outputs: `data/india/gold/india_forecasts.parquet` (16,938 cells) +
  `data/india/catalog/india_model_report.json` (`model_status: prototype`,
  `production_ready: false`, full metrics from held-out rows, limitations).
- The dashboard **India Forecast** page reads the report verbatim — it never
  labels the prototype as accurate, production-ready, guaranteed, or profitable.

## 7. Forecast page

Opens with the current market-source status. The **Market Intelligence** page
shows real observed AGMARKNET prices/arrivals; the **U.S. Forecast** page stays
gated on a live e-NAM feed: while that extension is not configured, it explains
*why* no price forecast exists, lists the exact upstream sources required
(`e-NAM`), and shows only weather-derived analytics plus the observed market
signal. No fabricated prediction is ever shown. The **India Forecast** page
(separate) shows the real crop-production prototype described above with
historical trend, forecast band, observed-vs-predicted scatter, methodology,
data period, and warnings/limitations.

## 8. Dashboard (`app.py` — Streamlit)

```bash
python -m streamlit run app.py
```

Ten pages, every figure sourced from generated repository data and priced in
₹ with Indian digit grouping:

| Page | Reads | Content |
|------|-------|---------|
| Home | region_summary, health_report, ml_status, bronze meta | Monitored districts, weather snapshot, data quality, ML/market status |
| My District | region_daily_features, summary, segments, geo | District selector (State/UT, then District/APMC): location, weather profile, segment assignment, explainable insight cards, daily detail |
| Market Core | region_daily_features, summary, segments | Cross-district market & weather outlook: exposure tiers and weather profiles (observed values only) |
| Market Intelligence | india_market_summary, india_region_features | Real observed AGMARKNET APMC prices/arrivals with weather context (no fabricated prices) |
| India Forecast | india_forecasts.parquet, india_model_report.json | Crop-production prototype (2023-24): historical trend, 95% band, observed-vs-predicted, methodology, data period, warnings/limitations read verbatim from the report |
| U.S. Forecast | ml_status.json, model_report.json | Honest gate: no price forecast while market not integrated; methodology + required upstream |
| Opportunity Scanner | region_segments.parquet, region_daily_features | Weather-exposure segments, district overview with weather signals, assignment reasons |
| Scenario Planner | (user assumptions) | Estimated revenue/cost/margin (₹/quintal) from user inputs, ±20% sensitivity |
| Data Health | health_report.json, bronze meta, silver/gold artifacts | Status, checks, latest ingestion, per-artifact source labels |
| Sources & Methodology | catalog/india/data_catalog.json, model_report.json | Dataset provenance, honest limitations, market extension point |

Shared helpers under `src/products/`:

- `insights.py` — explainable `WHAT / WHY / DATA PERIOD / SOURCE-METHOD` cards
  from observed Gold data only.
- `scenario.py` — deterministic estimate math (sellable quantity, revenue, total
  cost, margin, ±20% sensitivity) with input validation. Explicitly labeled
  **estimates under user assumptions**, never profit guarantees.
- `artifacts.py` — single source of truth mapping artifact keys to their locally
  generated path and their tracked `data/india/demo/` fallback.

### Deployment artifact fallback

Artifacts are resolved from the **locally generated** path first
(`data/india/gold/…`, `data/india/silver/…`, `data/india/catalog/…`). When not
generated (e.g. Streamlit Community Cloud, where generated files are
gitignored), the app falls back to the tracked real outputs under:

```
data/india/demo/
```

The `data/india/demo/` files are **real pipeline outputs** (gold features,
summary, segments, ml status, health report, prediction run, silver tables,
latest weather meta) — committed so the deployed app has data without running
PySpark. Data Health labels each artifact's source (`live generated artifact`
vs `tracked data/demo artifact`), so a deployed dashboard never pretends it ran
the pipeline on the hosting machine.

## 9. Airflow DAG (`dags/agripulse_dag.py`)

```text
ingest_weather ─────┐
ingest_market (echo)├──> bronze_to_silver ──┬──> silver_to_gold ─────────────────┐
ingest_geo (echo)  ─┘                       ├──> india_gold_summary ───> india_forecast ─┤
                                            └───────────────────────────────────────────┴──> monitor ──> prediction_pipeline
```

- Schedule: daily at 04:00 UTC; daily ingest tasks cover weather and the
  AGMARKNET mandi Bronze import (`src/ingestion/india_mandi.py`), while the
  `ingest_market (echo)` slot is reserved for a future live e-NAM feed.
- `india_gold_summary` runs the integrated India Gold builder
  (`src/transform/india_silver_to_gold.py`) — region x date features, crop
  summary and the AGMARKNET market summary — and writes graceful `no_data`
  manifests when the corresponding Silver inputs are absent.
- `india_forecast` runs the crop-production prototype
  (`src/ml/india_forecast.py`) between India Gold and Monitoring: a strict
  temporal split, gradient boosting vs persistence compared on validation,
  and `india_forecasts.parquet` + `catalog/india_model_report.json` written
  (or an honest `not_generated` report when the agriculture panel is absent).
- `PROJECT_ROOT` must point to the checked-out repo on the worker; `PYTHON_BIN`
  defaults to `python3`.
- Silver→Gold needs valid `JAVA_HOME`; Windows workers also set `HADOOP_HOME`
  to `hadoop-utils/` with `hadoop-utils/bin` on `PATH`.

Airflow itself only supports Linux/macOS (or WSL2/containers on Windows) —
importing `airflow` on native Windows fails inside Airflow's library, so run the
DAG on a Linux host. The DAG is covered by a dependency-light unit test
(`tests/test_dag.py`) that stubs the Airflow API surface, so its structure and
task graph are validated anywhere:

```
pytest tests/test_dag.py -v
```

## 10. Project structure

```
agripulse/
├── app.py                         Streamlit dashboard (10 pages)
├── run_pipeline.py                local orchestrator (all stages)
├── requirements.txt
├── README.md
│
├── catalog/
│   ├── pipeline_config.json       active profile (india), currency/units, market_source
│   ├── india_regions.json         10 monitored districts: ISO state codes + LGD district codes + coordinate anchors
│   └── india/
│       ├── commodities.json       16 Indian commodities (metadata only)
│       └── data_catalog.json      dataset provenance / status registry
│
├── data/
│   ├── india/
│   │   ├── bronze/                raw weather JSON + *.meta.json
│   │   ├── silver/                weather/geo parquet, market_status.json, _rejects/
│   │   ├── gold/                  region_daily_features/ (Delta, part-by-district),
│   │   │                          region_summary/, region_segments.parquet
│   │   ├── catalog/               health_report.json, ml_status.json,
│   │   │                          prediction_run.json
│   │   └── demo/                  tracked real pipeline outputs for deployments
│   │       ├── gold/              region_daily_features / region_summary /
│   │       │                      region_segments / india_crop_year /
│   │       │                      india_forecasts parquet
│   │       ├── silver/            weather / geo parquet
│   │       ├── catalog/           health_report.json, ml_status.json, prediction_run.json,
│   │       │                      india_model_report.json
│   │       └── bronze/            latest_weather.meta.json
│   └── existing/current/          frozen snapshot of the earlier US-model build
│
├── src/
│   ├── config.py                  country profile, India data layout, INR formatting
│   ├── ingestion/
│   │   ├── weather_api.py         live Open-Meteo weather ingestion
│   │   ├── weather_fixture_generator.py  sandbox fixture fallback
│   │   └── india_market_source.py e-NAM / AGMARKNET extension point (not configured)
│   ├── transform/
│   │   ├── bronze_to_silver.py    Pandas + Pandera cleaning/validation
│   │   └── silver_to_gold.py      PySpark + Spark SQL + Delta Lake
│   ├── quality/
│   │   ├── schemas.py             Pandera contracts (weather, GeoIndia, market)
│   │   └── monitor.py             freshness + row-count monitoring
│   ├── ml/
│   │   ├── demand_forecast_model.py      market-gated prototype forecast
│   │   ├── india_forecast.py    India crop-production prototype (temporal split)
│   │   ├── marketing_segmentation.py     rule-based weather-exposure segments
│   │   └── prediction_pipeline.py        unified ML orchestrator
│   └── products/
│       ├── artifacts.py           artifact key → live/demo path mapping
│       ├── insights.py            explainable district insight cards
│       └── scenario.py            scenario estimate + sensitivity math
│
├── dags/
│   └── agripulse_dag.py           Airflow DAG
│
├── scripts/
│   └── export_demo.py             refresh data/india/demo/ from real pipeline outputs
│
├── tests/
│   ├── test_india.py              India migration contracts (geo/commodities/INR)
│   ├── test_schemas.py            Pandera contract tests
│   ├── test_forecasting.py        forecast model + market-gate tests
│   ├── test_india_forecast.py     India crop-production forecast tests
│   ├── test_segmentation.py       weather-segmentation tests
│   ├── test_dag.py                Airflow DAG structure (stub-based)
│   └── test_products.py           scenario / insights / deployment-artifact contract
│
├── hadoop-utils/                  Windows Hadoop native libraries for PySpark
└── venv/                          Python virtual environment (not committed)
```

## 11. Setup

```bash
cd agripulse
python -m venv venv
# Windows: venv\Scripts\activate | Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
# Windows only: ensure JAVA_HOME and hadoop-utils/bin are configured
# (run_pipeline.py handles this automatically on this machine)
```

| Package | Used for |
|---------|----------|
| pandas, pyarrow | Silver cleaning, Gold reading, parquet IO |
| pandera | Data-quality contract enforcement |
| pyspark, delta-spark | Gold layer: PySpark + Spark SQL + Delta Lake |
| scikit-learn | market-gated LinearRegression + India crop-production HistGradientBoosting prototype (temporal split) |
| streamlit, plotly | dashboard |
| pytest | test runner |
| requests | weather_api.py live ingestion |
| apache-airflow | DAG orchestration (Airflow worker only) |

## 12. Running the pipeline

```bash
# full run — live weather pull from Open-Meteo for the 10 districts
# (falls back to the schema-identical fixture generator if unreachable)
python run_pipeline.py

# skip re-ingestion — regenerate silver/gold/ML from existing bronze
python run_pipeline.py --skip-ingestion
```

Individual stages:

```bash
python src/ingestion/weather_api.py --config catalog/india_regions.json --out data/india/bronze
python src/transform/bronze_to_silver.py
python src/transform/india_silver_to_gold.py
python src/transform/silver_to_gold.py
python src/ml/india_forecast.py       # crop-production prototype
python src/quality/monitor.py
python src/ml/prediction_pipeline.py
```

Update the tracked demo bundle after a successful run:

```bash
python scripts/export_demo.py
```

## 13. Running the tests

```bash
pytest tests/ -v
```

Expected: **all tests pass (no live APIs required)**. Coverage includes the
India contracts (real geography bounds, INR ₹ lakh/crore formatting, market
`not_configured` honesty), Pandera contracts, the market-gated forecast, the
India crop-production forecast (temporal split, prototype honesty), weather
segmentation, the DAG structure, and dashboard/artifact contracts.

## 14. Limitations

1. **Mandi data is historical AGMARKNET, not a live e-NAM feed.** Observed daily
   APMC prices/arrivals are real and attributed (see `src/ingestion/india_mandi.py`),
   but record freshness depends on the official dataset (this build covers
   `2026-04-01 → 2026-05-31` for the monitored districts, committed under
   `data/india/demo/`). A live e-NAM feed remains the documented extension point
   in `src/ingestion/india_market_source.py`. The product never fabricates prices
   or arrivals and never shows a price forecast under a `not_configured` market
   source.
2. **Forecast is gated by a live feed.** Because the model's target is the market
   price signal, no forecast is generated while the live market source is off;
   `ml_status.json` records the required upstream sources instead.
2b. **India crop-production forecast is a prototype.** It uses only the real
   DE&S/MoAFW Area-Production-Yield panel with a strict temporal split and
   honest model selection (gradient boosting did not beat last-year persistence
   on validation, so persistence is what is reported). Metrics come from held-out
   rows only; the 95% band is an approximate residual-based interval. Price and
   arrival forecasting was NOT attempted (61 days of daily mandi history is not a
   validation-capable series), and the model report (`india_model_report.json`)
   declares `production_ready: false`.
3. **Small sample.** The model architecture trains on a small monitored set
   (10 districts); any reported metrics are in-sample and not validated
   out-of-sample accuracy.
4. **Sandbox weather.** When Open-Meteo is unreachable (network-restricted
   sandbox), the fixture generator emits schema-identical synthetic weather from
   seeded seasonal baselines — realistic for a demo, not observational.
5. **Delta Lake.** `delta-spark` needs Maven Central access for its JVM
   connector; otherwise the pipeline falls back to partitioned Parquet with a
   logged warning. Delta works with zero code changes with normal internet.
6. **Windows + PySpark.** Spark on Windows needs `winutils.exe` + `hadoop.dll`
   and a valid `JAVA_HOME`; the repo bundles `hadoop-utils/bin` and
   `run_pipeline.py` sets the environment automatically.
7. **Monitoring freshness.** The 24-hour freshness window fails on a stale
   `--skip-ingestion` run; re-ingest (or generate fixtures) first.
8. **Airflow on Windows.** Airflow itself cannot import on native Windows; run
   the DAG on a Linux host/WSL2 or validate via `pytest tests/test_dag.py`.
9. **Scenario estimates.** The Scenario Planner is a deterministic estimate
   under the user's own assumptions (₹/quintal quantities, prices, costs,
   losses) — never a guarantee of revenue or profit.

## 15. Deployment (Streamlit Community Cloud)

Entry point `app.py` (`python -m streamlit run app.py`). The deployed instance
has no PySpark pipeline, so it reads the **tracked real outputs under
`data/india/demo/`**; locally, artifacts are read from the generated
`data/india/gold|silver|catalog` paths. Data Health labels every artifact's
source, so the deployed app never claims a live pipeline run it did not perform.
Refresh the demo bundle with `python scripts/export_demo.py`.

## 16. Demo Flow

1. `python run_pipeline.py --skip-ingestion` (or full run for a live weather pull).
2. `python -m streamlit run app.py`.
3. **Home** — monitored districts, weather snapshot, data quality, ML/market status.
4. **My District** — pick a State/UT then a District/APMC; read its location,
   weather profile, segment, and insight cards.
5. **Market Core** — compare weather across districts; exposure tiers and the
   honest market-source banner.
6. **Market Intelligence** — real observed AGMARKNET APMC prices/arrivals with
   weather context.
7. **India Forecast** — crop-production prototype (2023-24): historical trend,
   95% band, observed-vs-predicted, methodology, data period, and honest
   warnings/limitations read from `india_model_report.json`.
8. **U.S. Forecast** — see the market-gate status: why no price forecast exists
   yet, and exactly which upstream sources are required.
9. **Opportunity Scanner** — weather-exposure segments with per-district reasons.
10. **Scenario Planner** — enter ₹/quintal assumptions; view estimated
    revenue/cost/margin with ±20% sensitivity.
11. **Data Health** — status, checks, latest weather ingestion, artifact sources.
12. **Sources & Methodology** — provenance, honest limitations, market extension point.

Optional: `pytest tests/ -v` to verify the full validation suite offline.