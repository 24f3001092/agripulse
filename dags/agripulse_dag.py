"""
agripulse_dag.py
Airflow DAG orchestrating the AgriPulse INDIA pipeline daily.

Task dependency graph:

    ingest_weather ─┐
    ingest_market   ├──> bronze_to_silver ──────> silver_to_gold ──────────────┐
    ingest_geo     ─┘            │                                             │
                                 └──> india_gold_summary ──> india_forecast ───┤
                                                                               └──> monitor ──> prediction_pipeline

Drop this file into your Airflow $AIRFLOW_HOME/dags/ directory. It assumes the
AgriPulse repo is checked out at PROJECT_ROOT on the Airflow worker.

Environment notes:
  * On Linux/Airflow workers: JAVA_HOME must be set for PySpark (see
    worker-level env in airflow.cfg or --env option).
  * On Windows sandbox: also set HADOOP_HOME to hadoop-utils/ and prepend
    hadoop-utils/bin to PATH for PySpark's Hadoop native file ops. The local
    run_pipeline.py script handles this automatically; this DAG is the
    Airflow-side equivalent and assumes a properly configured worker env.

India climate profile (active country on this branch):
  * geography: India > State/UT > District > Mandi/APMC (catalog/india_regions.json)
  * weather:   live Open-Meteo API for monitored districts
  * mandi:     daily AGMARKNET APMC price/arrival record import
               (src/ingestion/india_mandi.py), summarized to Gold by the
               india_gold_summary task (region x date features, crop summary,
               market summary; graceful no_data manifests without bronze)
  * market:    e-NAM live feed -- OFFICIAL extension point. Until a loader is
               registered (india_market_source.py) and pipeline_config sets
               market_source: configured, the ingest_market task is a documented
               no-op and the market-price forecast honestly reports not_generated.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule

PROJECT_ROOT = "/opt/agripulse"  # adjust for your environment
PYTHON_BIN = "python3"
BASH_ENV_EXPORTS = (
    ""  # On Airflow workers, set JAVA_HOME + HADOOP_HOME via the worker's
    # own environment config, not inline here. This placeholder documents
    # that the Silver->Gold stage needs a valid JAVA_HOME and (on Windows)
    # HADOOP_HOME pointing to a hadoop-utils/bin with winutils.exe.
)

default_args = {
    "owner": "data-engineering",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": ["data-eng-alerts@example.com"],
}

with DAG(
    dag_id="agripulse_external_data_pipeline",
    description=(
        "Ingest India weather/geo data -> Silver -> Gold -> India crop-production "
        "forecast (prototype) -> monitor health -> weather-exposure segmentation "
        "+ market-gated forecast (e-NAM/AGMARKNET)"
    ),
    default_args=default_args,
    schedule_interval="0 4 * * *",  # daily at 04:00 UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["agripulse", "external-data", "commercial-it"],
) as dag:

    ingest_weather = BashOperator(
        task_id="ingest_weather",
        bash_command=(
            f"cd {PROJECT_ROOT} && {PYTHON_BIN} src/ingestion/weather_api.py "
            f"--config catalog/india_regions.json --out data/india/bronze"
        ),
    )

    ingest_market = BashOperator(
        task_id="ingest_market",
        bash_command=(
            "echo 'e-NAM (enam.gov.in) / AGMARKNET (agmarknet.gov.in) market pull -- "
            "NOT INTEGRATED. Register a loader in src/ingestion/india_market_source.py "
            "and set market_source=configured in catalog/pipeline_config.json. "
            "No market values are fabricated while unconfigured.'"
        ),
    )

    ingest_geo = BashOperator(
        task_id="ingest_geo",
        bash_command=(
            "echo 'India geography reference -- static in this build "
            "(catalog/india_regions.json); production would refine coordinates "
            "against an official gazetteer / boundary service.'"
        ),
    )

    bronze_to_silver = BashOperator(
        task_id="bronze_to_silver",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} src/transform/bronze_to_silver.py",
    )

    # India integrated agricultural Gold layer: region x date features, the
    # Location + Crop (+ Date/Year) crop summary, and the AGMARKNET market
    # summary (reusing mandi_to_gold). Graceful no_data manifests when the
    # corresponding Silver inputs are absent.
    india_gold_summary = BashOperator(
        task_id="india_gold_summary",
        bash_command=(
            f"cd {PROJECT_ROOT} && {PYTHON_BIN} src/transform/india_silver_to_gold.py"
        ),
    )

    silver_to_gold = BashOperator(
        task_id="silver_to_gold",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} src/transform/silver_to_gold.py",
    )

    # India crop-production forecast: strict temporal split, prototype only.
    # Trains on the 1997..2023 DE&S/MoAFW panel (india_crop_year); prices and
    # arrivals are NOT forecast (insufficient daily history) -- nothing is
    # fabricated. Writes gold/india_forecasts.parquet + catalog/india_model_report.json.
    india_forecast = BashOperator(
        task_id="india_forecast",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} src/ml/india_forecast.py",
    )

    monitor = BashOperator(
        task_id="monitor_pipeline_health",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} src/quality/monitor.py",
        # Run even on partial upstream failure to surface health status
        trigger_rule=TriggerRule.ALL_DONE,
    )

    prediction_pipeline = BashOperator(
        task_id="prediction_pipeline",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} src/ml/prediction_pipeline.py",
    )

    (
        [ingest_weather, ingest_market, ingest_geo]
        >> bronze_to_silver
    )
    bronze_to_silver >> silver_to_gold
    bronze_to_silver >> india_gold_summary >> india_forecast
    [silver_to_gold, india_forecast] >> monitor
    monitor >> prediction_pipeline
