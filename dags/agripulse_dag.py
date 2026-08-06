"""
agripulse_dag.py
Airflow DAG orchestrating the AgriPulse external data pipeline daily.

Mirrors the JD's "Exposure to Apache Airflow or other workflow
orchestration tools" preferred qualification. Drop this file into your
Airflow $AIRFLOW_HOME/dags/ directory; it assumes the AgriPulse repo is
checked out at /opt/agripulse on the Airflow worker (adjust PROJECT_ROOT
below for your environment, e.g. a Databricks Repos path or a mounted
volume).

Task dependency graph:

    ingest_weather ─┐
    ingest_market   ├──> bronze_to_silver ──> silver_to_gold ──> monitor ──> ml_handoff
    ingest_geo     ─┘

market/geo ingestion are shown as separate tasks for a real deployment
where they'd hit live APIs on their own schedule; in this repo they are
pre-fetched static files, so those tasks are no-ops here but kept in the
graph to reflect the real production shape.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

PROJECT_ROOT = "/opt/agripulse"  # adjust for your environment
PYTHON_BIN = "python3"

default_args = {
    "owner": "data-engineering",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": ["data-eng-alerts@example.com"],
}

with DAG(
    dag_id="agripulse_external_data_pipeline",
    description="Ingest weather/market/geo data, build Delta Lake gold features for marketing segmentation & sales forecasting",
    default_args=default_args,
    schedule_interval="0 4 * * *",  # daily at 04:00 UTC, ahead of business-hours model refresh
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["agripulse", "external-data", "commercial-it"],
) as dag:

    ingest_weather = BashOperator(
        task_id="ingest_weather",
        bash_command=(
            f"cd {PROJECT_ROOT} && {PYTHON_BIN} src/ingestion/weather_api.py "
            f"--config catalog/regions.json --out data/bronze"
        ),
    )

    ingest_market = BashOperator(
        task_id="ingest_market",
        bash_command=(
            "echo 'Market data source refresh -- replace with real API/SFTP pull "
            "when moving beyond the static demo CSV in data/bronze/us_ag_exports_raw.csv'"
        ),
    )

    ingest_geo = BashOperator(
        task_id="ingest_geo",
        bash_command=(
            "echo 'Geospatial reference data refresh -- static in this build, "
            "would hook to a geocoding/boundary service in production'"
        ),
    )

    bronze_to_silver = BashOperator(
        task_id="bronze_to_silver",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} src/transform/bronze_to_silver.py",
    )

    silver_to_gold = BashOperator(
        task_id="silver_to_gold",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} src/transform/silver_to_gold.py",
    )

    monitor = BashOperator(
        task_id="monitor_pipeline_health",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} src/quality/monitor.py",
        trigger_rule=TriggerRule.ALL_DONE,  # run even if upstream had partial failures, to surface them
    )

    ml_handoff = BashOperator(
        task_id="ml_handoff_demo",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} src/ml/demand_forecast_model.py",
    )

    [ingest_weather, ingest_market, ingest_geo] >> bronze_to_silver >> silver_to_gold >> monitor >> ml_handoff
