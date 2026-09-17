"""
agripulse_dag.py
Airflow DAG orchestrating the AgriPulse pipeline daily.

Task dependency graph:

    ingest_weather ─┐
    ingest_market   ├──> bronze_to_silver ──> silver_to_gold ──> monitor ──> prediction_pipeline
    ingest_geo     ─┘

Drop this file into your Airflow $AIRFLOW_HOME/dags/ directory. It assumes the
AgriPulse repo is checked out at PROJECT_ROOT on the Airflow worker.

Environment notes:
  * On Linux/Airflow workers: JAVA_HOME must be set for PySpark (see
    worker-level env in airflow.cfg or --env option).
  * On Windows sandbox: also set HADOOP_HOME to hadoop-utils/ and prepend
    hadoop-utils/bin to PATH for PySpark's Hadoop native file ops. The local
    run_pipeline.py script handles this automatically; this DAG is the
    Airflow-side equivalent and assumes a properly configured worker env.

market/geo ingestion are shown as separate tasks for a real deployment
where they'd hit live APIs on their own schedule; in this repo they are
pre-fetched static CSVs, so those tasks are no-ops here but kept in the
graph to reflect the production shape.
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
        "Ingest weather/market/geo data -> Silver -> Gold -> monitor health "
        "-> run prototype forecast + regional segmentation"
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
        >> silver_to_gold
        >> monitor
        >> prediction_pipeline
    )
