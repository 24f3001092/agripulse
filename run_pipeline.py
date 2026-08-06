"""
run_pipeline.py
Single-command orchestrator for the full AgriPulse pipeline:
    1. Ingest (weather fixture / market / geo already in bronze/)
    2. Bronze -> Silver (clean, validate)
    3. Silver -> Gold (join, aggregate, write Delta/Parquet)
    4. Monitor (freshness + row-count checks)

This is what dags/agripulse_dag.py calls stage-by-stage under Airflow;
running this file directly is the equivalent of a manual/local run.

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-ingestion   # reuse existing bronze data
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("run_pipeline")

BASE = Path(__file__).resolve().parent
SRC = BASE / "src"


def run_step(name: str, cmd: list, cwd: Path):
    logger.info(f"=== STAGE: {name} ===")
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        logger.error(f"Stage '{name}' failed with exit code {result.returncode} -- halting pipeline")
        sys.exit(result.returncode)
    logger.info(f"=== STAGE COMPLETE: {name} ===\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ingestion", action="store_true",
                         help="Reuse existing bronze/ data instead of re-fetching")
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_ingestion:
        run_step(
            "Ingestion: market + geo (real external sources, already static CSVs in bronze/)",
            [py, "-c", "print('Static sources already fetched to data/bronze/ -- see docs/README.md to re-fetch')"],
            BASE,
        )
        run_step(
            "Ingestion: weather",
            [py, str(SRC / "ingestion" / "weather_api.py"),
             "--config", str(BASE / "catalog" / "regions.json"),
             "--out", str(BASE / "data" / "bronze")],
            BASE,
        )

    run_step("Bronze -> Silver", [py, str(SRC / "transform" / "bronze_to_silver.py")], BASE)
    run_step("Silver -> Gold", [py, str(SRC / "transform" / "silver_to_gold.py")], BASE)
    run_step("Monitoring", [py, str(SRC / "quality" / "monitor.py")], BASE)
    run_step("ML handoff", [py, str(SRC / "ml" / "demand_forecast_model.py")], BASE)

    logger.info("Pipeline run complete.")


if __name__ == "__main__":
    main()
