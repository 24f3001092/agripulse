"""
run_pipeline.py

Single-command orchestrator for the full AgriPulse pipeline:

    1. Ingest        - weather fixture / live API (market + geo are static CSVs in bronze/)
    2. Bronze->Silver  - clean, validate (pandera)
    3. Silver->Gold    - join, aggregate, write Delta/Parquet (PySpark + Spark SQL)
    4. Monitor         - freshness + row-count health report
    5. Prediction      - export forecast + regional segmentation (ML outputs)

Usage:
    python run_pipeline.py                   # full run (re-ingests weather)
    python run_pipeline.py --skip-ingestion  # reuse existing bronze data

This is what dags/agripulse_dag.py calls stage-by-stage under Airflow; running
this file directly is the equivalent of a manual/local run.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("run_pipeline")

BASE = Path(__file__).resolve().parent
SRC = BASE / "src"
HADOOP_UTILS = BASE / "hadoop-utils" / "bin"


def resolve_java_home() -> str:
    """
    Return a valid JAVA_HOME. PySpark resolves the JVM from JAVA_HOME if it is
    set, so a stale/invalid JAVA_HOME (as seen on some dev machines) breaks the
    Spark stage. If the configured JAVA_HOME is unusable, fall back to locating
    `java` on PATH and deriving its home; return an empty string if none is found.
    """
    configured = os.environ.get("JAVA_HOME", "")
    if configured and (Path(configured) / "bin" / "java.exe").exists():
        return configured

    java_path = shutil.which("java")
    if java_path:
        java_home = Path(java_path).resolve().parent.parent
        logger.warning(
            f"JAVA_HOME ({configured!r}) is not a valid JDK root; using {java_home} "
            f"(derived from `java` on PATH)."
        )
        return str(java_home)

    logger.warning("No valid JAVA_HOME found; PySpark will fail if Java is not on PATH.")
    return configured


def build_subprocess_env() -> dict:
    """
    Environment for child stages. On Windows the Spark (JVM) stage additionally
    needs:
      * a valid JAVA_HOME   (see resolve_java_home)
      * winutils.exe + hadoop.dll for Hadoop native file ops -> HADOOP_HOME and
        PATH must point at the bundled hadoop-utils/bin directory.
    """
    env = dict(os.environ)
    env["JAVA_HOME"] = resolve_java_home() or env.get("JAVA_HOME", "")

    if HADOOP_UTILS.exists() and (HADOOP_UTILS / "winutils.exe").exists():
        env["HADOOP_HOME"] = str(HADOOP_UTILS.parent)
        env["PATH"] = str(HADOOP_UTILS) + os.pathsep + env.get("PATH", "")
    else:
        logger.warning(
            "hadoop-utils/winutils.exe not found; the Spark stage may fail on Windows "
            "(see README 'Limitations')."
        )
    return env


def latest_ingestion_status() -> str:
    """Read the newest weather ingestion's status from its meta file (e.g. 'success')."""
    meta_files = sorted((BASE / "data" / "bronze").glob("*.meta.json"))
    if not meta_files:
        return ""
    import json as _json
    try:
        return str(_json.loads(meta_files[-1].read_text()).get("status", ""))
    except Exception:
        return ""


def run_step(name: str, cmd: list, cwd: Path, env: dict):
    logger.info(f"=== STAGE: {name} ===")
    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if result.returncode != 0:
        logger.error(f"Stage '{name}' failed with exit code {result.returncode} -- halting pipeline")
        sys.exit(result.returncode)
    logger.info(f"=== STAGE COMPLETE: {name} ===\n")


def build_summary(ingested: bool) -> str:
    """Plain-text summary of every artifact the completed run should have produced."""
    lines = [
        "==================================================",
        " AgriPulse pipeline run complete",
        "==================================================",
        f" Ingestion executed            : {'yes' if ingested else 'skipped (--skip-ingestion)'}",
        " Expected outputs (%s):" % ("refreshed" if ingested else "verified present"),
    ]
    for rel in [
        "data/bronze/weather_raw_*.json",
        "data/silver/weather.parquet",
        "data/silver/ag_exports.parquet",
        "data/silver/geo.parquet",
        "data/gold/region_daily_features",
        "data/gold/region_summary",
        "catalog/health_report.json",
        "data/gold/forecast_results.parquet",
        "data/gold/region_segments.parquet",
        "catalog/model_report.json",
        "catalog/prediction_run.json",
    ]:
        marker = "OK" if _exists_loose(rel) else "MISSING"
        lines.append(f"   [{marker:7}] {rel}")
    lines.append("==================================================")
    return "\n".join(lines)


def _exists_loose(rel: str) -> bool:
    """Best-effort existence check that tolerates glob patterns."""
    if "*" in rel:
        return bool(list(BASE.glob(rel)))
    return (BASE / rel).exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ingestion", action="store_true",
                        help="Reuse existing bronze/ data instead of re-fetching")
    args = parser.parse_args()

    py = sys.executable
    env = build_subprocess_env()
    ingested = not args.skip_ingestion

    if ingested:
        run_step(
            "Ingestion: market + geo (real external sources, already static CSVs in bronze/)",
            [py, "-c", "print('Static sources already fetched to data/bronze/ -- see README.md to re-fetch')"],
            BASE, env,
        )
        run_step(
            "Ingestion: weather (live Open-Meteo API)",
            [py, str(SRC / "ingestion" / "weather_api.py"),
             "--config", str(BASE / "catalog" / "regions.json"),
             "--out", str(BASE / "data" / "bronze")],
            BASE, env,
        )
        # Honest sandbox fallback (see README 'Limitations'): if the live API
        # was unreachable, fall back to the repo's schema-identical synthetic
        # fixture generator so the rest of the pipeline can still be exercised.
        if latest_ingestion_status() != "success":
            logger.warning(
                "Live Open-Meteo ingestion did not report success (likely no outbound "
                "internet access in this sandbox). Falling back to the project's "
                "schema-identical fixture generator."
            )
            run_step(
                "Ingestion: weather (sandbox fixture fallback)",
                [py, str(SRC / "ingestion" / "weather_fixture_generator.py"),
                 "--config", str(BASE / "catalog" / "regions.json"),
                 "--out", str(BASE / "data" / "bronze")],
                BASE, env,
            )

    run_step("Bronze -> Silver", [py, str(SRC / "transform" / "bronze_to_silver.py")], BASE, env)
    run_step("Silver -> Gold", [py, str(SRC / "transform" / "silver_to_gold.py")], BASE, env)
    run_step("Monitoring", [py, str(SRC / "quality" / "monitor.py")], BASE, env)
    run_step("Prediction pipeline (forecast + segmentation)",
             [py, str(SRC / "ml" / "prediction_pipeline.py")], BASE, env)

    print()
    print(build_summary(ingested))
    logger.info("Pipeline run complete. Launch the dashboard with: streamlit run app.py")


if __name__ == "__main__":
    main()