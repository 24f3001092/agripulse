"""
run_pipeline.py

Single-command orchestrator for the full AgriPulse pipeline (active profile:
INDIA -- data/india/..., catalog/india/...):

    1. Ingest        - weather via live Open-Meteo API (or sandbox fixture)
                       + geography from catalog/india_regions.json
    2. Bronze->Silver  - clean, validate (pandera India contracts)
    3. Silver->Gold    - PySpark region features + integrated India agricultural
                         Gold layer (region x date features, crop summary,
                         market summary) under data/india/gold/
    4. India forecast  - one-year-ahead crop-production forecast (naive/gradient
                         boosting, temporal split; prototype, no fabrication)
    5. Monitor         - freshness, row counts + India artifact quality checks
                         (data/india/catalog/health_report.json)
    6. Prediction      - weather-exposure segmentation; market-price forecast is
                         GATED on the e-NAM/AGMARKNET source (ml_status.json)

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
sys.path.insert(0, str(SRC))
import config  # noqa: E402


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
    meta_files = sorted(config.BRONZE.glob("*.meta.json"))
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
    for rel, note in [
        ("data/india/bronze/weather_raw_*.json", ""),
        ("data/india/silver/weather.parquet", ""),
        ("data/india/silver/geo.parquet", ""),
        ("data/india/gold/region_daily_features", ""),
        ("data/india/gold/region_summary", ""),
        ("data/india/catalog/health_report.json", ""),
        ("data/india/gold/region_segments.parquet", ""),
        ("data/india/catalog/ml_status.json", ""),
        ("data/india/catalog/prediction_run.json", ""),
        ("data/india/gold/forecast_results.parquet", "[optional -- market-gated]"),
        ("data/india/catalog/model_report.json", "[optional -- market-gated]"),
        ("data/india/gold/india_market_summary/manifest.json", "[optional -- when mandi bronze exists]"),
        ("data/india/silver/india_mandi.parquet", "[optional -- when mandi bronze exists]"),
        ("data/india/silver/india_agriculture.parquet", "[optional -- when agriculture bronze exists]"),
        ("data/india/gold/india_region_features/india_region_features.parquet", "[optional -- when weather & mandi bronze exist]"),
        ("data/india/gold/india_region_features/manifest.json", "[optional -- when weather & mandi bronze exist]"),
        ("data/india/gold/india_crop_summary/india_crop_summary.parquet", "[optional -- when agriculture bronze exists]"),
        ("data/india/gold/india_crop_summary/manifest.json", "[optional -- when agriculture bronze exists]"),
        ("data/india/catalog/gold_lineage.json", ""),
        ("data/india/gold/india_forecasts.parquet", "[optional -- when agriculture bronze exists]"),
        ("data/india/catalog/india_model_report.json", "[optional -- when agriculture bronze exists]"),
    ]:
        marker = "OK" if _exists_loose(rel) else "MISSING"
        label = f"{rel} {note}".rstrip()
        lines.append(f"   [{marker:7}] {label}")
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
            "Ingestion: geography reference (catalog/india_regions.json -- official-source refinement pending)",
            [py, "-c", "print('India geography reference read from catalog/india_regions.json (see README for official-source refinement)')"],
            BASE, env,
        )
        run_step(
            "Ingestion: weather (live Open-Meteo API)",
            [py, str(SRC / "ingestion" / "weather_api.py"),
             "--config", str(config.REGIONS_CONFIG),
             "--out", str(config.BRONZE)],
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
                 "--config", str(config.REGIONS_CONFIG),
                 "--out", str(config.BRONZE)],
                BASE, env,
            )

    run_step("Bronze -> Silver", [py, str(SRC / "transform" / "bronze_to_silver.py")], BASE, env)
    run_step("Silver -> Gold (region features)", [py, str(SRC / "transform" / "silver_to_gold.py")], BASE, env)
    run_step("India integrated Gold (region features + crop summary + market summary)",
             [py, str(SRC / "transform" / "india_silver_to_gold.py")], BASE, env)
    run_step("India forecast (crop production, prototype -- temporal split, no fabrication)",
             [py, str(SRC / "ml" / "india_forecast.py")], BASE, env)
    run_step("Monitoring", [py, str(SRC / "quality" / "monitor.py")], BASE, env)
    run_step("Prediction pipeline (forecast + segmentation)",
             [py, str(SRC / "ml" / "prediction_pipeline.py")], BASE, env)

    print()
    print(build_summary(ingested))
    logger.info("Pipeline run complete. Launch the dashboard with: streamlit run app.py")


if __name__ == "__main__":
    main()