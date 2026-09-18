"""
export_demo.py
Export real, freshly-generated pipeline outputs into the tracked demo datasets
(data/india/demo/) that the deployed dashboard falls back to when local
generated files are absent (e.g. a fresh Streamlit Community Cloud checkout).

The demo files are REAL outputs (not fabricated): actual weather, geography,
Gold features, monitor report, ML status, segments, the AGMARKNET mandi silver,
the integrated India Gold layer (region x date features, crop summary, market
summary) and lineage produced by the last full pipeline run. Silver/Gold tables
that are Delta partition directories on disk are read via Spark and written as
single-file Parquet for the demo layer.

Agriculture is deliberately exported as the bounded REAL subset for the
monitored districts only (keeping the tracked demo layer small), matching the
documented demo philosophy in the README.

    python scripts/export_demo.py
"""

import json
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("export_demo")

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
import config  # noqa: E402

DEMO = config.DEMO


def export_gold_table(spark, table_name: str):
    from pyspark.sql import functions as F

    src = config.GOLD / table_name
    dest_dir = DEMO / "gold"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{table_name}.parquet"
    # The Gold tables are Delta directories; reading the part parquet files
    # (recursive lookup) avoids requiring a delta-enabled SparkSession here.
    df = spark.read.option("recursiveFileLookup", "true").parquet(str(src))
    if table_name == "region_daily_features":
        # Delta stores the partition column value only in the transaction log;
        # the on-disk part files carry it as a path segment. Reconstruct
        # region_name from the file path so the demo copy is self-contained.
        df = (
            df.withColumn("_path", F.input_file_name())
              .withColumn("region_name", F.regexp_extract("_path", r"region_name=([^/]*)", 1))
              .drop("_path")
        )
    df.coalesce(1).write.mode("overwrite").format("parquet").save(str(dest_dir / f"_{table_name}_tmp"))
    part = next((dest_dir / f"_{table_name}_tmp").glob("*.parquet"))
    shutil.move(str(part), str(dest))
    shutil.rmtree(str(dest_dir / f"_{table_name}_tmp"), ignore_errors=True)
    logger.info(f"Exported Gold {table_name} -> {dest}")


def export_silver_table(name: str):
    src = config.SILVER / f"{name}.parquet"
    if not src.exists():
        logger.warning(f"Silver {name} absent; skipping (expected, e.g. market not configured)")
        return
    dest = DEMO / "silver" / f"{name}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    logger.info(f"Exported Silver {name} -> {dest}")


def export_catalog_files():
    dest = DEMO / "catalog"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ["health_report.json", "ml_status.json", "prediction_run.json",
                 "gold_lineage.json", "india_model_report.json"]:
        src = config.CATALOG / name
        if src.exists():
            shutil.copy2(src, dest / name)
            logger.info(f"Exported catalog/{name} -> {dest / name}")


def export_flat_gold_output(name: str, table_dir: str):
    """Copy a freshly generated flat Parquet India-Gold output (plus manifest)."""
    src_dir = config.GOLD / table_dir
    if not src_dir.exists():
        logger.warning(f"Gold {table_dir} absent; skipping (expected when silver inputs missing)")
        return
    dest = DEMO / "gold" / f"{name}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_dir / f"{table_dir}.parquet", dest)
    logger.info(f"Exported Gold {name} -> {dest}")


def export_agriculture_subset():
    """Export the REAL agriculture rows for the monitored districts only.

    Keeps the tracked demo layer bounded (~7.5k rows) while the full silver
    table (~455k rows, several MB) stays a gitignored local artifact.
    """
    import pandas as pd

    src = config.SILVER / "india_agriculture.parquet"
    geo_src = config.SILVER / "geo.parquet"
    if not src.exists() or not geo_src.exists():
        logger.warning("Silver india_agriculture/geo absent; skipping")
        return
    agri = pd.read_parquet(src)
    monitored = set(pd.read_parquet(geo_src)["district"].astype(str).str.strip())
    subset = agri[agri["district"].astype(str).str.strip().isin(monitored)]
    dest_dir = DEMO / "silver"
    dest_dir.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(dest_dir / "india_agriculture.parquet", index=False)
    logger.info(f"Exported agriculture subset ({len(subset)} monitored rows) -> "
                f"{dest_dir / 'india_agriculture.parquet'}")


def export_crop_year_subset():
    """Monitored-district subset of the long-form crop/year production panel.

    The full india_crop_year table (455k rows) is the modelling input for the
    India forecast; the demo copy is intentionally bounded to the monitored
    districts so the tracked demo layer stays small.
    """
    import pandas as pd

    src = config.GOLD / "india_crop_summary" / "india_crop_year.parquet"
    geo_src = config.SILVER / "geo.parquet"
    if not src.exists() or not geo_src.exists():
        logger.warning("india_crop_year / geo absent; skipping")
        return
    panel = pd.read_parquet(src)
    monitored = set(pd.read_parquet(geo_src)["district"].astype(str).str.strip())
    subset = panel[panel["district"].astype(str).str.strip().isin(monitored)]
    dest_dir = DEMO / "gold"
    dest_dir.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(dest_dir / "india_crop_year.parquet", index=False)
    logger.info(f"Exported crop-year subset ({len(subset)} monitored rows) -> "
                f"{dest_dir / 'india_crop_year.parquet'}")


def export_bronze_meta():
    meta_files = sorted(config.BRONZE.glob("*.meta.json"))
    if not meta_files:
        return
    dest = DEMO / "bronze"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(meta_files[-1], dest / "latest_weather.meta.json")
    logger.info(f"Exported bronze meta -> {dest / 'latest_weather.meta.json'}")


def main():
    # Windows-only: PySpark needs a valid JAVA_HOME + hadoop-utils (winutils,
    # hadoop.dll) on PATH, exactly as run_pipeline.py resolves them.
    import os
    sys.path.insert(0, str(BASE))
    import run_pipeline  # noqa: E402  (repo-root sibling of scripts/)

    env = run_pipeline.build_subprocess_env()
    for k, v in env.items():
        os.environ[k] = v

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("agripulse-export-demo").master("local[*]").getOrCreate()
    try:
        DEMO.mkdir(parents=True, exist_ok=True)
        for table in ["region_daily_features", "region_summary"]:
            export_gold_table(spark, table)
    finally:
        spark.stop()

    export_silver_table("weather")
    export_silver_table("geo")
    export_silver_table("india_mandi")
    export_agriculture_subset()
    export_catalog_files()
    export_bronze_meta()

    export_flat_gold_output("india_region_features", "india_region_features")
    export_flat_gold_output("india_crop_summary", "india_crop_summary")

    market_src = config.GOLD / "india_market_summary"
    if market_src.exists():
        dest_dir = DEMO / "gold" / "india_market_summary"
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in market_src.glob("*"):
            if f.suffix in (".parquet", ".json"):
                shutil.copy2(f, dest_dir / f.name)
        logger.info(f"Exported Gold india_market_summary -> {dest_dir}")

    export_flat_gold_output("india_region_features", "india_region_features")
    export_flat_gold_output("india_crop_summary", "india_crop_summary")
    forecasts_src = config.GOLD / "india_forecasts.parquet"
    if forecasts_src.exists():
        dest = DEMO / "gold" / "india_forecasts.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(forecasts_src, dest)
        logger.info(f"Exported Gold india_forecasts -> {dest}")
    export_crop_year_subset()

    segments_src = config.GOLD / "region_segments.parquet"
    if segments_src.exists():
        dest = DEMO / "gold" / "region_segments.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(segments_src, dest)
        logger.info(f"Exported segments -> {dest}")

    logger.info("Demo export complete -> data/india/demo/")


if __name__ == "__main__":
    main()