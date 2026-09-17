"""
silver_to_gold.py
Joins the three Silver datasets (weather, ag_exports, geo) using PySpark +
Spark SQL, aggregates to a region-level daily feature table, and writes the
result as a Delta Lake table partitioned by region.

DELTA LAKE NOTE:
  This script tries to initialize a Delta-enabled SparkSession first. Delta
  Lake's JVM connector jar is fetched from Maven Central at runtime by
  `delta-spark` (there is no way around this -- it's how the library works
  everywhere, not just here). If Maven Central is unreachable (as in this
  sandbox, which allowlists pypi/npm/github but not repo1.maven.org), the
  script automatically falls back to writing partitioned Parquet instead,
  logs a clear warning, and continues -- so the pipeline never silently
  produces nothing. On Databricks or any normal machine, the Delta path
  just works and DELTA_AVAILABLE will be True.

Run:
    python silver_to_gold.py
"""

import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("silver_to_gold")

BASE = Path(__file__).resolve().parents[2]
SILVER = BASE / "data" / "silver"
GOLD = BASE / "data" / "gold"


def clean_output_dir(path: Path) -> None:
    """
    Fully remove a Gold output directory before overwriting it.

    Spark's Delta `mode("overwrite")` correctly replaces a *previous Delta
    table* but leaves stale data files behind when the existing directory was
    written by the plain-Parquet fallback (mixed-format upgrade path). Re-running
    the pipeline would then accumulate duplicate rows in the Gold layer and
    corrupt downstream ML row counts. Because this table is fully recomputed
    every run, deleting the stale directory first keeps every run idempotent.
    """
    if path.exists():
        shutil.rmtree(path)
        logger.info(f"Cleared stale Gold output dir: {path}")


def get_spark_session():
    """
    Attempts a Delta-enabled SparkSession. Falls back to plain Spark
    (Parquet output) if the Delta jar can't be resolved from Maven Central.
    Returns (spark, delta_available: bool).
    """
    from pyspark.sql import SparkSession

    try:
        from delta import configure_spark_with_delta_pip
        builder = (
            SparkSession.builder.appName("agripulse-silver-to-gold")
            .master("local[*]")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        # Force a trivial action to confirm the Delta jar actually resolved
        spark.sql("SELECT 1").collect()
        logger.info("Delta Lake JVM connector resolved successfully -- writing native Delta tables")
        return spark, True
    except Exception as e:
        logger.warning(f"Delta Lake unavailable in this environment ({type(e).__name__}); "
                        f"falling back to partitioned Parquet. Root cause: {str(e)[:200]}")
        spark = SparkSession.builder.appName("agripulse-silver-to-gold").master("local[*]").getOrCreate()
        return spark, False


def build_gold_table(spark, delta_available: bool):
    weather = spark.read.parquet(str(SILVER / "weather.parquet"))
    ag = spark.read.parquet(str(SILVER / "ag_exports.parquet"))
    geo = spark.read.parquet(str(SILVER / "geo.parquet"))

    for df, name in [(weather, "weather"), (ag, "ag_exports"), (geo, "geo")]:
        df.createOrReplaceTempView(name)

    # Real Spark SQL: join weather to geo (region<->state) and to ag export
    # market data, then aggregate to a region-level daily feature table.
    # This mirrors exactly what the JD calls "structuring external datasets
    # in Delta Lake for downstream ML" -- weather + market + geo joined into
    # one feature-ready table.
    gold_sql = """
        WITH region_geo AS (
            SELECT g.state_name, g.capital, g.latitude, g.longitude
            FROM geo g
        ),
        weather_enriched AS (
            SELECT
                w.region_name,
                w.date,
                w.temp_max_c,
                w.temp_min_c,
                (w.temp_max_c + w.temp_min_c) / 2 AS temp_avg_c,
                w.precipitation_mm,
                w.humidity_pct,
                w.windspeed_max_kmh
            FROM weather w
        )
        SELECT
            we.region_name,
            we.date,
            we.temp_max_c,
            we.temp_min_c,
            we.temp_avg_c,
            we.precipitation_mm,
            we.humidity_pct,
            we.windspeed_max_kmh,
            a.state_code,
            a.total_exports_musd,
            a.corn_musd,
            a.wheat_musd,
            a.cotton_musd,
            a.dairy_musd,
            rg.latitude,
            rg.longitude
        FROM weather_enriched we
        LEFT JOIN region_geo rg
            ON we.region_name = rg.state_name
        LEFT JOIN ag_exports a
            ON we.region_name = a.state_name
    """
    gold_df = spark.sql(gold_sql)

    logger.info(f"Gold feature table row count: {gold_df.count()}")
    gold_df.show(5, truncate=False)

    GOLD.mkdir(parents=True, exist_ok=True)
    gold_path = GOLD / "region_daily_features"
    clean_output_dir(gold_path)

    if delta_available:
        (gold_df.write.format("delta")
         .mode("overwrite")
         .partitionBy("region_name")
         .save(str(gold_path)))
        logger.info(f"Wrote Delta table -> {gold_path}")
    else:
        (gold_df.write.format("parquet")
         .mode("overwrite")
         .partitionBy("region_name")
         .save(str(gold_path)))
        logger.info(f"Wrote partitioned Parquet (Delta fallback) -> {gold_path}")

    # Region-level summary table: one row per region, aggregated -- the kind
    # of table a sales-forecasting model or BI dashboard would actually query.
    summary_sql = """
        SELECT
            region_name,
            state_code,
            ROUND(AVG(temp_avg_c), 2) AS avg_temp_c,
            ROUND(SUM(precipitation_mm), 1) AS total_precip_mm,
            ROUND(AVG(humidity_pct), 1) AS avg_humidity_pct,
            MAX(total_exports_musd) AS total_exports_musd,
            MAX(corn_musd) AS corn_musd,
            MAX(wheat_musd) AS wheat_musd,
            MAX(cotton_musd) AS cotton_musd
        FROM gold_view
        GROUP BY region_name, state_code
        ORDER BY total_exports_musd DESC
    """
    gold_df.createOrReplaceTempView("gold_view")
    summary_df = spark.sql(summary_sql)
    summary_df.show(20, truncate=False)

    summary_path = GOLD / "region_summary"
    clean_output_dir(summary_path)
    fmt = "delta" if delta_available else "parquet"
    summary_df.write.format(fmt).mode("overwrite").save(str(summary_path))
    logger.info(f"Wrote region summary table ({fmt}) -> {summary_path}")

    return gold_df, summary_df


def main():
    spark, delta_available = get_spark_session()
    try:
        build_gold_table(spark, delta_available)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
