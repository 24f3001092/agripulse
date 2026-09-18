"""
silver_to_gold.py
Joins the Silver datasets (weather, geo, and -- when configured -- market) for
the active country profile (India) using PySpark + Spark SQL, aggregates to a
district-level daily feature table, and writes the result as a Delta Lake table
partitioned by district.

GEOGRAPHY: India > State/UT > District > Mandi/APMC. The district (region_name)
is the monitoring unit; state/UT, district and mandi/APMC attributes ride along
so no downstream code needs to re-join the geography reference.

MARKET: the e-NAM / AGMARKNET market table is ONLY joined when a real loader is
configured (india_market_source). When absent, no market columns are produced
and nothing is fabricated. If present, market rows are pre-aggregated to
region x date (mean price per quintal, total arrivals in quintals) before join.

DELTA LAKE NOTE: tries a Delta-enabled SparkSession first; if Maven Central is
unreachable it falls back to partitioned Parquet with a clear warning. Delta
tables are fully recomputed every run (stale output dir removed for idempotency).

Run:
    python silver_to_gold.py
"""

import logging
import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("silver_to_gold")

BASE = Path(__file__).resolve().parents[2]
SILVER = config.SILVER
GOLD = config.GOLD


def clean_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        logger.info(f"Cleared stale Gold output dir: {path}")


def get_spark_session():
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
        spark.sql("SELECT 1").collect()
        logger.info("Delta Lake JVM connector resolved successfully -- writing native Delta tables")
        return spark, True
    except Exception as e:
        logger.warning(f"Delta Lake unavailable in this environment ({type(e).__name__}); "
                        f"falling back to partitioned Parquet. Root cause: {str(e)[:200]}")
        spark = SparkSession.builder.appName("agripulse-silver-to-gold").master("local[*]").getOrCreate()
        return spark, False


def build_gold_table(spark, delta_available: bool):
    market_enabled = (SILVER / "market.parquet").exists()

    weather = spark.read.parquet(str(SILVER / "weather.parquet"))
    geo = spark.read.parquet(str(SILVER / "geo.parquet"))
    for df, name in [(weather, "weather"), (geo, "geo")]:
        df.createOrReplaceTempView(name)

    if market_enabled:
        market = spark.read.parquet(str(SILVER / "market.parquet"))
        # Aggregate commodity-level market rows to region x date BEFORE joining so
        # the daily feature grain (one row per district per date) is preserved.
        market.createOrReplaceTempView("market_raw")
        spark.sql(
            """
            CREATE OR REPLACE TEMP VIEW market AS
            SELECT
                district AS region_name,
                date,
                ROUND(AVG(market_price_inr_per_quintal), 2) AS avg_market_price_inr_per_quintal,
                ROUND(SUM(arrivals_quintal), 2) AS total_arrivals_quintal
            FROM market_raw
            GROUP BY district, date
            """
        )

    # Real Spark SQL: weather joined to the India geography reference, with the
    # (optional) market signal joined per region/date. Mirrors the external-data
    # structuring requirement: weather + geo + market -> one feature-ready table.
    market_join = (
        "LEFT JOIN market m ON we.region_name = m.region_name AND we.date = m.date"
        if market_enabled else ""
    )
    market_cols = (
        ",\n            m.avg_market_price_inr_per_quintal,\n            m.total_arrivals_quintal" if market_enabled else ""
    )
    gold_sql = f"""
        WITH weather_enriched AS (
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
            rg.state_ut,
            rg.state_code,
            rg.district,
            rg.mandi_apmc,
            rg.latitude,
            rg.longitude{market_cols}
        FROM weather_enriched we
        LEFT JOIN geo rg
            ON we.region_name = rg.region_name
        {market_join}
    """
    gold_df = spark.sql(gold_sql)

    logger.info(f"Gold feature table row count: {gold_df.count()}")
    gold_df.show(5, truncate=False)

    GOLD.mkdir(parents=True, exist_ok=True)
    gold_path = GOLD / "region_daily_features"
    clean_output_dir(gold_path)
    fmt = "delta" if delta_available else "parquet"
    gold_df.write.format(fmt).mode("overwrite").partitionBy("region_name").save(str(gold_path))
    logger.info(f"Wrote Gold daily features ({fmt}) -> {gold_path}")

    market_summary = (
        ",\n            ROUND(AVG(avg_market_price_inr_per_quintal), 2) AS avg_market_price_inr_per_quintal,\n"
        "            ROUND(SUM(total_arrivals_quintal), 1) AS total_arrivals_quintal" if market_enabled else ""
    )
    summary_sql = f"""
        SELECT
            region_name,
            state_ut,
            state_code,
            district,
            mandi_apmc,
            ROUND(AVG(temp_avg_c), 2) AS avg_temp_c,
            ROUND(SUM(precipitation_mm), 1) AS total_precip_mm,
            ROUND(AVG(humidity_pct), 1) AS avg_humidity_pct{market_summary}
        FROM gold_view
        GROUP BY region_name, state_ut, state_code, district, mandi_apmc
        ORDER BY state_ut, region_name
    """
    gold_df.createOrReplaceTempView("gold_view")
    summary_df = spark.sql(summary_sql)
    summary_df.show(20, truncate=False)

    summary_path = GOLD / "region_summary"
    clean_output_dir(summary_path)
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