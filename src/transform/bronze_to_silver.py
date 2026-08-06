"""
bronze_to_silver.py
Cleans, standardizes, and validates the three Bronze datasets:
  1. Weather (Open-Meteo daily series, per region)
  2. Agricultural exports (state x commodity market data)
  3. Geospatial (state capital coordinates, used to join weather<->market)

Output: three Parquet files in data/silver/, each validated against its
pandera contract in src/quality/schemas.py. Any row failing validation is
written separately to data/silver/_rejects/ instead of being silently
dropped -- this is what "data quality monitoring" means in practice.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quality"))
from schemas import weather_schema, ag_exports_schema, geo_schema, validate_or_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("bronze_to_silver")

BASE = Path(__file__).resolve().parents[2]
BRONZE = BASE / "data" / "bronze"
SILVER = BASE / "data" / "silver"
REJECTS = SILVER / "_rejects"


def clean_weather() -> pd.DataFrame:
    """Flatten the nested Open-Meteo JSON (one record per region) into a tidy long table."""
    weather_files = sorted(f for f in BRONZE.glob("weather_raw_*.json") if not f.name.endswith(".meta.json"))
    if not weather_files:
        raise FileNotFoundError("No weather_raw_*.json found in bronze layer")
    latest = weather_files[-1]
    logger.info(f"Reading weather bronze file: {latest.name}")
    raw = json.loads(latest.read_text())

    rows = []
    for region_payload in raw:
        region = region_payload["_region_name"]
        daily = region_payload["daily"]
        n = len(daily["time"])
        for i in range(n):
            rows.append({
                "region_name": region,
                "date": daily["time"][i],
                "temp_max_c": daily["temperature_2m_max"][i],
                "temp_min_c": daily["temperature_2m_min"][i],
                "precipitation_mm": daily["precipitation_sum"][i],
                "humidity_pct": daily["relative_humidity_2m_mean"][i],
                "windspeed_max_kmh": daily["windspeed_10m_max"][i],
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    # standardize: drop exact duplicate region/date rows (can happen with overlapping forecast windows)
    before = len(df)
    df = df.drop_duplicates(subset=["region_name", "date"])
    logger.info(f"Weather: {before} raw rows -> {len(df)} after de-dup")
    return df


def clean_ag_exports() -> pd.DataFrame:
    path = BRONZE / "us_ag_exports_raw.csv"
    logger.info(f"Reading ag exports bronze file: {path.name}")
    df = pd.read_csv(path)

    # standardize column names (snake_case, units in name)
    df = df.rename(columns={
        "code": "state_code",
        "state": "state_name",
        "total exports": "total_exports_musd",
        "corn": "corn_musd",
        "wheat": "wheat_musd",
        "cotton": "cotton_musd",
        "dairy": "dairy_musd",
    })
    keep_cols = ["state_code", "state_name", "total_exports_musd",
                 "corn_musd", "wheat_musd", "cotton_musd", "dairy_musd"]
    df = df[keep_cols]

    # drop the aggregate "state" rows with no code / territories with nulls
    before = len(df)
    df = df.dropna(subset=["state_code", "state_name"])
    df = df[df["state_code"].str.len() == 2]
    logger.info(f"Ag exports: {before} raw rows -> {len(df)} after cleaning")
    return df


def clean_geo() -> pd.DataFrame:
    path = BRONZE / "us_state_geo_raw.csv"
    logger.info(f"Reading geo bronze file: {path.name}")
    df = pd.read_csv(path)
    df = df.rename(columns={"name": "state_name", "description": "capital"})
    before = len(df)
    df = df.dropna(subset=["state_name", "latitude", "longitude"])
    logger.info(f"Geo: {before} raw rows -> {len(df)} after cleaning")
    return df


def write_validated(df: pd.DataFrame, schema, name: str):
    SILVER.mkdir(parents=True, exist_ok=True)
    REJECTS.mkdir(parents=True, exist_ok=True)

    clean_df, error_report = validate_or_report(df, schema, name)
    if error_report:
        reject_path = REJECTS / f"{name}_rejects.json"
        reject_path.write_text(json.dumps(error_report, indent=2, default=str))
        logger.error(f"{name}: {error_report['failure_count']} validation failures written to {reject_path}")
        # In production this would raise a pipeline alert (email/Slack/PagerDuty).
        # We still write whatever passed validation piecemeal so partial data isn't lost.
        raise SystemExit(f"Data quality gate failed for {name} -- see {reject_path}")

    out_path = SILVER / f"{name}.parquet"
    # Spark's Parquet reader requires microsecond (not pandas' default
    # nanosecond) timestamp precision -- a common pandas<->Spark interop
    # gotcha in real pipelines. Coerce any datetime64[ns] columns before write.
    write_df = clean_df.copy()
    for col in write_df.select_dtypes(include=["datetime64[ns]"]).columns:
        write_df[col] = write_df[col].astype("datetime64[us]")
    write_df.to_parquet(out_path, index=False, coerce_timestamps="us", allow_truncated_timestamps=True)
    logger.info(f"{name}: {len(clean_df)} rows passed validation -> {out_path}")
    return clean_df


def main():
    weather_df = clean_weather()
    write_validated(weather_df, weather_schema, "weather")

    ag_df = clean_ag_exports()
    write_validated(ag_df, ag_exports_schema, "ag_exports")

    geo_df = clean_geo()
    write_validated(geo_df, geo_schema, "geo")

    logger.info("Bronze -> Silver complete for all 3 sources")


if __name__ == "__main__":
    main()
