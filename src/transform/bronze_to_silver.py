"""
bronze_to_silver.py
Cleans, standardizes, and validates the India Bronze datasets:

  1. Weather  - Open-Meteo daily series for monitored Indian districts
                (data/india/bronze/, live pull or sandbox fixture)
  2. Geography - India State/UT > District > Mandi/APMC reference
                (catalog/india_regions.json)
  3. Market (e-NAM / AGMARKNET) - ONLY when a real loader is configured; the
                market extension point currently returns not_configured, so a
                market_status marker is written instead of a fabricated table.
  4. Agriculture - DE&S/MoAFW crop Area/Production/Yield (GODL-India). Bronze
                under data/india/bronze/india_agriculture/; when no agriculture
                Bronze exists (fresh checkout / no data yet) an
                agriculture_status marker is written and the step is skipped
                gracefully so the rest of the pipeline stays independent.
  5. Mandi (market intelligence) - AGMARKNET daily APMC price/arrival records
                (GODL-India structured resource). Bronze under
                data/india/bronze/india_mandi/; missing bronze -> mandi_status
                marker and graceful skip.

Output: weather.parquet, geo.parquet, india_agriculture.parquet and
india_mandi.parquet in data/india/silver/, each validated against its pandera
contract in src/quality/schemas.py / src/quality/india_schemas.py. Any row
failing validation is written to data/india/silver/_rejects/ instead of being
silently dropped. No market prices/arrivals and no agriculture values are ever
invented.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "quality"))
sys.path.insert(0, str(SRC / "ingestion"))

import config  # noqa: E402
from schemas import validate_or_report, weather_schema, geo_india_schema, market_commodity_schema  # noqa: E402
from india_market_source import load_market_or_status, MARKET_COLUMNS  # noqa: E402
from india_agriculture import latest_normalized, DEDUP_KEY, AGRI_BRONZE_DIR  # noqa: E402
from india_mandi import latest_normalized as latest_mandi, DEDUP_KEY as MANDI_DEDUP_KEY, MANDI_BRONZE_DIR  # noqa: E402
from india_schemas import india_agriculture_schema, mandi_price_schema  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("bronze_to_silver")

BASE = Path(__file__).resolve().parents[2]
BRONZE = config.BRONZE
SILVER = config.SILVER
REJECTS = config.REJECTS


def clean_weather() -> pd.DataFrame:
    """Flatten the nested Open-Meteo JSON (one record per region) into a tidy long table."""
    weather_files = sorted(f for f in BRONZE.glob("weather_raw_*.json") if not f.name.endswith(".meta.json"))
    if not weather_files:
        raise FileNotFoundError(f"No weather_raw_*.json found in {BRONZE}")
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
    df = df.drop_duplicates(subset=["region_name", "date"])
    logger.info(f"Weather: {len(df)} rows after de-dup")
    return df


def clean_geo() -> pd.DataFrame:
    """Build the India geography Silver table from catalog/india_regions.json (reference data)."""
    regions = config.load_regions()
    df = pd.DataFrame([
        {
            "region_name": r["region_name"],
            "state_ut": r["state_ut"],
            "state_code": r["state_code"],
            "district": r["district"],
            "district_code": r.get("district_code"),
            "mandi_apmc": r["mandi_apmc"],
            "latitude": float(r["latitude"]),
            "longitude": float(r["longitude"]),
        }
        for r in regions
    ])
    logger.info(f"Geo: {len(df)} monitored India districts (State/UT > District > Mandi/APMC)")
    return df


def clean_market() -> pd.DataFrame | None:
    """
    Source market rows ONLY from a real e-NAM / AGMARKNET loader.
    Returns None when not configured (and writes a status marker).
    """
    result = load_market_or_status()
    if result["status"] != "configured" or result["df"] is None:
        marker = {
            "status": result["status"],
            "reason": result.get("reason", ""),
            "layers": ["bronze->silver skipped market table"],
        }
        (SILVER / "market_status.json").write_text(json.dumps(marker, indent=2))
        logger.info("Market table NOT generated (source not configured) -- marker written")
        return None
    df = result["df"]
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=[c for c in MARKET_COLUMNS if c != "variety"])
    return df


def clean_agriculture() -> pd.DataFrame | None:
    """
    Build the India agriculture Silver table from the newest normalized Bronze
    frame (data/india/bronze/india_agriculture/apy_normalized_*.parquet).

    Duplicates are removed on (state, district, crop, season, year): the same
    state/district/crop/year legitimately appears per season (Kharif, Rabi,
    Total, ...), so season is part of the identity. Returns None (and writes an
    agriculture_status marker) when no agriculture Bronze exists, so machines
    without the dataset still get a green pipeline.
    """
    bronze_file = latest_normalized(AGRI_BRONZE_DIR)
    if bronze_file is None:
        SILVER.mkdir(parents=True, exist_ok=True)
        marker = {
            "status": "no_data",
            "reason": "No agriculture Bronze data found; silver table skipped (run src/ingestion/india_agriculture.py first)",
            "layers": ["bronze->silver skipped india_agriculture table"],
        }
        (SILVER / "agriculture_status.json").write_text(json.dumps(marker, indent=2))
        logger.info("Agriculture table NOT generated (no bronze data) -- marker written")
        return None

    logger.info("Reading agriculture bronze file: %s", bronze_file.name)
    df = pd.read_parquet(bronze_file)
    before = len(df)
    df = df.drop_duplicates(subset=DEDUP_KEY, keep="first")
    dropped = before - len(df)
    if dropped:
        logger.info("Agriculture: removed %s duplicate rows (key: %s)", dropped, DEDUP_KEY)
    return df


def clean_mandi() -> pd.DataFrame | None:
    """
    Build the India mandi Silver table from the newest normalized Bronze frame
    (data/india/bronze/india_mandi/mandi_normalized_*.parquet).

    Duplicates are removed on the trade-day identity (date, state, district,
    apmc, commodity, variety). Rows whose reported prices contradict the
    AGMARKNET price ordering (min_price > modal_price or modal_price > max_price)
    are quarantined into silver/_rejects/india_mandi_invalid_price_rows.parquet
    instead of failing the table or being silently altered -- the source record
    stays visible and the validated subset continues to Silver.

    Returns None (and writes a mandi_status marker) when no mandi Bronze
    exists, so machines without market data still get a green pipeline.
    """
    bronze_file = latest_mandi(MANDI_BRONZE_DIR)
    if bronze_file is None:
        SILVER.mkdir(parents=True, exist_ok=True)
        marker = {
            "status": "no_data",
            "reason": "No mandi Bronze data found; silver table skipped (run src/ingestion/india_mandi.py first)",
            "layers": ["bronze->silver skipped india_mandi table"],
        }
        (SILVER / "mandi_status.json").write_text(json.dumps(marker, indent=2))
        logger.info("Mandi table NOT generated (no bronze data) -- marker written")
        return None

    logger.info("Reading mandi bronze file: %s", bronze_file.name)
    df = pd.read_parquet(bronze_file)
    before = len(df)
    df = df.drop_duplicates(subset=MANDI_DEDUP_KEY, keep="first")
    dropped = before - len(df)
    if dropped:
        logger.info("Mandi: removed %s duplicate rows (key: %s)", dropped, MANDI_DEDUP_KEY)

    # Quarantine self-contradicting price rows (real-source oddities) with
    # provenance -- never silently mutate or drop without a record.
    p = df["min_price"].notna() & df["modal_price"].notna() & (df["min_price"] > df["modal_price"])
    q = df["modal_price"].notna() & df["max_price"].notna() & (df["modal_price"] > df["max_price"])
    invalid = df[p | q]
    if len(invalid):
        REJECTS.mkdir(parents=True, exist_ok=True)
        reject_path = REJECTS / "india_mandi_invalid_price_rows.parquet"
        invalid.to_parquet(reject_path, index=False, coerce_timestamps="us",
                           allow_truncated_timestamps=True)
        logger.warning(
            "Mandi: quarantined %s contradictory price rows (min>modal or modal>max) -> %s",
            len(invalid), reject_path,
        )
        df = df.drop(index=invalid.index)

    # Enforce the exact Silver contract: drop source/fetch passthrough columns
    # (e.g. fetch_state, fetch_district) so the Silver table carries precisely
    # the documented mandi_price_schema columns.
    contract_cols = list(mandi_price_schema.columns.keys())
    extra = [c for c in df.columns if c not in contract_cols]
    if extra:
        logger.warning("Mandi: dropping %s non-contract passthrough columns from %s: %s",
                       len(extra), bronze_file.name, extra)
    return df[contract_cols]


def write_validated(df: pd.DataFrame, schema, name: str):
    SILVER.mkdir(parents=True, exist_ok=True)
    REJECTS.mkdir(parents=True, exist_ok=True)

    clean_df, error_report = validate_or_report(df, schema, name)
    if error_report:
        reject_path = REJECTS / f"{name}_rejects.json"
        reject_path.write_text(json.dumps(error_report, indent=2, default=str))
        logger.error(f"{name}: {error_report['failure_count']} validation failures written to {reject_path}")
        raise SystemExit(f"Data quality gate failed for {name} -- see {reject_path}")

    out_path = SILVER / f"{name}.parquet"
    write_df = clean_df.copy()
    for col in write_df.select_dtypes(include=["datetime64[ns]"]).columns:
        write_df[col] = write_df[col].astype("datetime64[us]")
    write_df.to_parquet(out_path, index=False, coerce_timestamps="us", allow_truncated_timestamps=True)
    logger.info(f"{name}: {len(clean_df)} rows passed validation -> {out_path}")
    return clean_df


def main():
    weather_df = clean_weather()
    write_validated(weather_df, weather_schema, "weather")

    geo_df = clean_geo()
    write_validated(geo_df, geo_india_schema, "geo")

    market_df = clean_market()
    if market_df is not None:
        write_validated(market_df, market_commodity_schema, "market")

    agriculture_df = clean_agriculture()
    if agriculture_df is not None:
        write_validated(agriculture_df, india_agriculture_schema, "india_agriculture")

    mandi_df = clean_mandi()
    if mandi_df is not None:
        write_validated(mandi_df, mandi_price_schema, "india_mandi")

    logger.info("Bronze -> Silver complete (India profile)")


if __name__ == "__main__":
    main()