"""
weather_api.py
Bronze-layer ingestion module: pulls daily agro-weather data from the
Open-Meteo Forecast + Historical Weather API for a configured list of
agricultural regions.

Real, runnable module — requires outbound internet access to
api.open-meteo.com (works on any normal machine, Databricks, Airflow
worker, etc). No API key needed; Open-Meteo is free for non-commercial use.

Usage:
    python weather_api.py --config ../../catalog/regions.json --out ../../data/bronze
"""

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("weather_ingestion")

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean,windspeed_10m_max"


@dataclass
class IngestionMetadata:
    """Data-catalog entry written alongside every raw pull."""
    source_name: str
    source_url: str
    ingested_at_utc: str
    region_count: int
    row_count: int
    status: str
    notes: str = ""


def fetch_region_weather(region_name: str, lat: float, lon: float,
                          past_days: int = 30, forecast_days: int = 7,
                          session: requests.Session = None) -> dict:
    """Fetch daily weather series for a single region. Retries handled by caller."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": DAILY_VARS,
        "timezone": "auto",
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    sess = session or requests
    resp = sess.get(OPEN_METEO_URL, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    payload["_region_name"] = region_name
    return payload


def fetch_all_regions(regions: List[dict], past_days: int = 30,
                       forecast_days: int = 7, max_retries: int = 3) -> List[dict]:
    """
    Fetch weather for every region in the catalog, with basic retry logic.
    A failed region is logged and skipped rather than failing the whole batch —
    mirrors production pipeline behavior (partial success + alerting, not
    all-or-nothing).
    """
    results = []
    failures = []
    with requests.Session() as session:
        for region in regions:
            name = region["region_name"]
            for attempt in range(1, max_retries + 1):
                try:
                    data = fetch_region_weather(
                        name, region["latitude"], region["longitude"],
                        past_days, forecast_days, session
                    )
                    results.append(data)
                    logger.info(f"OK  {name} (attempt {attempt})")
                    break
                except requests.exceptions.RequestException as e:
                    logger.warning(f"FAIL {name} attempt {attempt}/{max_retries}: {e}")
                    if attempt == max_retries:
                        failures.append({"region": name, "error": str(e)})
    if failures:
        logger.error(f"{len(failures)} region(s) failed after retries: {failures}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to regions.json")
    parser.add_argument("--out", required=True, help="Bronze output directory")
    parser.add_argument("--past-days", type=int, default=30)
    parser.add_argument("--forecast-days", type=int, default=7)
    args = parser.parse_args()

    regions = json.loads(Path(args.config).read_text())["regions"]
    logger.info(f"Fetching weather for {len(regions)} regions from Open-Meteo")

    raw_results = fetch_all_regions(regions, args.past_days, args.forecast_days)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_file = out_dir / f"weather_raw_{ts}.json"
    out_file.write_text(json.dumps(raw_results, indent=2))

    meta = IngestionMetadata(
        source_name="open-meteo-forecast-api",
        source_url=OPEN_METEO_URL,
        ingested_at_utc=ts,
        region_count=len(regions),
        row_count=len(raw_results),
        status="success" if len(raw_results) == len(regions) else "partial_failure",
    )
    meta_file = out_dir / f"weather_raw_{ts}.meta.json"
    meta_file.write_text(json.dumps(asdict(meta), indent=2))

    logger.info(f"Wrote {out_file}")
    logger.info(f"Wrote catalog metadata {meta_file}")


if __name__ == "__main__":
    main()
