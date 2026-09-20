"""
weather_fixture_generator.py

*** SANDBOX-ONLY UTILITY -- NOT PART OF THE PRODUCTION PIPELINE ***

This repo's real ingestion path is weather_api.py, which calls the live
Open-Meteo API for the ACTIVE country profile (India: catalog/india_regions.json).
In this specific execution sandbox, outbound network access is restricted to
package registries (pypi/npm/github) and does NOT include api.open-meteo.com
-- confirmed via a direct 403 (x-deny-reason: host_not_allowed).

To still demonstrate a fully working Bronze -> Silver -> Gold -> SQL -> ML
pipeline end-to-end in this environment, this script generates synthetic
daily weather records that exactly match Open-Meteo's real JSON response
schema (same keys, same units, same structure) using seeded randomness
with realistic post-monsoon seasonal means per monitored Indian district.

On a machine with normal internet access (your laptop, Databricks,
Airflow, GitHub Actions), delete this file and run weather_api.py instead
-- no other code changes needed, because downstream stages consume the
same schema either way.
"""

import json
import logging
import random
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from weather_api import IngestionMetadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("weather_fixture")

# Rough September/post-monsoon seasonal baselines per monitored Indian
# district (from the regional climate reference). Values are climatological
# guiding means for the SYNTHETIC SANDBOX FALLBACK ONLY -- the live API's
# numbers always win when available.
REGION_BASELINES = {
    "Nashik":     {"tmax": 30, "tmin": 21, "precip_p": 0.45, "rh": 78},
    "Ludhiana":   {"tmax": 33, "tmin": 22, "precip_p": 0.20, "rh": 62},
    "Kanpur":     {"tmax": 33, "tmin": 24, "precip_p": 0.28, "rh": 66},
    "Karnal":     {"tmax": 33, "tmin": 23, "precip_p": 0.25, "rh": 65},
    "Indore":     {"tmax": 30, "tmin": 21, "precip_p": 0.42, "rh": 74},
    "Rajkot":     {"tmax": 33, "tmin": 24, "precip_p": 0.28, "rh": 68},
    "Coimbatore": {"tmax": 32, "tmin": 23, "precip_p": 0.25, "rh": 70},
    "Hyderabad":  {"tmax": 31, "tmin": 23, "precip_p": 0.30, "rh": 70},
    "Bengaluru":  {"tmax": 29, "tmin": 20, "precip_p": 0.40, "rh": 74},
    "Kolkata":    {"tmax": 32, "tmin": 25, "precip_p": 0.40, "rh": 80},
}


def generate_region_series(region_name: str, seed: int, past_days: int = 30, forecast_days: int = 7) -> dict:
    rng = random.Random(seed)
    baseline = REGION_BASELINES.get(region_name, {"tmax": 30, "tmin": 21, "precip_p": 0.3, "rh": 70})

    total_days = past_days + forecast_days
    start = datetime.now(timezone.utc).date() - timedelta(days=past_days)

    dates, tmax, tmin, precip, rh, wind = [], [], [], [], [], []
    for i in range(total_days):
        d = start + timedelta(days=i)
        dates.append(d.isoformat())
        day_tmax = round(baseline["tmax"] + rng.uniform(-3.5, 3.5), 1)
        day_tmin = round(baseline["tmin"] + rng.uniform(-2.5, 2.5), 1)
        tmax.append(day_tmax)
        tmin.append(min(day_tmin, day_tmax - 1))
        precip.append(round(max(0, rng.gauss(0, 8)) if rng.random() < baseline["precip_p"] else 0.0, 1))
        rh.append(round(max(20, min(100, baseline["rh"] + rng.uniform(-10, 10))), 1))
        wind.append(round(max(0, rng.gauss(14, 5)), 1))

    return {
        "latitude": 0.0, "longitude": 0.0, "timezone": "auto",
        "daily": {
            "time": dates,
            "temperature_2m_max": tmax,
            "temperature_2m_min": tmin,
            "precipitation_sum": precip,
            "relative_humidity_2m_mean": rh,
            "windspeed_10m_max": wind,
        },
        "_region_name": region_name,
        "_synthetic": True,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--past-days", type=int, default=30)
    parser.add_argument("--forecast-days", type=int, default=7)
    args = parser.parse_args()

    regions = json.loads(Path(args.config).read_text())["regions"]
    logger.warning("Generating SYNTHETIC weather fixtures (sandbox network cannot reach api.open-meteo.com)")

    results = []
    for idx, region in enumerate(regions):
        series = generate_region_series(region["region_name"], seed=1000 + idx,
                                         past_days=args.past_days, forecast_days=args.forecast_days)
        series["latitude"] = region["latitude"]
        series["longitude"] = region["longitude"]
        results.append(series)
        logger.info(f"Generated fixture for {region['region_name']}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_file = out_dir / f"weather_raw_{ts}.json"
    out_file.write_text(json.dumps(results, indent=2))

    meta = IngestionMetadata(
        source_name="open-meteo-forecast-api",
        source_url="https://api.open-meteo.com/v1/forecast",
        ingested_at_utc=ts,
        region_count=len(regions),
        row_count=len(results),
        status="synthetic_fixture",
        notes="SANDBOX SUBSTITUTE: real API unreachable from this container. Schema-identical to live Open-Meteo response. Run weather_api.py on a network-unrestricted host for real data.",
    )
    meta_file = out_dir / f"weather_raw_{ts}.meta.json"
    meta_file.write_text(json.dumps(asdict(meta), indent=2))
    logger.info(f"Wrote {out_file}")
    logger.info(f"Wrote catalog metadata {meta_file}")


if __name__ == "__main__":
    main()
