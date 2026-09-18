"""
india_weather_fixture_generator.py

*** SANDBOX-ONLY UTILITY -- NOT PART OF THE PRODUCTION PIPELINE ***

Synthetic, clearly-labelled stand-ins for India Meteorological Department (IMD)
weather payloads. The real ingestion path for IMD data is imd_weather.py, which
calls the OFFICIAL IMD API (https://api.imd.gov.in/) using registered credentials
from the environment (AGRIPULSE_IMD_API_KEY + AGRIPULSE_IMD_TOKEN).

In this specific execution sandbox the IMD portal credentials are not available
(the API key must be registered, approved and bound to the server's public IP),
so live retrieval cannot be demonstrated. This script generates deterministic,
seeded fixture payloads that EXACTLY match the normalized Bronze schema produced
by imd_weather.py (same keys, same units, same structure) for every monitored
district in catalog/india_regions.json -- so publishers, tests and any pipeline
that consumes imd_* files behave identically whether the rows came from IMD or
from a fixture.

SAFETY RAILS (non-negotiable):
  * every file is written with provenance.is_fixture = true and status
    "synthetic_fixture"; the live path never sets these.
  * fixtures are generated only for the monitored districts in the catalog.
  * fixtures are NEVER presented as live IMD data anywhere in the app.

On a machine whose IMD API key has been approved and IP-bound, delete this file
and run imd_weather.py instead -- no downstream schema change is needed.

Usage:
    python india_weather_fixture_generator.py --out data/india/bronze/india_weather
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("imd_fixture_generator")

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from imd_weather import write_bronze  # noqa: E402

_FORECAST_TEXT = [
    "Partly cloudy sky",
    "Generally cloudy sky",
    "Generally cloudy sky with possibility of light rain",
    "Generally cloudy sky with possibility of thunderstorm and lightning",
    "Mainly clear sky",
    "Thundery development with possibility of rain",
    "Mostly clear sky",
]

_WEATHER_CODES = ["01", "02", "10", "21", "25", "29", "60", "95"]


def _rng(district_index: int) -> random.Random:
    return random.Random(7000 + district_index)


def _make_current(region: dict, seed_offset: int) -> dict:
    rng = _rng(seed_offset)
    temp = 30.0 + rng.uniform(-4, 4)
    return {
        "station_id": f"IMDFX{6000 + seed_offset}",
        "station": region["district"],
        "date": datetime.now(timezone.utc).date().isoformat(),
        "time_utc": "05:30:00",
        "temp_c": round(temp, 1),
        "humidity_pct": round(max(20, min(100, 72 + rng.uniform(-12, 12))), 1),
        "rainfall_24h_mm": round(max(0, rng.gauss(0, 6)), 1) if rng.random() < 0.35 else 0.0,
        "wind_speed_kmh": round(max(0, rng.gauss(12, 6)), 1),
        "wind_direction_code": str(rng.choice(["90", "180", "230", "270", "320", "360"])),
        "mslp_hpa": round(1008.0 + rng.uniform(-4, 4), 1),
        "weather_code": str(rng.choice(_WEATHER_CODES)),
        "nebulosity": str(rng.randint(1, 8)),
    }


def _make_forecast(region: dict, district_index: int) -> list:
    rng = _rng(district_index)
    today = datetime.now(timezone.utc).date()
    rows = []
    for day in range(1, 8):
        tmax = 31.0 + rng.uniform(-3, 3)
        tmin = 22.0 + rng.uniform(-2.5, 2.5)
        rows.append({
            "station_code": f"IMDFX{6000 + district_index}",
            "station_name": region["district"],
            "date": (today + timedelta(days=day - 1)).isoformat(),
            "forecast_day": day,
            "temp_max_c": round(tmax, 1),
            "temp_min_c": round(min(tmin, tmax - 1), 1),
            "forecast_text": rng.choice(_FORECAST_TEXT),
            "rh_0830_pct": round(max(20, min(100, 75 + rng.uniform(-10, 10))), 1) if day == 1 else None,
            "rh_1730_pct": round(max(20, min(100, 60 + rng.uniform(-8, 8))), 1) if day == 1 else None,
            "rainfall_24h_mm": round(max(0, rng.gauss(0, 5)), 1) if day == 1 else None,
        })
    return rows


_RAIN_CATEGORIES = ["LE", "E", "N", "D", "LD", "NR"]


def _make_rainfall(region: dict, district_index: int) -> dict:
    rng = _rng(district_index)
    today = datetime.now(timezone.utc).date()
    daily_actual = round(max(0, rng.gauss(2, 4)), 2)
    out = {
        "obj_id": str(10000 + district_index),
        "district": region["district"],
        "date": today.isoformat(),
        "daily_actual_mm": daily_actual if daily_actual >= 0.01 else 0.0,
        "daily_normal_mm": round(rng.uniform(0.5, 6.0), 2),
        "daily_departure_pct": round(rng.uniform(-100, 60), 1),
        "daily_category": rng.choice(_RAIN_CATEGORIES),
        "weekly_actual_mm": round(rng.uniform(0, 30), 2),
        "weekly_normal_mm": round(rng.uniform(5, 40), 2),
        "weekly_departure_pct": round(rng.uniform(-100, 40), 1),
        "weekly_category": rng.choice(_RAIN_CATEGORIES),
        "cumulative_actual_mm": round(rng.uniform(0, 90), 2),
        "cumulative_normal_mm": round(rng.uniform(20, 120), 2),
        "cumulative_departure_pct": round(rng.uniform(-60, 50), 1),
        "cumulative_category": rng.choice(_RAIN_CATEGORIES),
        "monthly_actual_mm": round(rng.uniform(0, 120), 2),
        "monthly_normal_mm": round(rng.uniform(30, 140), 2),
        "monthly_departure_pct": round(rng.uniform(-50, 40), 1),
        "monthly_category": rng.choice(_RAIN_CATEGORIES),
    }
    return out


_WARNING_CODES = {
    1: "No Warning",
    2: "Heavy Rain",
    3: "Heavy Snow",
    4: "Thunderstorm & Lightning, Squall etc",
    5: "Hailstorm",
    6: "Dust Storm",
    7: "Dust Raising Winds",
    8: "Strong Surface Winds",
    9: "Heat Wave",
    10: "Hot Day",
    11: "Warm Night",
    12: "Cold Wave",
    13: "Cold Day",
    14: "Ground Frost",
    15: "Fog",
    16: "Very Heavy Rain",
    17: "Extremely Heavy Rain",
}


def _make_warning(region: dict, district_index: int) -> dict:
    rng = _rng(district_index + 500)
    today = datetime.now(timezone.utc).date()
    day_codes = []
    for _day in range(1, 6):
        n = rng.randint(0, 2)
        codes = rng.sample(list(_WARNING_CODES.keys()), k=n) if n else [1]
        day_codes.append([int(c) for c in codes])
    out = {
        "obj_id": str(11000 + district_index),
        "district": region["district"],
        "date": today.isoformat(),
        "time_utc": "05:30:00",
    }
    for i, codes in enumerate(day_codes, start=1):
        out[f"day_{i}_codes"] = codes
        severity = max(codes) if codes else 1
        out[f"day_{i}_color"] = str({1: "4", **{v: "3" for v in range(2, 6)},
                                     **{v: "2" for v in range(6, 9)}, **{v: "1" for v in range(9, 18)}}.get(severity, "4"))
    return out


def _build_payload(kind: str, regions: list) -> dict:
    payload = {"current": _make_current, "forecast": _make_forecast,
               "rainfall": _make_rainfall, "warning": _make_warning}[kind]
    if kind == "forecast":
        rows = []
        for idx, region in enumerate(regions):
            rows.extend(_make_forecast(region, idx))
        return rows
    return [payload(region, idx) for idx, region in enumerate(regions)]


def main():
    parser = argparse.ArgumentParser(description="Generate SYNTHETIC IMD weather fixtures (sandbox only)")
    parser.add_argument("--out", help="output directory (default: data/india/bronze/india_weather)")
    parser.add_argument("--kind", default="all",
                        help="comma-separated kinds: current, forecast, rainfall, warning (default: all)")
    args = parser.parse_args()

    import config  # noqa: PLC0415
    regions = config.load_regions()
    logger.warning("Generating SYNTHETIC IMD fixtures for %d monitored districts "
                   "(IMD credentials unavailable in this sandbox).", len(regions))

    out_dir = Path(args.out) if args.out else config.BRONZE / "india_weather"
    kinds = ["current", "forecast", "rainfall", "warning"] if args.kind == "all" \
        else [k.strip() for k in args.kind.split(",")]

    for kind in kinds:
        records = _build_payload(kind, regions)
        write_bronze(kind, records, raw=records,
                     endpoint="https://api.imd.gov.in/api/v1/fixture",
                     location=None, state=None, out_dir=out_dir, is_fixture=True)
    logger.info("Fixtures written to %s", out_dir)


if __name__ == "__main__":
    main()