"""
test_schemas.py
Unit tests for the Pandera data-quality contracts (active profile: India).
Run with:
    pytest tests/test_schemas.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "quality"))
from schemas import (  # noqa: E402
    weather_schema,
    geo_india_schema,
    market_commodity_schema,
    validate_or_report,
)


def test_weather_schema_accepts_valid_row():
    df = pd.DataFrame([{
        "region_name": "Nashik", "date": "2026-09-01",
        "temp_max_c": 30.0, "temp_min_c": 18.0,
        "precipitation_mm": 2.5, "humidity_pct": 70.0, "windspeed_max_kmh": 12.0,
    }])
    clean, err = validate_or_report(df, weather_schema, "weather")
    assert err is None
    assert len(clean) == 1


def test_weather_schema_rejects_impossible_temp():
    df = pd.DataFrame([{
        "region_name": "Nashik", "date": "2026-09-01",
        "temp_max_c": 15.0, "temp_min_c": 25.0,  # min > max -- physically invalid
        "precipitation_mm": 2.5, "humidity_pct": 70.0, "windspeed_max_kmh": 12.0,
    }])
    clean, err = validate_or_report(df, weather_schema, "weather")
    assert err is not None
    assert err["failure_count"] >= 1


def test_weather_schema_rejects_humidity_out_of_range():
    df = pd.DataFrame([{
        "region_name": "Nashik", "date": "2026-09-01",
        "temp_max_c": 30.0, "temp_min_c": 18.0,
        "precipitation_mm": 2.5, "humidity_pct": 130.0,  # invalid: >100%
        "windspeed_max_kmh": 12.0,
    }])
    clean, err = validate_or_report(df, weather_schema, "weather")
    assert err is not None


def test_geo_india_schema_accepts_valid_row():
    df = pd.DataFrame([{
        "region_name": "Nashik", "state_ut": "Maharashtra",
        "state_code": "MH", "district": "Nashik", "mandi_apmc": "Nashik",
        "latitude": 19.9975, "longitude": 73.7898,
    }])
    clean, err = validate_or_report(df, geo_india_schema, "geo")
    assert err is None
    assert len(clean) == 1


def test_geo_india_schema_rejects_bad_state_code_length():
    df = pd.DataFrame([{
        "region_name": "Nashik", "state_ut": "Maharashtra",
        "state_code": "MHX", "district": "Nashik", "mandi_apmc": "Nashik",
        "latitude": 19.9975, "longitude": 73.7898,
    }])
    clean, err = validate_or_report(df, geo_india_schema, "geo")
    assert err is not None


def test_geo_india_schema_rejects_out_of_bounds_latitude():
    df = pd.DataFrame([{
        "region_name": "Nashik", "state_ut": "Maharashtra",
        "state_code": "MH", "district": "Nashik", "mandi_apmc": "Nashik",
        "latitude": 999.0, "longitude": 73.7898,  # invalid: outside India bounds
    }])
    clean, err = validate_or_report(df, geo_india_schema, "geo")
    assert err is not None


def test_market_schema_accepts_valid_row():
    df = pd.DataFrame([{
        "commodity": "Wheat", "variety": "Lokwan", "state_ut": "Maharashtra",
        "district": "Nashik", "mandi": "Nashik", "date": "2026-09-01",
        "market_price_inr_per_quintal": 2450.0, "arrivals_quintal": 1200.0,
    }])
    clean, err = validate_or_report(df, market_commodity_schema, "market")
    assert err is None


def test_market_schema_rejects_negative_price():
    df = pd.DataFrame([{
        "commodity": "Wheat", "variety": "Lokwan", "state_ut": "Maharashtra",
        "district": "Nashik", "mandi": "Nashik", "date": "2026-09-01",
        "market_price_inr_per_quintal": -10.0, "arrivals_quintal": 1200.0,
    }])
    clean, err = validate_or_report(df, market_commodity_schema, "market")
    assert err is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))