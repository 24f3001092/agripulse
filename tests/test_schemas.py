"""
test_schemas.py
Unit tests for the Pandera data-quality contracts. Run with:
    pytest tests/test_schemas.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "quality"))
from schemas import weather_schema, ag_exports_schema, geo_schema, validate_or_report


def test_weather_schema_accepts_valid_row():
    df = pd.DataFrame([{
        "region_name": "Iowa", "date": "2026-08-01",
        "temp_max_c": 30.0, "temp_min_c": 18.0,
        "precipitation_mm": 2.5, "humidity_pct": 70.0, "windspeed_max_kmh": 12.0,
    }])
    clean, err = validate_or_report(df, weather_schema, "weather")
    assert err is None
    assert len(clean) == 1


def test_weather_schema_rejects_impossible_temp():
    df = pd.DataFrame([{
        "region_name": "Iowa", "date": "2026-08-01",
        "temp_max_c": 15.0, "temp_min_c": 25.0,  # min > max -- physically invalid
        "precipitation_mm": 2.5, "humidity_pct": 70.0, "windspeed_max_kmh": 12.0,
    }])
    clean, err = validate_or_report(df, weather_schema, "weather")
    assert err is not None
    assert err["failure_count"] >= 1


def test_weather_schema_rejects_humidity_out_of_range():
    df = pd.DataFrame([{
        "region_name": "Iowa", "date": "2026-08-01",
        "temp_max_c": 30.0, "temp_min_c": 18.0,
        "precipitation_mm": 2.5, "humidity_pct": 130.0,  # invalid: >100%
        "windspeed_max_kmh": 12.0,
    }])
    clean, err = validate_or_report(df, weather_schema, "weather")
    assert err is not None


def test_ag_exports_schema_accepts_valid_row():
    df = pd.DataFrame([{
        "state_code": "IA", "state_name": "Iowa",
        "total_exports_musd": 11273.76, "corn_musd": 2529.8,
        "wheat_musd": 3.1, "cotton_musd": 0.0, "dairy_musd": 107.0,
    }])
    clean, err = validate_or_report(df, ag_exports_schema, "ag_exports")
    assert err is None


def test_ag_exports_schema_rejects_bad_state_code():
    df = pd.DataFrame([{
        "state_code": "IOWA", "state_name": "Iowa",  # invalid: not 2 chars
        "total_exports_musd": 11273.76, "corn_musd": 2529.8,
        "wheat_musd": 3.1, "cotton_musd": 0.0, "dairy_musd": 107.0,
    }])
    clean, err = validate_or_report(df, ag_exports_schema, "ag_exports")
    assert err is not None


def test_geo_schema_rejects_invalid_latitude():
    df = pd.DataFrame([{
        "state_name": "Iowa", "capital": "Des Moines",
        "latitude": 999.0,  # invalid: out of range
        "longitude": -93.6,
    }])
    clean, err = validate_or_report(df, geo_schema, "geo")
    assert err is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
