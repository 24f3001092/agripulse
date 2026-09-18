"""
test_segmentation.py

Unit tests for src/ml/marketing_segmentation.py (India deployment: weather-
exposure segmentation) using a small deterministic fixture that mirrors the
Gold-layer contract. No external APIs, no dependency on repository-generated
data, and no fabricated values.

Run with:
    pytest tests/test_segmentation.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC_ML = Path(__file__).resolve().parents[1] / "src" / "ml"
sys.path.insert(0, str(SRC_ML))

import marketing_segmentation as ms  # noqa: E402

# Two districts with clearly different weather-exposure profiles:
#   RainyWest  - wet, cool, humid   (high rainfall exposure)
#   DryEast    - dry, warm, dry-air (water-limited)
FIXTURE_PROFILES = [
    {"region_name": "RainyWest", "temp_avg_c": 24.0, "precipitation_mm": 200.0, "humidity_pct": 90.0},
    {"region_name": "DryEast", "temp_avg_c": 25.0, "precipitation_mm": 30.0, "humidity_pct": 45.0},
]


def make_gold_fixture(tmp_path: Path) -> Path:
    """Write daily rows for the two districts mirroring the Gold contract."""
    rows = []
    for d in range(3):
        for p in FIXTURE_PROFILES:
            rows.append({
                "region_name": p["region_name"],
                "date": pd.Timestamp("2026-08-01") + pd.DateOffset(days=d),
                "temp_max_c": p["temp_avg_c"] + 6.0, "temp_min_c": p["temp_avg_c"] - 6.0,
                "temp_avg_c": p["temp_avg_c"], "precipitation_mm": p["precipitation_mm"] / 3,
                "humidity_pct": p["humidity_pct"], "windspeed_max_kmh": 15.0,
                "state_ut": "Maharashtra", "state_code": "MH",
                "district": p["region_name"], "mandi_apmc": p["region_name"],
                "latitude": 19.9, "longitude": 73.8,
            })
    df = pd.DataFrame(rows)
    gold_dir = tmp_path / "region_daily_features"
    gold_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(gold_dir / "part-00000.parquet", index=False)
    return tmp_path


@pytest.fixture()
def seg_tmp(tmp_path, monkeypatch):
    gold_root = make_gold_fixture(tmp_path)
    monkeypatch.setattr(ms, "GOLD", gold_root)
    monkeypatch.setattr(ms, "BASE", tmp_path)
    return tmp_path


def test_segmentation_output_exists(seg_tmp):
    result = ms.run_segmentation()
    assert (ms.GOLD / "region_segments.parquet").exists()
    assert len(result["segments"]) == 2


def test_every_region_gets_a_segment(seg_tmp):
    result = ms.run_segmentation()
    segments = result["segments"]
    assert segments["region_name"].nunique() == 2
    assert segments["segment"].notna().all()


def test_segment_not_null(seg_tmp):
    result = ms.run_segmentation()
    assert result["segments"]["segment"].isna().sum() == 0
    assert (result["segments"]["segment"].astype(str).str.len() > 0).all()


def test_expected_columns_exist(seg_tmp):
    result = ms.run_segmentation()
    expected = ["region_name", "segment", "segment_reason", "exposure_tier", "dominant_factor"]
    for col in expected:
        assert col in result["segments"].columns


def test_segment_labels_reflect_weather_exposure(seg_tmp):
    result = ms.run_segmentation()
    seg = result["segments"].set_index("region_name")
    # RainyWest is wet + cool + humid -> high-rainfall exposure
    assert "High-Rainfall" in seg.loc["RainyWest", "segment"]
    assert seg.loc["RainyWest", "exposure_tier"] == "High-Exposure"
    # DryEast is dry -> water-limited
    assert "Water-Limited" in seg.loc["DryEast", "segment"]


def test_dominant_factor_is_observed_weather_driver(seg_tmp):
    result = ms.run_segmentation()
    seg = result["segments"].set_index("region_name")
    assert seg.loc["RainyWest", "dominant_factor"] == "precipitation"
    assert seg.loc["DryEast", "dominant_factor"] == "precipitation"
    assert "RainyWest" in seg.loc["RainyWest", "segment_reason"]


def test_reason_never_claims_forecast(seg_tmp):
    result = ms.run_segmentation()
    for reason in result["segments"]["segment_reason"]:
        assert "not a forecast" in reason.lower()