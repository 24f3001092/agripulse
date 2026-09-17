"""
test_segmentation.py

Unit tests for src/ml/marketing_segmentation.py using a small deterministic
fixture that mirrors the Gold-layer contract. No external APIs, no dependency
on repository-generated data.

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


def make_gold_fixture(tmp_path: Path) -> Path:
    """Two small regions with clearly different export/crop/weather profiles."""
    exports = {"Cornland": 11000.0, "CottonBay": 1500.0}
    rows = []
    for d in range(3):
        for region, ex in exports.items():
            is_corn = region == "Cornland"
            rows.append({
                "region_name": region,
                "date": pd.Timestamp("2026-08-01") + pd.DateOffset(days=d),
                "temp_max_c": 30.0, "temp_min_c": 18.0,
                "temp_avg_c": 24.0, "precipitation_mm": 30.0 if is_corn else 2.0,
                "humidity_pct": 70.0, "windspeed_max_kmh": 15.0,
                "state_code": "XX",
                "total_exports_musd": ex,
                "corn_musd": ex * 0.8 if is_corn else ex * 0.05,
                "wheat_musd": ex * 0.05,
                "cotton_musd": ex * 0.05 if is_corn else ex * 0.85,
                "dairy_musd": ex * 0.05,
                "latitude": 40.0, "longitude": -90.0,
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
    expected = ["region_name", "segment", "segment_reason", "export_tier", "dominant_crop"]
    for col in expected:
        assert col in result["segments"].columns


def test_dominant_crop_detection(seg_tmp):
    result = ms.run_segmentation()
    seg = result["segments"].set_index("region_name")
    assert seg.loc["Cornland", "dominant_crop"] == "corn"
    assert seg.loc["CottonBay", "dominant_crop"] == "cotton"
    assert "Cornland" in seg.loc["Cornland", "segment_reason"]