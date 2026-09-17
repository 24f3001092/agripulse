"""
test_forecasting.py

Unit tests for src/ml/demand_forecast_model.py using a small deterministic
fixture that mirrors the Gold-layer contract (region_daily_features). No
external APIs and no dependency on repository-generated data.

Run with:
    pytest tests/test_forecasting.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_ML = Path(__file__).resolve().parents[1] / "src" / "ml"
sys.path.insert(0, str(SRC_ML))

import demand_forecast_model as dfm  # noqa: E402


def make_gold_fixture(tmp_path: Path, n_regions: int = 6, n_days: int = 3) -> Path:
    """Create a synthetic region_daily_features dataset matching the Gold contract."""
    exports = {
        "RegionA": 12000.0, "RegionB": 9000.0, "RegionC": 7000.0,
        "RegionD": 5000.0, "RegionE": 3000.0, "RegionF": 1000.0,
    }
    rows = []
    for i in range(n_days):
        for region in list(exports)[:n_regions]:
            rows.append({
                "region_name": region,
                "date": pd.Timestamp("2026-08-01") + pd.DateOffset(days=i),
                "temp_max_c": 30.0 + i, "temp_min_c": 18.0 + i,
                "temp_avg_c": 24.0 + i, "precipitation_mm": 2.0 + i,
                "humidity_pct": 70.0, "windspeed_max_kmh": 15.0,
                "state_code": "XX", "total_exports_musd": exports[region],
                "corn_musd": exports[region] * 0.3, "wheat_musd": exports[region] * 0.2,
                "cotton_musd": exports[region] * 0.1, "dairy_musd": exports[region] * 0.05,
                "latitude": 40.0, "longitude": -90.0,
            })
    df = pd.DataFrame(rows)
    gold_dir = tmp_path / "region_daily_features"
    gold_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(gold_dir / "part-00000.parquet", index=False)
    return gold_dir.parent


@pytest.fixture()
def gold_tmp(tmp_path, monkeypatch):
    """Point the module at a temp Gold/Catalog location with synthetic data."""
    gold_root = make_gold_fixture(tmp_path)
    monkeypatch.setattr(dfm, "GOLD", gold_root)
    monkeypatch.setattr(dfm, "CATALOG", tmp_path / "catalog")
    monkeypatch.setattr(dfm, "BASE", tmp_path)
    return tmp_path


def test_gold_fixture_loads(gold_tmp):
    df = dfm.load_gold_features()
    assert len(df) == 6 * 3
    assert set(["region_name", "temp_avg_c", "precipitation_mm", "humidity_pct",
                "windspeed_max_kmh", "total_exports_musd"]).issubset(df.columns)


def test_required_feature_columns_exist(gold_tmp, tmp_path):
    df = dfm.load_gold_features()
    region_df = dfm.build_region_level_dataset(df)
    missing = [c for c in dfm.FEATURE_COLS if c not in region_df.columns]
    assert missing == []
    assert dfm.TARGET_COL in region_df.columns


def test_predictions_generated_and_finite(gold_tmp):
    df = dfm.load_gold_features()
    region_df = dfm.build_region_level_dataset(df)
    model_info = dfm.train_model(region_df)
    forecast_df = dfm.build_forecast_frame(region_df, model_info)

    assert len(forecast_df) == len(region_df)
    assert forecast_df["predicted_exports_musd"].notna().all()
    assert np.isfinite(forecast_df["predicted_exports_musd"].to_numpy()).all()
    assert np.isfinite(forecast_df["actual_exports_musd"].to_numpy()).all()


def test_expected_output_fields_exist(gold_tmp):
    df = dfm.load_gold_features()
    region_df = dfm.build_region_level_dataset(df)
    model_info = dfm.train_model(region_df)
    forecast_df = dfm.build_forecast_frame(region_df, model_info)

    expected = ["region_name", "actual_exports_musd", "predicted_exports_musd",
                "difference_musd", "prediction_error_pct"]
    for col in expected:
        assert col in forecast_df.columns


def test_small_sample_flagged_in_sample(gold_tmp):
    df = dfm.load_gold_features()
    region_df = dfm.build_region_level_dataset(df)
    # 6 regions < MIN_ROWS_FOR_VALID_SPLIT -> metrics must be labelled in-sample
    model_info = dfm.train_model(region_df)
    assert model_info["metrics"]["metrics_are_in_sample"] is True
    assert model_info["small_sample_warning"] is not None


def test_write_forecast_outputs(gold_tmp):
    df = dfm.load_gold_features()
    region_df = dfm.build_region_level_dataset(df)
    model_info = dfm.train_model(region_df)
    forecast_df = dfm.build_forecast_frame(region_df, model_info)
    dfm.write_forecast_outputs(forecast_df, model_info)

    assert (dfm.GOLD / "forecast_results.parquet").exists()
    assert (dfm.CATALOG / "model_report.json").exists()
    written = pd.read_parquet(dfm.GOLD / "forecast_results.parquet")
    assert len(written) == len(forecast_df)


def test_prediction_error_pct_null_when_actual_zero(gold_tmp, tmp_path, monkeypatch):
    df = dfm.load_gold_features()
    region_df = dfm.build_region_level_dataset(df)
    bad = region_df.copy()
    bad.loc[0, "total_exports_musd"] = 0.0
    model_info = dfm.train_model(bad)
    forecast_df = dfm.build_forecast_frame(bad, model_info)
    assert forecast_df.loc[0, "prediction_error_pct"] is None or np.isnan(forecast_df.loc[0, "prediction_error_pct"])