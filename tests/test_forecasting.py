"""
test_forecasting.py

Unit tests for src/ml/demand_forecast_model.py using a small deterministic
fixture that mirrors the Gold-layer contract (region_daily_features with a
market-price signal column). No external APIs and no dependency on
repository-generated data.

The market-gating rule is tested explicitly: while the e-NAM / AGMARKNET market
source is not configured, run_forecast() writes ml_status.json with
model_status = not_generated and does NOT produce forecast outputs.

Run with:
    pytest tests/test_forecasting.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_ML = Path(__file__).resolve().parents[1] / "src" / "ml"
sys.path.insert(0, str(SRC_ML))

import demand_forecast_model as dfm  # noqa: E402

PRICES = {
    "RegionA": 2600.0, "RegionB": 2400.0, "RegionC": 2100.0,
    "RegionD": 1800.0, "RegionE": 1500.0, "RegionF": 1200.0,
}


def make_gold_fixture(tmp_path: Path, n_regions: int = 6, n_days: int = 3) -> Path:
    """Create a synthetic region_daily_features dataset matching the Gold contract."""
    rows = []
    for i in range(n_days):
        for region in list(PRICES)[:n_regions]:
            rows.append({
                "region_name": region,
                "date": pd.Timestamp("2026-08-01") + pd.DateOffset(days=i),
                "temp_max_c": 30.0 + i, "temp_min_c": 18.0 + i,
                "temp_avg_c": 24.0 + i, "precipitation_mm": 2.0 + i,
                "humidity_pct": 70.0, "windspeed_max_kmh": 15.0,
                "state_ut": "Maharashtra", "state_code": "MH",
                "district": region, "mandi_apmc": region,
                "latitude": 19.9, "longitude": 73.8,
                "avg_market_price_inr_per_quintal": PRICES[region] + i,
                "total_arrivals_quintal": 5000.0 + region.encode("ascii")[0],
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


@pytest.fixture()
def market_configured(monkeypatch):
    """Simulate the e-NAM / AGMARKNET source being configured."""
    monkeypatch.setattr(dfm, "market_enabled", lambda: True)


@pytest.fixture()
def market_not_configured(monkeypatch):
    monkeypatch.setattr(dfm, "market_enabled", lambda: False)


def test_gold_fixture_loads(gold_tmp):
    df = dfm.load_gold_features()
    assert len(df) == 6 * 3
    assert set(["region_name", "temp_avg_c", "precipitation_mm", "humidity_pct",
                "windspeed_max_kmh"]).issubset(df.columns)


def test_required_feature_columns_exist(gold_tmp):
    df = dfm.load_gold_features()
    region_df = dfm.build_region_level_dataset(df)
    missing = [c for c in dfm.FEATURE_COLS if c not in region_df.columns]
    assert missing == []
    assert dfm.TARGET_COL in region_df.columns


def test_predictions_generated_and_finite(gold_tmp, market_configured):
    df = dfm.load_gold_features()
    region_df = dfm.build_region_level_dataset(df)
    model_info = dfm.train_model(region_df)
    forecast_df = dfm.build_forecast_frame(region_df, model_info)

    assert len(forecast_df) == len(region_df)
    assert forecast_df["predicted_signal_inr_per_quintal"].notna().all()
    assert np.isfinite(forecast_df["predicted_signal_inr_per_quintal"].to_numpy()).all()
    assert np.isfinite(forecast_df["actual_signal_inr_per_quintal"].to_numpy()).all()


def test_expected_output_fields_exist(gold_tmp, market_configured):
    df = dfm.load_gold_features()
    region_df = dfm.build_region_level_dataset(df)
    model_info = dfm.train_model(region_df)
    forecast_df = dfm.build_forecast_frame(region_df, model_info)

    expected = ["region_name", "actual_signal_inr_per_quintal",
                "predicted_signal_inr_per_quintal", "difference_inr_per_quintal",
                "prediction_error_pct"]
    for col in expected:
        assert col in forecast_df.columns


def test_small_sample_flagged_in_sample(gold_tmp, market_configured):
    df = dfm.load_gold_features()
    region_df = dfm.build_region_level_dataset(df)
    # 6 regions < MIN_ROWS_FOR_VALID_SPLIT -> metrics must be labelled in-sample
    model_info = dfm.train_model(region_df)
    assert model_info["metrics"]["metrics_are_in_sample"] is True
    assert model_info["small_sample_warning"] is not None


def test_write_forecast_outputs(gold_tmp, market_configured):
    df = dfm.load_gold_features()
    region_df = dfm.build_region_level_dataset(df)
    model_info = dfm.train_model(region_df)
    forecast_df = dfm.build_forecast_frame(region_df, model_info)
    dfm.write_forecast_outputs(forecast_df, model_info)

    assert (dfm.GOLD / "forecast_results.parquet").exists()
    assert (dfm.CATALOG / "model_report.json").exists()
    assert (dfm.CATALOG / "ml_status.json").exists() is False
    written = pd.read_parquet(dfm.GOLD / "forecast_results.parquet")
    assert len(written) == len(forecast_df)


def test_prediction_error_pct_null_when_actual_zero(gold_tmp, market_configured):
    df = dfm.load_gold_features()
    region_df = dfm.build_region_level_dataset(df)
    bad = region_df.copy()
    bad.loc[0, dfm.TARGET_COL] = 0.0
    model_info = dfm.train_model(bad)
    forecast_df = dfm.build_forecast_frame(bad, model_info)
    assert np.isnan(forecast_df.loc[0, "prediction_error_pct"])


def test_market_gate_writes_ml_status_and_no_forecast(gold_tmp, market_not_configured):
    result = dfm.run_forecast()
    assert result["gated"] is True
    assert result["forecast"].empty

    status_path = dfm.CATALOG / "ml_status.json"
    assert status_path.exists()
    status = json.loads(status_path.read_text())
    assert status["model_status"] == "not_generated"

    assert not (dfm.GOLD / "forecast_results.parquet").exists()
    assert not (dfm.CATALOG / "model_report.json").exists()


def test_market_configured_path_produces_outputs(gold_tmp, market_configured):
    result = dfm.run_forecast()
    assert result["gated"] is False
    assert (dfm.GOLD / "forecast_results.parquet").exists()
    assert (dfm.CATALOG / "model_report.json").exists()
    report = json.loads((dfm.CATALOG / "model_report.json").read_text())
    assert report["target_variable"] == dfm.TARGET_COL