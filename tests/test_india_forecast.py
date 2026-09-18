"""
test_india_forecast.py

Phase 7 -- India crop-production forecasting.

Validates that the India forecast module:
  * audits the real historical dimensions and chooses production as the ONLY
    defensible target (prices/arrivals have a 61-day daily window, so they are
    explicitly NOT forecast);
  * uses a STRICT temporal train/validate/test split (no random shuffle, train
    years strictly precede validation and test years -- no leakage);
  * never fabricates metrics or a production-ready claim;
  * reports MAE / RMSE / R2 / MAPE from held-out rows only;
  * writes india_forecasts.parquet + india_model_report.json (or an honest
    no_data report when the agriculture panel is absent).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "src" / "ml"))
import india_forecast as if_  # noqa: E402


YEARS = [f"{y}-{y + 1}" for y in range(1997, 2023)]


def _synthetic_panel(n_cells: int = 40, seed: int = 7) -> pd.DataFrame:
    """Small deterministic crop-year panel: 26 years, several cells.

    Column names match the post-load schema consumed by the forecast module
    (production_t / area_ha / yield_tph).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cells):
        state = f"S{c % 3}"
        district = f"D{c % 12}"
        crop = f"C{c % 8}"
        season = c % 2 and "Kharif" or "Rabi"
        base = 100.0 + 50.0 * (c % 5)
        prod = base
        for i, y in enumerate(range(1997, 2023)):
            prod = prod * 1.03 + rng.normal(0, 4)
            rows.append({
                "state": state, "district": district, "crop": crop, "season": season,
                "year": YEARS[i], "year_start": y,
                "production_t": max(prod, 1.0), "area_ha": 50.0, "yield_tph": prod / 50.0,
                "state_code": 1, "district_code": 100 + c % 12, "crop_type": "Foodgrains",
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel():
    return _synthetic_panel()


@pytest.fixture(scope="module")
def built(panel):
    return if_.prepare_dataset(panel)


@pytest.fixture(scope="module")
def fit(built):
    return if_.run_fit(built)


def test_audit_chooses_production_as_target():
    agri = pd.DataFrame({
        "year": [f"{y}-{y+1}" for y in range(1997, 2023)],
        "state": ["X"] * 26, "district": ["D"] * 26, "crop": ["Wheat"] * 26,
    })
    mandi = pd.DataFrame({"latest_date": pd.to_datetime(["2026-04-01"]),
                          "days_of_history": [61]})
    weather = pd.DataFrame({"date": pd.to_datetime(["2026-08-19"])})
    result = if_.audit(agri=agri, mandi=mandi, weather=weather)
    assert result["target_decision"].startswith("production")
    assert result["agriculture"]["n_years"] == 26
    assert result["mandi"]["verdict"] == "insufficient history for temporal validation"
    assert result["weather"]["verdict"].startswith("single recent window")


def test_audit_insufficient_history_targets_nothing():
    agri = pd.DataFrame({"year": ["2020-2021"], "state": ["X"], "district": ["D"], "crop": ["W"]})
    result = if_.audit(agri=agri)
    assert "no valid target" in result["target_decision"]


def test_temporal_split_never_shuffles_time():
    splits = if_.temporal_splits(YEARS)
    assert sorted(splits["train"] + splits["validate"] + splits["test"]) == sorted(YEARS)
    assert splits["train"] == sorted(splits["train"])
    assert max(splits["train"]) < min(splits["validate"])
    assert max(splits["validate"]) < min(splits["test"])
    assert len(splits["test"]) == if_.SPLIT_TEST_YEARS
    assert len(splits["validate"]) == if_.SPLIT_VAL_YEARS


def test_lags_are_strictly_within_cell():
    # Rebuild lags on a tiny panel to assert within-cell ordering directly.
    raw = _synthetic_panel(n_cells=4, seed=3)
    with_lags = if_.build_features(raw)
    cell = with_lags[(with_lags["state"] == "S0") & (with_lags["district"] == "D0")
                     & (with_lags["crop"] == "C0")].sort_values("year_start")
    assert not cell.empty
    # lag1 at row t equals production at row t-1 within the same cell
    # (built keeps every observed row EXCEPT each cell's first -- so compare
    # against the raw panel's own lag sequence).
    raw_cell = raw[(raw["state"] == "S0") & (raw["district"] == "D0")
                   & (raw["crop"] == "C0")].sort_values("year_start")
    assert (cell["lag1_production"].values == raw_cell["production_t"].shift(1).dropna().values).all()
    # The first observation of a cell has no lag and is therefore never a
    # training target (it is removed by build_features).
    assert (cell["year_offset"] >= 1).all()
    assert len(with_lags) <= len(raw)  # boundary rows were dropped, not invented


def test_forecast_rows_only_for_next_unobserved_year(fit, built):
    forecast = if_.build_forecast_frame(built, fit, fit["model"],
                                        residual_std_log=0.5)
    assert (forecast["forecast_year"].map(str) == if_.FORECAST_YEAR).all()
    assert (forecast["latest_observed_year"].map(str) == if_.LAG1_ANCHOR_YEAR).all()


def test_forecast_columns_and_nonnegative(fit, built):
    forecast = if_.build_forecast_frame(built, fit, fit["model"],
                                        residual_std_log=0.5)
    expected = ["state", "district", "crop", "season", "forecast_year",
                "observed_last_tonnes", "forecast_production_tonnes",
                "forecast_lower_tonnes", "forecast_upper_tonnes", "model_used"]
    assert all(c in forecast.columns for c in expected)
    assert (forecast["forecast_production_tonnes"] >= 0).all()
    assert (forecast["forecast_lower_tonnes"] <= forecast["forecast_production_tonnes"]).all()
    # chosen model (naive or boosting) always tagged
    assert forecast["model_used"].iloc[0] in {"naive_persistence-prototype",
                                              "gradient_boosting-prototype"}


def test_metrics_present_and_sane(fit):
    m = fit["test_metrics"]
    assert m["mae_t"] >= 0
    assert m["rmse_t"] >= 0
    assert m["r2_vs_observed_mean"] <= 1.0
    assert m["n_rows"] > 0
    assert "mape_median_pct" in m  # may be None when <1t rows absent
    # naive baseline evaluated on the exact same held-out rows
    assert fit["naive_metrics"]["test"]["n_rows"] == m["n_rows"]
    assert fit["naive_metrics"]["validation"]["n_rows"] > 0


def test_report_contains_required_fields(fit):
    report = if_.build_report(fit, {"agriculture": {"years": {"min": "1997-1998", "max": "2022-2023"},
                                                    "n_years": 26, "rows": 1, "n_states": 34,
                                                    "n_districts": 737, "n_crops": 115}},
                              pd.DataFrame(), 0.5)
    for key in ("target", "features", "data_period", "training_rows",
                "validation_method", "metrics", "limitations"):
        assert key in report, key
    assert "production" in report["target"]["variable"]
    assert report["model_status"] == "prototype"
    assert report["production_ready"] is False


def test_report_never_claims_price_forecast(fit):
    report = if_.build_report(fit, dict(agriculture={"years": {"min": "1997-1998", "max": "2022-2023"},
                                                     "n_years": 26, "rows": 1, "n_states": 34,
                                                     "n_districts": 737, "n_crops": 115}),
                              pd.DataFrame(), 0.5)
    assert report["target"]["variable"] != "price"
    joined = " ".join(report["limitations"]).lower()
    assert "price" in joined
    assert "not attempted" in joined


def test_no_data_gate_writes_honest_report(tmp_path, monkeypatch):
    monkeypatch.setattr(if_, "CROP_YEAR_PATH", tmp_path / "missing" / "crop_year.parquet")
    out = tmp_path / "gold" / "india_forecasts.parquet"
    report_out = tmp_path / "catalog" / "india_model_report.json"
    monkeypatch.setattr(if_, "FORECAST_OUT", out)
    monkeypatch.setattr(if_, "REPORT_OUT", report_out)
    result = if_.run_forecast()
    assert result["status"] == "no_data"
    assert not out.exists()
    payload = json.loads(report_out.read_text())
    assert payload["model_status"] == "not_generated"
    assert "agriculture history unavailable" in payload["reason"]


def test_full_run_writes_outputs_with_monkeypatched_paths(tmp_path, monkeypatch, panel):
    crop_year_path = tmp_path / "gold" / "india_crop_summary" / "india_crop_year.parquet"
    crop_year_path.parent.mkdir(parents=True)
    monkeypatch.setattr(if_, "CROP_YEAR_PATH", crop_year_path)
    out = tmp_path / "gold" / "india_forecasts.parquet"
    report_out = tmp_path / "catalog" / "india_model_report.json"
    monkeypatch.setattr(if_, "FORECAST_OUT", out)
    monkeypatch.setattr(if_, "REPORT_OUT", report_out)
    raw = panel.rename(columns={"production_t": "production", "area_ha": "area", "yield_tph": "yield"})
    raw.to_parquet(crop_year_path, index=False)
    result = if_.run_forecast()
    assert result["status"] == "generated"
    assert out.exists()
    assert report_out.exists()

    fc = pd.read_parquet(out)
    assert (fc["forecast_year"] == if_.FORECAST_YEAR).all()
    report = json.loads(report_out.read_text())
    assert report["model_status"] == "prototype"
    assert report["production_ready"] is False
    assert "test_mae_t" in report["metrics"]
    assert "test_rmse_t" in report["metrics"]
    assert "test_r2_vs_observed_mean" in report["metrics"]

    # Strict temporal discipline on the real fit artefacts.
    splits = report["splits"]
    assert max(splits["train"]) < min(splits["validate"])
    assert max(splits["validate"]) < min(splits["test"])


def test_validation_samples_are_held_out_cases(fit):
    samples = if_.validation_samples(fit, cap=50)
    assert 0 < len(samples) <= 50
    test_years = if_.temporal_splits(YEARS)["test"]
    for s in samples:
        assert s["year"] in test_years
        assert s["observed_tonnes"] >= 0 and s["predicted_tonnes"] >= 0