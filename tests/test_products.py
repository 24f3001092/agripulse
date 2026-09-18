"""
test_products.py

Unit tests for the public product-layer modules (src/products/*), for the
deployment-readiness contract of the tracked data/india/demo/ artifacts, and
for the India market extension point (src/ingestion/india_market_source.py).

Rules honored here the same way the dashboard honors them:
  * No live APIs - every test reads local code or the tracked data/india/demo files.
  * No invented values - real-artifact tests assert on the demo copies that are
    committed to the repo for Streamlit Community Cloud deployment.
  * India reality - INR (₹), per quintal, weather-exposure segments, and the
    honest not_configured market state.

Covered:
  * scenario.py   - estimate_scenario math, validation, sensitivity frames
  * insights.py   - build_region_insights cards and tercile labeling
  * artifacts.py  - REQUIRED_ARTIFACTS <-> data/india/demo deployment-mapping checks
  * india_market_source.py - the documented e-NAM / AGMARKNET extension point

Run with:
    pytest tests/test_products.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

from products import artifacts, insights, scenario  # noqa: E402
import config  # noqa: E402
from ingestion import india_market_source  # noqa: E402

REPO_ROOT = BASE
DEMO_ROOT = BASE / "data" / "india" / "demo"


# --------------------------------------------------------------------------- #
# Scenario planner (currency-agnostic math; presented as INR in the dashboard)
# --------------------------------------------------------------------------- #
def test_estimate_scenario_basic_math():
    est = scenario.estimate_scenario(
        quantity=1000, unit_price=100,
        unit_seed_cost=30, unit_fertilizer_cost=0, unit_labor_cost=0,
        unit_transport_cost=20, unit_storage_cost=10, unit_other_cost=0,
        loss_pct=5
    )
    assert est["errors"] == []
    assert est["sellable_quantity"] == pytest.approx(950.0)
    assert est["revenue"] == pytest.approx(95000.0)
    assert est["total_cost"] == pytest.approx(60000.0)
    assert est["margin"] == pytest.approx(35000.0)
    assert est["margin_pct"] == pytest.approx(36.842105263157894)


def test_estimate_scenario_no_loss():
    est = scenario.estimate_scenario(
        quantity=100, unit_price=50,
        unit_seed_cost=5, unit_fertilizer_cost=0, unit_labor_cost=0,
        unit_transport_cost=5, unit_storage_cost=0, unit_other_cost=0,
        loss_pct=0
    )
    assert est["sellable_quantity"] == pytest.approx(100.0)
    assert est["revenue"] == pytest.approx(5000.0)


def test_estimate_scenario_zero_revenue_margin_pct_is_nan():
    est = scenario.estimate_scenario(
        quantity=100, unit_price=0,
        unit_seed_cost=5, unit_fertilizer_cost=0, unit_labor_cost=0,
        unit_transport_cost=0, unit_storage_cost=0, unit_other_cost=0,
        loss_pct=0
    )
    assert est["errors"] == []
    assert est["revenue"] == pytest.approx(0.0)
    assert np.isnan(est["margin_pct"])
    assert est["margin"] == pytest.approx(-500.0)


def test_estimate_scenario_zero_quantity_error():
    est = scenario.estimate_scenario(quantity=0, unit_price=100)
    assert any("Quantity" in e for e in est["errors"])
    assert est["revenue"] == 0.0


def test_estimate_scenario_negative_price_error():
    est = scenario.estimate_scenario(
        quantity=100, unit_price=-10,
        unit_seed_cost=1, unit_fertilizer_cost=0, unit_labor_cost=0,
        unit_transport_cost=0, unit_storage_cost=0, unit_other_cost=0,
    )
    assert any("price" in e.lower() for e in est["errors"])


def test_estimate_scenario_negative_cost_errors():
    est = scenario.estimate_scenario(
        quantity=100, unit_price=10,
        unit_seed_cost=-1, unit_fertilizer_cost=-2, unit_labor_cost=-3,
        unit_transport_cost=0, unit_storage_cost=-4, unit_other_cost=0,
    )
    for label in ("seed/input", "fertilizer", "labor", "storage"):
        assert any(label.capitalize() in e for e in est["errors"])


def test_estimate_scenario_loss_100_is_invalid():
    est = scenario.estimate_scenario(quantity=100, unit_price=10, loss_pct=100)
    assert any("loss" in e.lower() for e in est["errors"])


def test_estimate_scenario_invalid_inputs_zero_out_derived_columns():
    est = scenario.estimate_scenario(
        quantity=0, unit_price=100,
        unit_seed_cost=0, unit_fertilizer_cost=0, unit_labor_cost=0,
        unit_transport_cost=0, unit_storage_cost=0, unit_other_cost=0,
        loss_pct=5
    )
    assert est["sellable_quantity"] == 0.0
    assert est["revenue"] == 0.0
    assert est["total_cost"] == 0.0
    assert est["margin"] == 0.0


def test_run_sensitivity_baseline_row_matches_estimate():
    baseline = scenario.estimate_scenario(
        1000, 100,
        unit_seed_cost=30, unit_fertilizer_cost=0, unit_labor_cost=0,
        unit_transport_cost=20, unit_storage_cost=10, unit_other_cost=0,
        loss_pct=5
    )
    sens = scenario.run_sensitivity(
        1000, 100,
        unit_seed_cost=30, unit_fertilizer_cost=0, unit_labor_cost=0,
        unit_transport_cost=20, unit_storage_cost=10, unit_other_cost=0,
        loss_pct=5
    )
    assert not sens.empty
    base_row = sens[sens["variation"] == "Baseline"].iloc[0]
    assert base_row["margin"] == pytest.approx(baseline["margin"])
    assert base_row["margin_pct"] == pytest.approx(baseline["margin_pct"])


def test_run_sensitivity_has_expected_variations():
    sens = scenario.run_sensitivity(
        1000, 100,
        unit_seed_cost=30, unit_fertilizer_cost=0, unit_labor_cost=0,
        unit_transport_cost=20, unit_storage_cost=10, unit_other_cost=0,
        loss_pct=5
    )
    assert set(sens["variation"]) == {
        "Baseline", "Price +10%", "Price -10%", "Quantity +10%",
        "Quantity -10%", "Costs +10%", "Costs -10%",
    }


def test_run_sensitivity_columns_and_deltas():
    sens = scenario.run_sensitivity(
        1000, 100,
        unit_seed_cost=30, unit_fertilizer_cost=0, unit_labor_cost=0,
        unit_transport_cost=20, unit_storage_cost=10, unit_other_cost=0,
        loss_pct=5
    )
    assert {"variation", "unit_price", "revenue", "total_cost", "margin",
            "margin_pct", "margin_pct_delta", "break_even_price"} <= set(sens.columns)
    assert not any(c.endswith("_musd") for c in sens.columns)
    assert sens["margin_pct_delta"].notna().all()


def test_run_sensitivity_price_increase_raises_margin():
    sens = scenario.run_sensitivity(
        1000, 100,
        unit_seed_cost=30, unit_fertilizer_cost=0, unit_labor_cost=0,
        unit_transport_cost=20, unit_storage_cost=10, unit_other_cost=0,
        loss_pct=5
    ).set_index("variation")
    assert sens.loc["Price +10%", "margin"] > sens.loc["Baseline", "margin"]
    assert sens.loc["Price -10%", "margin"] < sens.loc["Baseline", "margin"]


def test_run_sensitivity_invalid_inputs_returns_empty_frame():
    sens = scenario.run_sensitivity(
        quantity=0, unit_price=100,
        unit_seed_cost=0, unit_fertilizer_cost=0, unit_labor_cost=0,
        unit_transport_cost=0, unit_storage_cost=0, unit_other_cost=0,
    )
    assert sens.empty


# --------------------------------------------------------------------------- #
# Insights (India: weather + exposure segment; market cards only when configured)
# --------------------------------------------------------------------------- #
def _daily_fixture() -> pd.DataFrame:
    regions = ["Alpha", "Beta", "Gamma"]
    market = {
        "Alpha": {"Wheat_inr_per_quintal": 2400.0, "Onion_inr_per_quintal": 1800.0, "Tomato_inr_per_quintal": 900.0},
        "Beta": {"Wheat_inr_per_quintal": 2200.0, "Onion_inr_per_quintal": 1200.0, "Tomato_inr_per_quintal": 700.0},
        "Gamma": {"Wheat_inr_per_quintal": 2100.0, "Onion_inr_per_quintal": 1100.0, "Tomato_inr_per_quintal": 600.0},
    }
    rows = []
    for region in regions:
        for i in range(3):
            rows.append({
                "region_name": region,
                "date": pd.Timestamp("2026-09-01") + pd.DateOffset(days=i),
                "temp_max_c": 30.0 + i, "temp_min_c": 18.0 + i,
                "temp_avg_c": 25.0 + i, "precipitation_mm": 5.0 + i,
                "humidity_pct": 65.0, "windspeed_max_kmh": 14.0,
                **market[region],
            })
    return pd.DataFrame(rows)


def _summary_fixture() -> pd.DataFrame:
    return pd.DataFrame([
        {"region_name": "Alpha", "avg_temp_c": 26.0},
        {"region_name": "Beta", "avg_temp_c": 27.0},
        {"region_name": "Gamma", "avg_temp_c": 28.0},
    ])


def _segments_fixture() -> pd.DataFrame:
    return pd.DataFrame([
        {"region_name": "Alpha", "segment": "High-Rainfall > Cool-Regime > Humid",
         "segment_reason": "Alpha observed 60mm precipitation (top tercile).",
         "exposure_tier": "High-Exposure", "dominant_factor": "precipitation"},
        {"region_name": "Beta", "segment": "Moderate-Rainfall > Moderate-Heat > Humid",
         "segment_reason": "Beta observed 12mm precipitation (mid tercile).",
         "exposure_tier": "Moderate-Exposure", "dominant_factor": "temperature"},
        {"region_name": "Gamma", "segment": "Water-Limited > Heat-Stress > Dry-Air",
         "segment_reason": "Gamma observed 3mm precipitation (bottom tercile).",
         "exposure_tier": "High-Exposure", "dominant_factor": "precipitation"},
    ])


def _forecast_fixture() -> pd.DataFrame:
    return pd.DataFrame([
        {"region_name": "Alpha", "actual_signal_inr_per_quintal": 2400.0, "predicted_signal_inr_per_quintal": 2380.0},
        {"region_name": "Beta", "actual_signal_inr_per_quintal": 2200.0, "predicted_signal_inr_per_quintal": 2300.0},
        {"region_name": "Gamma", "actual_signal_inr_per_quintal": 2100.0, "predicted_signal_inr_per_quintal": 2050.0},
    ])


def test_build_region_insights_weather_card():
    cards = insights.build_region_insights(
        _daily_fixture(), _summary_fixture(), _segments_fixture(), _forecast_fixture(), "Alpha"
    )
    weather = next(c for c in cards if c["title"].startswith("Weather exposure:"))
    assert "Alpha" in weather["what"]
    assert "C" in weather["what"] and "mm" in weather["what"]
    assert weather["source_method"].startswith("Computed from data/india/gold/region_daily_features")


def test_build_region_insights_commodity_mix_card():
    cards = insights.build_region_insights(
        _daily_fixture(), _summary_fixture(), _segments_fixture(), _forecast_fixture(), "Alpha"
    )
    crop_card = next(c for c in cards if c["title"].startswith("Commodity mix:"))
    assert "wheat" in crop_card["what"].lower()
    assert "/quintal" in crop_card["what"]


def test_build_region_insights_segment_and_forecast_cards():
    cards = insights.build_region_insights(
        _daily_fixture(), _summary_fixture(), _segments_fixture(), _forecast_fixture(), "Beta"
    )
    assert any(c["title"].startswith("Weather-exposure tier:") for c in cards)
    forecast_card = next(c for c in cards if c["title"].startswith("Model signal:"))
    assert "prototype" in forecast_card["title"].lower()
    assert "in-sample" in forecast_card["why"].lower()


def test_build_region_insights_empty_region_returns_empty_list():
    cards = insights.build_region_insights(
        _daily_fixture(), _summary_fixture(), _segments_fixture(), _forecast_fixture(), "Nonexistent"
    )
    assert cards == []


def test_build_region_insights_tolerates_missing_frames():
    cards = insights.build_region_insights(
        _daily_fixture(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "Alpha"
    )
    titles = {c["title"] for c in cards}
    assert not any(t.startswith("Weather-exposure tier:") for t in titles)
    assert not any(t.startswith("Model signal:") for t in titles)


def test_build_region_insights_skips_commodity_card_without_market_columns():
    daily = _daily_fixture()[[
        "region_name", "date", "temp_max_c", "temp_min_c", "temp_avg_c",
        "precipitation_mm", "humidity_pct", "windspeed_max_kmh",
    ]]
    cards = insights.build_region_insights(
        daily, _summary_fixture(), _segments_fixture(), _forecast_fixture(), "Alpha"
    )
    assert not any(t.startswith("Commodity mix:") for t in {c["title"] for c in cards})


def test_tercile_label_thresholds():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert insights._tercile_label(1.0, series) == "Low"
    assert insights._tercile_label(3.5, series) == "Moderate"
    assert insights._tercile_label(6.0, series) == "High"


def test_tercile_label_empty_series_returns_na():
    assert insights._tercile_label(1.0, pd.Series(dtype=float)) == "n/a"


# --------------------------------------------------------------------------- #
# Deployment artifacts mapping (data/india/demo as tracked source of truth)
# --------------------------------------------------------------------------- #
def test_required_artifacts_all_have_tracked_demo_copies():
    missing = artifacts.missing_demo_artifacts(REPO_ROOT)
    assert missing == [], f"demo artifacts missing under repo root: {missing}"


def test_demo_artifacts_present_detects_missing(tmp_path):
    assert artifacts.demo_artifacts_present(tmp_path) is False
    assert len(artifacts.missing_demo_artifacts(tmp_path)) == len(artifacts.REQUIRED_ARTIFACTS)


def test_demo_artifacts_present_true_on_populated_tree(tmp_path):
    for key, (live_rel, demo_rel) in artifacts.REQUIRED_ARTIFACTS.items():
        p = tmp_path / demo_rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    assert artifacts.demo_artifacts_present(tmp_path) is True


def test_every_required_artifact_maps_to_existing_demo_file():
    for key, (live_rel, demo_rel) in artifacts.REQUIRED_ARTIFACTS.items():
        assert (REPO_ROOT / demo_rel).is_file(), f"{key} demo path is not a file: {demo_rel}"


def test_optional_artifacts_not_required_for_deployment():
    # forecast / model / market are market-gated; their absence must not fail
    # the deployment contract today.
    assert "forecast" in artifacts.OPTIONAL_ARTIFACTS
    assert "model" in artifacts.OPTIONAL_ARTIFACTS
    assert "silver_market" in artifacts.OPTIONAL_ARTIFACTS


# --------------------------------------------------------------------------- #
# Demo artifact schema contracts (no live APIs, no fabricated data)
# --------------------------------------------------------------------------- #
def test_demo_gold_daily_columns_and_shape():
    df = pd.read_parquet(DEMO_ROOT / "gold" / "region_daily_features.parquet")
    expected = {
        "region_name", "date", "temp_max_c", "temp_min_c", "temp_avg_c",
        "precipitation_mm", "humidity_pct", "windspeed_max_kmh",
        "state_ut", "state_code", "district", "mandi_apmc",
        "latitude", "longitude",
    }
    assert expected <= set(df.columns)
    assert not df.empty
    assert df["date"].isnull().sum() == 0
    assert df["region_name"].nunique() >= 10
    # No U.S.-style export columns anywhere in the active India profile.
    assert not any(c in df.columns for c in ["total_exports_musd", "corn_musd", "wheat_musd"])


def test_demo_gold_summary_columns():
    df = pd.read_parquet(DEMO_ROOT / "gold" / "region_summary.parquet")
    assert {"region_name", "avg_temp_c", "total_precip_mm", "avg_humidity_pct"} <= set(df.columns)
    assert not df.empty
    assert (df["total_precip_mm"] >= 0).all()


def test_demo_segments_columns():
    df = pd.read_parquet(DEMO_ROOT / "gold" / "region_segments.parquet")
    assert {"region_name", "segment", "segment_reason", "exposure_tier", "dominant_factor"} <= set(df.columns)
    assert not df.empty
    assert df["exposure_tier"].notna().all()


def test_demo_ml_status_reports_not_generated_honestly():
    import json
    status = json.loads((DEMO_ROOT / "catalog" / "ml_status.json").read_text())
    assert status["model_status"] == "not_generated"
    assert "reason" in status and status["reason"]
    assert isinstance(status["required_upstream"], list) and status["required_upstream"]


def test_demo_health_report_status_key():
    import json
    report = json.loads((DEMO_ROOT / "catalog" / "health_report.json").read_text())
    assert "status" in report
    assert report["status"] in {"PASS", "FAIL"}
    assert (report.get("market_source") or {}).get("status") == "not_configured"


def test_demo_has_no_market_parquet_yet():
    # Market data is an extension point: no market file may exist until a real
    # e-NAM / AGMARKNET loader is configured.
    assert not (DEMO_ROOT / "silver" / "market.parquet").exists()


# --------------------------------------------------------------------------- #
# India market extension point (e-NAM / AGMARKNET -- no live source yet)
# --------------------------------------------------------------------------- #
def test_india_market_column_contract_is_documented():
    assert india_market_source.MARKET_COLUMNS == [
        "commodity", "variety", "state_ut", "district", "mandi",
        "date", "market_price_inr_per_quintal", "arrivals_quintal",
    ]


def test_india_market_source_status_is_not_configured():
    assert india_market_source.market_source_status() == "not_configured"


def test_india_market_load_never_fabricates():
    result = india_market_source.load_market_data()
    assert result["status"] == "not_configured"
    assert result["df"] is None
    assert "reason" in result