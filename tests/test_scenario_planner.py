"""
test_scenario_planner.py

Phase 8 -- India Agricultural Scenario Planner tests.

Validates that the scenario planner:
  * computes correct math for normal case
  * rejects negative quantities, prices, costs
  * rejects impossible percentages (>=100% loss)
  * handles zero quantity
  * computes break-even price correctly
  * runs sensitivity with ±10% variations
  * computes risk indicators from AgriPulse signals
"""

import numpy as np
import pandas as pd
import pytest

from src.products import scenario


def test_normal_case():
    """Basic scenario with all fields populated."""
    est = scenario.estimate_scenario(
        quantity=1000.0,
        unit_price=2500.0,
        unit_seed_cost=500.0,
        unit_fertilizer_cost=300.0,
        unit_labor_cost=200.0,
        unit_transport_cost=150.0,
        unit_storage_cost=100.0,
        unit_other_cost=50.0,
        loss_pct=5.0,
    )
    assert not est["errors"]
    assert est["sellable_quantity"] == 950.0  # 1000 * (1 - 5%)
    assert est["revenue"] == 950.0 * 2500.0  # 2,375,000
    total_cost_per_q = 500 + 300 + 200 + 150 + 100 + 50  # 1300
    assert est["total_cost"] == 1000.0 * total_cost_per_q  # 1,300,000
    assert est["margin"] == est["revenue"] - est["total_cost"]  # 1,075,000
    assert est["margin_pct"] == (est["margin"] / est["revenue"]) * 100.0  # ~45.26%
    assert est["break_even_price"] == est["total_cost"] / est["sellable_quantity"]  # ~1368.42


def test_zero_quantity_rejected():
    """Quantity must be > 0."""
    est = scenario.estimate_scenario(
        quantity=0.0,
        unit_price=2500.0,
        unit_seed_cost=500.0,
    )
    assert est["errors"]
    assert "Quantity must be greater than 0." in est["errors"]


def test_negative_price_rejected():
    """Selling price cannot be negative."""
    est = scenario.estimate_scenario(
        quantity=1000.0,
        unit_price=-100.0,
        unit_seed_cost=500.0,
    )
    assert est["errors"]
    assert "Selling price cannot be negative." in est["errors"]


def test_negative_costs_rejected():
    """All cost fields cannot be negative."""
    for cost_field in [
        "unit_seed_cost", "unit_fertilizer_cost", "unit_labor_cost",
        "unit_transport_cost", "unit_storage_cost", "unit_other_cost"
    ]:
        kwargs = {
            "quantity": 1000.0,
            "unit_price": 2500.0,
            cost_field: -50.0,
        }
        est = scenario.estimate_scenario(**kwargs)
        assert est["errors"]
        assert any("cannot be negative" in err.lower() for err in est["errors"])


def test_loss_100_percent_rejected():
    """Expected loss must be < 100%."""
    est = scenario.estimate_scenario(
        quantity=1000.0,
        unit_price=2500.0,
        unit_seed_cost=500.0,
        loss_pct=100.0,
    )
    assert est["errors"]
    assert "Expected loss must be between 0 and below 100 percent." in est["errors"]


def test_loss_negative_rejected():
    """Expected loss cannot be negative."""
    est = scenario.estimate_scenario(
        quantity=1000.0,
        unit_price=2500.0,
        unit_seed_cost=500.0,
        loss_pct=-1.0,
    )
    assert est["errors"]
    assert "Expected loss must be between 0 and below 100 percent." in est["errors"]


def test_zero_loss_accepted():
    """Zero loss percentage is valid."""
    est = scenario.estimate_scenario(
        quantity=1000.0,
        unit_price=2500.0,
        unit_seed_cost=500.0,
        loss_pct=0.0,
    )
    assert not est["errors"]
    assert est["sellable_quantity"] == 1000.0


def test_break_even_price_calculation():
    """Break-even price = total_cost / sellable_quantity."""
    est = scenario.estimate_scenario(
        quantity=500.0,
        unit_price=2000.0,
        unit_seed_cost=400.0,
        unit_fertilizer_cost=200.0,
        unit_labor_cost=100.0,
        unit_transport_cost=50.0,
        unit_storage_cost=50.0,
        unit_other_cost=0.0,
        loss_pct=10.0,
    )
    assert not est["errors"]
    sellable = 500 * 0.9  # 450
    total_cost = 500 * (400 + 200 + 100 + 50 + 50 + 0)  # 400,000
    expected_be = total_cost / sellable  # 400000 / 450 = 888.888...
    assert np.isclose(est["break_even_price"], expected_be)


def test_break_even_nan_when_sellable_zero():
    """Break-even is nan when sellable quantity is 0 (not tested directly since quantity>0)."""
    # With 99.99% loss, sellable is tiny but not zero
    est = scenario.estimate_scenario(
        quantity=1000.0,
        unit_price=2500.0,
        unit_seed_cost=500.0,
        loss_pct=99.9,
    )
    # sellable = 1000 * 0.001 = 1
    assert pd.notna(est["break_even_price"])


def test_sensitivity_variations():
    """Sensitivity runs with ±10% on price, quantity, costs."""
    sens = scenario.run_sensitivity(
        quantity=1000.0,
        unit_price=2500.0,
        unit_seed_cost=500.0,
        unit_fertilizer_cost=300.0,
        unit_labor_cost=200.0,
        unit_transport_cost=150.0,
        unit_storage_cost=100.0,
        unit_other_cost=50.0,
        loss_pct=5.0,
    )
    assert len(sens) == 7  # Baseline + 6 variations
    variations = sens["variation"].tolist()
    expected_vars = [
        "Baseline", "Price +10%", "Price -10%",
        "Quantity +10%", "Quantity -10%",
        "Costs +10%", "Costs -10%",
    ]
    assert variations == expected_vars

    # Baseline row should match estimate
    baseline = sens[sens["variation"] == "Baseline"].iloc[0]
    est = scenario.estimate_scenario(
        quantity=1000.0, unit_price=2500.0,
        unit_seed_cost=500.0, unit_fertilizer_cost=300.0,
        unit_labor_cost=200.0, unit_transport_cost=150.0,
        unit_storage_cost=100.0, unit_other_cost=50.0,
        loss_pct=5.0,
    )
    assert np.isclose(baseline["revenue"], est["revenue"])
    assert np.isclose(baseline["margin"], est["margin"])
    assert np.isclose(baseline["break_even_price"], est["break_even_price"])

    # Price +10% should increase margin
    price_up = sens[sens["variation"] == "Price +10%"].iloc[0]
    assert price_up["margin"] > baseline["margin"]

    # Price -10% should decrease margin
    price_down = sens[sens["variation"] == "Price -10%"].iloc[0]
    assert price_down["margin"] < baseline["margin"]

    # Costs +10% should decrease margin
    cost_up = sens[sens["variation"] == "Costs +10%"].iloc[0]
    assert cost_up["margin"] < baseline["margin"]


def test_sensitivity_empty_on_invalid():
    """Sensitivity returns empty DataFrame when baseline has errors."""
    sens = scenario.run_sensitivity(
        quantity=0.0,  # invalid
        unit_price=2500.0,
        unit_seed_cost=500.0,
    )
    assert sens.empty


def test_compute_risk_indicators_break_even_vs_market():
    """Risk indicator when break-even differs from market price."""
    est = {
        "break_even_price": 2000.0,
        "sellable_quantity": 950.0,
    }
    # Market price below break-even by >10%
    risk = scenario.compute_risk_indicators(est, market_price=1500.0)
    assert any(r["indicator"] == "Break-even above market" and r["severity"] == "caution" for r in risk)

    # Market price above break-even by >10%
    risk = scenario.compute_risk_indicators(est, market_price=2500.0)
    assert any(r["indicator"] == "Break-even below market" and r["severity"] == "info" for r in risk)

    # Market price within 10% of break-even -> no indicator
    risk = scenario.compute_risk_indicators(est, market_price=1950.0)
    assert not any(r["indicator"] in ("Break-even above market", "Break-even below market") for r in risk)


def test_compute_risk_indicators_price_trend():
    """Risk indicators for price trends."""
    est = {"break_even_price": 2000.0}
    # Declining trend
    risk = scenario.compute_risk_indicators(est, price_trend_pct=-10.0)
    assert any(r["indicator"] == "Declining price trend" and r["severity"] == "warning" for r in risk)
    # Rising trend
    risk = scenario.compute_risk_indicators(est, price_trend_pct=10.0)
    assert any(r["indicator"] == "Rising price trend" and r["severity"] == "info" for r in risk)
    # Flat trend
    risk = scenario.compute_risk_indicators(est, price_trend_pct=2.0)
    assert not any("price trend" in r["indicator"].lower() for r in risk)


def test_compute_risk_indicators_weather():
    """Risk indicators for weather exposure."""
    est = {"break_even_price": 2000.0}
    risk = scenario.compute_risk_indicators(est, weather_exposure_tier="High")
    assert any(r["indicator"] == "High weather exposure" and r["severity"] == "caution" for r in risk)
    risk = scenario.compute_risk_indicators(est, weather_exposure_tier="Medium")
    assert any(r["indicator"] == "Medium weather exposure" and r["severity"] == "info" for r in risk)
    risk = scenario.compute_risk_indicators(est, weather_exposure_tier="Low")
    assert not any("weather exposure" in r["indicator"].lower() for r in risk)


def test_compute_risk_indicators_production_context():
    """Risk indicator includes production context when provided."""
    est = {"break_even_price": 2000.0}
    risk = scenario.compute_risk_indicators(est, production_context="Forecast 2023-24: 5000 tonnes")
    assert any(r["indicator"] == "Crop production context" for r in risk)


def test_compute_risk_indicators_arrival_trend():
    """Risk indicators for arrival trends."""
    est = {"break_even_price": 2000.0}
    risk = scenario.compute_risk_indicators(est, arrival_trend_pct=-15.0)
    assert any(r["indicator"] == "Declining arrivals" and r["severity"] == "warning" for r in risk)
    risk = scenario.compute_risk_indicators(est, arrival_trend_pct=15.0)
    assert any(r["indicator"] == "Increasing arrivals" and r["severity"] == "info" for r in risk)