"""Scenario planner logic for AgriPulse — pure, deterministic, Streamlit-free.

A "scenario estimate" combines user-supplied production assumptions:

    sellable_quantity = quantity * (1 - loss_pct / 100)
    revenue           = sellable_quantity * unit_price
    total_cost        = quantity * (unit_input_cost + unit_transport_cost + unit_storage_cost)
    estimated_margin  = revenue - total_cost

Every number is labeled an ESTIMATE under explicit user assumptions (for
AgriPulse India, figures are presented in INR, costs on a per-quintal basis
where the user chooses). This module never predicts profit and never invents
market data. Sensitivity analysis re-runs the estimate with price / quantity /
cost variations so the user can see how the estimated margin moves.

Run with:
    pytest tests/test_scenario_planner.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_scenario_inputs(
    quantity: float,
    unit_price: float,
    unit_seed_cost: float = 0.0,
    unit_fertilizer_cost: float = 0.0,
    unit_labor_cost: float = 0.0,
    unit_transport_cost: float = 0.0,
    unit_storage_cost: float = 0.0,
    unit_other_cost: float = 0.0,
    loss_pct: float = 0.0,
) -> list[str]:
    """Return human-readable validation errors, empty list when inputs are valid."""
    errors: list[str] = []
    if quantity is None or quantity <= 0:
        errors.append("Quantity must be greater than 0.")
    if unit_price is None or unit_price < 0:
        errors.append("Selling price cannot be negative.")
    for label, val in [
        ("seed/input cost", unit_seed_cost),
        ("fertilizer cost", unit_fertilizer_cost),
        ("labor cost", unit_labor_cost),
        ("transport cost", unit_transport_cost),
        ("storage cost", unit_storage_cost),
        ("other cost", unit_other_cost),
    ]:
        if val is None or val < 0:
            errors.append(f"{label.capitalize()} cannot be negative.")
    if loss_pct is None or not (0 <= loss_pct < 100):
        errors.append("Expected loss must be between 0 and below 100 percent.")
    return errors


def estimate_scenario(
    quantity: float,
    unit_price: float,
    unit_seed_cost: float = 0.0,
    unit_fertilizer_cost: float = 0.0,
    unit_labor_cost: float = 0.0,
    unit_transport_cost: float = 0.0,
    unit_storage_cost: float = 0.0,
    unit_other_cost: float = 0.0,
    loss_pct: float = 0.0,
) -> dict:
    """Estimate revenue, total cost and margin from user-supplied assumptions.

    Returns a dict with keys: errors, sellable_quantity, revenue, total_cost,
    margin, margin_pct, break_even_price. margin_pct is relative to revenue
    (nan when revenue is 0). break_even_price is the price needed to cover
    total cost at the sellable quantity (nan when sellable_quantity is 0).
    """
    errors = validate_scenario_inputs(
        quantity, unit_price, unit_seed_cost, unit_fertilizer_cost,
        unit_labor_cost, unit_transport_cost, unit_storage_cost,
        unit_other_cost, loss_pct
    )
    if errors:
        return {
            "errors": errors,
            "sellable_quantity": 0.0,
            "revenue": 0.0,
            "total_cost": 0.0,
            "margin": 0.0,
            "margin_pct": np.nan,
            "break_even_price": np.nan,
        }

    sellable_quantity = quantity * (1.0 - loss_pct / 100.0)
    total_cost_per_quintal = (
        unit_seed_cost + unit_fertilizer_cost + unit_labor_cost +
        unit_transport_cost + unit_storage_cost + unit_other_cost
    )
    revenue = sellable_quantity * unit_price
    total_cost = quantity * total_cost_per_quintal
    margin = revenue - total_cost
    margin_pct = (margin / revenue * 100.0) if revenue > 0 else np.nan
    break_even_price = (total_cost / sellable_quantity) if sellable_quantity > 0 else np.nan

    return {
        "errors": [],
        "sellable_quantity": sellable_quantity,
        "revenue": revenue,
        "total_cost": total_cost,
        "margin": margin,
        "margin_pct": margin_pct,
        "break_even_price": break_even_price,
    }


def compute_risk_indicators(
    estimate: dict,
    market_price: float | None = None,
    price_trend_pct: float | None = None,
    weather_exposure_tier: str | None = None,
    production_context: str | None = None,
    arrival_trend_pct: float | None = None,
) -> list[dict]:
    """Compute scenario risk indicators from AgriPulse signals.

    Each indicator is a dict with: indicator, severity (info/warning/caution),
    detail. These are analytical observations, not guarantees.
    """
    indicators: list[dict] = []

    # Break-even vs market price
    if market_price is not None and pd.notna(estimate.get("break_even_price")):
        be = estimate["break_even_price"]
        if be > 0 and market_price > 0:
            pct_diff = (market_price - be) / be * 100.0
            if pct_diff < -10:
                indicators.append({
                    "indicator": "Break-even above market",
                    "severity": "caution",
                    "detail": (
                        f"Your break-even price (₹{be:,.0f}/quintal) is "
                        f"{abs(pct_diff):.0f}% above the latest market price "
                        f"(₹{market_price:,.0f}/quintal)."
                    ),
                })
            elif pct_diff > 10:
                indicators.append({
                    "indicator": "Break-even below market",
                    "severity": "info",
                    "detail": (
                        f"Your break-even price (₹{be:,.0f}/quintal) is "
                        f"{pct_diff:.0f}% below the latest market price "
                        f"(₹{market_price:,.0f}/quintal)."
                    ),
                })

    # Price trend
    if price_trend_pct is not None:
        if price_trend_pct < -5:
            indicators.append({
                "indicator": "Declining price trend",
                "severity": "warning",
                "detail": f"Recent price trend shows a {price_trend_pct:.1f}% decline (7-day window).",
            })
        elif price_trend_pct > 5:
            indicators.append({
                "indicator": "Rising price trend",
                "severity": "info",
                "detail": f"Recent price trend shows a {price_trend_pct:.1f}% increase (7-day window).",
            })

    # Weather exposure
    if weather_exposure_tier == "High":
        indicators.append({
            "indicator": "High weather exposure",
            "severity": "caution",
            "detail": "This district shows high weather exposure (analytical label from observed weather).",
        })
    elif weather_exposure_tier == "Medium":
        indicators.append({
            "indicator": "Medium weather exposure",
            "severity": "info",
            "detail": "This district shows medium weather exposure (analytical label from observed weather).",
        })

    # Production context
    if production_context:
        indicators.append({
            "indicator": "Crop production context",
            "severity": "info",
            "detail": production_context,
        })

    # Arrival trend
    if arrival_trend_pct is not None:
        if arrival_trend_pct < -10:
            indicators.append({
                "indicator": "Declining arrivals",
                "severity": "warning",
                "detail": f"Recent arrivals trend shows a {arrival_trend_pct:.1f}% decline.",
            })
        elif arrival_trend_pct > 10:
            indicators.append({
                "indicator": "Increasing arrivals",
                "severity": "info",
                "detail": f"Recent arrivals trend shows a {arrival_trend_pct:.1f}% increase.",
            })

    return indicators


def run_sensitivity(
    quantity: float,
    unit_price: float,
    unit_seed_cost: float = 0.0,
    unit_fertilizer_cost: float = 0.0,
    unit_labor_cost: float = 0.0,
    unit_transport_cost: float = 0.0,
    unit_storage_cost: float = 0.0,
    unit_other_cost: float = 0.0,
    loss_pct: float = 0.0,
) -> pd.DataFrame:
    """Re-run the estimate under price / quantity / cost variations.

    Returns one row per variation with columns:
    variation, quantity, unit_price, revenue, total_cost, margin, margin_pct,
    margin_pct_delta (basis points vs baseline), sellable_quantity,
    break_even_price.
    """
    baseline = estimate_scenario(
        quantity, unit_price, unit_seed_cost, unit_fertilizer_cost,
        unit_labor_cost, unit_transport_cost, unit_storage_cost,
        unit_other_cost, loss_pct
    )
    if baseline["errors"]:
        return pd.DataFrame()

    def _run(pf: float, qf: float, cf: float) -> dict:
        return estimate_scenario(
            quantity * qf,
            unit_price * pf,
            unit_seed_cost * cf,
            unit_fertilizer_cost * cf,
            unit_labor_cost * cf,
            unit_transport_cost * cf,
            unit_storage_cost * cf,
            unit_other_cost * cf,
            loss_pct,
        )

    variations = [
        ("Baseline", 1.0, 1.0, 1.0),
        ("Price +10%", 1.1, 1.0, 1.0),
        ("Price -10%", 0.9, 1.0, 1.0),
        ("Quantity +10%", 1.0, 1.1, 1.0),
        ("Quantity -10%", 1.0, 0.9, 1.0),
        ("Costs +10%", 1.0, 1.0, 1.1),
        ("Costs -10%", 1.0, 1.0, 0.9),
    ]
    rows = []
    for label, pf, qf, cf in variations:
        est = _run(pf, qf, cf)
        rows.append({
            "variation": label,
            "quantity": est["sellable_quantity"],
            "unit_price": unit_price * pf,
            "revenue": est["revenue"],
            "total_cost": est["total_cost"],
            "margin": est["margin"],
            "margin_pct": est["margin_pct"],
            "break_even_price": est["break_even_price"],
            "margin_pct_delta": (est["margin_pct"] - baseline["margin_pct"]) if pd.notna(baseline["margin_pct"]) else np.nan,
        })
    return pd.DataFrame(rows)