"""Explainable insight generation for AgriPulse India — pure and deterministic.

Every insight is derived ONLY from observed Gold-layer data (plus the analytical
segments/ml-status artifacts produced by the ML stages). Each card exposes:

    WHAT           the indicator read off the data
    WHY            the actual numbers behind it
    DATA PERIOD    the observed window the numbers come from
    SOURCE/METHOD  the artifact and derivation used

Insights are analytical observations about available data — never promises
about future yield, price, revenue or profit. No value is hard-coded: all
benchmarks (e.g. tercile thresholds) are computed from the observed cohort.

Market-backed insights (commodity mix / price signal) appear ONLY when a real
e-NAM / AGMARKNET source is configured; otherwise they are absent rather than
made up.

Run with:
    pytest tests/test_products.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _tercile_label(value: float, series: pd.Series, low: float = 0.333, high: float = 0.667) -> str:
    """Classify a value against tercile thresholds of the observed cohort."""
    arr = np.asarray(series.to_numpy(dtype=float), dtype=float)
    if len(arr) == 0:
        return "n/a"
    lo = float(np.quantile(arr, low))
    hi = float(np.quantile(arr, high))
    if value < lo:
        return "Low"
    if value > hi:
        return "High"
    return "Moderate"


def _level_of(levels: list[str]) -> str:
    """Combine 'Low'/'Moderate'/'High' levels, keeping the most attention-worthy."""
    if "High" in levels:
        return "High"
    if "Low" in levels:
        return "Low"
    return "Moderate"


def _fmt_period(start: pd.Timestamp, end: pd.Timestamp) -> str:
    d0, d1 = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    if d0 == d1:
        return d0.isoformat()
    return f"{d0.isoformat()} → {d1.isoformat()}"


def _market_commodity_cols(daily: pd.DataFrame) -> list[str]:
    """Market commodity price columns (INR per quintal) present in the Gold frame."""
    return [c for c in daily.columns if c.endswith("_inr_per_quintal")]


def build_region_insights(
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    segments: pd.DataFrame,
    forecast: pd.DataFrame,
    region: str,
) -> list[dict]:
    """Return a list of insight cards for one district.

    Each card: {title, what, why, period, source_method}. Safe to call with a
    subset of frames — missing frames simply remove their insight. District
    counts and cohort references come from the actual observed data.
    """
    region_daily = daily[daily["region_name"] == region]
    if region_daily.empty:
        return []

    insights: list[dict] = []

    # ---- Weather exposure --------------------------------------------------
    if "temp_avg_c" in daily.columns and "precipitation_mm" in daily.columns:
        cohort_temps = daily.groupby("region_name", observed=True)["temp_avg_c"].mean()
        cohort_precip = daily.groupby("region_name", observed=True)["precipitation_mm"].sum()
        region_temp = float(region_daily["temp_avg_c"].mean())
        region_precip = float(region_daily["precipitation_mm"].sum())
        temp_level = _tercile_label(region_temp, cohort_temps)
        precip_level = _tercile_label(region_precip, cohort_precip)
        overall = _level_of([temp_level, precip_level])
        insights.append({
            "title": f"Weather exposure: {overall}",
            "what": (
                f"{region} recorded {region_temp:.1f}°C average temperature and "
                f"{region_precip:.0f} mm total precipitation over the window."
            ),
            "why": (
                f"Compared with the {len(cohort_precip)} monitored districts "
                f"(cohort terciles), temperature is {temp_level} and precipitation "
                f"is {precip_level}, giving an overall {overall} weather-exposure "
                f"signal. Levels are relative to the observed cohort only."
            ),
            "period": _fmt_period(region_daily["date"].min(), region_daily["date"].max()),
            "source_method": "Computed from data/india/gold/region_daily_features (mean temperature, summed precipitation).",
        })

    # ---- Commodity market mix (e-NAM / AGMARKNET; only when configured) ------
    commodity_cols = _market_commodity_cols(region_daily)
    if commodity_cols:
        values = [float(region_daily[c].max()) for c in commodity_cols]
        total = sum(values)
        order = sorted(range(len(commodity_cols)), key=lambda i: values[i], reverse=True)
        best = commodity_cols[order[0]]
        best_name = best.replace("_inr_per_quintal", "").replace("_", " ").title()
        best_share = values[order[0]] / total if total > 0 else 0.0
        dominant = best_name if best_share >= 0.40 else "diversified"
        comm_summary = ", ".join(
            f"{commodity_cols[i].replace('_inr_per_quintal', '').replace('_', ' ').title()} ₹{round(values[i]):,}" for i in order
        )
        insights.append({
            "title": f"Commodity mix: {best_name}" if dominant != "diversified" else "Commodity mix: diversified",
            "what": (
                f"Highest observed price signal in {region} is {best_name} at ₹{values[order[0]]:,.0f}/quintal "
                f"({best_share * 100:.0f}% of the observed basket)."
            ),
            "why": f"Observed commodity prices (INR/quintal): {comm_summary}.",
            "period": _fmt_period(region_daily["date"].min(), region_daily["date"].max()),
            "source_method": "Computed from data/india/gold/region_daily_features market columns (official e-NAM/AGMARKNET source when configured).",
        })

    # ---- Weather-exposure segment ------------------------------------------
    seg_row = (
        segments[segments["region_name"] == region]
        if isinstance(segments, pd.DataFrame) and "region_name" in segments.columns
        else pd.DataFrame()
    )
    if not seg_row.empty:
        seg = seg_row.iloc[0]
        insights.append({
            "title": f"Weather-exposure tier: {seg.get('exposure_tier', 'n/a')}",
            "what": (
                f"{region} is grouped as {seg.get('segment', 'n/a')} "
                f"(exposure tier {seg.get('exposure_tier', 'n/a')}, dominant driver "
                f"{seg.get('dominant_factor', 'n/a')})."
            ),
            "why": str(seg.get("segment_reason", "See segment_reason.")),
            "period": "Observed weather window (see reason for tercile thresholds).",
            "source_method": "Read from data/india/gold/region_segments.parquet (analytical, rule-based; not a validated agronomic classification).",
        })

    # ---- Market price model signal (only when a real market feed exists) -----
    fc_row = (
        forecast[forecast["region_name"] == region]
        if isinstance(forecast, pd.DataFrame) and "region_name" in forecast.columns
        else pd.DataFrame()
    )
    if not fc_row.empty and "predicted_signal_inr_per_quintal" in fc_row.columns:
        fc = fc_row.iloc[0]
        actual = float(fc.get("actual_signal_inr_per_quintal", float("nan")))
        predicted = float(fc.get("predicted_signal_inr_per_quintal", float("nan")))
        insights.append({
            "title": "Model signal: prototype (in-sample)",
            "what": (
                f"Prototype model estimates the market-price signal for {region} on weather "
                f"features at ₹{predicted:,.0f}/quintal vs the observed ₹{actual:,.0f}/quintal."
            ),
            "why": "Linear-regression prototype; metrics are in-sample and must not be read as out-of-sample accuracy.",
            "period": _fmt_period(region_daily["date"].min(), region_daily["date"].max()),
            "source_method": "Read from data/india/gold/forecast_results.parquet + data/india/catalog/model_report.json.",
        })

    return insights