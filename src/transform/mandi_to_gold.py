"""
mandi_to_gold.py

Gold-layer market-intelligence summary for India mandi (APMC) prices.

Reads the Silver table data/india/silver/india_mandi.parquet and writes
data/india/gold/india_market_summary/ containing:

  * india_market_summary.parquet -- per market/commodity/variety aggregates:
      latest modal price + latest min/max, latest arrival,
      7-day average modal price, price change vs 7-day average,
      arrival change vs 7-day average, days of history, and state-level
      latest-modal benchmark for comparison.
  * state_market_summary.parquet  -- per state latest-modal benchmark.
  * manifest.json                 -- provenance, period, metrics and sources.

Honesty rules
-------------
  * Only metrics supported by the available history are computed. A single-day
    history yields latest prices/arrival but NO change metrics (null).
  * Prices are compared only within the same price_unit, arrivals only within
    the same arrival_units (AGMARKNET reports Rs./Quintal, Rs./Unit,
    Rs./Bundle; Metric Tonnes, Bundle, Nos) -- never across incompatible units.
  * When the Silver mandi table is absent, a no_data manifest is written and
    the stage exits 0 (graceful skip).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("mandi_to_gold")

SILVER_MANDI = config.SILVER / "india_mandi.parquet"
GOLD_SUMMARY_DIR = config.GOLD / "india_market_summary"

GROUP_KEY = ["state", "district", "apmc", "commodity", "variety", "price_unit"]


def _winsorised_change(latest: float, baseline: float) -> float | None:
    if baseline is None or pd.isna(baseline) or baseline == 0 or pd.isna(latest):
        return None
    return (latest - baseline) / baseline * 100.0


def build_market_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per market/commodity/variety (see module docstring)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.as_unit("ns")
    df = df.sort_values(["state", "district", "apmc", "commodity", "variety", "date"])

    rows = []
    for key, grp in df.groupby(GROUP_KEY, observed=True):
        k = dict(zip(GROUP_KEY, key))
        last_date = grp["date"].max()
        latest = grp.loc[grp["date"] == last_date].iloc[0]
        n_days = grp["date"].dt.normalize().nunique()
        latest_arrival_units = str(latest.get("arrival_units") or "") if len(grp) else ""

        window = grp[(last_date - grp["date"]).dt.days <= 6]
        n_window_days = int(window["date"].dt.normalize().nunique())
        week_modal = window["modal_price"].dropna()
        week_arrival = window["arrival_quantity"].dropna()

        latest_modal = float(latest["modal_price"]) if pd.notna(latest["modal_price"]) else None
        latest_arrival = float(latest["arrival_quantity"]) if pd.notna(latest["arrival_quantity"]) else None

        avg_modal_7d = float(week_modal.mean()) if len(week_modal) else None
        avg_arrival_7d = float(week_arrival.mean()) if len(week_arrival) else None
        # A change vs a 7-day window needs an actual window (>=2 trading days);
        # otherwise it is reported as unsupported (None), never as 0% movement.
        has_window = n_window_days >= 2

        rows.append({
            "state": k["state"], "district": k["district"], "apmc": k["apmc"],
            "commodity": k["commodity"], "variety": k["variety"],
            "price_unit": k["price_unit"],
            "latest_date": last_date,
            "days_of_history": int(n_days),
            "latest_min_price": float(latest["min_price"]) if pd.notna(latest["min_price"]) else None,
            "latest_modal_price": latest_modal,
            "latest_max_price": float(latest["max_price"]) if pd.notna(latest["max_price"]) else None,
            "latest_arrival_quantity": latest_arrival,
            "avg_modal_price_7d": avg_modal_7d,
            "avg_arrival_quantity_7d": avg_arrival_7d,
            "modal_price_change_pct_7d": _winsorised_change(latest_modal, avg_modal_7d) if has_window else None,
            "arrival_change_pct_7d": _winsorised_change(latest_arrival, avg_arrival_7d) if has_window else None,
            "arrival_units": latest_arrival_units,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        # latest modal per state on the most recent day (within compatible unit)
        state_latest = (
            out.sort_values("latest_date")
               .groupby(["state", "price_unit"], observed=True)
               .tail(1)[["state", "price_unit", "latest_modal_price"]]
               .rename(columns={"latest_modal_price": "state_latest_modal_price"})
        )
        out = out.merge(state_latest, on=["state", "price_unit"], how="left")
        out["modal_pct_vs_state_latest"] = out.apply(
            lambda r: _winsorised_change(r["latest_modal_price"], r["state_latest_modal_price"]),
            axis=1,
        )
    return out


def write_gold(df: pd.DataFrame, manifest: dict) -> Path:
    GOLD_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(GOLD_SUMMARY_DIR / "india_market_summary.parquet", index=False,
                  coerce_timestamps="us", allow_truncated_timestamps=True)
    if not df.empty and "latest_modal_price" in df.columns:
        state = (
            df.groupby(["state", "price_unit"], observed=True)
              .agg(latest_modal_price=("latest_modal_price", "first"),
                   markets=("apmc", "nunique"),
                   commodities=("commodity", "nunique"))
              .reset_index()
        )
        state.to_parquet(GOLD_SUMMARY_DIR / "state_market_summary.parquet", index=False,
                         coerce_timestamps="us", allow_truncated_timestamps=True)
    (GOLD_SUMMARY_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Mandi Gold summary: %s markets/commodities rows -> %s",
                len(df), GOLD_SUMMARY_DIR)
    return GOLD_SUMMARY_DIR / "manifest.json"


def main():
    if not SILVER_MANDI.exists():
        GOLD_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
        manifest = {
            "status": "no_data",
            "reason": "Silver india_mandi.parquet absent; market summary skipped (run bronze->silver first)",
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
        (GOLD_SUMMARY_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info("Mandi Gold summary skipped (no silver data) -- manifest written")
        return

    df = pd.read_parquet(SILVER_MANDI)
    summary = build_market_summary(df)
    period = (df["date"].min(), df["date"].max())

    manifest = {
        "status": "success",
        "dataset": "agmarknet-daily-apmc",
        "source": ("AGMARKNET (Directorate of Marketing & Inspection, MoAFW) daily APMC "
                   "price & arrival records via the India Data Portal -- GODL-India"),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "period": [period[0].strftime("%Y-%m-%d"), period[1].strftime("%Y-%m-%d")],
        "last_data_date": period[1].strftime("%Y-%m-%d"),
        "silver_rows": int(len(df)),
        "summary_rows": int(len(summary)),
        "n_markets": int(df["apmc"].nunique()) if not df.empty else 0,
        "n_commodities": int(df["commodity"].nunique()) if not df.empty else 0,
        "n_states": int(df["state"].nunique()) if not df.empty else 0,
        "metrics": [
            "latest_modal_price", "latest_min_price", "latest_max_price",
            "latest_arrival_quantity", "avg_modal_price_7d",
            "modal_price_change_pct_7d", "arrival_change_pct_7d",
            "state_latest_modal_price", "modal_pct_vs_state_latest",
        ],
        "metrics_rule": "change metrics are null when the history does not support them "
                        "(fewer than one comparison day or zero baseline)",
        "units_rule": "prices compared only within the same price_unit and arrivals only "
                      "within the same arrival_units",
        "no_traded_quantity": ("AGMARKNET does not report a traded quantity; traded_quantity "
                               "is never synthesized"),
        "disclaimer": ("Observed wholesale prices and arrivals from official AGMARKNET "
                       "records. Analytical signals only -- not investment advice and no "
                       "guaranteed outcomes."),
    }
    write_gold(summary, manifest)
    logger.info("Mandi Gold complete: %s summary rows", len(summary))


if __name__ == "__main__":
    main()