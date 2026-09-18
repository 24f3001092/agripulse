"""
india_market_source.py

Market price / arrival extension point for AgriPulse India.

AUTHORITATIVE SOURCES (see README 'Data sources'):
  1. e-NAM (National Agriculture Market)   - https://enam.gov.in
  2. AGMARKNET                             - https://agmarknet.gov.in
  3. Open Government Data Platform (India) - https://data.gov.in

This module defines the CANONICAL Indian market-row contract these official
sources report (prices per quintal, arrivals in quintals, INR) and the single
place a real loader gets registered. NO market data is generated here, and no
AVERAGE PRICES / ARRIVALS are ever fabricated: until a loader is implemented
and the pipeline config switches market_source to 'configured', all market
outputs are honestly reported as 'not_configured'.

Canonical market grain (one row per commodity x variety x mandi x date):
    commodity               : commodity name (e.g. 'Wheat')
    variety                 : variety (nullable)
    state_ut                : State / Union Territory
    district                : district
    mandi                   : mandi / APMC
    date                    : trade date
    market_price_inr_per_quintal : modal/average price in INR per quintal
    arrivals_quintal        : quantity arrived, in quintals

Implementing a loader:
    1. Write a function returning a DataFrame with the columns above.
    2. Register it in MARKET_LOADERS.
    3. Set Market Source = configured in catalog/pipeline_config.json
    4. Re-run the pipeline so Bronze/Silver/Gold consume it.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path

import pandas as pd

from config import PIPELINE_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("india_market_source")

# Columns contract for e-NAM / AGMARKNET-style market rows.
MARKET_COLUMNS = [
    "commodity",
    "variety",
    "state_ut",
    "district",
    "mandi",
    "date",
    "market_price_inr_per_quintal",
    "arrivals_quintal",
]

# Registry: source key -> loader callable. Empty until a real loader is wired.
MARKET_LOADERS: dict[str, callable] = {}


def market_source_status() -> str:
    """'configured' only when an official loader is registered AND pipeline_config enables it."""
    if not MARKET_LOADERS:
        return "not_configured"
    try:
        cfg = json.loads(PIPELINE_CONFIG.read_text()) if PIPELINE_CONFIG.exists() else {}
        cfg_status = str(cfg.get("market_source", "not_configured"))
    except (OSError, json.JSONDecodeError):
        cfg_status = "not_configured"
    return cfg_status


def load_market_data() -> dict:
    """
    Load the registered official market source.

    Returns {'status': 'not_configured', ...} when no loader is registered
    (never fabricates rows). When configured, returns the validated Bronze
    DataFrame. This is the single entry point used by bronze_to_silver.
    """
    status = market_source_status()
    if status != "configured":
        logger.warning(
            "Market data source is NOT configured (e-NAM/AGMARKNET loader pending). "
            "No market prices or arrivals will be produced -- nothing is fabricated."
        )
        return {
            "status": "not_configured",
            "reason": ("No e-NAM / AGMARKNET loader registered yet. "
                       "Register a loader in india_market_source.MARKET_LOADERS and set "
                       "market_source=configured in catalog/pipeline_config.json."),
            "df": None,
        }
    loader = next(iter(MARKET_LOADERS.values()))
    frame = loader()
    missing = [c for c in MARKET_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Registered market loader returned a frame missing columns: {missing}")
    return {"status": "configured", "df": frame, "reason": ""}


# Backwards-compatible alias so the pipeline stage can import one name.
def load_market_or_status() -> dict:
    return load_market_data()