"""Extension point for future agricultural & market datasets.

The current pipeline uses a static 2011 US agricultural export cross-section
(data/bronze/us_ag_exports_raw.csv) as a market signal. That gives export value
per state/crop but NO time series for production, yield, acreage, or price.

This module defines the target interface for richer agricultural data — the
schema the Gold/full pipeline should ultimately expose — plus a documented,
runnable extension point. No data is invented here; nothing is loaded until a
concrete source is implemented.

Implementing a real source (e.g. a USDA NASS QuickStats public API reader):

    1. Write a loader that returns a DataFrame with the columns below.
    2. Register it in ``load_agricultural_dataset`` behind a ``source`` key.
    3. Re-run the pipeline so the new source flows through Bronze/Silver/Gold.

Expected canonical agricultural grain (one row per region per crop per period):
    region   : string region identifier (joins to geo.region_name)
    crop     : string crop identifier (e.g. corn)
    date     : date or year-end anchor for the observation
    production: float production volume
    yield    : float yield per unit area
    acreage  : float area planted/harvested
    price    : float unit price (same units as production)
    exports  : float export value (same units as pipeline export figures)
"""

from __future__ import annotations

import pandas as pd

EXPECTED_COLUMNS = [
    "region",
    "crop",
    "date",
    "production",
    "yield",
    "acreage",
    "price",
    "exports",
]


def load_agricultural_dataset(source: str | None = None, **kwargs) -> pd.DataFrame:
    """Load an agricultural/market dataset through the extension point.

    A concrete implementation must return a DataFrame matching
    EXPECTED_COLUMNS. The default implementation is intentionally not
    implemented: connect an official public source (e.g. USDA NASS QuickStats)
    here when the pipeline is expanded beyond the static export cross-section.
    """
    raise NotImplementedError(
        "No agricultural dataset source is configured yet. "
        "Implement a loader returning EXPECTED_COLUMNS and register it here; see module docstring."
    )