"""
india_mandi.py

India Mandi (APMC market) price & arrival intelligence -- ingestion (Bronze).

Official data source
--------------------
AGMARKNET (Agricultural Marketing Information Network), Directorate of
Marketing & Inspection, Ministry of Agriculture & Farmers Welfare -- daily
wholesale price and arrival reports from Agricultural Produce Market
Committees (APMCs) across India (3,231 markets).

The data is exposed as a *structured official resource* on the India Data
Portal (GODL / Open Data Commons Attribution License) CKAN, which mirrors
agmarknet.gov.in. This module consumes the CKAN datastore_search API (a real,
documented, structured endpoint) -- no page scraping is performed.

      resource : APMC Arrivals And Prices
      dataset  : https://ckan.indiadataportal.com/dataset/1749d812-35b4-4635-ac29-dd3116ed31b0
      resource : .../resource/13f9964a-9398-4d49-a7b8-2f57c2396c48
      full CSV : .../resource/13f9964a-9398-4d49-a7b8-2f57c2396c48/download/apmc-arrivals-and-prices.csv
      OGD page : https://www.data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi

e-NAM
-----
The e-NAM portal (enam.gov.in) presents live prices through its web dashboard.
It does not publish a public, documented REST API for third parties, and its
dashboard tables are loaded client-side. This module therefore does NOT invent
e-NAM endpoints and does NOT scrape e-NAM pages; the AGMARKNET structured
datastore (the "AGMARKNET information exposed through official systems"
preferred by the product brief) is used instead. A customer-supplied
authoritative endpoint can be wired via AGRIPULSE_MANDI_API_URL.

Honesty model
-------------
  * Only fields actually available in the source are retained. AGMARKNET does
    not report a traded quantity, so traded_quantity is NEVER synthesized.
  * Source units are preserved (price_unit, arrival_units).
  * Missing values in the source stay null -- nothing is fabricated.
  * fetch_mandi_prices() returns the LATEST AVAILABLE date in the data
    (currently frozen at the AGMARKNET/IDP publication date), labelled with
    is_live=False -- it is never presented as live real-time data.

Canonical Bronze columns
------------------------
date, state (state_name), state_code, district (district_name), district_code,
apmc (market_center_name), market_center_code, commodity (commodity_name),
commodity_id, variety, grade, min_price, modal_price, max_price, price_unit,
arrival_quantity, arrival_units
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("india_mandi")

# --------------------------------------------------------------------------- #
# Source metadata
# --------------------------------------------------------------------------- #
SOURCE_NAME = "agmarknet-daily-apmc"
SOURCE_LABEL = (
    "AGMARKNET (Agricultural Marketing Information Network), Directorate of "
    "Marketing & Inspection, Ministry of Agriculture & Farmers Welfare, GoI -- "
    "daily APMC market price & arrival reports"
)
SOURCE_LICENSE = "Government Open Data License (GODL-India) / Open Data Commons Attribution"
OGD_CATALOG_URL = "https://www.data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi"
IDP_CKAN_BASE = os.environ.get("AGRIPULSE_MANDI_API_URL", "https://ckan.indiadataportal.com")
IDP_DATASET_ID = "1749d812-35b4-4635-ac29-dd3116ed31b0"
IDP_RESOURCE_ID = "13f9964a-9398-4d49-a7b8-2f57c2396c48"
FULL_CSV_URL = (
    f"{IDP_CKAN_BASE}/dataset/{IDP_DATASET_ID}/resource/{IDP_RESOURCE_ID}/"
    f"download/apmc-arrivals-and-prices.csv"
)
SOURCE_UNITS = {
    "price": "INR (Rs./Quintal, Rs./Unit or Rs./Bundle as reported in price_unit)",
    "arrival": "arrival_quantity as reported in arrival_units (Metric Tonnes | Bundle | Nos)",
}

ENV_FILE = "AGRIPULSE_MANDI_FILE"
ENV_API_URL = "AGRIPULSE_MANDI_API_URL"

# API paging: CKAN datastore_search cap observed on this portal.
PAGE_SIZE = 10_000
DEFAULT_ROW_LIMIT = 250_000

MANDI_BRONZE_DIR = config.BRONZE / "india_mandi"

RAW_COLUMNS = [
    "date", "state_name", "state_code", "district_name", "district_code",
    "market_center_name", "market_center_code", "category", "year_of_establishment",
    "latitude", "longitude", "commodity_name", "commodity_id", "variety",
    "grade", "arrival_quantity", "arrival_units", "min_price", "max_price",
    "modal_price", "price_unit",
]
REQUIRED_SOURCE_COLUMNS = [
    "date", "state_name", "district_name", "market_center_name",
    "commodity_name", "min_price", "modal_price", "max_price",
]

# Canonical Bronze/Silver columns (only fields actually available in the source;
# AGMARKNET has no traded quantity).
NORMALIZED_COLUMNS = [
    "date", "state", "state_code", "district", "district_code",
    "apmc", "market_center_code", "commodity", "commodity_id",
    "variety", "grade", "min_price", "modal_price", "max_price",
    "price_unit", "arrival_quantity", "arrival_units",
]
# Business key for a single reported trade day at one market for one
# commodity/variety (deduplication identity).
DEDUP_KEY = ["date", "state", "district", "apmc", "commodity", "variety"]


# --------------------------------------------------------------------------- #
# Typed errors
# --------------------------------------------------------------------------- #
class MandiError(Exception):
    """Base error for the India mandi market ingestion."""


class MandiConnectionError(MandiError):
    """Source unreachable."""


class MandiTimeoutError(MandiConnectionError):
    """Source timed out."""


class MandiHttpError(MandiError):
    """Source returned a non-success status."""


class MandiInvalidSource(MandiError):
    """Local source file missing, empty, or missing required columns."""


class MandiEmptyResponse(MandiError):
    """Source returned no rows for the requested filters."""


# --------------------------------------------------------------------------- #
# IDP CKAN datastore client (structured endpoint -- no scraping)
# --------------------------------------------------------------------------- #
class _DatastoreClient:
    """Tiny CKAN datastore_search reader. `session` is injectable for tests."""

    def __init__(self, base_url: str = IDP_CKAN_BASE, resource_id: str = IDP_RESOURCE_ID,
                 session=None, timeout: tuple = (10, 120)):
        self.base_url = base_url.rstrip("/")
        self.resource_id = resource_id
        self.session = session
        self.timeout = timeout

    def _request(self, base: str, params: dict) -> dict:
        import requests

        use = self.session or requests
        try:
            resp = use.get(base, params=params, timeout=self.timeout)
        except requests.exceptions.Timeout as exc:
            raise MandiTimeoutError(f"Timeout querying {base}") from exc
        except requests.exceptions.RequestException as exc:
            raise MandiConnectionError(f"Connection error querying {base}: {exc}") from exc
        if resp.status_code != 200:
            raise MandiHttpError(f"Datastore returned HTTP {resp.status_code}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise MandiInvalidSource(f"Non-JSON response from {base}") from exc
        if "error" in payload:
            raise MandiInvalidSource(f"Datastore error: {str(payload['error'])[:300]}")
        result = payload.get("result") or {}
        records = result.get("records") or []
        if not records:
            raise MandiEmptyResponse(f"No rows for query {params}")
        return result

    def latest_date(self, filters: dict | None = None) -> str:
        result = self._request(f"{self.base_url}/api/3/action/datastore_search", {
            "resource_id": self.resource_id,
            "limit": 1,
            "sort": "date desc",
            **({"filters": json.dumps(filters)} if filters else {}),
        })
        return str(result["records"][0]["date"])

    def page(self, offset: int, limit: int, filters: dict | None = None) -> dict:
        params: dict = {
            "resource_id": self.resource_id,
            "limit": limit,
            "offset": offset,
        }
        if filters:
            params["filters"] = json.dumps(filters)
        return self._request(f"{self.base_url}/api/3/action/datastore_search", params)


# --------------------------------------------------------------------------- #
# Fetching: current (latest available) and historical
# --------------------------------------------------------------------------- #
def _build_filters(state=None, district=None, commodity=None, latest_date=None) -> dict:
    filters = {}
    if state:
        filters["state_name"] = state
    if district:
        filters["district_name"] = district
    if commodity:
        filters["commodity_name"] = commodity
    if latest_date:
        filters["date"] = latest_date
    return filters


def fetch_mandi_prices(state=None, district=None, commodity=None,
                       session=None, timeout: tuple = (10, 120),
                       row_limit: int = 5_000):
    """Latest available price/arrival records for the optional filters.

    Returns (DataFrame, metadata). The data is labelled is_live=False: it is the
    most recent date published in the AGMARKNET/IDP structured dataset, not a
    real-time e-NAM feed (see module docstring).
    """
    client = _DatastoreClient(session=session, timeout=timeout)
    base_filters = _build_filters(state=state, district=district, commodity=commodity)
    latest = client.latest_date(base_filters or None)
    result = client.page(0, limit=min(row_limit, PAGE_SIZE),
                         filters=_build_filters(state=state, district=district,
                                                commodity=commodity, latest_date=latest))
    df = pd.DataFrame(result["records"])
    return df, {
        "source": SOURCE_NAME,
        "is_live": False,
        "latest_date": latest,
        "fetched_rows": int(len(df)),
        "total_available": int(result.get("total", len(df))),
        "filters": base_filters,
    }


def fetch_historical_mandi_prices(state=None, district=None, commodity=None,
                                  start_date=None, end_date=None,
                                  session=None, timeout: tuple = (10, 120),
                                  row_limit: int = DEFAULT_ROW_LIMIT):
    """Pull historical AGMARKNET price/arrival records (paged, bounded).

    Filters are exact-match on state/district/commodity; `start_date`/`end_date`
    are honoured client-side so any CKAN instance works. Returns
    (DataFrame, metadata). Raises MandiEmptyResponse when no rows match.
    """
    client = _DatastoreClient(session=session, timeout=timeout)
    base_filters = _build_filters(state=state, district=district, commodity=commodity)
    collected: list[dict] = []
    total = None
    for offset in range(0, row_limit, PAGE_SIZE):
        want = min(PAGE_SIZE, row_limit - len(collected))
        if want <= 0:
            break
        result = client.page(offset, limit=want, filters=base_filters or None)
        total = int(result.get("total", 0))
        collected.extend(result["records"])
        logger.info("Mandi: fetched %s / %s rows (offset=%s)", len(collected), total, offset)
        if len(result["records"]) < want:
            break
    if not collected:
        raise MandiEmptyResponse("No AGMARKNET rows matched the requested filters")

    df = pd.DataFrame(collected)
    if start_date or end_date:
        df["_d"] = pd.to_datetime(df["date"], errors="coerce")
        if start_date:
            df = df[df["_d"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["_d"] <= pd.Timestamp(end_date)]
        df = df.drop(columns=["_d"]).reset_index(drop=True)

    return df, {
        "source": SOURCE_NAME,
        "is_live": False,
        "resource_id": IDP_RESOURCE_ID,
        "filters": base_filters,
        "start_date": start_date,
        "end_date": end_date,
        "fetched_rows": int(len(df)),
        "total_available": int(total or len(df)),
    }


# --------------------------------------------------------------------------- #
# Documented import/CSV interface (fallback when API access is unavailable)
# --------------------------------------------------------------------------- #
def load_mandi_csv(path: str | Path) -> pd.DataFrame:
    """Read a local AGMARKNET/IDP-shaped CSV (documented import interface).

    Column spellings from the official CSV and the CKAN datastore are both
    accepted. Required: date, state_name, district_name, market_center_name,
    commodity_name, min_price, modal_price, max_price (arrival optional).
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise MandiInvalidSource(f"Mandi CSV missing: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise MandiInvalidSource(f"Unreadable mandi CSV {csv_path}: {exc}") from exc
    if df.empty:
        raise MandiInvalidSource(f"Mandi CSV is empty: {csv_path}")
    missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise MandiInvalidSource(
            f"Mandi CSV missing required columns {missing} (have {sorted(df.columns)})"
        )
    return df


def resolve_source() -> tuple[str | None, Path | None]:
    """(api_url, local_file) per environment preference."""
    local = os.environ.get(ENV_FILE)
    if local and Path(local).exists():
        return None, Path(local)
    url = os.environ.get(ENV_API_URL) or IDP_CKAN_BASE
    return url, None


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical Bronze mandi frame from the raw source frame."""
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["date"], errors="coerce")

    for src, dst in [
        ("state_name", "state"), ("district_name", "district"),
        ("market_center_name", "apmc"), ("commodity_name", "commodity"),
        ("state_code", "state_code"), ("district_code", "district_code"),
        ("market_center_code", "market_center_code"), ("commodity_id", "commodity_id"),
    ]:
        out[dst] = df[src].fillna("").astype(str).str.strip() if src in df.columns else ""

    out["variety"] = df["variety"].fillna("").astype(str).str.strip() if "variety" in df.columns else ""
    out["grade"] = df["grade"].fillna("").astype(str).str.strip() if "grade" in df.columns else ""

    for col in ["min_price", "modal_price", "max_price"]:
        out[col] = pd.to_numeric(df.get(col), errors="coerce") if col in df.columns else None
    out["price_unit"] = (df["price_unit"].fillna("").astype(str).str.strip()
                         if "price_unit" in df.columns else "")
    out["arrival_quantity"] = (pd.to_numeric(df.get("arrival_quantity"), errors="coerce")
                               if "arrival_quantity" in df.columns else None)
    out["arrival_units"] = (df["arrival_units"].fillna("").astype(str).str.strip()
                            if "arrival_units" in df.columns else "")

    if out["date"].isna().any():
        logger.warning("Mandi: dropped %s rows with unparseable date", int(out["date"].isna().sum()))
        out = out[out["date"].notna()]
    return out[NORMALIZED_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Bronze writer with provenance
# --------------------------------------------------------------------------- #
def write_bronze(df: pd.DataFrame, meta_extra: dict | None = None,
                 raw_df: pd.DataFrame | None = None, out_dir: Path | None = None) -> Path:
    """Write raw+normalized Bronze artifacts with full provenance."""
    out_dir = Path(out_dir or MANDI_BRONZE_DIR)
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = None
    if raw_df is not None and not raw_df.empty:
        raw_path = raw_dir / f"agmarknet_raw_{now}.csv"
        raw_df.to_csv(raw_path, index=False)

    norm_path = out_dir / f"mandi_normalized_{now}.parquet"
    df.to_parquet(norm_path, index=False, coerce_timestamps="us",
                  allow_truncated_timestamps=True)

    extra = dict(meta_extra or {})
    period = extra.get("period")
    latest = extra.get("latest_date")
    meta = {
        "dataset": SOURCE_NAME,
        "status": "success",
        "source": SOURCE_LABEL,
        "license": SOURCE_LICENSE,
        "source_kind": extra.get("source_kind", "imported-csv"),
        "source_url": extra.get("source_url", FULL_CSV_URL),
        "resource_id": extra.get("resource_id"),
        "filters": extra.get("filters"),
        "raw_file": raw_path.name if raw_path else None,
        "normalized_file": norm_path.name,
        "retrieved_at": extra.get("retrieved_at", now),
        "row_count": int(len(df)),
        "columns": NORMALIZED_COLUMNS,
        "units": SOURCE_UNITS,
        "period": period,
        "latest_date": latest,
        "notes": [
            "traded_quantity is NOT available in the AGMARKNET source and is never synthesized",
            "e-NAM live prices are not used: e-NAM publishes no public documented REST API and its portal is not scraped",
            "fetch_mandi_prices returns the latest AVAILABLE date (is_live=False), not real-time data",
            "missing source values are preserved as null (never fabricated)",
        ],
    }
    (out_dir / f"mandi_normalized_{now}.meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Mandi Bronze: %s rows -> %s", len(df), norm_path)
    return norm_path


def latest_normalized(out_dir: Path | None = None) -> Path | None:
    """Newest normalized Bronze parquet (or None when absent)."""
    files = sorted((out_dir or MANDI_BRONZE_DIR).glob("mandi_normalized_*.parquet"))
    return files[-1] if files else None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Iterable[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Ingest AGMARKNET APMC price/arrival data (Bronze).")
    parser.add_argument("--file", default=None, help="Local AGMARKNET/IDP CSV (or AGRIPULSE_MANDI_FILE)")
    parser.add_argument("--state", default=None, help="Filter: state_name")
    parser.add_argument("--district", default=None, help="Filter: district_name")
    parser.add_argument("--commodity", default=None, help="Filter: commodity_name")
    parser.add_argument("--start-date", default=None, help="ISO start date (inclusive)")
    parser.add_argument("--end-date", default=None, help="ISO end date (inclusive)")
    parser.add_argument("--latest", action="store_true", help="Fetch only the latest available date")
    parser.add_argument("--limit", type=int, default=DEFAULT_ROW_LIMIT, help="Max rows")
    parser.add_argument("--out", default=None, help="Override Bronze output directory")
    args = parser.parse_args(argv)

    raw_meta: dict = {}
    if args.file:
        raw = load_mandi_csv(args.file)
        fetch_df = raw
        raw_meta = {"source_kind": "imported-csv", "source_url": args.file}
        logger.info("Mandi: read %s rows from local file %s", len(raw), args.file)
    else:
        if args.latest:
            fetch_df, m = fetch_mandi_prices(state=args.state, district=args.district,
                                             commodity=args.commodity, row_limit=args.limit)
        else:
            fetch_df, m = fetch_historical_mandi_prices(
                state=args.state, district=args.district, commodity=args.commodity,
                start_date=args.start_date, end_date=args.end_date, row_limit=args.limit)
        raw_meta = {
            "source_kind": "live-structured-api",
            "source_url": f"{IDP_CKAN_BASE}/dataset/{IDP_DATASET_ID}/resource/{IDP_RESOURCE_ID}",
            "resource_id": IDP_RESOURCE_ID,
            "is_live": m.get("is_live"),
            "latest_date": m.get("latest_date"),
        }

    norm = normalize(fetch_df)
    period = (norm["date"].min(), norm["date"].max()) if len(norm) else (None, None)
    raw_meta.update({
        "filters": {"state": args.state, "district": args.district, "commodity": args.commodity},
        "period": [p.strftime("%Y-%m-%d") if hasattr(p, "strftime") else p for p in period],
    })
    write_bronze(norm, meta_extra=raw_meta, raw_df=fetch_df,
                 out_dir=Path(args.out) if args.out else None)
    logger.info("Done: %s rows (%s..%s)", len(norm), period[0] or "-", period[1] or "-")


if __name__ == "__main__":
    main()