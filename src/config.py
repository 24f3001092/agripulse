"""
config.py

Active AgriPulse country profile and data-layout configuration.

Everything that is country-dependent in the pipeline and the dashboard reads
this module (or the catalog JSON it points at) instead of hard-coding paths,
currency or units. On this branch the active profile is INDIA:

  data layout : data/india/{bronze,silver,gold,demo,catalog}
  geography   : India > State/UT > District > Mandi/APMC   (catalog/india_regions.json)
  commodities : catalog/india/commodities.json             (metadata only, no prices)
  currency    : INR (Rupee, symbol U+20B9), prices per quintal (100 kg)
  market feed : live e-NAM -- extension point, not configured; AGMARKNET daily
                APMC prices/arrivals ARE integrated into the India mandi pipeline

The previous U.S.-focused build and its data were moved to data/existing/current/
without deletion (see data/existing/current/README.md); this module is the
authoritative place the migrated code reads paths from.
"""

from __future__ import annotations

import json
import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# Country / profile
# --------------------------------------------------------------------------- #
def _active_country() -> str:
    env_country = os.environ.get("AGRIPULSE_COUNTRY", "").strip().lower()
    cfg_path = BASE / "catalog" / "pipeline_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            return str(cfg.get("country") or env_country or "india").lower()
        except (OSError, json.JSONDecodeError):
            pass
    return env_country or "india"


COUNTRY: str = _active_country()


# --------------------------------------------------------------------------- #
# Data layout (data/<country>/ on this branch: data/india/)
# --------------------------------------------------------------------------- #
DATA_ROOT = BASE / "data" / COUNTRY
BRONZE = DATA_ROOT / "bronze"
SILVER = DATA_ROOT / "silver"
GOLD = DATA_ROOT / "gold"
CATALOG = DATA_ROOT / "catalog"          # generated reports (health, ml_status, ...)
DEMO = DATA_ROOT / "demo"
REJECTS = SILVER / "_rejects"

CATALOG_CFG = BASE / "catalog"
REGIONS_CONFIG = CATALOG_CFG / f"{COUNTRY}_regions.json"
COMMODITIES_CONFIG = CATALOG_CFG / COUNTRY / "commodities.json"
PIPELINE_CONFIG = CATALOG_CFG / "pipeline_config.json"
DATA_CATALOG_JSON = CATALOG_CFG / COUNTRY / "data_catalog.json"

GEOGRAPHY_LEVELS = ["country", "state_ut", "district", "mandi_apmc"]
DEFAULT_REGION_COUNT = 10  # monitored Indian districts (catalog/india_regions.json)

CURRENCY_CODE = "INR"
CURRENCY_SYMBOL = "\u20b9"          # ₹
PRICE_BASIS_LABEL = "per quintal (100 kg)"


# --------------------------------------------------------------------------- #
# Market data source status (e-NAM / AGMARKNET extension point)
# --------------------------------------------------------------------------- #
def market_source_status() -> str:
    """Return 'configured' when a live market loader is wired, else 'not_configured'.

    No market values are EVER fabricated. Until a real e-NAM / AGMARKNET source
    is implemented (see src/ingestion/india_market_source.py), the pipeline and
    dashboard report 'not_configured' and do not produce market-based outputs.
    """
    if PIPELINE_CONFIG.exists():
        try:
            return str(json.loads(PIPELINE_CONFIG.read_text()).get("market_source", "not_configured"))
        except (OSError, json.JSONDecodeError):
            return "not_configured"
    return "not_configured"


def market_enabled() -> bool:
    return market_source_status() == "configured"


# --------------------------------------------------------------------------- #
# Config loaders (India geography + commodity reference)
# --------------------------------------------------------------------------- #
def load_regions() -> list[dict]:
    """India monitored districts: India > State/UT > District > Mandi/APMC.

    Reads the canonical location catalog (catalog/india_regions.json on this
    branch), which carries ISO 3166-2:IN state codes and LGD district codes.
    Records are normalized to the legacy/pipeline shape so Silver/Gold and the
    dashboard keep working unchanged; the new catalog fields (state, district_code,
    mandi, apmc) are preserved alongside for geo-aware consumers.

    Neither this function nor the catalog invents codes: district codes are LGD
    values sourced from the Open Government Data district master, and mandi/APMC
    codes are omitted until verified against an official e-NAM / AGMARKNET source.
    """
    raw = json.loads(REGIONS_CONFIG.read_text())
    records = raw if isinstance(raw, list) else raw["regions"]
    out: list[dict] = []
    for r in records:
        district = str(r["district"])
        mandi = r.get("mandi") or r.get("mandi_apmc") or ""
        apmc = r.get("apmc") or r.get("mandi_apmc") or ""
        out.append({
            "region_name": str(r.get("region_name") or district),
            "state": str(r.get("state") or r.get("state_ut") or ""),
            "state_ut": str(r.get("state_ut") or r.get("state") or ""),
            "state_code": str(r["state_code"]),
            "district": district,
            "district_code": r.get("district_code"),
            "mandi_apmc": mandi or apmc,
            "mandi": mandi,
            "apmc": apmc,
            "latitude": float(r["latitude"]),
            "longitude": float(r["longitude"]),
        })
    return out


def load_commodities() -> list[dict]:
    """Indian commodity reference metadata (names/categories only -- no prices)."""
    raw = json.loads(COMMODITIES_CONFIG.read_text())
    return raw if isinstance(raw, list) else raw["commodities"]


def region_hierarchy_label(region: dict) -> str:
    """Human label: 'Nashik (Maharashtra) -- APMC Nashik'."""
    parts = [region["region_name"]]
    if region.get("state_ut"):
        parts[0] = f"{region['region_name']} ({region['state_ut']})"
    if region.get("mandi_apmc"):
        parts.append(f"APMC {region['mandi_apmc']}")
    return " -- ".join(parts)


# --------------------------------------------------------------------------- #
# Location catalog helpers (State > District > Mandi cascade + validation)
# --------------------------------------------------------------------------- #
def load_location_catalog() -> list[dict]:
    """Alias for load_regions(): the reusable India location catalog records."""
    return load_regions()


def catalog_states() -> list[str]:
    """Sorted State/UT names present in the monitored location catalog."""
    return sorted({r["state_ut"] for r in load_regions() if r.get("state_ut")})


def districts_for_state(state_ut: str) -> list[dict]:
    """Catalog records belonging to one State/UT (empty list when unknown)."""
    return [r for r in load_regions() if r.get("state_ut") == state_ut]


def mandi_options_for(region: dict) -> list[str]:
    """Deduplicated mandi/APMC names for a record (names only -- never codes)."""
    names = [region.get("mandi"), region.get("apmc"), region.get("mandi_apmc")]
    return sorted({str(n) for n in names if n})


def is_valid_district_state(district: str, state_ut: str) -> bool:
    """True when (district, State/UT) is a real pair in the catalog."""
    return any(r["district"] == district and r["state_ut"] == state_ut
               for r in load_regions())


def is_lgd_district_code(state_code: str, district_code) -> bool:
    """True when a record with this state_code carries this LGD district code."""
    dc = str(district_code)
    return any(r["state_code"] == state_code and str(r.get("district_code")) == dc
               for r in load_regions())


def validate_catalog() -> list[str]:
    """Return catalog data-integrity problems; empty list means the catalog is clean.

    Checks: unique region_name, unique (state_code, district) pairs, unique LGD
    district codes, ISO 2-letter state codes, all-digit district codes, Pakistan/US
    geography leakage, and coordinates inside Indian bounds.
    """
    problems: list[str] = []
    records = load_regions()
    names = [r["region_name"] for r in records]
    if len(names) != len(set(names)):
        problems.append("duplicate region_name values")
    seen_geo: set[tuple[str, str]] = set()
    seen_codes: set[str] = set()
    for r in records:
        geo_key = (r["state_code"], r["district"])
        if geo_key in seen_geo:
            problems.append(f"duplicate geography record: {r['state_code']} / {r['district']}")
        seen_geo.add(geo_key)
        dc = r.get("district_code")
        if dc is None or str(dc) == "" or not str(dc).isdigit():
            problems.append(f"district_code missing/invalid for {r['region_name']}: {dc!r}")
        else:
            if dc in seen_codes:
                problems.append(f"duplicate LGD district_code: {dc}")
            seen_codes.add(dc)
        if len(r["state_code"]) != 2:
            problems.append(f"state_code not 2 chars for {r['region_name']}")
        lat, lon = float(r["latitude"]), float(r["longitude"])
        if not (6.0 <= lat <= 37.0) or not (68.0 <= lon <= 98.0):
            problems.append(f"coordinates outside India for {r['region_name']}")
        st = r.get("state_ut", "")
        if st in {"Iowa", "Texas", "California", "Wisconsin"}:
            problems.append(f"U.S. state leaked into catalog: {st}")
    return problems


# --------------------------------------------------------------------------- #
# INR / Indian number formatting (pure, copy-free, no locale dependency)
# --------------------------------------------------------------------------- #
def inr_number(value, decimals: int = 0) -> str:
    """Format with Indian digit grouping: 1234567 -> '12,34,567'."""
    d = Decimal(str(value)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)
    sign = "-" if d < 0 else ""
    whole = str(int(abs(d)))

    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.append(head[-2:])
            head = head[:-2]
        if head:
            groups.append(head)
        whole = ",".join(reversed(groups)) + "," + tail

    frac = str(abs(d)).split(".", 1)[1] if decimals else ""
    out = whole + (f".{frac}" if decimals else "")
    return f"{sign}{out}"


def format_inr(value, decimals: int = 0) -> str:
    """'1234567' -> '₹12,34,567'."""
    return f"{CURRENCY_SYMBOL}{inr_number(value, decimals)}"


def format_price_per_quintal(value, decimals: int = 0) -> str:
    return f"{format_inr(value, decimals)} / quintal"


def inr_compact(value, decimals: int = 1) -> str:
    """Compact Indian units of account: rupees, lakh (1e5), crore (1e7).

    Uses only the RAW figure -- never news or market claims -- purely a
    presentation helper. Example: 12,500,000 -> '₹1.3 crore'.
    """
    v = float(value)
    av = abs(v)
    if av >= 1e7:
        return f"{format_inr(v / 1e7, decimals)} crore"
    if av >= 1e5:
        return f"{format_inr(v / 1e5, decimals)} lakh"
    return format_inr(v, 0)


# --------------------------------------------------------------------------- #
# Weather window constants (Open-Meteo pull: 30 observed + 7 forecast days)
# --------------------------------------------------------------------------- #
WEATHER_PAST_DAYS = 30
WEATHER_FORECAST_DAYS = 7