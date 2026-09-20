"""
test_india.py

India-migration specific unit tests: active country profile, real geography,
commodity metadata, INR formatting, India data layout, and the honest
market-gating behaviour of the pipeline.

Rules honored:
  * No live APIs -- everything reads committed config/data files or pure logic.
  * No fabricated market values -- the market source must report not_configured.
  * U.S.-specific references (states, USD export columns, 2011 datasets) must not
    leak into the active India profile's contracts.

Run with:
    pytest tests/test_india.py -v
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

import config  # noqa: E402
from config import format_inr, inr_compact, inr_number  # noqa: E402


# --------------------------------------------------------------------------- #
# Active profile
# --------------------------------------------------------------------------- #
def test_active_country_is_india():
    assert config.COUNTRY == "india"
    assert config.GEOGRAPHY_LEVELS == ["country", "state_ut", "district", "mandi_apmc"]


def test_data_layout_points_at_data_india():
    assert config.BRONZE.name == "bronze"
    assert config.DATA_ROOT.name == "india"
    assert config.GOLD.name == "gold"
    assert config.CATALOG.name == "catalog"


def test_market_source_is_not_configured():
    assert config.market_source_status() == "not_configured"
    assert config.market_enabled() is False


def test_regions_config_exists_and_loads():
    regions = config.load_regions()
    assert isinstance(regions, list) and len(regions) > 0
    assert config.REGIONS_CONFIG.exists()


def test_commodities_config_exists_and_loads():
    commodities = config.load_commodities()
    assert isinstance(commodities, list) and len(commodities) > 0
    assert config.COMMODITIES_CONFIG.exists()


def test_pipeline_config_species():
    cfg = json.loads(config.PIPELINE_CONFIG.read_text())
    assert cfg["country"] == "india"
    assert cfg["currency"]["symbol"] == "\u20b9"
    assert cfg["units"]["price_basis"].startswith("per quintal (100 kg)")
    assert cfg["market_source"] == "not_configured"


# --------------------------------------------------------------------------- #
# Real geography integrity (catalog/india_regions.json)
# --------------------------------------------------------------------------- #
def test_region_names_unique():
    regions = config.load_regions()
    names = [r["region_name"] for r in regions]
    assert len(names) == len(set(names)), "region_name must be unique"


@pytest.mark.parametrize("field", ["region_name", "state_ut", "state_code", "district", "mandi_apmc"])
def test_region_fields_complete(field):
    for r in config.load_regions():
        assert str(r.get(field, "")).strip(), f"{field} missing in {r}"


def test_state_codes_are_two_characters():
    for r in config.load_regions():
        assert len(r["state_code"]) == 2


def test_india_coordinates_within_bounds():
    for r in config.load_regions():
        lat, lon = float(r["latitude"]), float(r["longitude"])
        assert 6.0 <= lat <= 37.0, f"latitude out of India bounds: {r['region_name']} {lat}"
        assert 68.0 <= lon <= 98.0, f"longitude out of India bounds: {r['region_name']} {lon}"


def test_first_level_geography_is_state_ut_not_country_of_exports():
    # Sanity check that US-era naming did not leak into the active geography.
    st = [r["state_ut"] for r in config.load_regions()]
    assert "Iowa" not in st and "Texas" not in st
    assert any("Maharashtra" in s for s in st)


# --------------------------------------------------------------------------- #
# Commodity metadata (factual only; no prices)
# --------------------------------------------------------------------------- #
CATEGORIES = {"cereal", "pulse", "oilseed", "commercial", "horticulture", "sugarcane"}


def test_commodities_have_name_and_category():
    for c in config.load_commodities():
        assert c["commodity"].strip()
        assert c.get("category") in CATEGORIES


def test_commodities_are_indian_references():
    names = [c["commodity"] for c in config.load_commodities()]
    assert "Wheat" in names and "Rice (Paddy)" in names
    assert "Sugarcane" in names and "Onion" in names


def test_commodity_metadata_contains_no_price_fields():
    for c in config.load_commodities():
        assert not any(k in c for k in ("price", "cost", "value_musd", "exports"))


# --------------------------------------------------------------------------- #
# INR formatting (pure presentation logic)
# --------------------------------------------------------------------------- #
def test_inr_number_indian_grouping():
    assert inr_number(1234567) == "12,34,567"
    assert inr_number(100000) == "1,00,000"
    assert inr_number(999) == "999"


def test_format_inr_symbol_and_grouping():
    assert format_inr(1234567) == "\u20b912,34,567"


def test_inr_compact_lakh_and_crore():
    assert inr_compact(12500000) == "\u20b91.3 crore"
    assert inr_compact(250000) == "\u20b92.5 lakh"


# --------------------------------------------------------------------------- #
# Demo artifacts sanity for the India deployment contract
# --------------------------------------------------------------------------- #
def test_demo_gold_carries_india_geography():
    df = pd.read_parquet(BASE / "data" / "india" / "demo" / "gold" / "region_daily_features.parquet")
    assert {"state_ut", "district", "mandi_apmc"} <= set(df.columns)
    assert df["state_code"].map(len).eq(2).all()
    assert (df["latitude"] >= 6).all() and (df["latitude"] <= 37).all()
    assert (df["longitude"] >= 68).all() and (df["longitude"] <= 98).all()


def test_demo_ml_status_tracks_market_gate():
    status = json.loads((BASE / "data" / "india" / "demo" / "catalog" / "ml_status.json").read_text())
    assert status["model_status"] == "not_generated"
    assert any("e-NAM" in s or "AGMARKNET" in s for s in status.get("required_upstream", []))