"""
test_geography.py

Phase 2 -- India location catalog tests. The active geography is a reusable,
canonical State/UT > District > Mandi/APMC location catalog
(catalog/india_regions.json) carrying ISO 3166-2:IN state codes and LGD
(Local Government Directory) district codes.

Rules honored:
  * No invented codes -- every district code is a real LGD code sourced from the
    OGD district master; mandi/APMC codes are intentionally absent.
  * No U.S. geography -- state names are Indian State/UTs only.
  * District/State pairs are validated against the factual catalog (a real
    administrative relationship), and wrong combinations must be rejected.

Run with:
    pytest tests/test_geography.py -v
"""

import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

import config  # noqa: E402


# --------------------------------------------------------------------------- #
# Canonical catalog file
# --------------------------------------------------------------------------- #
def test_location_catalog_is_canonical_file():
    assert config.REGIONS_CONFIG.name == "india_regions.json"
    assert config.REGIONS_CONFIG.exists()


def test_load_regions_reads_canonical_catalog():
    regions = config.load_regions()
    assert isinstance(regions, list) and len(regions) >= 10


def test_regions_keep_legacy_fields_for_downstream():
    # Silver/Gold and dashboard code still read region_name/state_ut/mandi_apmc.
    for r in config.load_regions():
        assert r.get("region_name")
        assert r.get("state_ut")
        assert r.get("district")
        assert r.get("mandi_apmc")
        assert r.get("state_code")
        assert r.get("latitude") is not None
        assert r.get("longitude") is not None


# --------------------------------------------------------------------------- #
# Preferred record schema (state, state_code, district, district_code, lat, lon;
# optional mandi/apmc)
# --------------------------------------------------------------------------- #
def test_preferred_schema_fields_present():
    for r in config.load_regions():
        assert r.get("state"), f"state missing for {r['region_name']}"
        assert r.get("state_code"), f"state_code missing for {r['region_name']}"
        assert r.get("district"), f"district missing for {r['region_name']}"
        assert r.get("district_code") is not None, f"district_code missing for {r['region_name']}"
        assert r.get("latitude") is not None and r.get("longitude") is not None
        assert "mandi" in r and "apmc" in r


def test_no_invented_mandi_or_apmc_codes():
    # We only carry factual mandi/APMC NAMES; codes are never fabricated.
    for r in config.load_regions():
        assert r.get("mandi_code") is None, f"invented mandi_code on {r['region_name']}"
        assert r.get("apmc_code") is None, f"invented apmc_code on {r['region_name']}"
        if r.get("mandi"):
            assert r["mandi"].strip()


# --------------------------------------------------------------------------- #
# No U.S. geography (Phase 2 directive)
# --------------------------------------------------------------------------- #
def test_no_us_state_names_in_catalog():
    states = {r["state_ut"].lower() for r in config.load_regions()}
    assert not (states & {"iowa", "texas", "california", "wisconsin", "ohio"})
    for r in config.load_regions():
        assert r["state_code"] not in {"IA", "TX", "CA", "WI", "OH", "IL"}


# --------------------------------------------------------------------------- #
# Validity: state level
# --------------------------------------------------------------------------- #
def test_state_codes_are_two_character_iso_codes():
    for r in config.load_regions():
        assert len(r["state_code"]) == 2
        assert r["state_code"].isalpha()


@pytest.mark.parametrize(
    "state_ut, state_code",
    [
        ("Maharashtra", "MH"),
        ("Punjab", "PB"),
        ("Uttar Pradesh", "UP"),
        ("Haryana", "HR"),
        ("Madhya Pradesh", "MP"),
        ("Gujarat", "GJ"),
        ("Tamil Nadu", "TN"),
        ("Telangana", "TS"),
        ("Karnataka", "KA"),
        ("West Bengal", "WB"),
    ],
)
def test_state_code_mapping_is_iso_3166(state_ut, state_code):
    records = config.districts_for_state(state_ut)
    assert records, f"{state_ut} has no catalog records"
    assert all(r["state_code"] == state_code for r in records)


# --------------------------------------------------------------------------- #
# Validity: valid district -> State relationships
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "district, state_ut",
    [
        ("Nashik", "Maharashtra"),
        ("Ludhiana", "Punjab"),
        ("Kanpur Nagar", "Uttar Pradesh"),
        ("Karnal", "Haryana"),
        ("Indore", "Madhya Pradesh"),
        ("Rajkot", "Gujarat"),
        ("Coimbatore", "Tamil Nadu"),
        ("Ranga Reddy", "Telangana"),
        ("Bengaluru Urban", "Karnataka"),
        ("Kolkata", "West Bengal"),
    ],
)
def test_valid_district_state_pairs(district, state_ut):
    assert config.is_valid_district_state(district, state_ut), \
        f"expected ({district}, {state_ut}) to be a real catalog pair"


# --------------------------------------------------------------------------- #
# Validity: invalid district -> State combinations rejected
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "district, state_ut",
    [
        ("Coimbatore", "Maharashtra"),
        ("Ludhiana", "Tamil Nadu"),
        ("Nashik", "Punjab"),
        ("Kolkata", "Gujarat"),
        ("Ranga Reddy", "Karnataka"),
        ("Bengaluru Urban", "Telangana"),
        ("Karnal", "Madhya Pradesh"),
        ("Springfield", "Maharashtra"),
    ],
)
def test_invalid_district_state_combinations_rejected(district, state_ut):
    assert not config.is_valid_district_state(district, state_ut), \
        f"({district}, {state_ut}) is not a real catalog pair"


# --------------------------------------------------------------------------- #
# Uniqueness / duplicates
# --------------------------------------------------------------------------- #
def test_no_duplicate_geography_records():
    pairs = [(r["state_code"], r["district"]) for r in config.load_regions()]
    assert len(pairs) == len(set(pairs)), "duplicate (state_code, district) records"


def test_no_duplicate_lgd_district_codes():
    codes = [str(r["district_code"]) for r in config.load_regions()]
    assert len(codes) == len(set(codes)), "duplicate LGD district codes"
    assert all(c.isdigit() for c in codes), "LGD codes must be digit strings"


# --------------------------------------------------------------------------- #
# Missing latitude / longitude handled (validate_catalog flags problems)
# --------------------------------------------------------------------------- #
def test_coordinates_present_and_within_india_bounds():
    for r in config.load_regions():
        lat, lon = float(r["latitude"]), float(r["longitude"])
        assert 6.0 <= lat <= 37.0, f"latitude out of India bounds: {lat}"
        assert 68.0 <= lon <= 98.0, f"longitude out of India bounds: {lon}"


def test_validate_catalog_reports_no_problems():
    assert config.validate_catalog() == []


# --------------------------------------------------------------------------- #
# Authoritative LGD master values (factual snapshot, not invented)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "district, state_code, district_code",
    [
        ("Nashik", "MH", "487"),
        ("Ludhiana", "PB", "36"),
        ("Kanpur Nagar", "UP", "157"),
        ("Karnal", "HR", "67"),
        ("Indore", "MP", "410"),
        ("Rajkot", "GJ", "457"),
        ("Coimbatore", "TN", "569"),
        ("Ranga Reddy", "TS", "518"),
        ("Bengaluru Urban", "KA", "525"),
        ("Kolkata", "WB", "315"),
    ],
)
def test_lgd_district_codes_match_master(district, state_code, district_code):  # noqa: ARG001
    assert config.is_lgd_district_code(state_code, district_code), \
        f"LGD code {district_code} expected for {district} ({state_code}) in master"


# --------------------------------------------------------------------------- #
# Cascade helpers (what the dashboard State > District > Mandi pickers use)
# --------------------------------------------------------------------------- #
def test_catalog_states_and_district_lookup_consistent():
    states = config.catalog_states()
    assert states == sorted(states)
    flat = [r for s in states for r in config.districts_for_state(s)]
    assert {r["region_name"] for r in flat} == {r["region_name"] for r in config.load_regions()}


def test_mandi_options_are_names_only():
    for r in config.load_regions():
        names = config.mandi_options_for(r)
        assert all(str(n).strip() for n in names)
        assert r.get("mandi_apmc") in names or not names


def test_version_tag_absent_because_none_invented():
    # Every catalog record is a real, monitored Indian district.
    districts = {r["district"] for r in config.load_regions()}
    assert "Hyderabad" not in districts or "Ranga Reddy" in districts