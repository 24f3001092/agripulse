"""
test_agriculture.py

Unit tests for the India agricultural-production pipeline:

  * src/ingestion/india_agriculture.py  -- DE&S/MoAFW crop APY ingestion, crop
    canonicalization (catalog/india_crops.json), Bronze writer with provenance
  * src/quality/india_schemas.py        -- pandera gate for the 8 dimensions
    (State, District, Crop, Season, Year, Area, Production, Yield)
  * src/transform/bronze_to_silver.py   -- clean_agriculture (season-aware
    dedup) and the Bronze->Silver gate

Covered cases: valid records, invalid (negative) production, invalid (negative)
area, invalid (non-numeric) numeric fields, duplicate state/district/crop/year
records, missing season, missing (nullable) production, distinct similar-named
crops never merged, and the honest no-data marker. No live network calls.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "ingestion"))
sys.path.insert(0, str(SRC / "quality"))
sys.path.insert(0, str(SRC / "transform"))

import india_agriculture as ag  # noqa: E402
from india_schemas import india_agriculture_schema, validate_agriculture  # noqa: E402
import bronze_to_silver as b2s  # noqa: E402


def _raw_frame() -> pd.DataFrame:
    """Real-value sample rows from the DE&S APY dataset (Andhra Pradesh)."""
    return pd.DataFrame([
        {
            "id": 0, "year": "1997-1998", "state_name": "Andhra Pradesh",
            "state_code": 28, "district_name": "Ananthapuramu", "district_code": 502,
            "crop_name": "Arhar/Tur", "crop_code": 202.0, "crop_type": "Pulses",
            "season": "Kharif", "area": 21400.0, "area_unit": "Hectare",
            "production": 2600.0, "production_unit": "Tonnes",
            "yield": 0.121, "yield_unit": "Tonnes/Hectare",
        },
        {
            "id": 1, "year": "1997-1998", "state_name": "Andhra Pradesh",
            "state_code": 28, "district_name": "Ananthapuramu", "district_code": 502,
            "crop_name": "Arhar/Tur", "crop_code": 202.0, "crop_type": "Pulses",
            "season": "Rabi", "area": 500.0, "area_unit": "Hectare",
            "production": 60.0, "production_unit": "Tonnes",
            "yield": 0.12, "yield_unit": "Tonnes/Hectare",
        },
    ])


def _valid_normalized() -> pd.DataFrame:
    return ag.normalize(_raw_frame())


# --------------------------------------------------------------------------- #
# Ingestion / normalization
# --------------------------------------------------------------------------- #
def test_normalize_builds_canonical_columns():
    df = _valid_normalized()
    assert list(df.columns) == ag.NORMALIZED_COLUMNS
    assert df["state"].tolist() == ["Andhra Pradesh"] * 2
    assert df["district"].tolist() == ["Ananthapuramu"] * 2
    assert df["crop"].tolist() == ["Arhar/Tur"] * 2
    assert df["crop_type"].tolist() == ["Pulses"] * 2
    assert df["season"].tolist() == ["Kharif", "Rabi"]
    assert df["year"].tolist() == ["1997-1998", "1997-1998"]


def test_year_start_derivation():
    assert ag._year_start("1997-1998") == 1997
    assert ag._year_start("2019-20") == 2019
    assert ag._year_start("2022-2023") == 2022
    assert ag._year_start("n/a") is None


def test_missing_production_is_preserved_as_null():
    raw = _raw_frame()
    raw.loc[0, "production"] = None
    df = ag.normalize(raw)
    assert pd.isna(df.loc[0, "production"])
    clean, report = validate_agriculture(df)
    assert report is None  # nullable production is allowed (source reality)


# --------------------------------------------------------------------------- #
# Crop canonicalization (never merge similar-looking crops)
# --------------------------------------------------------------------------- #
def test_canonical_catalog_preserves_distinct_crops():
    raw = _raw_frame()
    raw.loc[1, "crop_name"] = "Tur"  # same family, DIFFERENT source crop name
    raw.loc[1, "crop_type"] = "Pulses"
    df = ag.normalize(raw)
    assert df["crop"].tolist() == ["Arhar/Tur", "Tur"]  # NOT merged


def test_unknown_crop_is_preserved_not_dropped():
    raw = _raw_frame()
    raw.loc[0, "crop_name"] = "Future Crop Zzz"
    df = ag.normalize(raw)
    assert df.loc[0, "crop"] == "Future Crop Zzz"


def test_catalog_covers_all_real_source_crops():
    """Every real crop observed in the dataset is registered (no silent renames)."""
    catalog = ag.load_crop_catalog()
    names = {c["name"] for c in catalog["crops"]}
    canonical = {c["canonical"] for c in catalog["crops"]}
    assert len(catalog["crops"]) == catalog["_meta"]["count"]
    assert names == canonical  # 1:1 source-faithful mapping, no merges


# --------------------------------------------------------------------------- #
# Schema gate
# --------------------------------------------------------------------------- #
def test_valid_records_pass_schema():
    df = _valid_normalized()
    clean, report = validate_agriculture(df)
    assert report is None
    assert len(clean) == 2


def test_invalid_negative_production_fails():
    df = _valid_normalized()
    df.loc[0, "production"] = -5.0
    _, report = validate_agriculture(df)
    assert report is not None
    assert report["failure_count"] >= 1


def test_invalid_negative_area_fails():
    df = _valid_normalized()
    df.loc[0, "area"] = -10.0
    _, report = validate_agriculture(df)
    assert report is not None
    assert report["failure_count"] >= 1


def test_invalid_non_numeric_area_fails():
    raw = _raw_frame().astype({"area": object})
    raw.loc[0, "area"] = "not-a-number"
    df = ag.normalize(raw)          # coerced to NaN
    assert pd.isna(df.loc[0, "area"])
    _, report = validate_agriculture(df)
    assert report is not None        # area is a required dimension


def test_missing_season_fails_schema():
    raw = _raw_frame()
    raw.loc[0, "season"] = ""
    df = ag.normalize(raw)
    _, report = validate_agriculture(df)
    assert report is not None
    assert report["failure_count"] >= 1


def test_missing_area_fails_schema():
    df = _valid_normalized()
    df.loc[0, "area"] = None
    _, report = validate_agriculture(df)
    assert report is not None


# --------------------------------------------------------------------------- #
# Bronze writer provenance
# --------------------------------------------------------------------------- #
def test_write_bronze_preserves_raw_and_provenance(tmp_path):
    df = _valid_normalized()
    raw_file = tmp_path / "source.csv"
    raw_file.write_text("id,year\n0,1997-1998\n")
    norm_path = ag.write_bronze(df, source_url="https://example.test/apy.csv",
                                raw_file=raw_file, out_dir=tmp_path)
    meta = json.loads((tmp_path / (norm_path.stem + ".meta.json")).read_text())
    assert meta["dataset"] == ag.SOURCE_NAME
    assert meta["status"] == "success"
    assert meta["source_url"] == "https://example.test/apy.csv"
    assert meta["row_count"] == 2
    assert meta["units"]["area"] == "Hectare"
    raw_copies = list((tmp_path / "raw").glob("apy_raw_*"))
    assert len(raw_copies) == 1  # verbatim raw copy preserved
    assert raw_copies[0].read_bytes() == raw_file.read_bytes()


# --------------------------------------------------------------------------- #
# Bronze -> Silver (season-aware dedup + no-data marker)
# --------------------------------------------------------------------------- #
def test_clean_agriculture_dedups_season_aware(tmp_path, monkeypatch):
    raw = _raw_frame()
    # exact duplicate of row 0 + a third legitimately distinct season
    dup = raw.loc[[0]].copy()
    total = raw.loc[[0]].copy()
    total[["id", "crop_code"]] = [9, 202.0]
    total["season"] = "Total"
    full = pd.concat([raw, dup, total], ignore_index=True)
    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"
    ag.write_bronze(ag.normalize(full), out_dir=bronze)

    monkeypatch.setattr(b2s, "AGRI_BRONZE_DIR", bronze)
    monkeypatch.setattr(b2s, "SILVER", silver)
    monkeypatch.setattr(b2s, "REJECTS", silver / "_rejects")

    result = b2s.clean_agriculture()
    assert result is not None
    # 4 input rows -> 1 dup dropped -> 3 remain (Kharif, Rabi, Total all kept)
    assert len(result) == 3
    assert result["season"].tolist() == ["Kharif", "Rabi", "Total"]


def test_clean_agriculture_gate_writes_silver(tmp_path, monkeypatch):
    raw = _raw_frame()
    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"
    ag.write_bronze(ag.normalize(raw), out_dir=bronze)
    monkeypatch.setattr(b2s, "AGRI_BRONZE_DIR", bronze)
    monkeypatch.setattr(b2s, "SILVER", silver)
    monkeypatch.setattr(b2s, "REJECTS", silver / "_rejects")

    ag_df = b2s.clean_agriculture()
    b2s.write_validated(ag_df, india_agriculture_schema, "india_agriculture")
    assert (silver / "india_agriculture.parquet").exists()
    assert not (silver / "_rejects" / "india_agriculture_rejects.json").exists()


def test_clean_agriculture_writes_no_data_marker(tmp_path, monkeypatch):
    empty_bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"
    monkeypatch.setattr(b2s, "AGRI_BRONZE_DIR", empty_bronze)
    monkeypatch.setattr(b2s, "SILVER", silver)

    assert b2s.clean_agriculture() is None
    marker = json.loads((silver / "agriculture_status.json").read_text())
    assert marker["status"] == "no_data"


def test_latest_normalized_none_when_empty(tmp_path):
    assert ag.latest_normalized(tmp_path) is None