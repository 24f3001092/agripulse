"""
test_india_gold.py

Unit tests for the integrated India agricultural Gold layer
(src/transform/india_silver_to_gold.py) and the India artifact quality checks
(src/quality/monitor.py):

  * india_region_features -- region x date grid: weather block outer-joined
    with the observed APMC market block, canonical units only
  * india_crop_summary    -- Location + Crop (+ Date/Year) integrated summary
    (APY + market + weather blocks; exact name-normalized joins only)
  * india_crop_year       -- long-form (state, district, crop, season, year)
  * no_data manifests     -- graceful skip when Silver inputs are absent
  * lineage manifests     -- source / input / output / schema / generated_at
  * monitor India checks  -- duplicates, missing geography, impossible values,
    staleness; clean data passes, planted defects are reported

No live networks and no dependence on the repo's generated data (all inputs are
small synthetic Silver frames written to tmp_path).
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "transform"))
sys.path.insert(0, str(SRC / "quality"))

import india_silver_to_gold as ig  # noqa: E402
import mandi_to_gold  # noqa: E402
import monitor  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic silver frames
# --------------------------------------------------------------------------- #
def _geo() -> pd.DataFrame:
    return pd.DataFrame([
        {"region_name": "Bengaluru", "state_ut": "Karnataka", "state_code": "KA",
         "district": "Bengaluru Urban", "district_code": "525",
         "mandi_apmc": "Yeshwanthpur", "latitude": 13.0, "longitude": 77.5},
        {"region_name": "Karnal", "state_ut": "Haryana", "state_code": "HR",
         "district": "Karnal", "district_code": "67",
         "mandi_apmc": "Karnal", "latitude": 29.6, "longitude": 76.9},
    ])


def _weather() -> pd.DataFrame:
    return pd.DataFrame([
        {"region_name": "Bengaluru", "date": pd.Timestamp("2026-08-01"),
         "temp_max_c": 30.1, "temp_min_c": 20.0, "precipitation_mm": 0.5,
         "humidity_pct": 74.0, "windspeed_max_kmh": 12.0},
        {"region_name": "Bengaluru", "date": pd.Timestamp("2026-08-02"),
         "temp_max_c": 31.0, "temp_min_c": 20.5, "precipitation_mm": 2.0,
         "humidity_pct": 78.0, "windspeed_max_kmh": 14.0},
        {"region_name": "Karnal", "date": pd.Timestamp("2026-08-01"),
         "temp_max_c": 34.0, "temp_min_c": 24.0, "precipitation_mm": 0.0,
         "humidity_pct": 60.0, "windspeed_max_kmh": 9.0},
    ])


def _mandi() -> pd.DataFrame:
    return pd.DataFrame([
        {"date": pd.Timestamp("2026-04-01"), "state": "Karnataka", "state_code": "29",
         "district": "Bengaluru Urban", "district_code": "572", "apmc": "Yeshwanthpur",
         "market_center_code": "1", "commodity": "Onion", "commodity_id": "101",
         "variety": "Local", "grade": "FAQ", "min_price": 1100.0, "modal_price": 1200.0,
         "max_price": 1400.0, "price_unit": "Rs./Quintal",
         "arrival_quantity": 200.0, "arrival_units": "Metric Tonnes"},
        {"date": pd.Timestamp("2026-04-01"), "state": "Karnataka", "state_code": "29",
         "district": "Bengaluru Urban", "district_code": "572", "apmc": "Shivajinagar",
         "market_center_code": "2", "commodity": "Tomato", "commodity_id": "102",
         "variety": "Local", "grade": "FAQ", "min_price": 900.0, "modal_price": 1000.0,
         "max_price": 1200.0, "price_unit": "Rs./Quintal",
         "arrival_quantity": 150.0, "arrival_units": "Metric Tonnes"},
        {"date": pd.Timestamp("2026-04-02"), "state": "Karnataka", "state_code": "29",
         "district": "Bengaluru Urban", "district_code": "572", "apmc": "Yeshwanthpur",
         "market_center_code": "1", "commodity": "Onion", "commodity_id": "101",
         "variety": "Local", "grade": "FAQ", "min_price": 1050.0, "modal_price": 1180.0,
         "max_price": 1350.0, "price_unit": "Rs./Unit",  # excluded: non-canonical unit
         "arrival_quantity": 40.0, "arrival_units": "Bundle"},  # excluded: non-canonical unit
        {"date": pd.Timestamp("2026-04-01"), "state": "Karnataka", "state_code": "29",
         "district": "Bengaluru Urban", "district_code": "572", "apmc": "Yeshwanthpur",
         "market_center_code": "1", "commodity": "Onion", "commodity_id": "101",
         "variety": "Local", "grade": "FAQ", "min_price": 1120.0, "modal_price": 1230.0,
         "max_price": 1450.0, "price_unit": "Rs./Quintal",
         "arrival_quantity": 210.0, "arrival_units": "Metric Tonnes"},
    ])


def _agri() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": 1, "year": "2020-2021", "year_start": 2020, "state": "Karnataka",
         "state_code": 29, "district": "Bengaluru Urban", "district_code": 572,
         "crop": "Onion", "crop_type": "Vegetables", "season": "Kharif",
         "area": 100.0, "production": 2000.0, "yield": 20.0},
        {"id": 2, "year": "2021-2022", "year_start": 2021, "state": "Karnataka",
         "state_code": 29, "district": "Bengaluru Urban", "district_code": 572,
         "crop": "Onion", "crop_type": "Vegetables", "season": "Kharif",
         "area": 120.0, "production": 2400.0, "yield": 20.0},
        {"id": 3, "year": "2020-2021", "year_start": 2020, "state": "Karnataka",
         "state_code": 29, "district": "Bengaluru Urban", "district_code": 572,
         "crop": "Carrot", "crop_type": "Vegetables", "season": "Rabi",
         "area": 50.0, "production": 1000.0, "yield": 20.0},
        {"id": 4, "year": "2021-2022", "year_start": 2021, "state": "Haryana",
         "state_code": 6, "district": "Karnal", "district_code": 67,
         "crop": "Wheat", "crop_type": "Cereals", "season": "Rabi",
         "area": 300.0, "production": None, "yield": None},
    ])


# --------------------------------------------------------------------------- #
# india_region_features
# --------------------------------------------------------------------------- #
def test_region_features_columns_and_honest_outer_join():
    df = ig.build_region_features(_weather(), _mandi(), _geo())
    assert list(df.columns) == ig.REGION_FEATURES_COLUMNS
    # weather window (Aug) + mandi window (Apr) per region, outer join.
    # The 04-02 mandi record is Rs./Unit + Bundle (non-canonical) so it forms no
    # market block and is honestly absent -- the union of canonical blocks is 3.
    assert set(df["region_name"]) == {"Bengaluru", "Karnal"}
    bengaluru = df[df["region_name"] == "Bengaluru"]
    assert len(bengaluru) == 3  # 04-01 (canonical market) + 08-01 + 08-02 (weather)
    assert bengaluru["date"].min().strftime("%Y-%m-%d") == "2026-04-01"
    assert bengaluru["date"].max().strftime("%Y-%m-%d") == "2026-08-02"
    # mandi-only day keeps honest null weather.
    apr_row = bengaluru[bengaluru["date"].dt.strftime("%Y-%m-%d") == "2026-04-01"].iloc[0]
    assert pd.isna(apr_row["temp_max_c"])
    assert apr_row["total_arrival_tonnes"] == 560.0  # Bundle arrival excluded
    aug_row = bengaluru[bengaluru["date"].dt.strftime("%Y-%m-%d") == "2026-08-01"].iloc[0]
    assert pd.isna(aug_row["median_modal_price_rq"])
    assert aug_row["temp_max_c"] == pytest.approx(30.1)


def test_region_features_canonical_units_only():
    df = ig.build_region_features(_weather(), _mandi(), _geo())
    bengaluru = df[df["region_name"] == "Bengaluru"]
    apr01 = bengaluru[bengaluru["date"].dt.strftime("%Y-%m-%d") == "2026-04-01"].iloc[0]
    # Rs./Quintal modal prices on 04-01: 1200, 1000, 1230 -> median 1200.
    assert apr01["median_modal_price_rq"] == pytest.approx(1200.0)
    # Rs./Unit price (1180) excluded.
    assert apr01["n_price_records"] == 3
    # Metric Tonnes arrivals only: 200 + 150 (+210 in a second Onion row) -> 560.
    assert apr01["total_arrival_tonnes"] == pytest.approx(560.0)
    assert apr01["n_arrival_records"] == 3
    assert apr01["district"] == "Bengaluru Urban"
    assert apr01["state_code"] == "KA"


def test_region_features_location_recovery_for_mandi_only_dates():
    # Mandi-only dates fall outside the weather window; the geo fallback must
    # still name the district/state via the geo reference.
    df = ig.build_region_features(_weather(), _mandi(), _geo())
    bengaluru = df[(df["region_name"] == "Bengaluru")
                   & (df["date"].dt.strftime("%Y-%m-%d") == "2026-04-01")].iloc[0]
    assert bengaluru["state"] == "Karnataka"
    assert bengaluru["district"] == "Bengaluru Urban"


def test_region_features_never_invents_traded_quantity():
    df = ig.build_region_features(_weather(), _mandi(), _geo())
    assert "traded_quantity" not in df.columns
    assert "arrival_units" not in df.columns  # arrivals already canonicalized


# --------------------------------------------------------------------------- #
# india_crop_year + india_crop_summary
# --------------------------------------------------------------------------- #
def test_crop_year_long_form_dedupes_and_keeps_identity():
    agri = pd.concat([_agri(), _agri().iloc[[0]]], ignore_index=True)  # an exact dup
    cy = ig.build_crop_year(agri)
    assert len(cy[cy["state"] == "Karnataka"]) == 3  # dup dropped, one row per key
    assert list(cy.columns[:6]) == ["state", "district", "crop", "crop_type", "season", "year_start"]
    assert cy[["state", "district", "crop", "crop_type", "season", "year_start"]].duplicated().sum() == 0


def test_crop_summary_agriculture_block():
    cs = ig.build_crop_summary(_agri(), _mandi(), ig.build_region_features(_weather(), _mandi(), _geo()))
    onion = cs[(cs["crop"] == "Onion") & (cs["district"] == "Bengaluru Urban")].iloc[0]
    assert onion["years_observed"] == 2
    assert onion["latest_year"] == 2021
    assert onion["latest_area_ha"] == pytest.approx(120.0)
    assert onion["latest_production_t"] == pytest.approx(2400.0)
    assert onion["latest_yield_t_per_ha"] == pytest.approx(20.0)
    assert onion["total_area_ha"] == pytest.approx(220.0)
    assert onion["n_year_season_rows"] == 2


def test_crop_summary_market_block_exact_match_only():
    cs = ig.build_crop_summary(_agri(), _mandi(), ig.build_region_features(_weather(), _mandi(), _geo()))
    onion = cs[(cs["crop"] == "Onion") & (cs["district"] == "Bengaluru Urban")].iloc[0]
    assert onion["market_status"] == "matched"
    assert onion["matched_commodity"] == "Onion"
    assert onion["n_market_records"] == 2          # only the Rs./Quintal rows
    assert onion["n_markets"] == 1
    assert onion["latest_modal_price"] is not None
    carrot = cs[(cs["crop"] == "Carrot") & (cs["district"] == "Bengaluru Urban")].iloc[0]
    assert carrot["market_status"] == "no_records"  # no exact commodity 'Carrot'
    assert carrot["matched_commodity"] is None
    assert pd.isna(carrot["latest_modal_price"])


def test_crop_summary_weather_and_warning_columns():
    cs = ig.build_crop_summary(_agri(), _mandi(), ig.build_region_features(_weather(), _mandi(), _geo()))
    onion = cs[(cs["crop"] == "Onion") & (cs["district"] == "Bengaluru Urban")].iloc[0]
    assert onion["weather_status"] == "reported"
    assert onion["obs_temp_max_c"] == pytest.approx(30.55)
    wheat = cs[(cs["crop"] == "Wheat") & (cs["district"] == "Karnal")].iloc[0]
    assert wheat["weather_status"] == "reported"
    assert wheat["obs_temp_max_c"] == pytest.approx(34.0)
    # IMD warning loader is not configured -> column exists, is explicitly null.
    assert "weather_warning" in cs.columns
    assert cs["weather_warning"].isna().all()


def test_crop_summary_no_invented_values():
    cs = ig.build_crop_summary(_agri(), _mandi(), ig.build_region_features(_weather(), _mandi(), _geo()))
    assert "traded_quantity" not in cs.columns
    assert (cs["latest_production_t"].dropna() >= 0).all()
    assert (cs["total_area_ha"].dropna() >= 0).all()


# --------------------------------------------------------------------------- #
# Name normalization
# --------------------------------------------------------------------------- #
def test_normalize_is_case_and_whitespace_insensitive():
    assert ig._normalize("  Onion ") == "onion"
    assert ig._normalize("ONION") == "onion"
    assert ig._normalize("Ground Nut") == ig._normalize(" ground nut ")


# --------------------------------------------------------------------------- #
# Graceful no_data + lineage manifests
# --------------------------------------------------------------------------- #
def test_region_features_no_data_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(ig, "WEATHER", tmp_path / "nope.parquet")
    monkeypatch.setattr(ig, "MANDI", tmp_path / "nope2.parquet")
    monkeypatch.setattr(ig, "GEO", tmp_path / "nope3.parquet")
    monkeypatch.setattr(ig, "REGION_FEATURES_DIR", tmp_path / "gold" / "india_region_features")
    manifest = ig.run_region_features()
    assert manifest["status"] == "no_data"
    written = json.loads((tmp_path / "gold" / "india_region_features" / "manifest.json").read_text())
    assert written["missing"]


def test_crop_summary_no_data_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(ig, "AGRI", tmp_path / "nope.parquet")
    monkeypatch.setattr(ig, "CROP_SUMMARY_DIR", tmp_path / "gold" / "india_crop_summary")
    manifest = ig.run_crop_summary()
    assert manifest["status"] == "no_data"


def test_real_build_writes_manifests_with_lineage_fields(tmp_path, monkeypatch):
    agri_out = tmp_path / "silver" / "india_agriculture.parquet"
    mandi_out = tmp_path / "silver" / "india_mandi.parquet"
    weather_out = tmp_path / "silver" / "weather.parquet"
    geo_out = tmp_path / "silver" / "geo.parquet"
    for p, df in [(agri_out, _agri()), (mandi_out, _mandi()),
                  (weather_out, _weather()), (geo_out, _geo())]:
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)

    rf_dir = tmp_path / "gold" / "india_region_features"
    cs_dir = tmp_path / "gold" / "india_crop_summary"
    monkeypatch.setattr(ig, "AGRI", agri_out)
    monkeypatch.setattr(ig, "MANDI", mandi_out)
    monkeypatch.setattr(ig, "WEATHER", weather_out)
    monkeypatch.setattr(ig, "GEO", geo_out)
    monkeypatch.setattr(ig, "REGION_FEATURES_DIR", rf_dir)
    monkeypatch.setattr(ig, "CROP_SUMMARY_DIR", cs_dir)
    monkeypatch.setattr(ig, "CATALOG", tmp_path / "catalog")
    monkeypatch.setattr(ig, "LINEAGE_PATH", tmp_path / "catalog" / "gold_lineage.json")

    ig.run_region_features()
    ig.run_crop_summary()

    for manifest_path in (rf_dir / "manifest.json", cs_dir / "manifest.json"):
        m = json.loads(manifest_path.read_text())
        for key in ("source", "input", "output", "schema", "generated_at", "status"):
            assert key in m, f"{manifest_path.name} missing {key}"
            assert key != "input" or m[key], f"{manifest_path.name} input empty"
        assert m["status"] == "success"
        assert m["schema"]

    cs = pd.read_parquet(cs_dir / "india_crop_summary.parquet")
    assert not cs.empty
    assert "weather_warning" in cs.columns

    rg_manifest = json.loads((rf_dir / "manifest.json").read_text())
    cs_manifest = json.loads((cs_dir / "manifest.json").read_text())
    outputs = [
        ig._lineage_entry(rg_manifest, rf_dir, "india_region_features", "r"),
        ig._lineage_entry({"status": "no_data"}, tmp_path / "gold" / "india_market_summary",
                          "india_market_summary", "m"),
        ig._lineage_entry(cs_manifest, cs_dir, "india_crop_summary", "c"),
    ]
    ig.run_lineage(outputs)

    lineage = json.loads((tmp_path / "catalog" / "gold_lineage.json").read_text())
    assert {o["id"] for o in lineage["outputs"]} == {
        "india_region_features", "india_market_summary", "india_crop_summary"}
    for out in lineage["outputs"]:
        for key in ("source", "input", "output", "schema", "generated_at", "status"):
            assert key in out


def test_lineage_feature_set_contains_required_fields():
    for field in ["state", "district", "mandi", "crop", "season", "year",
                  "production", "area", "yield", "price_min", "price_modal",
                  "price_max", "arrival", "temperature", "rainfall", "humidity",
                  "weather_warning"]:
        assert field in ig.FEATURE_SET


# --------------------------------------------------------------------------- #
# mandi_to_gold reuse (single source of truth)
# --------------------------------------------------------------------------- #
def test_market_summary_reuse_via_run(tmp_path, monkeypatch):
    mandi_path = tmp_path / "silver" / "india_mandi.parquet"
    mandi_path.parent.mkdir(parents=True, exist_ok=True)
    gold_dir = tmp_path / "gold" / "india_market_summary"
    monkeypatch.setattr(mandi_to_gold, "SILVER_MANDI", mandi_path)
    monkeypatch.setattr(mandi_to_gold, "GOLD_SUMMARY_DIR", gold_dir)
    monkeypatch.setattr(ig, "MARKET_SUMMARY_DIR", gold_dir)
    _mandi().to_parquet(mandi_path, index=False)

    manifest = ig.run_market_summary()
    assert manifest["status"] == "success"
    summary = pd.read_parquet(gold_dir / "india_market_summary.parquet")
    onion = summary[(summary["commodity"] == "Onion") & (summary["district"] == "Bengaluru Urban")].iloc[0]
    # Single trade date -> latest metrics present, change metrics honestly null.
    assert onion["latest_modal_price"] == pytest.approx(1200.0)
    assert pd.isna(onion["modal_price_change_pct_7d"])


# --------------------------------------------------------------------------- #
# Monitor: India artifact checks
# --------------------------------------------------------------------------- #
def _write_manifest(manifest_dir, generated_hours_ago=0.0):
    import datetime as _dt
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "success",
        "generated_at": (_dt.datetime.now(_dt.timezone.utc)
                         - _dt.timedelta(hours=generated_hours_ago)).isoformat(),
        "period": ["2026-04-01", "2026-05-31"],
        "last_data_date": "2026-05-31",
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest_dir / "manifest.json"


def test_monitor_clean_india_data_passes(tmp_path, monkeypatch):
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    silver.mkdir(parents=True, exist_ok=True)
    gold.mkdir(parents=True, exist_ok=True)
    for name, frame in [("india_mandi.parquet",
                         _mandi().drop_duplicates(subset=["date", "state", "district",
                                                          "apmc", "commodity", "variety"])),
                        ("india_agriculture.parquet", _agri()),
                        ("geo.parquet", _geo())]:
        frame.to_parquet(silver / name, index=False)

    rf = ig.build_region_features(_weather(), _mandi(), _geo())
    (gold / "india_region_features").mkdir(parents=True, exist_ok=True)
    rf.to_parquet(gold / "india_region_features" / "india_region_features.parquet", index=False)
    cs = ig.build_crop_summary(_agri(), _mandi(), rf)
    (gold / "india_crop_summary").mkdir(parents=True, exist_ok=True)
    cs.to_parquet(gold / "india_crop_summary" / "india_crop_summary.parquet", index=False)
    for sub in ("india_region_features", "india_crop_summary", "india_market_summary"):
        _write_manifest(gold / sub)

    monkeypatch.setattr(monitor, "SILVER", silver)
    monkeypatch.setattr(monitor, "GOLD", gold)
    result = monitor.check_india_artifacts()
    assert result["issues"] == []
    assert all(c["status"] != "FAIL" for c in result["checks"])


def test_monitor_detects_planted_defects(tmp_path, monkeypatch):
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    silver.mkdir(parents=True, exist_ok=True)
    gold.mkdir(parents=True, exist_ok=True)
    mandi_dup = pd.concat([_mandi(), _mandi().iloc[[0]]], ignore_index=True)
    mandi_dup.to_parquet(silver / "india_mandi.parquet", index=False)
    _agri().to_parquet(silver / "india_agriculture.parquet", index=False)
    _geo().to_parquet(silver / "geo.parquet", index=False)

    rf = ig.build_region_features(_weather(), _mandi(), _geo())
    dup_rf = pd.concat([rf, rf.iloc[[0]]], ignore_index=True)
    dup_rf.loc[0, "temp_max_c"] = 99.9  # impossible value
    (gold / "india_region_features").mkdir(parents=True, exist_ok=True)
    dup_rf.to_parquet(gold / "india_region_features" / "india_region_features.parquet", index=False)
    _write_manifest(gold / "india_region_features", generated_hours_ago=50.0)
    cs = ig.build_crop_summary(_agri(), _mandi(), rf)
    (gold / "india_crop_summary").mkdir(parents=True, exist_ok=True)
    cs.to_parquet(gold / "india_crop_summary" / "india_crop_summary.parquet", index=False)
    _write_manifest(gold / "india_crop_summary")
    _write_manifest(gold / "india_market_summary")

    monkeypatch.setattr(monitor, "SILVER", silver)
    monkeypatch.setattr(monitor, "GOLD", gold)
    result = monitor.check_india_artifacts()
    by_check = {(c["check"], c["dataset"]): c["status"] for c in result["checks"]}
    assert by_check[("duplicates", "india_mandi")] == "FAIL"
    assert by_check[("duplicates", "india_region_features")] == "FAIL"
    assert by_check[("impossible_values", "india_region_features")] == "FAIL"
    assert by_check[("staleness", "india_region_features")] == "FAIL"
    failing = {(i["check"], i["dataset"]) for i in result["issues"]}
    assert ("duplicates", "india_mandi") in failing
    assert ("impossible_values", "india_region_features") in failing