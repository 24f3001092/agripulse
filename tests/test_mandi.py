"""
test_mandi.py

Unit tests for the India mandi market-intelligence pipeline:

  * src/ingestion/india_mandi.py   -- AGMARKNET daily APMC price/arrival
    ingestion (structured IDP/CKAN datastore, documented CSV import, Bronze
    writer with provenance)
  * src/quality/india_schemas.py   -- mandi_price_schema gate
  * src/transform/bronze_to_silver.py -- clean_mandi (dedup, silver gate)
  * src/transform/mandi_to_gold.py -- market-intelligence aggregates

Covered cases: invalid (negative) price, negative arrival, duplicate trade
record, missing commodity, missing date, valid records, unit preservation,
no traded-quantity invention, CSV import, provenance, no-data markers, and
Gold aggregates. No live network calls (fake datastore transports only).
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

import india_mandi as mandi  # noqa: E402
from india_schemas import mandi_price_schema, validate_mandi  # noqa: E402
import bronze_to_silver as b2s  # noqa: E402
import mandi_to_gold  # noqa: E402


def _raw_record(**overrides) -> dict:
    rec = {
        "date": "2026-05-14",
        "state_name": "Tamil Nadu",
        "state_code": "33",
        "district_name": "Coimbatore",
        "district_code": "538",
        "market_center_name": "Coimbatore Apmc",
        "market_center_code": "9991",
        "category": "Wholesale Market",
        "commodity_name": "Onion",
        "commodity_id": "123",
        "variety": "Local",
        "grade": "FAQ",
        "arrival_quantity": 125.0,
        "arrival_units": "Metric Tonnes",
        "min_price": 1100.0,
        "modal_price": 1200.0,
        "max_price": 1400.0,
        "price_unit": "Rs./Quintal",
    }
    rec.update(overrides)
    return rec


def _raw_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def _valid_normalized() -> pd.DataFrame:
    return mandi.normalize(_raw_frame([_raw_record()]))


# --------------------------------------------------------------------------- #
# Ingestion / normalization
# --------------------------------------------------------------------------- #
def test_normalize_maps_columns_and_preserves_units():
    df = _valid_normalized()
    assert list(df.columns) == mandi.NORMALIZED_COLUMNS
    assert df.loc[0, "state"] == "Tamil Nadu"
    assert df.loc[0, "district"] == "Coimbatore"
    assert df.loc[0, "apmc"] == "Coimbatore Apmc"
    assert df.loc[0, "commodity"] == "Onion"
    assert df.loc[0, "price_unit"] == "Rs./Quintal"
    assert df.loc[0, "arrival_units"] == "Metric Tonnes"
    assert df.loc[0, "state_code"] == "33"  # identifiers stay strings
    assert str(df.loc[0, "date"])[:10] == "2026-05-14"


def test_traded_quantity_never_invented():
    df = _valid_normalized()
    assert "traded_quantity" not in df.columns
    assert "traded_quantity" not in mandi.NORMALIZED_COLUMNS


def test_missing_source_columns_are_dropped_not_fabricated():
    raw = _raw_frame([_raw_record()]).drop(columns=["arrival_quantity", "arrival_units"])
    df = mandi.normalize(raw)
    assert pd.isna(df.loc[0, "arrival_quantity"])


def test_load_mandi_csv_documented_import_interface(tmp_path):
    path = tmp_path / "agmarknet.csv"
    _raw_frame([_raw_record(), _raw_record(date="2026-05-15")]).to_csv(path, index=False)
    df = mandi.load_mandi_csv(path)
    assert len(df) == 2


def test_load_mandi_csv_missing_file(tmp_path):
    with pytest.raises(mandi.MandiInvalidSource):
        mandi.load_mandi_csv(tmp_path / "nope.csv")


# --------------------------------------------------------------------------- #
# Schema gate
# --------------------------------------------------------------------------- #
def test_valid_mandi_record_passes():
    df = _valid_normalized()
    clean, report = validate_mandi(df)
    assert report is None
    assert len(clean) == 1


def test_invalid_negative_price_fails():
    df = _valid_normalized()
    df.loc[0, "min_price"] = -5.0
    _, report = validate_mandi(df)
    assert report is not None
    assert report["failure_count"] >= 1


def test_invalid_negative_arrival_fails():
    df = _valid_normalized()
    df.loc[0, "arrival_quantity"] = -10.0
    _, report = validate_mandi(df)
    assert report is not None
    assert report["failure_count"] >= 1


def test_modal_outside_min_max_fails():
    df = _valid_normalized()
    df.loc[0, "modal_price"] = df.loc[0, "max_price"] + 500
    _, report = validate_mandi(df)
    assert report is not None


def test_missing_commodity_fails():
    df = _valid_normalized()
    df.loc[0, "commodity"] = ""
    _, report = validate_mandi(df)
    assert report is not None
    assert report["failure_count"] >= 1


def test_missing_date_fails():
    df = _valid_normalized()
    df.loc[0, "date"] = None
    _, report = validate_mandi(df)
    assert report is not None


def test_unparseable_source_date_dropped_before_schema():
    raw = _raw_frame([_raw_record(), _raw_record(date="not-a-date")])
    df = mandi.normalize(raw)
    assert len(df) == 1
    params = []
    for col in ["traded_quantity"]:
        assert col not in df.columns
    # date col has no nulls after dropping unparseable rows
    assert not df["date"].isna().any()


# --------------------------------------------------------------------------- #
# Duplicate trade record (clean_mandi)
# --------------------------------------------------------------------------- #
def test_clean_mandi_dedups_duplicate_trade_record(tmp_path, monkeypatch):
    raw = _raw_frame([_raw_record(), _raw_record()])  # exact duplicate trade day
    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"
    mandi.write_bronze(mandi.normalize(raw), out_dir=bronze, raw_df=raw,
                       meta_extra={"source_kind": "imported-csv"})

    monkeypatch.setattr(b2s, "MANDI_BRONZE_DIR", bronze)
    monkeypatch.setattr(b2s, "SILVER", silver)
    monkeypatch.setattr(b2s, "REJECTS", silver / "_rejects")

    result = b2s.clean_mandi()
    assert result is not None
    assert len(result) == 1  # one duplicate removed on trade-day identity

    # business key equality with real trade identity
    assert len(mandi.normalize(raw).drop_duplicates(subset=mandi.DEDUP_KEY, keep="first")) == 1


def test_clean_mandi_quarantines_contradictory_price_ordering(tmp_path, monkeypatch):
    raw = _raw_frame([
        _raw_record(),
        _raw_record(date="2026-05-13", min_price=9000.0, modal_price=8200.0),  # min > modal
        _raw_record(date="2026-05-12", modal_price=9000.0, max_price=8200.0),  # modal > max
    ])
    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"
    mandi.write_bronze(mandi.normalize(raw), out_dir=bronze, raw_df=raw,
                       meta_extra={"source_kind": "imported-csv"})
    monkeypatch.setattr(b2s, "MANDI_BRONZE_DIR", bronze)
    monkeypatch.setattr(b2s, "SILVER", silver)
    monkeypatch.setattr(b2s, "REJECTS", silver / "_rejects")

    df = b2s.clean_mandi()
    assert df is not None
    assert len(df) == 1  # the valid row survives
    # contradicting rows are preserved transparently, not silently dropped
    rejects = pd.read_parquet(silver / "_rejects" / "india_mandi_invalid_price_rows.parquet")
    assert len(rejects) == 2
    # the quarantined frame still passes the schema (clean table is writable)
    clean, report = validate_mandi(df)
    assert report is None
    assert len(clean) == 1


def test_clean_mandi_writes_no_data_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(b2s, "MANDI_BRONZE_DIR", tmp_path / "bronze")
    monkeypatch.setattr(b2s, "SILVER", tmp_path / "silver")
    assert b2s.clean_mandi() is None
    marker = json.loads((tmp_path / "silver" / "mandi_status.json").read_text())
    assert marker["status"] == "no_data"


def test_clean_mandi_gate_writes_silver(tmp_path, monkeypatch):
    raw = _raw_frame([_raw_record()])
    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"
    mandi.write_bronze(mandi.normalize(raw), out_dir=bronze, raw_df=raw,
                       meta_extra={"source_kind": "imported-csv"})
    monkeypatch.setattr(b2s, "MANDI_BRONZE_DIR", bronze)
    monkeypatch.setattr(b2s, "SILVER", silver)
    monkeypatch.setattr(b2s, "REJECTS", silver / "_rejects")

    df = b2s.clean_mandi()
    b2s.write_validated(df, mandi_price_schema, "india_mandi")
    assert (silver / "india_mandi.parquet").exists()
    assert not (silver / "_rejects" / "india_mandi_rejects.json").exists()


# --------------------------------------------------------------------------- #
# Provenance / ensure no live calls
# --------------------------------------------------------------------------- #
def test_write_bronze_provenance_no_traded_quantity(tmp_path):
    raw = _raw_frame([_raw_record()])
    norm = mandi.normalize(raw)
    path = mandi.write_bronze(norm, out_dir=tmp_path, raw_df=raw,
                              meta_extra={"source_kind": "imported-csv", "period": ["2026-05-14", "2026-05-14"]})
    meta = json.loads((tmp_path / (path.stem + ".meta.json")).read_text())
    assert meta["dataset"] == mandi.SOURCE_NAME
    assert meta["source_kind"] == "imported-csv"
    assert meta["status"] == "success"
    assert meta["row_count"] == 1
    assert any("traded_quantity" in n for n in meta["notes"])
    assert (tmp_path / "raw" / "agmarknet_raw_*.csv") or True
    assert list((tmp_path / "raw").glob("agmarknet_raw_*.csv"))


# --------------------------------------------------------------------------- #
# API client (fake transport only -- no network)
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses: list, fail_with=None):
        self.responses = list(responses)
        self.fail_with = fail_with
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if self.fail_with is not None:
            raise self.fail_with
        if not self.responses:
            return _FakeResp({"success": False, "error": {"name": "SearchQueryError"}})
        return self.responses.pop(0)


def test_fetch_historical_mandi_prices_uses_canned_transport(monkeypatch):
    rec = {"date": "2026-05-14", "state_name": "Tamil Nadu",
           "district_name": "Coimbatore", "market_center_name": "Coimbatore Apmc",
           "commodity_name": "Onion", "variety": "Local", "min_price": 100.0,
           "modal_price": 110.0, "max_price": 130.0, "arrival_quantity": 5.0,
           "price_unit": "Rs./Quintal", "arrival_units": "Metric Tonnes"}
    page = {"success": True, "result": {"total": 2, "records": [rec, dict(rec, date="2026-05-15")]}}
    session = _FakeSession([_FakeResp(page)])
    df, meta = mandi.fetch_historical_mandi_prices(state="Tamil Nadu", district="Coimbatore",
                                                   session=session)
    assert len(df) == 2
    assert meta["source"] == mandi.SOURCE_NAME
    assert meta["is_live"] is False
    assert session.calls  # transport exercised; no module-level requests path


def test_fetches_classify_requests_exceptions(monkeypatch):
    import requests
    session = _FakeSession([], fail_with=requests.exceptions.Timeout())
    with pytest.raises(mandi.MandiTimeoutError):
        mandi.fetch_historical_mandi_prices(session=session)
    session = _FakeSession([], fail_with=requests.exceptions.ConnectionError())
    with pytest.raises(mandi.MandiConnectionError):
        mandi.fetch_historical_mandi_prices(session=session)
    session = _FakeSession([_FakeResp({}, status=403)])
    with pytest.raises(mandi.MandiHttpError):
        mandi.fetch_historical_mandi_prices(session=session)
    session = _FakeSession([])
    with pytest.raises(mandi.MandiInvalidSource):
        mandi.fetch_historical_mandi_prices(session=session)


# --------------------------------------------------------------------------- #
# Gold market-intelligence aggregates
# --------------------------------------------------------------------------- #
def _silver_frame() -> pd.DataFrame:
    rows = [
        _raw_record(date="2026-05-10", modal_price=1000.0),
        _raw_record(date="2026-05-12", modal_price=1100.0),
        _raw_record(date="2026-05-14", modal_price=1200.0),
    ]
    import pandas as pd
    return mandi.normalize(_raw_frame(rows))


def test_gold_summary_latest_and_7d_average(tmp_path):
    df = _silver_frame()
    summary = mandi_to_gold.build_market_summary(df)
    row = summary.iloc[0]
    assert row["latest_modal_price"] == 1200.0
    assert row["days_of_history"] == 3
    assert row["avg_modal_price_7d"] == pytest.approx(1100.0)
    assert row["modal_price_change_pct_7d"] == pytest.approx((1200 - 1100) / 1100 * 100.0)
    assert row["price_unit"] == "Rs./Quintal"
    assert "state_latest_modal_price" in summary.columns


def test_gold_summary_single_day_has_no_change_metrics():
    df = mandi.normalize(_raw_frame([_raw_record()]))
    summary = mandi_to_gold.build_market_summary(df)
    row = summary.iloc[0]
    assert row["days_of_history"] == 1
    assert row["latest_modal_price"] == 1200.0
    assert row["modal_price_change_pct_7d"] is None  # not supported by history
    assert row["arrival_change_pct_7d"] is None


def test_gold_manifest_written(tmp_path, monkeypatch):
    silver = tmp_path / "silver"
    silver.mkdir(parents=True)
    _silver_frame().to_parquet(silver / "india_mandi.parquet")
    monkeypatch.setattr(mandi_to_gold, "SILVER_MANDI", silver / "india_mandi.parquet")
    monkeypatch.setattr(mandi_to_gold, "GOLD_SUMMARY_DIR", tmp_path / "gold" / "india_market_summary")
    mandi_to_gold.main()
    manifest = json.loads((tmp_path / "gold" / "india_market_summary" / "manifest.json").read_text())
    assert manifest["status"] == "success"
    assert manifest["n_commodities"] == 1
    assert manifest["no_traded_quantity"]
    assert "not investment advice" in manifest["disclaimer"].lower()


def test_gold_skips_with_no_data_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(mandi_to_gold, "SILVER_MANDI", tmp_path / "silver" / "india_mandi.parquet")
    monkeypatch.setattr(mandi_to_gold, "GOLD_SUMMARY_DIR", tmp_path / "gold" / "india_market_summary")
    mandi_to_gold.main()
    manifest = json.loads((tmp_path / "gold" / "india_market_summary" / "manifest.json").read_text())
    assert manifest["status"] == "no_data"