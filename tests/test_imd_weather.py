"""
test_imd_weather.py

Unit tests for the IMD weather ingestion module (src/ingestion/imd_weather.py)
and its sandbox fixture generator (india_weather_fixture_generator.py).

Covered failure modes: valid pulls, timeout, connection error, HTTP 401/403
(auth), HTTP 5xx, malformed JSON, missing fields, invalid location, and empty
responses. NO live network calls are made: every test injects a canned request
transport so the suite is deterministic and offline-safe.

Run with:
    pytest tests/test_imd_weather.py -v
"""

import json
import sys
from pathlib import Path

import pytest
import requests

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "ingestion"))

import imd_weather  # noqa: E402
from imd_weather import (  # noqa: E402
    ImdAuthError,
    ImdClient,
    ImdConnectionError,
    ImdEmptyResponse,
    ImdHttpError,
    ImdInvalidLocation,
    ImdMalformedResponse,
    ImdTimeoutError,
    ENV_API_KEY,
    ENV_TOKEN,
    fetch_current_weather,
    fetch_district_rainfall,
    fetch_district_warning,
    fetch_forecast,
    imd_availability,
    write_bronze,
)


# --------------------------------------------------------------------------- #
# Canned, schema-accurate IMD payloads (from the public API reference)
# --------------------------------------------------------------------------- #
CURRENT_ROWS = [
    {
        "Station Id": "42182",
        "Station": "PUNE",
        "Date of Observation": "2026-09-18",
        "Time of Observation": "05:30",
        "M.S.L.P": "1005.2",
        "Wind Direction": "270",
        "Wind Speed": "11",
        "Temperature": "26.4",
        "Weather Code": "21",
        "Nebulosity": "5",
        "Humidity": "74",
        "Last 24 hrs Rainfall": "0.00",
    }
]

FORECAST_ROW = {
    "Date": "2026-09-18",
    "Station_Code": "42182",
    "Station_Name": "NASHIK",
    "Today_Max_temp": "31.0",
    "Today_Min_temp": "21.5",
    "Todays_Forecast": "Generally cloudy sky",
    "Past_24_hrs_Rainfall": "0.0",
    "Relative_Humidity_at_0830": "76",
    "Relative_Humidity_at_1730": "58",
    "Day_2_Max_Temp": "32.0",
    "Day_2_Min_temp": "22.0",
    "Day_2_Forecast": "Partly cloudy sky",
    "Day_3_Max_Temp": "32.5",
    "Day_3_Min_temp": "22.4",
    "Day_3_Forecast": "Partly cloudy sky",
    "Day_4_Max_Temp": "30.0",
    "Day_4_Min_temp": "21.0",
    "Day_4_Forecast": "Rain possible",
    "Day_5_Max_Temp": "29.0",
    "Day_5_Min_temp": "20.5",
    "Day_5_Forecast": "Rain possible",
    "Day_6_Max_Temp": "29.5",
    "Day_6_Min_temp": "20.8",
    "Day_6_Forecast": "Cloudy",
    "Day_7_Max_Temp": "30.5",
    "Day_7_Min_temp": "21.2",
    "Day_7_Forecast": "Partly cloudy sky",
}

RAINFALL_ROWS = [
    {
        "OBJ_ID": "164",
        "District": "ADILABAD",
        "Date": "2026-09-18",
        "Daily Actual": "0.00",
        "Daily Normal": "1.70",
        "Daily Departure Per": "-100%",
        "Daily Category": "NR",
        "Weekly Actual": "0.00",
        "Weekly Normal": "1.70",
        "Weekly Departure Per": "-100%",
        "Weekly Category": "NR",
        "Cumulative Actual": "0.00",
        "Cumulative Normal": "11.60",
        "Cumulative Departure Per": "-100%",
        "Cumulative Category": "NR",
        "Monthly Actual": "5.10",
        "Monthly Normal": "5.00",
        "Monthly Departure Per": "1%",
        "Monthly Category": "N",
    },
    {
        "OBJ_ID": "479",
        "District": "NASHIK",
        "Date": "2026-09-18",
        "Daily Actual": "3.20",
        "Daily Normal": "2.10",
        "Daily Departure Per": "52%",
        "Daily Category": "E",
        "Weekly Actual": "12.00",
        "Weekly Normal": "10.00",
        "Weekly Departure Per": "20%",
        "Weekly Category": "E",
        "Cumulative Actual": "55.00",
        "Cumulative Normal": "50.00",
        "Cumulative Departure Per": "10%",
        "Cumulative Category": "N",
        "Monthly Actual": "60.00",
        "Monthly Normal": "55.00",
        "Monthly Departure Per": "9%",
        "Monthly Category": "N",
    },
]

WARNING_ROWS = [
    {
        "Obj_id": "479",
        "Date": "2026-09-18",
        "UTC": "03:30",
        "District": "NASHIK",
        "Day_1": "4,17",
        "Day_2": "2",
        "Day_3": "1",
        "Day_4": "1",
        "Day_5": "1",
        "Day1_Color": "1",
        "Day2_Color": "1",
        "Day3_Color": "4",
        "Day4_Color": "4",
        "Day5_Color": "4",
    }
]


# --------------------------------------------------------------------------- #
# Fake transport (deterministic, offline)
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, ValueError):
            raise self._payload
        return self._payload


class _FakeSession:
    """Records every call; replays canned responses (or raises canned errors)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        item = self._responses.pop(0) if self._responses else (200, [])
        if isinstance(item, Exception):
            raise item
        status_code, payload = item
        return _FakeResponse(status_code, payload)


def _client(responses, api_key="TEST-KEY", token="TEST-TOKEN", timeout=5):
    session = _FakeSession(responses)
    client = ImdClient(api_key=api_key, token=token, timeout=timeout, session=session)
    return client, session


# --------------------------------------------------------------------------- #
# Availability gate (never touches the network)
# --------------------------------------------------------------------------- #
def test_availability_not_configured_without_credentials(monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    monkeypatch.delenv(ENV_TOKEN, raising=False)
    report = imd_availability()
    assert report["configured"] is False
    assert report["service"] == "imd"
    assert ENV_API_KEY in report["missing_env"]
    assert ENV_TOKEN in report["missing_env"]


def test_availability_configured_from_env(monkeypatch):
    monkeypatch.setenv(ENV_API_KEY, "K")
    monkeypatch.setenv(ENV_TOKEN, "T")
    report = imd_availability()
    assert report["configured"] is True


def test_availability_explicit_client_needs_both():
    report = imd_availability(ImdClient(api_key="K", token=None))
    assert report["configured"] is False


def test_fetch_without_credentials_never_hits_transport():
    client = ImdClient(api_key="K", token="")
    with pytest.raises(ImdAuthError):
        client._pull("current")


# --------------------------------------------------------------------------- #
# HTTP / error plumbing
# --------------------------------------------------------------------------- #
def test_timeout_raises_imd_timeout():
    client, _session = _client([requests.Timeout("slow")])
    with pytest.raises(ImdTimeoutError):
        client._pull("current")


def test_connection_error_raises_imd_connection():
    client, _session = _client([requests.ConnectionError("refused")])
    with pytest.raises(ImdConnectionError):
        client._pull("current")


def test_http_401_raises_auth_error():
    client, _session = _client([(401, {"detail": "unauthorized"})])
    with pytest.raises(ImdAuthError):
        client._pull("current")


def test_http_403_raises_auth_error():
    client, _session = _client([(403, {"detail": "forbidden"})])
    with pytest.raises(ImdAuthError):
        client._pull("current")


def test_http_500_raises_imd_http_error():
    client, _session = _client([(500, {})])
    with pytest.raises(ImdHttpError) as excinfo:
        client._pull("current")
    assert excinfo.value.status_code == 500


def test_malformed_json_raises_imd_malformed():
    client, _session = _client([(200, ValueError("not json"))])
    with pytest.raises(ImdMalformedResponse):
        client._pull("current")


def test_empty_response_raises_imd_empty():
    client, _session = _client([(200, [])])
    with pytest.raises(ImdEmptyResponse):
        client._pull("current")


def test_empty_envelope_raises_imd_empty():
    client, _session = _client([(200, {"data": []})])
    with pytest.raises(ImdEmptyResponse):
        client._pull("rainfall")


def test_request_headers_and_params_are_sent():
    client, session = _client([(200, CURRENT_ROWS)])
    fetch_current_weather(station_id="42182", client=client)
    call = session.calls[0]
    assert call["headers"][imd_weather.API_KEY_HEADER] == "TEST-KEY"
    assert call["headers"][imd_weather.AUTH_HEADER] == "Bearer TEST-TOKEN"
    assert call["params"] == {"id": "42182"}
    assert call["timeout"] == 5


# --------------------------------------------------------------------------- #
# current_wx
# --------------------------------------------------------------------------- #
def test_fetch_current_weather_valid():
    client, _session = _client([(200, CURRENT_ROWS)])
    rows = fetch_current_weather(client=client)
    assert len(rows) == 1
    row = rows[0]
    assert row["station_id"] == "42182"
    assert row["station"] == "PUNE"
    assert row["date"] == "2026-09-18"
    assert row["temp_c"] == 26.4
    assert row["humidity_pct"] == 74.0
    assert row["rainfall_24h_mm"] == 0.0
    assert row["mslp_hpa"] == 1005.2
    assert row["weather_code"] == "21"


def test_fetch_current_weather_missing_fields():
    client, _session = _client([(200, [{"foo": "bar"}])])
    with pytest.raises(ImdMalformedResponse):
        fetch_current_weather(client=client)


def test_fetch_current_weather_empty():
    client, _session = _client([(200, [])])
    with pytest.raises(ImdEmptyResponse):
        fetch_current_weather(client=client)


# --------------------------------------------------------------------------- #
# cityforecast
# --------------------------------------------------------------------------- #
def test_fetch_forecast_expands_seven_days_from_raw_list():
    client, session = _client([(200, [FORECAST_ROW])])
    rows = fetch_forecast(client=client)
    assert len(rows) == 7
    assert [r["forecast_day"] for r in rows] == [1, 2, 3, 4, 5, 6, 7]
    day1 = rows[0]
    assert day1["date"] == "2026-09-18"
    assert day1["temp_max_c"] == 31.0
    assert day1["temp_min_c"] == 21.5
    assert day1["rh_0830_pct"] == 76.0
    assert day1["rainfall_24h_mm"] == 0.0
    day2 = rows[1]
    assert day2["date"] == "2026-09-19"
    assert day2["temp_max_c"] == 32.0
    assert day2["rh_0830_pct"] is None
    day7 = rows[6]
    assert day7["date"] == "2026-09-24"
    assert day7["forecast_text"] == "Partly cloudy sky"


def test_fetch_forecast_accepts_envelope_with_data_key():
    client, session = _client([(200, {"status": True, "data": [FORECAST_ROW]})])
    rows = fetch_forecast(client=client)
    assert len(rows) == 7


def test_fetch_forecast_unparseable_date_raises_malformed():
    bad = dict(FORECAST_ROW, Date="18/09/2026")
    client, session = _client([(200, [bad])])
    with pytest.raises(ImdMalformedResponse):
        fetch_forecast(client=client)


# --------------------------------------------------------------------------- #
# districtrainfall
# --------------------------------------------------------------------------- #
def test_fetch_district_rainfall_all():
    client, session = _client([(200, RAINFALL_ROWS)])
    rows = fetch_district_rainfall(client=client)
    assert len(rows) == 2
    nashik = [r for r in rows if r["district"] == "NASHIK"][0]
    assert nashik["daily_actual_mm"] == 3.2
    assert nashik["daily_departure_pct"] == 52.0
    assert nashik["daily_category"] == "E"
    assert nashik["obj_id"] == "479"


def test_fetch_district_rainfall_filter_case_insensitive():
    client, session = _client([(200, RAINFALL_ROWS)])
    rows = fetch_district_rainfall(district="nashik", client=client)
    assert len(rows) == 1
    assert rows[0]["district"] == "NASHIK"


def test_fetch_district_rainfall_invalid_location():
    client, session = _client([(200, RAINFALL_ROWS)])
    with pytest.raises(ImdInvalidLocation) as excinfo:
        fetch_district_rainfall(district="Narnia", client=client)
    assert "Narnia" in str(excinfo.value)
    assert "ADILABAD" in str(excinfo.value)


def test_fetch_district_rainfall_empty():
    client, session = _client([(200, [])])
    with pytest.raises(ImdEmptyResponse):
        fetch_district_rainfall(client=client)


# --------------------------------------------------------------------------- #
# districtwarning
# --------------------------------------------------------------------------- #
def test_fetch_district_warning_valid_and_codes_parsed():
    client, session = _client([(200, WARNING_ROWS)])
    rows = fetch_district_warning(district="Nashik", client=client)
    assert len(rows) == 1
    row = rows[0]
    assert row["district"] == "NASHIK"
    assert row["day_1_codes"] == [4, 17]
    assert row["day_2_codes"] == [2]
    assert row["day_3_codes"] == [1]
    assert row["day_1_color"] == "1"


def test_fetch_district_warning_invalid_location():
    client, session = _client([(200, WARNING_ROWS)])
    with pytest.raises(ImdInvalidLocation):
        fetch_district_warning(district="Mumbai", client=client)


def test_fetch_district_warning_empty():
    client, session = _client([(200, [])])
    with pytest.raises(ImdEmptyResponse):
        fetch_district_warning(client=client)


# --------------------------------------------------------------------------- #
# Bronze writer provenance (per-pull payload + .meta.json)
# --------------------------------------------------------------------------- #
def test_write_bronze_provenance(tmp_path):
    records = [{"district": "NASHIK", "daily_actual_mm": 3.2}]
    path = write_bronze(
        "rainfall", records, raw=records,
        endpoint="https://api.imd.gov.in/api/v1/districtrainfall",
        location="Nashik", state="Maharashtra",
        retrieved_at="20260918T053000Z", out_dir=tmp_path,
    )
    assert path.exists()
    payload = json.loads(path.read_text())
    prov = payload["provenance"]
    assert prov["source"] == "imd"
    assert prov["endpoint"] == "https://api.imd.gov.in/api/v1/districtrainfall"
    assert prov["retrieved_at"] == "2026-09-18T05:30:00Z"
    assert prov["location"] == "Nashik"
    assert prov["date"] == "2026-09-18"
    assert prov["is_fixture"] is False
    assert payload["records"] == records

    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    assert meta["status"] == "success"
    assert meta["row_count"] == 1
    assert meta["is_fixture"] is False


def test_write_bronze_labelled_fixture(tmp_path):
    path = write_bronze(
        "rainfall", [], raw=[], endpoint="https://api.imd.gov.in/api/v1/fixture",
        out_dir=tmp_path, is_fixture=True,
    )
    payload = json.loads(path.read_text())
    assert payload["provenance"]["source"] == "imd-fixture"
    assert payload["provenance"]["is_fixture"] is True
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["status"] == "synthetic_fixture"
    assert meta["is_fixture"] is True


# --------------------------------------------------------------------------- #
# Sandbox fixture generator: same schema, clearly labelled
# --------------------------------------------------------------------------- #
def _load_fixture_generator():
    import importlib
    return importlib.import_module("india_weather_fixture_generator")


def test_fixture_generator_produces_imd_schema():
    mod = _load_fixture_generator()
    regions = [
        {"district": "Nashik", "region_name": "Nashik", "state_ut": "Maharashtra"},
        {"district": "Coimbatore", "region_name": "Coimbatore", "state_ut": "Tamil Nadu"},
    ]

    current = mod._build_payload("current", regions)
    assert len(current) == 2
    assert current[0]["station"] == "Nashik"
    assert set(current[0]) >= {"temp_c", "humidity_pct", "date", "weather_code"}

    forecast = mod._build_payload("forecast", regions)
    assert len(forecast) == 14
    assert set(forecast[0]) >= {"station_name", "forecast_day", "temp_max_c", "forecast_text"}

    rainfall = mod._build_payload("rainfall", regions)
    assert rainfall[0]["district"] == "Nashik"
    assert set(rainfall[0]) >= {"daily_actual_mm", "daily_category", "weekly_category"}

    warning = mod._build_payload("warning", regions)
    assert warning[0]["district"] == "Nashik"
    assert set(warning[0]) >= {"day_1_codes", "day_1_color", "time_utc"}


def test_fixture_generator_writes_labelled_fixture(tmp_path):
    mod = _load_fixture_generator()
    records = mod._make_rainfall({"district": "Nashik"}, 0)
    path = mod.write_bronze(
        "rainfall", [records], raw=[records],
        endpoint="https://api.imd.gov.in/api/v1/fixture",
        out_dir=tmp_path, is_fixture=True,
    )
    payload = json.loads(path.read_text())
    assert payload["provenance"]["source"] == "imd-fixture"
    assert "NEVER presented as live data" in payload["provenance"]["note"]