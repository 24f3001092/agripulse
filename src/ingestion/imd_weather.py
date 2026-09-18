"""
imd_weather.py
Bronze-layer ingestion module for the India Meteorological Department (IMD).

Official API portal    : https://api.imd.gov.in/
Official API reference : https://api.imd.gov.in/public/api_reference.html
Registration / keys    : https://api.imd.gov.in/public/register.php

Endpoints used (documented in the public API reference):
    current weather  : https://api.imd.gov.in/api/v1/current_wx
    city forecast    : https://api.imd.gov.in/api/v1/cityforecast
    district rainfall: https://api.imd.gov.in/api/v1/districtrainfall
    district warnings: https://api.imd.gov.in/api/v1/districtwarning

Auth model (per the IMD portal user guide): a registered user receives a DEV/PROD
API key bound to the public IP of the calling server plus a bearer token (JWT)
minted from the registered email + password. Every request must carry both.

Credentials are NEVER hard-coded. They are read from environment variables:
    AGRIPULSE_IMD_API_KEY       IMD API key (supplied by the portal after approval)
    AGRIPULSE_IMD_TOKEN         pre-minted bearer token (JWT)
    AGRIPULSE_IMD_API_EMAIL     registered email (informational / for manual JWT minting)
    AGRIPULSE_IMD_API_PASSWORD  registered password (informational / for manual JWT minting)

The IMD portal's JWT minting endpoint is not published in the public API
reference, so this module does NOT invent one: it expects a pre-minted token in
AGRIPULSE_IMD_TOKEN (a short main() fallback hint is printed when the token is
absent). Without credentials, imd_availability() reports not_configured and NOTHING
is requested from IMD; no data is ever fabricated.

District rainfall / warnings are fetched WITHOUT an id and filtered client-side by
district name (and optionally state) because the public reference does not publish
a stable IMD object-id table. Filtering is name-based and case-insensitive; unknown
districts raise ImdInvalidLocation rather than guessing an id.

Output: every successful pull writes a JSON payload + a provenance .meta.json into
data/<country>/bronze/india_weather/ (India profile: data/india/bronze/india_weather/),
i.e. one directory per IMD pull under the active profile's bronze layer. The files
are named imd_<kind>_<UTC>_<location>.json and never match the Open-Meteo
weather_raw_* glob that bronze_to_silver consumes, so the two pipelines stay
independent. Provenance records source, endpoint, retrieved_at, location, date and
an is_fixture flag (set by the sandbox fixture generator, never by the live path).

Usage:
    python imd_weather.py --kind current --station-id 42182
    python imd_weather.py --kind rainfall --district Nashik
    python imd_weather.py --kind all --district Nashik --state Maharashtra
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("imd_weather_ingestion")

# --------------------------------------------------------------------------- #
# Documented endpoints (api.imd.gov.in/public/api_reference.html)
# --------------------------------------------------------------------------- #
IMD_BASE_URL = "https://api.imd.gov.in/api/v1"
ENDPOINT_CURRENT = f"{IMD_BASE_URL}/current_wx"
ENDPOINT_FORECAST = f"{IMD_BASE_URL}/cityforecast"
ENDPOINT_RAINFALL = f"{IMD_BASE_URL}/districtrainfall"
ENDPOINT_WARNING = f"{IMD_BASE_URL}/districtwarning"
ENDPOINT_FORECAST_MAPPING = f"{IMD_BASE_URL}/cityforecast_mapping"

# --------------------------------------------------------------------------- #
# Credentials come exclusively from the environment (never hard-coded).
# --------------------------------------------------------------------------- #
ENV_API_KEY = "AGRIPULSE_IMD_API_KEY"
ENV_TOKEN = "AGRIPULSE_IMD_TOKEN"
ENV_EMAIL = "AGRIPULSE_IMD_API_EMAIL"
ENV_PASSWORD = "AGRIPULSE_IMD_API_PASSWORD"

# Header names used by IMD. The API key travels in the "apikey" header and the
# pre-minted JWT in the standard "Authorization: Bearer <token>" header, per the
# IMD API portal user guide. They are module constants so a documented change on
# IMD's side can be applied in exactly one place.
API_KEY_HEADER = "apikey"
AUTH_HEADER = "Authorization"

DEFAULT_TIMEOUT_SECONDS = 20


# --------------------------------------------------------------------------- #
# Typed errors (one per failure mode the caller needs to react to)
# --------------------------------------------------------------------------- #
class ImdError(Exception):
    """Base class for all IMD ingestion failures."""


class ImdAuthError(ImdError):
    """Credentials missing/invalid or the server rejected them (401/403)."""


class ImdConnectionError(ImdError):
    """Network-level failure (DNS, connection refused, TLS, ...)."""


class ImdTimeoutError(ImdError):
    """The request exceeded the client timeout."""


class ImdHttpError(ImdError):
    """The API returned a non-2xx status code."""

    def __init__(self, message: str, status_code: int, url: str):
        super().__init__(message)
        self.status_code = status_code
        self.url = url


class ImdEmptyResponse(ImdError):
    """The endpoint returned no usable data for the requested scope."""


class ImdMalformedResponse(ImdError):
    """The payload is not a JSON object/list of the documented weather records."""


class ImdInvalidLocation(ImdError):
    """A requested district/station does not appear in the IMD response."""


# --------------------------------------------------------------------------- #
# Small parsing helpers
# --------------------------------------------------------------------------- #
def _num(value):
    """Best-effort float parse. Returns None when the value is empty/unparseable.

    IMD returns numbers as strings that sometimes carry a trailing '%' or stray
    whitespace (e.g. "-100%", "0.00"). Missing markers come back as "N/A", "" or
    "-" and must translate to None, not to a fabricated number.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"n/a", "na", "--", "-", "nil", "nd", "nan", "none"}:
        return None
    s = s.rstrip("%").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _clean_text(value):
    if value is None:
        return ""
    return re.sub(r"[\r\n]+", " ", str(value)).strip()


def _loc_key(value):
    """Normalized comparison key for location names ('Kanpur Nagar' -> 'KANPURNAGAR')."""
    return re.sub(r"\s+", "", str(value or "").upper())


# --------------------------------------------------------------------------- #
# Response shape handling
# --------------------------------------------------------------------------- #
def _coerce_records(payload):
    """Tolerate the response envelopes seen across IMD endpoints.

    Some endpoints return a bare JSON list, others a single object, and some wrap
    records under a 'data' key (e.g. the sunmoon sample in the reference).
    Returns a list of record dicts (possibly empty).
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return payload["data"]
        if payload:
            return [payload]
    return []


def _require_records(records, endpoint):
    if not isinstance(records, list):
        raise ImdMalformedResponse(f"{endpoint}: expected a JSON list of weather records")
    for rec in records:
        if not isinstance(rec, dict):
            raise ImdMalformedResponse(f"{endpoint}: record is not a JSON object: {rec!r}")
    return records


# --------------------------------------------------------------------------- #
# Record normalizers (documented IMD fields -> stable Bronze schema)
# --------------------------------------------------------------------------- #
def _norm_current(rec: dict) -> dict:
    raw_station = rec.get("Station") or rec.get("Station Id") or rec.get("station_id")
    if raw_station is None:
        raise ImdMalformedResponse("current_wx: record missing required 'Station' field")
    return {
        "station_id": _clean_text(rec.get("Station Id") or rec.get("station_id")),
        "station": _clean_text(raw_station),
        "date": _clean_text(rec.get("Date of Observation") or rec.get("Date")),
        "time_utc": _clean_text(rec.get("Time of Observation") or rec.get("Time")),
        "temp_c": _num(rec.get("Temperature") or rec.get("temperature")),
        "humidity_pct": _num(rec.get("Humidity") or rec.get("humidity")),
        "rainfall_24h_mm": _num(rec.get("Last 24 hrs Rainfall") or rec.get("Rainfall")),
        "wind_speed_kmh": _num(rec.get("Wind Speed")),
        "wind_direction_code": _clean_text(rec.get("Wind Direction")),
        "mslp_hpa": _num(rec.get("M.S.L.P") or rec.get("MSLP") or rec.get("mslp")),
        "weather_code": _clean_text(rec.get("Weather Code")),
        "nebulosity": _clean_text(rec.get("Nebulosity")),
    }


def _norm_forecast(rec: dict) -> list:
    raw_name = rec.get("Station_Name") or rec.get("Station")
    if raw_name is None:
        raise ImdMalformedResponse("cityforecast: record missing required 'Station_Name' field")
    station_code = _clean_text(rec.get("Station_Code"))
    station_name = _clean_text(raw_name)
    base_date = rec.get("Date")
    rows = []
    observed_fields = {
        1: {
            "temp_max_c": rec.get("Today_Max_temp"),
            "temp_min_c": rec.get("Today_Min_temp"),
            "forecast_text": rec.get("Todays_Forecast"),
            "rh_0830_pct": rec.get("Relative_Humidity_at_0830"),
            "rh_1730_pct": rec.get("Relative_Humidity_at_1730"),
            "rainfall_24h_mm": rec.get("Past_24_hrs_Rainfall"),
        }
    }
    for day in range(2, 8):
        observed_fields[day] = {
            "temp_max_c": rec.get(f"Day_{day}_Max_Temp"),
            "temp_min_c": rec.get(f"Day_{day}_Min_temp"),
            "forecast_text": rec.get(f"Day_{day}_Forecast"),
            "rh_0830_pct": None,
            "rh_1730_pct": None,
            "rainfall_24h_mm": None,
        }

    date_0 = None
    try:
        date_0 = datetime.strptime(_clean_text(base_date), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        raise ImdMalformedResponse(
            f"cityforecast: unparseable issue date for station {station_name}: {base_date!r}"
        ) from None

    for day in range(1, 8):
        f = observed_fields[day]
        rows.append({
            "station_code": station_code,
            "station_name": station_name,
            "date": (date_0 + timedelta(days=day - 1)).isoformat(),
            "forecast_day": day,
            "temp_max_c": _num(f["temp_max_c"]),
            "temp_min_c": _num(f["temp_min_c"]),
            "forecast_text": _clean_text(f.get("forecast_text")),
            "rh_0830_pct": _num(f["rh_0830_pct"]),
            "rh_1730_pct": _num(f["rh_1730_pct"]),
            "rainfall_24h_mm": _num(f["rainfall_24h_mm"]),
        })
    return rows


_RAINFALL_GROUPS = (
    ("daily", "Daily Actual", "Daily Normal", "Daily Departure Per", "Daily Category"),
    ("weekly", "Weekly Actual", "Weekly Normal", "Weekly Departure Per", "Weekly Category"),
    ("cumulative", "Cumulative Actual", "Cumulative Normal", "Cumulative Departure Per", "Cumulative Category"),
    ("monthly", "Monthly Actual", "Monthly Normal", "Monthly Departure Per", "Monthly Category"),
)


def _norm_rainfall(rec: dict) -> dict:
    raw_district = rec.get("District")
    if raw_district is None:
        raise ImdMalformedResponse("districtrainfall: record missing required 'District' field")
    out = {
        "obj_id": _clean_text(rec.get("OBJ_ID") or rec.get("Obj_id") or rec.get("obj_id")),
        "district": _clean_text(raw_district),
        "date": _clean_text(rec.get("Date")),
    }
    for prefix, actual_key, normal_key, dep_key, cat_key in _RAINFALL_GROUPS:
        out[f"{prefix}_actual_mm"] = _num(rec.get(actual_key))
        out[f"{prefix}_normal_mm"] = _num(rec.get(normal_key))
        out[f"{prefix}_departure_pct"] = _num(rec.get(dep_key))
        out[f"{prefix}_category"] = _clean_text(rec.get(cat_key))
    return out


def _norm_warning(rec: dict) -> dict:
    raw_district = rec.get("District")
    if raw_district is None:
        raise ImdMalformedResponse("districtwarning: record missing required 'District' field")
    out = {
        "obj_id": _clean_text(rec.get("Obj_id") or rec.get("OBJ_ID") or rec.get("obj_id")),
        "district": _clean_text(raw_district),
        "date": _clean_text(rec.get("Date")),
        "time_utc": _clean_text(rec.get("UTC")),
    }
    for day in range(1, 6):
        raw_codes = rec.get(f"Day_{day}")
        codes = []
        for token in str(raw_codes or "").split(","):
            token = _clean_text(token)
            if not token:
                continue
            try:
                codes.append(int(token))
            except ValueError:
                codes.append(token)
        out[f"day_{day}_codes"] = codes
        out[f"day_{day}_color"] = _clean_text(rec.get(f"Day{day}_Color"))
    return out


_NORMALIZERS = {
    "current": _norm_current,
    "forecast": _norm_forecast,
    "rainfall": _norm_rainfall,
    "warning": _norm_warning,
}

_FILTER_FIELD = {
    "rainfall": "district",
    "warning": "district",
}


# --------------------------------------------------------------------------- #
# HTTP client with injectable transport (tests never touch the live API)
# --------------------------------------------------------------------------- #
class ImdClient:
    def __init__(self, api_key: Optional[str] = None, token: Optional[str] = None,
                 base_url: str = IMD_BASE_URL, timeout: int = DEFAULT_TIMEOUT_SECONDS,
                 session: Optional[requests.Session] = None):
        self.api_key = (api_key or os.environ.get(ENV_API_KEY, "") or "").strip()
        self.token = (token or os.environ.get(ENV_TOKEN, "") or "").strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session if session is not None else requests.Session()

    def is_configured(self) -> bool:
        return bool(self.api_key and self.token)

    def _headers(self) -> dict:
        return {API_KEY_HEADER: self.api_key, AUTH_HEADER: f"Bearer {self.token}"}

    def _endpoint(self, name: str) -> str:
        return {
            "current": ENDPOINT_CURRENT,
            "forecast": ENDPOINT_FORECAST,
            "rainfall": ENDPOINT_RAINFALL,
            "warning": ENDPOINT_WARNING,
        }[name]

    def _get(self, name: str, params: Optional[dict] = None):
        url = self._endpoint(name)
        try:
            resp = self._session.get(url, params=params, headers=self._headers(), timeout=self.timeout)
        except requests.Timeout as e:
            raise ImdTimeoutError(f"{name}: request to {url} timed out after {self.timeout}s") from e
        except requests.RequestException as e:
            raise ImdConnectionError(f"{name}: request to {url} failed: {e}") from e
        if resp.status_code in (401, 403):
            raise ImdAuthError(
                f"{name}: IMD rejected credentials (HTTP {resp.status_code}). "
                "Checked env vars: AGRIPULSE_IMD_API_KEY + AGRIPULSE_IMD_TOKEN. "
                "Confirm the API key is bound to this server's public IP and the token is valid."
            )
        if resp.status_code >= 400:
            raise ImdHttpError(f"{name}: HTTP {resp.status_code}", resp.status_code, url)
        try:
            return resp.json()
        except ValueError as e:
            raise ImdMalformedResponse(f"{name}: response is not valid JSON") from e

    def _pull(self, name: str, params: Optional[dict] = None,
              location: Optional[str] = None, state: Optional[str] = None) -> dict:
        """Fetch one endpoint and normalize records. Returns {endpoint, records, raw}."""
        if not self.is_configured():
            raise ImdAuthError(
                "IMD weather is not configured: set AGRIPULSE_IMD_API_KEY and AGRIPULSE_IMD_TOKEN "
                "(register at https://api.imd.gov.in/public/register.php and mint a JWT per the portal "
                "user guide). No IMD request was attempted."
            )
        endpoint = self._endpoint(name)
        payload = self._get(name, params)
        records = _require_records(_coerce_records(payload), name)
        if not records:
            raise ImdEmptyResponse(f"{name}: endpoint returned no records for {location or 'unscoped pull'}")

        normalized = []
        for rec in records:
            result = _NORMALIZERS[name](rec)
            if isinstance(result, list):
                normalized.extend(result)
            else:
                normalized.append(result)

        if location is not None and name in _FILTER_FIELD:
            filter_field = _FILTER_FIELD[name]
            key = _loc_key(location)
            matched = [r for r in normalized if key and _loc_key(r.get(filter_field)) == key]
            if not matched:
                available = sorted({str(r.get(filter_field)) for r in normalized})
                raise ImdInvalidLocation(
                    f"{name}: district '{location}' not found in the IMD response "
                    f"(state filter: {state or 'none'}). Available: {', '.join(available[:25])}"
                )
            normalized = matched

        return {"endpoint": endpoint, "records": normalized, "raw": payload}


def fetch_current_weather(station_id: Optional[str] = None, client: Optional[ImdClient] = None) -> list:
    """IMD current weather for a station (or all stations when id is omitted)."""
    c = client or ImdClient()
    params = {"id": station_id} if station_id else None
    return c._pull("current", params, location=station_id)["records"]


def fetch_forecast(station_id: Optional[str] = None, client: Optional[ImdClient] = None) -> list:
    """IMD 7-day city forecast for a station (or all stations when id is omitted)."""
    c = client or ImdClient()
    params = {"id": station_id} if station_id else None
    return c._pull("forecast", params, location=station_id)["records"]


def fetch_district_rainfall(district: Optional[str] = None, state: Optional[str] = None,
                            client: Optional[ImdClient] = None) -> list:
    """IMD district-wise rainfall; optionally filtered to one district (name-based)."""
    c = client or ImdClient()
    return c._pull("rainfall", location=district, state=state)["records"]


def fetch_district_warning(district: Optional[str] = None, state: Optional[str] = None,
                           client: Optional[ImdClient] = None) -> list:
    """IMD district-wise warnings; optionally filtered to one district (name-based)."""
    c = client or ImdClient()
    return c._pull("warning", location=district, state=state)["records"]


def imd_availability(client: Optional[ImdClient] = None) -> dict:
    """Report whether live IMD access is configured. NEVER makes a network call."""
    c = client if client is not None else ImdClient()
    missing = []
    if not c.api_key:
        missing.append(ENV_API_KEY)
    if not c.token:
        missing.append(ENV_TOKEN)
    if missing:
        return {
            "service": "imd",
            "configured": False,
            "missing_env": missing,
            "reason": (
                "Live IMD access needs registered credentials (register at "
                "https://api.imd.gov.in/public/register.php). Set the missing env vars; "
                "until then the dashboard/pipeline report IMD as not_configured and no "
                "live IMD request or fabricated value is generated."
            ),
        }
    return {"service": "imd", "configured": True, "api_key_header": API_KEY_HEADER, "auth_header": AUTH_HEADER}


# --------------------------------------------------------------------------- #
# Bronze writer with provenance (data/<country>/bronze/india_weather/)
# --------------------------------------------------------------------------- #
def _default_out_dir() -> Optional[Path]:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import config  # noqa: PLC0415
        return config.BRONZE / "india_weather"
    except Exception:  # pragma: no cover - only reached on a broken install
        return None


def write_bronze(kind: str, records: list, raw, endpoint: str,
                 location: Optional[str] = None, state: Optional[str] = None,
                 retrieved_at: Optional[str] = None,
                 is_fixture: bool = False, out_dir: Optional[Path] = None) -> Path:
    """Persist one IMD pull + provenance. Returns the written .json path.

    The active profile layout is data/india/bronze/india_weather/ (i.e.
    config.BRONZE. It is the India-profile equivalent of a bare data/bronze/
    directory, keeping every pull inside the country profile used by this branch.
    """
    at = retrieved_at or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    iso_at = f"{at[:4]}-{at[4:6]}-{at[6:8]}T{at[9:11]}:{at[11:13]}:{at[13:15]}Z"
    loc = re.sub(r"[^A-Za-z0-9_-]", "_", _loc_key(location)) if location else None

    payload = {
        "provenance": {
            "source": "imd-fixture" if is_fixture else "imd",
            "endpoint": endpoint,
            "retrieved_at": iso_at,
            "location": location,
            "date": iso_at[:10],
            "state": state,
            "is_fixture": is_fixture,
            "note": ("SYNTHETIC SANDBOX FIXTURE -- identical schema to the live IMD payload, "
                     "NEVER presented as live data.") if is_fixture else "",
        },
        "records": records,
        "raw": raw,
    }
    dest = Path(out_dir) if out_dir else Path(_default_out_dir() or "india_weather")
    dest.mkdir(parents=True, exist_ok=True)
    slug = "".join(p for p in [kind, at, loc] if p)
    out_file = dest / f"imd_{slug}.json"

    status = "synthetic_fixture" if is_fixture else "success"
    meta = {
        "source": payload["provenance"]["source"],
        "source_url": endpoint,
        "retrieved_at": iso_at,
        "location": location,
        "state": state,
        "date": iso_at[:10],
        "row_count": len(records),
        "status": status,
        "is_fixture": is_fixture,
    }
    meta_file = dest / f"imd_{slug}.meta.json"

    out_file.write_text(json.dumps(payload, indent=2))
    meta_file.write_text(json.dumps(meta, indent=2))
    logger.info(f"Wrote {out_file}")
    logger.info(f"Wrote provenance metadata {meta_file}")
    return out_file


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_kinds(value: str) -> list:
    kinds = value.split(",")
    if "all" in {k.lower() for k in kinds}:
        return ["current", "forecast", "rainfall", "warning"]
    return [k.strip().lower() for k in kinds if k.strip().lower() in _NORMALIZERS]


def main():
    parser = argparse.ArgumentParser(description="Fetch IMD weather into the bronze layer")
    parser.add_argument("--kind", default="all",
                        help="comma-separated kinds: current, forecast, rainfall, warning (default: all)")
    parser.add_argument("--district", help="district name filter (rainfall/warning)")
    parser.add_argument("--state", help="state/UT name (metadata + context only)")
    parser.add_argument("--station-id", help="IMD station id (current/forecast)")
    parser.add_argument("--out", help="override the bronze india_weather output directory")
    args = parser.parse_args()

    avail = imd_availability()
    if not avail["configured"]:
        logger.error("IMD not configured: %s", avail["reason"])
        sys.exit(1)

    client = ImdClient()
    out_dir = Path(args.out) if args.out else _default_out_dir()

    for kind in _parse_kinds(args.kind):
        location = args.district if kind in ("rainfall", "warning") else args.station_id
        try:
            result = client._pull(kind, location=location, state=args.state)
        except ImdError as e:
            logger.error("%s fetch failed: %s", kind, e)
            continue
        write_bronze(kind, result["records"], result["raw"], result["endpoint"],
                     location=location, state=args.state, out_dir=out_dir)
    logger.info("IMD ingestion complete. Output directory: %s", out_dir)


if __name__ == "__main__":
    main()