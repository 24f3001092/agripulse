"""
india_agriculture.py

India Agricultural Production ingestion (Bronze layer).

Authoritative dataset:
    Directorate of Economics & Statistics (DE&S), Ministry of Agriculture &
    Farmers Welfare -- "District-wise, season-wise crop production statistics"
    (Area/Production/Yield by State > District > Crop > Season > Year, from 1997).

    OGD catalog : https://data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0
    GODL-India  : data.gov.in licensed (Government Open Data License).

    On sandboxes where the data.gov.in file host is unreachable, the same DES
    APY dataset is acquired through the India Data Portal CKAN resource
    (source_url below), or through a local copy via AGRIPULSE_AGRICULTURE_FILE.

Fields (source):
    id, year ("1997-1998" crop year), state_name, state_code, district_name,
    district_code (DES code -- NOT the LGD code), crop_name, crop_code,
    crop_type, season (Kharif/Rabi/Summer/Autumn/Winter/Whole Year/Total),
    area (Hectare), production (Tonnes), yield (Tonnes/Hectare).

Normalized Bronze frame adds year_start (int crop-year start) and applies the
canonical crop registry (catalog/india_crops.json). Canonicalization is
source-faithful: crops are NEVER merged just because names look similar, and
unknown future crops are preserved as-is (never dropped, never renamed).

No values are fabricated: missing production/yield are kept as nulls.
"""

from __future__ import annotations

import datetime as _dt
import functools as _functools
import json
import logging
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("india_agriculture")

# --------------------------------------------------------------------------- #
# Source metadata
# --------------------------------------------------------------------------- #
SOURCE_NAME = "des-moafw-crop-apy"
SOURCE_LABEL = (
    "Directorate of Economics & Statistics, Ministry of Agriculture & Farmers Welfare (GODL-India)"
)
SOURCE_CATALOG = "https://data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0"
# DE&S APY resource on the India Data Portal CKAN (GODL-India). Redirects to a
# signed object (s3.ap-southeast-1.wasabisys.com) that expires after ~1 hour --
# always use this CKAN URL, never the presigned link.
DEFAULT_SOURCE_URL = (
    "https://ckandev.indiadataportal.com/dataset/"
    "f2bbc28c-6c7c-462b-9064-ea4c4213d466/resource/"
    "f980409d-49a2-42ae-9eb0-182365005c04/download/crop-wise-area-production-yield.csv"
)
SOURCE_UNITS = {
    "area": "Hectare",
    "production": "Tonnes",
    "yield": "Tonnes/Hectare",
}

RAW_COLUMNS = [
    "id", "year", "state_name", "state_code", "district_name", "district_code",
    "crop_name", "crop_code", "crop_type", "season", "area",
    "area_unit", "production", "production_unit", "yield", "yield_unit",
]
REQUIRED_SOURCE_COLUMNS = [
    "year", "state_name", "district_name", "crop_name", "season", "area",
]

# Canonical Bronze/Silver columns (source selection, unit-suffixed values).
NORMALIZED_COLUMNS = [
    "id", "year", "year_start", "state", "state_code", "district",
    "district_code", "crop", "crop_type", "season", "area", "production", "yield",
]
DEDUP_KEY = ["state", "district", "crop", "season", "year"]

AGRI_BRONZE_DIR = config.BRONZE / "india_agriculture"
CATALOG_FILE = SRC.parent / "catalog" / "india_crops.json"

ENV_URL = "AGRIPULSE_AGRICULTURE_URL"
ENV_FILE = "AGRIPULSE_AGRICULTURE_FILE"


# --------------------------------------------------------------------------- #
# Typed errors
# --------------------------------------------------------------------------- #
class AgricultureError(Exception):
    """Base error for the India agriculture ingestion."""


class AgricultureConnectionError(AgricultureError):
    """Source unreachable."""


class AgricultureTimeoutError(AgricultureConnectionError):
    """Source timed out while downloading."""


class AgricultureDownloadError(AgricultureError):
    """Source returned a non-success HTTP status."""


class AgricultureInvalidSource(AgricultureError):
    """Local source file missing, empty, or missing required columns."""


# --------------------------------------------------------------------------- #
# Crop registry (source-faithful canonicalization)
# --------------------------------------------------------------------------- #
@_functools.lru_cache(maxsize=1)
def load_crop_catalog() -> dict:
    """Load catalog/india_crops.json (canonical registry derived from real data)."""
    if not CATALOG_FILE.exists():
        raise AgricultureInvalidSource(f"Crop catalog missing: {CATALOG_FILE}")
    with CATALOG_FILE.open(encoding="utf-8") as f:
        return json.load(f)


@_functools.lru_cache(maxsize=1)
def _canonical_map() -> dict:
    catalog = load_crop_catalog()
    by_name = {c["name"]: c for c in catalog.get("crops", [])}
    by_canonical = {c["canonical"]: c for c in catalog.get("crops", [])}
    mapping = dict(by_name)
    mapping.update(by_canonical)
    return mapping


def canonical_crop(name: object) -> tuple[str, str]:
    """Return (canonical_crop, category) for a source crop name.

    Source-faithful: names are only normalized for whitespace/case. Crops are
    NEVER merged on similar looks, and unknown crops are preserved verbatim so
    future crop additions never silently corrupt the catalog.
    """
    raw = " ".join(str(name).split()).strip()
    if not raw:
        return "", ""
    mapping = _canonical_map()  # cached per call-chain; fine for modest use
    entry = mapping.get(raw)
    if entry is None:
        logger.warning("Agriculture: crop %r not in catalog -- preserved as-is", raw)
        return raw, ""
    return str(entry.get("canonical", raw)), str(entry.get("category", "") or "")


def _year_start(year: object) -> int | None:
    text = " ".join(str(year).split()).strip()
    digits = ""
    for ch in text:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if len(digits) >= 4 else None


# --------------------------------------------------------------------------- #
# Acquisition (network or local file)
# --------------------------------------------------------------------------- #
def download_source(url: str, dest: Path, timeout: tuple = (10, 600)) -> Path:
    """Download the raw source CSV verbatim to `dest` (bytes preserved)."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
    except requests.exceptions.Timeout as exc:
        raise AgricultureTimeoutError(f"Timeout downloading {url}") from exc
    except requests.exceptions.RequestException as exc:
        raise AgricultureConnectionError(f"Connection error downloading {url}: {exc}") from exc

    if resp.status_code != 200:
        raise AgricultureDownloadError(
            f"Source returned HTTP {resp.status_code} for {url}"
        )
    with dest.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    logger.info("Agriculture: downloaded %s (%s bytes) -> %s", url, dest.stat().st_size, dest)
    return dest


def resolve_source(env=None) -> tuple[str | None, Path | None]:
    """Return (source_url, local_file) according to env/order of preference."""
    env = dict(env or __import__("os").environ)
    local = env.get(ENV_FILE)
    if local and Path(local).exists():
        return None, Path(local)
    url = env.get(ENV_URL) or DEFAULT_SOURCE_URL
    return url, None


def load_source_csv(path: Path) -> pd.DataFrame:
    """Read and structurally validate the raw APY CSV."""
    if not Path(path).exists():
        raise AgricultureInvalidSource(f"Source file missing: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise AgricultureInvalidSource(f"Unreadable CSV {path}: {exc}") from exc
    if df.empty:
        raise AgricultureEmptySource(path)  # noqa: F821 -- defined below
    missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise AgricultureInvalidSource(
            f"Source missing required columns {missing} (have {sorted(df.columns)})"
        )
    return df


class AgricultureEmptySource(AgricultureInvalidSource):
    """Source file contains no rows."""


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical Bronze agriculture frame from the raw CSV frame."""
    out = pd.DataFrame()
    out["id"] = df["id"] if "id" in df.columns else pd.Series(range(len(df)), index=df.index)
    out["year"] = df["year"].astype(str).str.strip()
    out["year_start"] = out["year"].map(_year_start)
    out["state"] = df["state_name"].astype(str).str.strip()
    out["state_code"] = pd.to_numeric(df["state_code"], errors="coerce").astype("Int64")
    out["district"] = df["district_name"].astype(str).str.strip()
    out["district_code"] = pd.to_numeric(df["district_code"], errors="coerce").astype("Int64")

    crops = df["crop_name"].map(canonical_crop)
    out["crop"] = crops.map(lambda x: x[0])
    out["crop_type"] = crops.map(lambda x: x[1])
    if "crop_type" in df.columns:
        out["crop_type"] = df["crop_type"].fillna("").astype(str).str.strip()

    out["season"] = df["season"].astype(str).str.strip()

    def _optional_numeric(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(pd.NA, index=df.index, dtype="float64")
        return pd.to_numeric(df[col], errors="coerce")

    out["area"] = pd.to_numeric(df["area"], errors="coerce")
    out["production"] = _optional_numeric("production")
    out["yield"] = _optional_numeric("yield")

    missing_prod = int(out["production"].isna().sum())
    logger.info(
        "Agriculture: normalized %s rows; %s rows with missing production (kept null)",
        len(out), missing_prod,
    )
    return out[NORMALIZED_COLUMNS]


# --------------------------------------------------------------------------- #
# Bronze writer with provenance
# --------------------------------------------------------------------------- #
def write_bronze(df: pd.DataFrame, source_url: str | None = None,
                 raw_file: Path | None = None, out_dir: Path | None = None) -> Path:
    """Write raw+normalized Bronze artifacts with full provenance."""
    out_dir = Path(out_dir or AGRI_BRONZE_DIR)
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_copy = None
    if raw_file is not None and Path(raw_file).exists():
        raw_copy = raw_dir / f"apy_raw_{now}{Path(raw_file).suffix}"
        raw_copy.write_bytes(Path(raw_file).read_bytes())

    norm_path = out_dir / f"apy_normalized_{now}.parquet"
    df.to_parquet(norm_path, index=False,
                  coerce_timestamps="us", allow_truncated_timestamps=True)

    meta = {
        "dataset": SOURCE_NAME,
        "status": "success",
        "source": SOURCE_LABEL,
        "source_catalog": SOURCE_CATALOG,
        "source_url": source_url or ("local-file:" + (str(raw_file) if raw_file else "unknown")),
        "raw_file": raw_copy.name if raw_copy else None,
        "normalized_file": norm_path.name,
        "ingested_at": now,
        "row_count": int(len(df)),
        "columns": NORMALIZED_COLUMNS,
        "units": SOURCE_UNITS,
        "notes": [
            "production/yield missing in source are preserved as null (never fabricated)",
            "district_code is the DE&S APY district code, NOT the LGD code",
            "district_code is the DE&S APY district code, NOT the LGD code (LGD crosswalk lives in catalog/india_regions.json)",
        ],
    }
    meta_path = out_dir / f"apy_normalized_{now}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Agriculture Bronze: %s rows -> %s", len(df), norm_path)
    return norm_path


def latest_normalized(out_dir: Path | None = None) -> Path | None:
    """Newest normalized Bronze parquet (or None when absent)."""
    files = sorted((out_dir or AGRI_BRONZE_DIR).glob("apy_normalized_*.parquet"))
    return files[-1] if files else None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Ingest DE&S/MoAFW crop Area-Production-Yield data (Bronze).")
    parser.add_argument("--url", default=None, help="Override source URL (or set AGRIPULSE_AGRICULTURE_URL)")
    parser.add_argument("--file", default=None, help="Local source CSV (or set AGRIPULSE_AGRICULTURE_FILE)")
    parser.add_argument("--out", default=None, help="Override Bronze output directory")
    args = parser.parse_args(argv)

    url, local = resolve_source({ENV_URL: args.url, ENV_FILE: args.file} if (args.url or args.file) else {})
    if local is None and url is not None:
        tmp = Path(config.DATA_ROOT / "bronze" / "india_agriculture" / "raw_work") / "apy_download.csv"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(b"")
        download_source(url, tmp)
        local = tmp

    df = load_source_csv(local)
    norm = normalize(df)
    write_bronze(norm, source_url=url, raw_file=local, out_dir=Path(args.out) if args.out else None)
    logger.info("Done. Reset the file: %s rows, %s columns.", len(norm), len(norm.columns))


if __name__ == "__main__":
    main()