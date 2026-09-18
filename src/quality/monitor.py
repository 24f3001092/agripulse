"""
monitor.py
Basic pipeline monitoring: checks data freshness (is Bronze data recent
enough?) and row-count anomalies (did a source suddenly return far fewer
rows than expected?). Mirrors the JD's "Support the implementation of
basic monitoring and alerting for pipeline and data quality issues."

India artifact quality checks (added with the integrated Gold layer):
  * staleness      - gold manifests refreshed recently (pipeline processing);
                     the frozen AGMARKNET publication gap is informational
  * duplicates     - trade-day / season-aware identity duplicates
  * missing_geo    - empty state/district; districts absent from the geo catalog
  * impossible     - values outside physical/economic bounds (temperatures,
                     negative prices/arrivals/area/production/yield)
  * missing_prices - records without a modal price (informational: legitimate
                     nulls are surfaced, not silently hidden)
  * missing_weather- weather-window rows without temperature/humidity

In a real deployment this would push to Slack/PagerDuty/email; here it
writes a structured JSON health report and exits non-zero on failure so
it can gate an Airflow DAG or CI pipeline.

Active profile: India (data/india/...). Market data is optional -- while
the e-NAM / AGMARKNET loader is not configured, the health report merely
notes market as not_configured (this is NOT treated as a failure; nothing
is fabricated). Missing mandi/agriculture tables and missing weather values
similarly surface as informative INFO checks, never as fabricated data.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("monitor")

BASE = Path(__file__).resolve().parents[2]
BRONZE = config.BRONZE
SILVER = config.SILVER
GOLD = config.GOLD
CATALOG = config.CATALOG

# Minimum expected row counts derived from the ACTIVE country profile
# (India: one row per monitored district per weather-window day).
_N_REGIONS = config.DEFAULT_REGION_COUNT
EXPECTED_MIN_ROWS = {
    "weather": _N_REGIONS * 28,   # n districts x weather window
    "geo": _N_REGIONS - 1,        # India geo reference table
}

FRESHNESS_MAX_AGE_HOURS = 24

MANDI_TRADE_DAY_KEY = ["date", "state", "district", "apmc", "commodity", "variety"]
AGRI_IDENTITY_KEY = ["state", "district", "crop", "season", "year"]


def check_freshness() -> list:
    issues = []
    meta_files = sorted(BRONZE.glob("*.meta.json"))
    if not meta_files:
        issues.append({"check": "freshness", "status": "FAIL", "detail": "No ingestion metadata found in bronze layer"})
        return issues

    latest_meta = json.loads(meta_files[-1].read_text())
    ingested_at = datetime.strptime(latest_meta["ingested_at_utc"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - ingested_at).total_seconds() / 3600

    if age_hours > FRESHNESS_MAX_AGE_HOURS:
        issues.append({
            "check": "freshness", "status": "FAIL",
            "detail": f"Latest weather ingestion is {age_hours:.1f}h old (max allowed {FRESHNESS_MAX_AGE_HOURS}h)"
        })
    else:
        logger.info(f"Freshness OK: latest ingestion {age_hours:.1f}h ago")
    return issues


def check_row_counts() -> list:
    issues = []
    for name, min_rows in EXPECTED_MIN_ROWS.items():
        path = SILVER / f"{name}.parquet"
        if not path.exists():
            issues.append({"check": "row_count", "dataset": name, "status": "FAIL", "detail": "Silver file missing"})
            continue
        df = pd.read_parquet(path)
        if len(df) < min_rows:
            issues.append({
                "check": "row_count", "dataset": name, "status": "FAIL",
                "detail": f"{len(df)} rows < expected minimum {min_rows}"
            })
        else:
            logger.info(f"Row count OK for {name}: {len(df)} rows (min {min_rows})")
    return issues


def _empty_str_count(df: pd.DataFrame, cols: list[str]) -> int:
    return int(df[cols].fillna("").astype(str).apply(lambda s: s.str.strip() == "").any(axis=1).sum())


def _stale_hours(manifest: dict) -> float | None:
    stamp = manifest.get("generated_at")
    if not stamp:
        return None
    try:
        generated = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - generated).total_seconds() / 3600


def check_india_artifacts() -> dict:
    """
    India artifact quality checks over the Silver + Gold layers.

    Returns {"checks": [...], "observed": {...}, "issues": [...]} where
    `checks` carries PASS/INFO/FAIL per category and `issues` holds ONLY the
    gate-failing (FAIL) entries.
    """
    checks: list[dict] = []
    observed: dict = {}
    issues: list[dict] = []

    def add(category: str, dataset: str, status: str, detail: str):
        checks.append({"check": category, "dataset": dataset, "status": status, "detail": detail})
        if status == "FAIL":
            issues.append({"check": category, "dataset": dataset, "status": status, "detail": detail})

    # --- duplicates ------------------------------------------------------- #
    mandi_path = SILVER / "india_mandi.parquet"
    if mandi_path.exists():
        m = pd.read_parquet(mandi_path)
        observed["india_mandi_rows"] = int(len(m))
        dupes = int(m.duplicated(subset=MANDI_TRADE_DAY_KEY).sum())
        observed["india_mandi_duplicates"] = dupes
        add("duplicates", "india_mandi",
            "FAIL" if dupes else "PASS",
            f"{dupes} duplicate trade-day rows" if dupes else "no duplicate trade-day rows")
    else:
        add("duplicates", "india_mandi", "INFO", "silver india_mandi absent (no market data yet)")

    agri_path = SILVER / "india_agriculture.parquet"
    if agri_path.exists():
        a = pd.read_parquet(agri_path)
        observed["india_agriculture_rows"] = int(len(a))
        key = [c for c in AGRI_IDENTITY_KEY if c in a.columns]
        dupes = int(a.duplicated(subset=key).sum())
        observed["india_agriculture_duplicates"] = dupes
        add("duplicates", "india_agriculture",
            "FAIL" if dupes else "PASS",
            f"{dupes} season-aware duplicate rows" if dupes else "no season-aware duplicate rows")
    else:
        add("duplicates", "india_agriculture", "INFO", "silver india_agriculture absent")

    rf_path = GOLD / "india_region_features" / "india_region_features.parquet"
    if rf_path.exists():
        rf = pd.read_parquet(rf_path)
        observed["region_features_rows"] = int(len(rf))
        dupes = int(rf.duplicated(subset=["region_name", "date"]).sum())
        add("duplicates", "india_region_features",
            "FAIL" if dupes else "PASS",
            f"{dupes} duplicate (region_name, date) rows" if dupes else "no duplicate (region_name, date) rows")
    else:
        add("duplicates", "india_region_features", "INFO", "india_region_features absent")

    crop_path = GOLD / "india_crop_summary" / "india_crop_summary.parquet"
    if crop_path.exists():
        cs = pd.read_parquet(crop_path)
        observed["crop_summary_rows"] = int(len(cs))
        dupes = int(cs.duplicated(subset=["state", "district", "crop"]).sum())
        add("duplicates", "india_crop_summary",
            "FAIL" if dupes else "PASS",
            f"{dupes} duplicate (state, district, crop) rows" if dupes else "no duplicate (state, district, crop) rows")
    else:
        add("duplicates", "india_crop_summary", "INFO", "india_crop_summary absent")

    # --- missing geography ----------------------------------------------- #
    geo_path = SILVER / "geo.parquet"
    known_districts: set[str] = set()
    if geo_path.exists():
        known_districts = {str(d).casefold() for d in pd.read_parquet(geo_path)["district"].dropna()}

    if rf_path.exists():
        empty_geo = _empty_str_count(rf, ["state", "district"])
        add("missing_geo", "india_region_features",
            "FAIL" if empty_geo else "PASS",
            f"{empty_geo} rows with empty state/district" if empty_geo else "no empty state/district")
    if crop_path.exists():
        empty_geo = _empty_str_count(cs, ["state", "district"])
        add("missing_geo", "india_crop_summary",
            "FAIL" if empty_geo else "PASS",
            f"{empty_geo} rows with empty state/district" if empty_geo else "no empty state/district")
        if known_districts:
            unknown = int((~cs["district"].astype(str).str.casefold().isin(known_districts)).sum())
            observed["crop_summary_districts_not_in_geo"] = unknown
            add("missing_geo", "india_crop_summary",
                "FAIL" if unknown == len(cs) else "PASS",
                f"{unknown}/{len(cs)} crop rows without a monitored district "
                "(expected: weather/market blocks only exist for monitored districts)")

    # --- impossible values ---------------------------------------------- #
    if rf_path.exists():
        bad: list[str] = []
        if rf["temp_max_c"].notna().any() and ((rf["temp_max_c"] > 60) | (rf["temp_max_c"] < -60)).any():
            bad.append("temperature out of [-60, 60] C")
        if rf["humidity_pct"].notna().any() and ((rf["humidity_pct"] < 0) | (rf["humidity_pct"] > 100)).any():
            bad.append("humidity outside [0, 100]")
        if rf["median_modal_price_rq"].notna().any() and (rf["median_modal_price_rq"] < 0).any():
            bad.append("negative median modal price")
        if rf["total_arrival_tonnes"].notna().any() and (rf["total_arrival_tonnes"] < 0).any():
            bad.append("negative total arrival")
        add("impossible_values", "india_region_features",
            "FAIL" if bad else "PASS", "; ".join(bad) if bad else "no impossible values")
    if crop_path.exists():
        bad = []
        for col in ("latest_area_ha", "latest_production_t", "latest_yield_t_per_ha",
                    "latest_modal_price", "obs_temp_max_c", "obs_temp_min_c"):
            if col in cs.columns and cs[col].notna().any() and (cs[col] < 0).any():
                bad.append(f"{col} < 0")
        add("impossible_values", "india_crop_summary",
            "FAIL" if bad else "PASS", "; ".join(bad) if bad else "no impossible values")

    # --- missing prices (informational except when price channel vanished) #
    if mandi_path.exists():
        missing_price = int(m["modal_price"].isna().sum())
        observed["india_mandi_null_modal"] = missing_price
        price_days = int(rf["n_price_records"].fillna(0).gt(0).sum()) if rf_path.exists() else None
        observed["region_market_days"] = price_days
        status = "INFO" if missing_price or (price_days is None or price_days == 0) else "PASS"
        detail = (f"{missing_price} rows without modal price "
                  f"(AGMARKNET allows nulls)") if missing_price else (
                  "no market price days observed" if price_days == 0 else "price channel present")
        add("missing_prices", "india_mandi", status, detail)

    # --- missing weather (inside the weather window only) ---------------- #
    if rf_path.exists():
        window_dates = rf.loc[rf["temp_max_c"].notna(), "date"].drop_duplicates()
        if window_dates.empty:
            add("missing_weather", "india_region_features", "INFO", "no weather window rows present")
        else:
            in_window = rf[rf["date"].isin(window_dates)]
            missing_temp = int(in_window["temp_max_c"].isna().sum())
            missing_humidity = int(in_window["humidity_pct"].isna().sum())
            add("missing_weather", "india_region_features",
                "FAIL" if (missing_temp or missing_humidity) else "PASS",
                f"{missing_temp} weather-window rows missing temperature, "
                f"{missing_humidity} missing humidity" if (missing_temp or missing_humidity)
                else "weather-window rows carry complete temperature/humidity")

    # --- staleness (pipeline processing; publication gap is informational) #
    for name, table_dir in (("india_region_features", GOLD / "india_region_features"),
                            ("india_crop_summary", GOLD / "india_crop_summary"),
                            ("india_market_summary", GOLD / "india_market_summary")):
        manifest_path = table_dir / "manifest.json"
        if not manifest_path.exists():
            add("staleness", name, "INFO", "gold manifest absent")
            continue
        manifest = json.loads(manifest_path.read_text())
        age = _stale_hours(manifest)
        if age is None:
            add("staleness", name, "INFO", "gold manifest has no parseable generated_at")
        elif age > FRESHNESS_MAX_AGE_HOURS:
            add("staleness", name, "FAIL",
                f"{name} manifest generated {age:.1f}h ago (max allowed {FRESHNESS_MAX_AGE_HOURS}h)")
        else:
            add("staleness", name, "PASS", f"{name} manifest refreshed {age:.1f}h ago")
        last_data = manifest.get("last_data_date") or manifest.get("period")
        if last_data:
            observed[f"{name}_last_data"] = last_data

    return {"checks": checks, "observed": observed, "issues": issues}


def main():
    all_issues = check_freshness() + check_row_counts()
    india = check_india_artifacts()
    all_issues += india["issues"]

    # Market readiness: informational only -- absence is honest, not a fault.
    market = {
        "status": config.market_source_status(),
        "note": ("Market prices/arrivals are produced ONLY by an official e-NAM / AGMARKNET "
                 "loader once integrated. No market values are fabricated.")
    }

    CATALOG.mkdir(parents=True, exist_ok=True)
    report_path = CATALOG / "health_report.json"
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "country": config.COUNTRY,
        "status": "FAIL" if all_issues else "PASS",
        "issues": all_issues,
        "market_source": market,
        "india": {
            "artifact_checks": india["checks"],
            "observed": india["observed"],
        },
    }
    report_path.write_text(json.dumps(report, indent=2))

    if all_issues:
        logger.error(f"{len(all_issues)} monitoring issue(s) found -- see {report_path}")
        for issue in all_issues:
            logger.error(f"  - {issue}")
        sys.exit(1)
    else:
        logger.info(f"All monitoring checks passed -- report written to {report_path}")


if __name__ == "__main__":
    main()