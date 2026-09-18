"""
india_silver_to_gold.py

Integrated Indian agricultural Gold layer: Location + Crop + Date/Year.

Combines the three India Silver tables into one Gold layer:

  * data/india/gold/india_region_features/   -- region x date features: weather
      (temperature, precipitation, humidity, wind) joined with the observed
      APMC market state (median modal price in Rs./Quintal, total arrival in
      Metric Tonnes) reported on the canonical units only.
  * data/india/gold/india_market_summary/    -- per market/commodity/variety
      aggregates (reused from mandi_to_gold.py, the single source of truth).
  * data/india/gold/india_crop_summary/      -- Location + Crop (+ Date/Year)
      integrated summary: the Area-Production-Yield agriculture block joined
      with the observed market block and the observed weather block, plus the
      long-form india_crop_year.parquet at (state, district, crop, season,
      year_start) grain.

Everything the Gold layer reports comes from fields the Silver sources actually
carry. The feature set is therefore:

  State, District, Mandi, Crop, Season, Year, Production, Area, Yield,
  Price (Min/Modal/Max), Arrival, Temperature, Rainfall, Humidity,
  Weather Warning.

Honesty rules (same contract as the rest of the manual feed):
  * Market prices are averaged/compared ONLY within Rs./Quintal and arrivals
    ONLY within Metric Tonnes -- never mixing AGMARKNET's other units.
  * No traded_quantity is ever synthesized.
  * The IMD weather-warning loader is not configured, so the weather_warning
    column of the crop summary is intentionally null and the manifest records
    weather_warning_status=not_ingested. No fabricated advisory values.
  * Crop -> commodity matching is an exact, name-normalized join (casefold +
    whitespace collapse) with NO fuzzy matching; unmatched crops are reported
    as market_status=no_records, never guessed.
  * When a Silver input is absent, a no_data manifest is written for that
    output and the stage exits 0 (graceful skip).

Weather (Aug 2026 window) and mandi (Apr-May 2026 window) overlap on region and
date only where they actually do; the region_features table is an honest outer
join per region so both windows stay visible with nulls where a source has no
record.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import config  # noqa: E402
import mandi_to_gold  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("india_silver_to_gold")

SILVER = config.SILVER
GOLD = config.GOLD
CATALOG = config.CATALOG

WEATHER = SILVER / "weather.parquet"
AGRI = SILVER / "india_agriculture.parquet"
MANDI = SILVER / "india_mandi.parquet"
GEO = SILVER / "geo.parquet"

REGION_FEATURES_DIR = GOLD / "india_region_features"
CROP_SUMMARY_DIR = GOLD / "india_crop_summary"
MARKET_SUMMARY_DIR = GOLD / "india_market_summary"
LINEAGE_PATH = CATALOG / "gold_lineage.json"

# Canonical AGMARKNET units for cross-market comparison.
PRICE_UNIT = "Rs./Quintal"
ARRIVAL_UNIT = "Metric Tonnes"

FEATURE_SET = [
    "state", "district", "mandi", "crop", "season", "year",
    "production", "area", "yield",
    "price_min", "price_modal", "price_max", "arrival",
    "temperature", "rainfall", "humidity", "weather_warning",
]

REGION_FEATURES_COLUMNS = [
    "region_name", "state", "state_code", "district", "date",
    "temp_max_c", "temp_min_c", "temp_avg_c", "precipitation_mm",
    "humidity_pct", "windspeed_max_kmh",
    "median_modal_price_rq", "n_price_records", "n_price_markets",
    "n_price_commodities", "total_arrival_tonnes", "n_arrival_records",
    "n_arrival_markets",
]

WEATHER_OUTPUT_COLUMNS = [
    "temp_max_c", "temp_min_c", "temp_avg_c", "precipitation_mm",
    "humidity_pct", "windspeed_max_kmh",
]


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _normalize(value) -> str:
    """Exact-match normalization: casefold + trim + collapse whitespace."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).casefold().strip())


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_df = df.copy()
    for col in write_df.select_dtypes(include=["datetime64[ns]"]).columns:
        write_df[col] = write_df[col].astype("datetime64[us]")
    write_df.to_parquet(path, index=False, coerce_timestamps="us",
                        allow_truncated_timestamps=True)
    logger.info("Wrote %s (%s rows) -> %s", path.name, len(df), path)


def _region_from_geo(geo: pd.DataFrame) -> dict:
    """district name -> (region_name, state, state_code) from the geo table."""
    mapping: dict[str, dict] = {}
    for row in geo.to_dict(orient="records"):
        mapping.setdefault(str(row["district"]).casefold(), {
            "region_name": row["region_name"],
            "state": row["state_ut"],
            "state_code": row["state_code"],
        })
    return mapping


def build_region_features(weather: pd.DataFrame, mandi: pd.DataFrame,
                          geo: pd.DataFrame) -> pd.DataFrame:
    """Region x date features (weather block outer-joined with mandi block)."""
    region_map = _region_from_geo(geo)

    w = weather.copy()
    w["date"] = pd.to_datetime(w["date"]).dt.as_unit("ns")
    w = w.merge(
        geo[["region_name", "state_ut", "state_code", "district"]]
           .drop_duplicates("region_name")
           .rename(columns={"state_ut": "state"}),
        on="region_name", how="left",
    )
    w["temp_avg_c"] = (w["temp_max_c"] + w["temp_min_c"]) / 2.0

    m = mandi.copy()
    m["date"] = pd.to_datetime(m["date"]).dt.as_unit("ns")
    m["_district_fold"] = m["district"].astype(str).str.casefold()
    m["_region"] = m["_district_fold"].map(lambda d: region_map[d]["region_name"] if d in region_map else d)
    m["_state"] = m["_district_fold"].map(lambda d: region_map[d]["state"] if d in region_map else pd.NA)
    m["_state_code"] = m["_district_fold"].map(lambda d: region_map[d]["state_code"] if d in region_map else pd.NA)
    if not m.empty:
        m = m.dropna(subset=["date"])

    price = m[m["price_unit"] == PRICE_UNIT].copy()
    arrival = m[m["arrival_units"] == ARRIVAL_UNIT].copy()

    # Market blocks per region x date, canonical units only.
    price_out = pd.DataFrame(columns=["region_name", "date", "median_modal_price_rq",
                                      "n_price_records", "n_price_markets",
                                      "n_price_commodities"])
    if not price.empty:
        price_out = price.groupby(["_region", "date"], observed=True).agg(
            median_modal_price_rq=("modal_price", "median"),
            n_price_records=("modal_price", "count"),
            n_price_markets=("apmc", "nunique"),
            n_price_commodities=("commodity", "nunique"),
        ).reset_index().rename(columns={"_region": "region_name"})
    arrival_out = pd.DataFrame(columns=["_region", "date", "total_arrival_tonnes",
                                        "n_arrival_records", "n_arrival_markets"])
    if not arrival.empty:
        arrival_out = arrival.groupby(["_region", "date"], observed=True).agg(
            total_arrival_tonnes=("arrival_quantity", "sum"),
            n_arrival_records=("arrival_quantity", "count"),
            n_arrival_markets=("apmc", "nunique"),
        ).reset_index()

    # Outer join: weather window + market window per region x date, honest nulls.
    merged = w.merge(price_out, on=["region_name", "date"], how="outer")
    merged = merged.merge(arrival_out.rename(columns={"_region": "region_name"}),
                          on=["region_name", "date"], how="outer")

    # Recover location identity for mandi-only dates (outside the weather window)
    # from the geo reference -- the outer join leaves them null on the weather side.
    geo_by_region = (geo.drop_duplicates("region_name")[["region_name", "state_ut", "state_code", "district"]]
                     .rename(columns={"state_ut": "state"}))
    merged = merged.merge(geo_by_region, on="region_name", how="left", suffixes=("", "_geo"))
    for col in ("state", "state_code", "district"):
        merged[col] = merged[col].fillna(merged[f"{col}_geo"])
        merged = merged.drop(columns=[f"{col}_geo"])

    out = merged.copy()
    out["state"] = out["state"].fillna(pd.NA)
    out["state_code"] = out["state_code"].fillna(pd.NA)
    out["district"] = out["district"].fillna(pd.NA)
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["region_name", "date"]).reset_index(drop=True)
    return out[REGION_FEATURES_COLUMNS]


def _market_block(district_commodity: pd.DataFrame) -> pd.DataFrame:
    """Per (district, commodity) market aggregates within Rs./Quintal."""
    df = district_commodity[district_commodity["price_unit"] == PRICE_UNIT].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "district", "commodity", "n_market_records", "n_markets",
            "latest_market_date", "latest_modal_price", "latest_min_price",
            "latest_max_price", "avg_modal_price_7d", "modal_price_change_pct_7d",
        ])
    df["date"] = pd.to_datetime(df["date"]).dt.as_unit("ns")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["district", "commodity", "date"])

    rows = []
    for (district, commodity), grp in df.groupby(["district", "commodity"], observed=True):
        latest_date = grp["date"].max()
        latest = grp.loc[grp["date"] == latest_date].sort_values("apmc").iloc[0]
        window = grp[(latest_date - grp["date"]).dt.days <= 6]
        n_window_days = int(window["date"].dt.normalize().nunique())
        week_modal = window["modal_price"].dropna()
        latest_modal = float(latest["modal_price"]) if pd.notna(latest["modal_price"]) else None
        avg_modal_7d = float(week_modal.mean()) if len(week_modal) else None
        change = None
        if n_window_days >= 2 and avg_modal_7d not in (None, 0) and latest_modal is not None:
            change = (latest_modal - avg_modal_7d) / avg_modal_7d * 100.0
        rows.append({
            "district": district, "commodity": commodity,
            "n_market_records": int(grp["modal_price"].count()),
            "n_markets": int(grp["apmc"].nunique()),
            "latest_market_date": latest_date,
            "latest_modal_price": latest_modal,
            "latest_min_price": float(latest["min_price"]) if pd.notna(latest["min_price"]) else None,
            "latest_max_price": float(latest["max_price"]) if pd.notna(latest["max_price"]) else None,
            "avg_modal_price_7d": avg_modal_7d,
            "modal_price_change_pct_7d": change,
        })
    return pd.DataFrame(rows)


def _weather_block(region_features: pd.DataFrame) -> pd.DataFrame:
    """Per-district observed-weather means from the region features table."""
    wcols = [c for c in WEATHER_OUTPUT_COLUMNS if c != "temp_avg_c"]
    df = region_features.copy()
    if df.empty:
        return pd.DataFrame(columns=["district", "obs_temp_max_c", "obs_temp_min_c",
                                     "obs_precip_mm", "obs_humidity_pct", "weather_status"])
    has = df[df["temp_max_c"].notna()]["district"].dropna().unique()
    agg = (df[df["district"].notna()]
             .groupby("district", observed=True)[wcols]
             .mean()
             .rename(columns={"temp_max_c": "obs_temp_max_c",
                              "temp_min_c": "obs_temp_min_c",
                              "precipitation_mm": "obs_precip_mm",
                              "humidity_pct": "obs_humidity_pct"})
             .reset_index())
    reported = set(map(str.casefold, map(str, has)))
    agg["weather_status"] = agg["district"].astype(str).str.casefold().map(
        lambda d: "reported" if d in reported else "no_records")
    return agg


def build_crop_year(agri: pd.DataFrame) -> pd.DataFrame:
    """Long-form Location + Crop + Date(Year) grain, deduplicated."""
    df = agri.copy()
    df["year_start"] = pd.to_numeric(df["year_start"], errors="coerce")
    df = df.dropna(subset=["year_start"])
    df["year_start"] = df["year_start"].astype(int)
    key = ["state", "district", "crop", "crop_type", "season", "year_start"]
    df = df.drop_duplicates(subset=[c for c in key if c != "crop_type"])
    df = df.sort_values(key).reset_index(drop=True)
    cols = key + ["year", "area", "production", "yield", "state_code", "district_code"]
    return df[[c for c in cols if c in df.columns]]


def build_crop_summary(agri: pd.DataFrame, mandi: pd.DataFrame,
                       region_features: pd.DataFrame) -> pd.DataFrame:
    """Integrated Location + Crop (+ Date/Year) summary with ag + market + weather blocks."""
    crop_year = build_crop_year(agri)

    rows = []
    for (state, district, crop, crop_type), g in crop_year.groupby(
            ["state", "district", "crop", "crop_type"], observed=True):
        gs = g.sort_values(["year_start", "season"])
        latest = gs.iloc[-1]
        rows.append({
            "state": state, "district": district, "crop": crop, "crop_type": crop_type,
            "state_code": _first_non_null(g["state_code"]),
            "district_code": _first_non_null(g["district_code"]),
            "n_year_season_rows": int(len(g)),
            "years_observed": int(g["year_start"].nunique()),
            "first_year": int(g["year_start"].min()),
            "last_year": int(g["year_start"].max()),
            "latest_year": int(latest["year_start"]),
            "latest_season": str(latest["season"]),
            "latest_area_ha": _float_or_none(latest.get("area")),
            "latest_production_t": _float_or_none(latest.get("production")),
            "latest_yield_t_per_ha": _float_or_none(latest.get("yield")),
            "total_area_ha": float(g["area"].sum()) if "area" in g else None,
            "total_production_t": _sum_or_none(g["production"]) if "production" in g else None,
        })
    summary = pd.DataFrame(rows)

    # Precompute market + weather lookups (dict) so the crop loop is O(rows).
    mkt = _market_block(mandi)
    market_by = {}
    if not mkt.empty:
        for rec in mkt.to_dict(orient="records"):
            market_by[(str(rec["district"]).casefold(), _normalize(rec["commodity"]))] = rec

    wb = _weather_block(region_features)
    weather_by = {}
    if not wb.empty:
        for rec in wb.to_dict(orient="records"):
            weather_by[str(rec["district"]).casefold()] = rec

    merged = []
    for row in summary.to_dict(orient="records"):
        district_fold = str(row["district"]).casefold()
        market_row = market_by.get((district_fold, _normalize(row["crop"])))
        weather_row = weather_by.get(district_fold)

        merged.append({
            **row,
            "market_status": "matched" if market_row is not None else "no_records",
            "matched_commodity": market_row["commodity"] if market_row is not None else None,
            "n_market_records": int(market_row["n_market_records"]) if market_row is not None else 0,
            "n_markets": int(market_row["n_markets"]) if market_row is not None else 0,
            "latest_market_date": market_row["latest_market_date"] if market_row is not None else pd.NaT,
            "latest_modal_price": _float_or_none(market_row["latest_modal_price"]) if market_row is not None else None,
            "latest_min_price": _float_or_none(market_row["latest_min_price"]) if market_row is not None else None,
            "latest_max_price": _float_or_none(market_row["latest_max_price"]) if market_row is not None else None,
            "avg_modal_price_7d": _float_or_none(market_row["avg_modal_price_7d"]) if market_row is not None else None,
            "modal_price_change_pct_7d": _float_or_none(market_row["modal_price_change_pct_7d"]) if market_row is not None else None,
            "weather_status": weather_row["weather_status"] if weather_row is not None else "no_records",
            "obs_temp_max_c": _float_or_none(weather_row["obs_temp_max_c"]) if weather_row is not None else None,
            "obs_temp_min_c": _float_or_none(weather_row["obs_temp_min_c"]) if weather_row is not None else None,
            "obs_precip_mm": _float_or_none(weather_row["obs_precip_mm"]) if weather_row is not None else None,
            "obs_humidity_pct": _float_or_none(weather_row["obs_humidity_pct"]) if weather_row is not None else None,
            "weather_warning": None,
        })
    out = pd.DataFrame(merged)
    out["latest_market_date"] = pd.to_datetime(out["latest_market_date"])
    return out


def _float_or_none(v):
    if v is None or pd.isna(v):
        return None
    return float(v)


def _first_non_null(s) -> object | None:
    vals = s.dropna()
    return None if vals.empty else vals.iloc[0]


def _sum_or_none(s) -> float | None:
    vals = s.dropna()
    return float(vals.sum()) if not vals.empty else None


def _write_manifest(dir_path: Path, manifest: dict) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def make_manifest(dataset: str, source: str, inputs: list[str], outputs: list[str],
                  schema: list[str], meta: dict) -> dict:
    manifest = {
        "status": "success",
        "dataset": dataset,
        "source": source,
        "input": inputs,
        "output": outputs,
        "schema": schema,
        "generated_at": _now_iso(),
        "feature_set": FEATURE_SET,
    }
    manifest.update(meta)
    return manifest


def run_region_features() -> dict:
    """Build india_region_features/ from weather + mandi + geo."""
    present = [p for p in (WEATHER, MANDI, GEO) if p.exists()]
    missing = [str(p) for p in (WEATHER, MANDI, GEO) if not p.exists()]
    if len(present) < 3:
        manifest = {"status": "no_data",
                    "reason": "India Silver weather/mandi/geo not all present; region features skipped",
                    "missing": missing, "generated_at": _now_iso()}
        REGION_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        (REGION_FEATURES_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info("Region features skipped (missing inputs: %s)", missing)
        return manifest

    weather = pd.read_parquet(WEATHER)
    mandi = pd.read_parquet(MANDI)
    geo = pd.read_parquet(GEO)
    df = build_region_features(weather, mandi, geo)
    out_path = REGION_FEATURES_DIR / "india_region_features.parquet"
    _write_parquet(out_path, df)

    n_weather_days = int(df["temp_max_c"].notna().sum())
    n_price_days = int(df["n_price_records"].fillna(0).gt(0).sum())
    manifest = make_manifest(
        dataset="india_region_features",
        source=("Open-Meteo daily weather for monitored India districts (GODL "
                "context) + AGMARKNET daily APMC price/arrival records via the "
                "India Data Portal -- GODL-India; location mapping from "
                "catalog/india_regions.json"),
        inputs=[str(WEATHER), str(MANDI), str(GEO)],
        outputs=[str(out_path), str(REGION_FEATURES_DIR / "manifest.json")],
        schema=REGION_FEATURES_COLUMNS,
        meta={
            "period": [df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d")],
            "regions": int(df["region_name"].nunique()),
            "region_days": int(len(df)),
            "weather_day_records": n_weather_days,
            "market_day_records": n_price_days,
            "units_rule": ("market prices aggregated ONLY within Rs./Quintal "
                           "(median modal) and arrivals ONLY within Metric "
                           "Tonnes (sum); other AGMARKNET units excluded"),
            "windows_note": ("weather and mandi windows are reported on the same "
                             "region x date grid with an honest outer join; where "
                             "one source has no record the cells are null"),
            "no_traded_quantity": "AGMARKNET does not report traded quantity; none is synthesized",
        },
    )
    _write_manifest(REGION_FEATURES_DIR, manifest)
    logger.info("Region features complete: %s rows", len(df))
    return manifest


def run_market_summary() -> dict:
    """Rebuild india_market_summary/ via mandi_to_gold (single source of truth)."""
    mandi_to_gold.main()
    manifest_path = mandi_to_gold.GOLD_SUMMARY_DIR / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {"status": "no_data", "reason": "market summary not produced"}


def run_crop_summary() -> dict:
    """Build india_crop_summary/ from agriculture + mandi + region features."""
    if not AGRI.exists():
        manifest = {"status": "no_data",
                    "reason": "India Silver agriculture table absent; crop summary skipped "
                              "(run src/ingestion/india_agriculture.py first)",
                    "generated_at": _now_iso()}
        CROP_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
        (CROP_SUMMARY_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info("Crop summary skipped (no agriculture silver)")
        return manifest

    agri = pd.read_parquet(AGRI)
    region_features_path = REGION_FEATURES_DIR / "india_region_features.parquet"
    if region_features_path.exists():
        region_features = pd.read_parquet(region_features_path)
    else:
        region_features = pd.DataFrame(columns=REGION_FEATURES_COLUMNS)
    if MANDI.exists():
        mandi = pd.read_parquet(MANDI)
    else:
        mandi = pd.DataFrame(columns=["district", "commodity", "price_unit",
                                      "modal_price", "min_price", "max_price",
                                      "arrival_quantity", "apmc", "date"])

    summary = build_crop_summary(agri, mandi, region_features)
    crop_year = build_crop_year(agri)

    summary_path = CROP_SUMMARY_DIR / "india_crop_summary.parquet"
    crop_year_path = CROP_SUMMARY_DIR / "india_crop_year.parquet"
    _write_parquet(summary_path, summary)
    _write_parquet(crop_year_path, crop_year)

    n_weather = int(summary["weather_status"].eq("reported").sum())
    n_market = int(summary["market_status"].eq("matched").sum())
    manifest = make_manifest(
        dataset="india_crop_summary",
        source=("DE&S/MoAFW crop Area-Production-Yield (GODL-India) + AGMARKNET "
                "daily APMC price/arrival records via the India Data Portal "
                "(GODL-India) + Open-Meteo daily weather for monitored districts"),
        inputs=[str(AGRI), str(MANDI), str(REGION_FEATURES_DIR / "india_region_features.parquet")],
        outputs=[str(summary_path), str(crop_year_path),
                 str(CROP_SUMMARY_DIR / "manifest.json")],
        schema=[str(c) for c in summary.columns],
        meta={
            "grain": "state, district, crop (summary) and state, district, crop, "
                     "season, year_start (india_crop_year.parquet)",
            "crop_rows": int(len(summary)),
            "crop_year_rows": int(len(crop_year)),
            "with_market_match": n_market,
            "with_weather": n_weather,
            "market_match_rule": ("exact name-normalized join between agriculture "
                                  "crop and AGMARKNET commodity (casefold + "
                                  "whitespace collapse); NO fuzzy matching"),
            "weather_warning_status": "not_ingested",
            "weather_warning_note": ("IMD weather-warning loader is not configured; "
                                     "the weather_warning column is intentionally "
                                     "null -- no fabricated advisory values"),
            "units_rule": ("market prices aggregated ONLY within Rs./Quintal; "
                           "arrivals ONLY within Metric Tonnes"),
            "no_traded_quantity": "AGMARKNET does not report traded quantity; none is synthesized",
            "disclaimer": ("Observed wholesale prices and arrivals from official "
                           "AGMARKNET records and DE&S/MoAFW statistics. Analytical "
                           "signals only -- not investment advice."),
        },
    )
    _write_manifest(CROP_SUMMARY_DIR, manifest)
    logger.info("Crop summary complete: %s rows (%s market-matched)",
                len(summary), n_market)
    return manifest


def run_lineage(outputs: list[dict]) -> dict:
    """Write data/india/catalog/gold_lineage.json (source/input/output/schema/generated_at)."""
    CATALOG.mkdir(parents=True, exist_ok=True)
    lineage = {
        "country": config.COUNTRY,
        "layer": "gold",
        "feature_set": FEATURE_SET,
        "generated_at": _now_iso(),
        "outputs": outputs,
    }
    LINEAGE_PATH.write_text(json.dumps(lineage, indent=2))
    logger.info("Gold lineage -> %s", LINEAGE_PATH)
    return lineage


def _lineage_entry(manifest: dict, table_dir: Path, dataset: str, title: str) -> dict:
    return {
        "id": dataset,
        "title": title,
        "source": manifest.get("source"),
        "input": manifest.get("input"),
        "output": manifest.get("output"),
        "schema": manifest.get("schema"),
        "status": manifest.get("status"),
        "generated_at": manifest.get("generated_at"),
        "manifest": str(table_dir / "manifest.json"),
    }


def main():
    region_manifest = run_region_features()
    market_manifest = run_market_summary()
    crop_manifest = run_crop_summary()

    outputs = [
        _lineage_entry(region_manifest, REGION_FEATURES_DIR, "india_region_features",
                       "India region x date features (weather + observed market block)"),
        _lineage_entry(market_manifest, MARKET_SUMMARY_DIR, "india_market_summary",
                       "India market/commodity/variety price & arrival summary"),
        _lineage_entry(crop_manifest, CROP_SUMMARY_DIR, "india_crop_summary",
                       "India Location + Crop (+ Date/Year) integrated crop summary"),
    ]
    run_lineage(outputs)
    logger.info("India integrated Gold layer complete.")


if __name__ == "__main__":
    main()