"""
india_forecast.py

India crop-production forecasting, built around the REAL multi-year panel that
exists in this product (Phase 7 -- do NOT reuse the U.S. market model):

  * DE&S / MoAFW Area-Production-Yield: 26 crop years (1997-98 .. 2022-23),
    737 districts, 115 crops, Kharif/Rabi/Summer/Autumn/Winter/Whole Year/Total.
  * AGMARKNET daily mandi prices/arrivals: only 2026-04-01 .. 2026-05-31 (61
    days) -- NOT enough daily history for a temporally validated price/arrival
    forecast.
  * Open-Meteo weather: a single recent window (once-daily, disjoint from the
    agriculture years) -- NOT usable as a historical feature.

Audit conclusion (documented in the model report):
  * TARGET  = production (tonnes) for a (state, district, crop, season) cell
              in the next unobserved year -- the only dimension with decades
              of temporal history.
  * Model   = sklearn HistGradientBoostingRegressor (interpretable gradient
              boosting, depth-limited) on year/lag covariates, compared
              honestly against a naive persistence baseline (last observed
              production).
  * Horizon = one year ahead (2023-24), per cell.
  * Honesty = strict temporal split (no random shuffle: every training year
              strictly precedes validation and test years); out-of-sample
              metrics only; prototype label; no fabricated price/arrival
              values ever; no claim of production-readiness.

Outputs:
  data/india/gold/india_forecasts.parquet
  data/india/catalog/india_model_report.json

Run:
    python src/ml/india_forecast.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("india_forecast")

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

GOLD = config.GOLD
CATALOG = config.CATALOG

CROP_YEAR_PATH = GOLD / "india_crop_summary" / "india_crop_year.parquet"
FORECAST_OUT = GOLD / "india_forecasts.parquet"
REPORT_OUT = CATALOG / "india_model_report.json"

# A cell needs this many observed years before it is forecastable at all.
MIN_CELL_YEARS = 6
# Leading part of the sorted year grid kept for training (validation and test
# are the most recent 3 + 2 years respectively).
SPLIT_VAL_YEARS = 3
SPLIT_TEST_YEARS = 2

CELL_KEYS = ["state", "district", "crop", "season"]
# Native categorical cardinalities must stay <= HistGradientBoosting's 255-cap;
# district/state are therefore numeric-encoded (ordinals) instead.
CATEGORICAL_COLS = ["crop", "season"]
NUMERIC_FEATURES = [
    "year_offset",
    "district_ordinal",
    "state_ordinal",
    "lag1_production",
    "roll3_production",
    "lag1_area_log",
    "lag1_yield",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_COLS
TARGET_COL = "production_t"
FORECAST_YEAR = "2023-2024"
LAG1_ANCHOR_YEAR = "2022-2023"  # forecast lag1 must exist for this observed year


def audit(agri=None, mandi=None, weather=None) -> dict:
    """Dimension audit of the available historical data (Phase 7 FIRST AUDIT).

    DataFrames may be passed positionally for unit testing; when None each
    dimension is loaded from its live path (missing files become empty frames).
    """
    if mandi is None:
        m_path = GOLD / "india_market_summary" / "india_market_summary.parquet"
        m = pd.read_parquet(m_path) if m_path.exists() else None
    else:
        m = mandi
    if weather is None:
        w_path = GOLD / "india_region_features" / "india_region_features.parquet"
        weather = pd.read_parquet(w_path) if w_path.exists() else None
    if agri is None:
        a_path = config.SILVER / "india_agriculture.parquet"
        agri = pd.read_parquet(a_path) if a_path.exists() else pd.DataFrame()

    def _minmax(s):
        if s is None or len(s) == 0:
            return None
        return {"min": str(s.min()), "max": str(s.max()), "n": int(s.nunique())}

    years = _minmax(agri["year"]) if not agri.empty and "year" in agri else None
    mandi_dates = _minmax(m["latest_date"]) if m is not None and "latest_date" in m else None
    mandi_days = int(m["days_of_history"].sum()) if m is not None and "days_of_history" in m else 0
    weather_dates = _minmax(weather["date"]) if weather is not None and "date" in weather else None

    report = {
        "agriculture": {
            "rows": int(len(agri)),
            "years": years,
            "n_years": int(agri["year"].nunique()) if not agri.empty and "year" in agri else 0,
            "n_states": int(agri["state"].nunique()) if not agri.empty and "state" in agri else 0,
            "n_districts": int(agri["district"].nunique()) if not agri.empty and "district" in agri else 0,
            "n_crops": int(agri["crop"].nunique()) if not agri.empty and "crop" in agri else 0,
            "unit": "tonnes (production), hectare (area), tonnes/hectare (yield)",
        },
        "mandi": {
            "summary_rows": int(len(m)) if m is not None else 0,
            "date_window": mandi_dates,
            "total_record_days": int(mandi_days),
            "unit": "INR/quintal (prices), Metric Tonnes (arrivals)",
            "verdict": "insufficient history for temporal validation",
        },
        "weather": {
            "region_date_rows": int(len(weather)) if weather is not None else 0,
            "date_window": weather_dates,
            "verdict": "single recent window, temporally disjoint from agriculture years",
        },
    }

    sufficient = years is not None and years["n"] >= 10
    report["target_decision"] = (
        "production (tonnes, next unobserved year 2023-2024)"
        if sufficient
        else "no valid target -- agriculture history insufficient"
    )
    return report


def load_crop_year() -> pd.DataFrame:
    df = pd.read_parquet(CROP_YEAR_PATH)
    # year_start is the numeric anchor (e.g. 1997, 2022) already present in Silver.
    df = df.sort_values(CELL_KEYS + ["year_start"]).reset_index(drop=True)
    for col in CELL_KEYS:
        df[col] = df[col].astype(str).str.strip()
    df["production"] = pd.to_numeric(df["production"], errors="coerce")
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
    df = df.rename(columns={"production": TARGET_COL, "area": "area_ha", "yield": "yield_tph"})
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell lag features (strictly same-cell, chronologically ordered)."""
    df = df.copy()
    df["year_offset"] = df.groupby(CELL_KEYS, observed=True)["year_start"].transform(
        lambda s: s - s.min()
    )
    # Ordinal encodings consistent across the whole panel (fit AND forecast grid).
    df["district_ordinal"] = df["district"].astype("category").cat.codes
    df["state_ordinal"] = df["state"].astype("category").cat.codes

    def _log(x: pd.Series) -> pd.Series:
        with np.errstate(divide="ignore"):
            return np.log1p(pd.to_numeric(x, errors="coerce"))

    df = df.sort_values(CELL_KEYS + ["year_start"]).reset_index(drop=True)
    g = df.groupby(CELL_KEYS, observed=True)
    df["lag1_production"] = g[TARGET_COL].shift(1)
    df["lag2_production"] = g[TARGET_COL].shift(2)
    df["lag1_area_ha"] = g["area_ha"].shift(1)
    df["lag1_yield"] = g["yield_tph"].shift(1)
    df["lag1_available"] = df["lag1_production"].notna().astype(int)
    # Mean of the 3 previous years' production in the cell (excludes current year).
    df["roll3_production"] = df.groupby(CELL_KEYS, observed=True)[TARGET_COL].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )

    with np.errstate(divide="ignore"):
        df["lag1_area_log"] = np.log1p(df["lag1_area_ha"])

    # A target row needs a real observed production value; rows without lag1
    # (first observation of a cell) are lag SOURCES, not training targets.
    df = df[df[TARGET_COL].notna() & (df["lag1_available"] == 1)].copy()
    return df


def cell_history_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Rows per (state, district, crop, season) over the full panel."""
    return df.groupby(CELL_KEYS, observed=True).size().rename("n_year_season_rows").reset_index()


def temporal_splits(years: list[str]) -> dict[str, list[str]]:
    """Strict temporal split: every train year < every val < every test year."""
    ordered = sorted(years)
    split = len(ordered) - SPLIT_VAL_YEARS - SPLIT_TEST_YEARS
    return {
        "train": ordered[:split],
        "validate": ordered[split:split + SPLIT_VAL_YEARS],
        "test": ordered[split + SPLIT_VAL_YEARS:],
    }


def _grade_on(set_df: pd.DataFrame, y_true, pred) -> dict:
    """Metrics on the ORIGINAL tonnes scale for a set of rows."""
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mae = mean_absolute_error(y_true, pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    r2 = r2_score(y_true, pred)
    mae_pct = mae / max(float(np.mean(y_true)), 1e-9)
    pos = y_true >= 1.0
    mape = None
    mape_pos = int(pos.sum())
    safe = np.where(np.isfinite(pred) & (pred > 0), pred, np.nan)
    if pos.any():
        rel = np.abs((y_true[pos] - np.where(np.isfinite(safe[pos]), safe[pos], np.nan))) / y_true[pos]
        mape = float(np.nanmedian(rel) * 100.0)  # robust median APE
    return {
        "mae_t": round(float(mae), 2),
        "rmse_t": round(float(rmse), 2),
        "r2_vs_observed_mean": round(float(r2), 4),
        "mae_pct_of_mean": round(float(mae_pct) * 100.0, 4),
        "mape_median_pct": mape,
        "mape_on_rows": int(mape_pos),
        "n_rows": int(len(set_df)),
    }


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering + eligible-cell filtering (>= MIN_CELL_YEARS forecastable rows)."""
    built = build_features(df)
    history = cell_history_counts(built)
    eligible = set(map(tuple, history.loc[history["n_year_season_rows"] >= MIN_CELL_YEARS, CELL_KEYS].values))
    built = built[built[CELL_KEYS].apply(tuple, axis=1).isin(eligible)].copy()
    logger.info(f"Prepared {len(built)} forecastable rows across {len(history)} cells"
                f" ({int((history['n_year_season_rows'] >= MIN_CELL_YEARS).sum())} eligible)")
    return built


def run_fit(built: pd.DataFrame) -> dict:
    """Strict temporal split + GBM-vs-naive comparison on a prepared dataset."""
    if built.empty:
        raise ValueError("no forecastable rows after feature engineering")

    splits = temporal_splits(list(built["year"].unique()))
    train = built[built["year"].isin(splits["train"])]
    val = built[built["year"].isin(splits["validate"])]
    test = built[built["year"].isin(splits["test"])]

    def _fit_and_predict(X, y):
        model = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=300, max_depth=3, learning_rate=0.06,
            categorical_features=[ALL_FEATURES.index(c) for c in CATEGORICAL_COLS],
            random_state=42,
        )
        model.fit(X, y)
        return model

    X_train = train[ALL_FEATURES]
    y_train = train[TARGET_COL]
    model = _fit_and_predict(X_train, y_train)

    pred_val = model.predict(val[ALL_FEATURES])
    pred_test = model.predict(test[ALL_FEATURES])

    # Naive persistence baseline: last observed production in the cell.
    naive_val = val["lag1_production"].fillna(0)
    naive_test = test["lag1_production"].fillna(0)

    val_metrics = _grade_on(val, val[TARGET_COL], pred_val)
    test_metrics = _grade_on(test, test[TARGET_COL], pred_test)
    naive_val_metrics = _grade_on(val, val[TARGET_COL], naive_val)
    naive_test_metrics = _grade_on(test, test[TARGET_COL], naive_test)

    # Model selection on the VALIDATION set only; the test set is used once.
    better_mae = val_metrics["mae_t"] <= naive_val_metrics["mae_t"]
    chosen = "gradient_boosting" if better_mae else "naive_persistence"
    logger.info(
        f"Val MAE: model {val_metrics['mae_t']} vs naive {naive_val_metrics['mae_t']} (t) "
        f"-> choosing {chosen}"
    )

    return {
        "model": model,
        "splits": splits,
        "train_rows": int(len(train)),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "naive_metrics": {"validation": naive_val_metrics, "test": naive_test_metrics},
        "chosen_on_validation": chosen,
        "beat_naive_on_val": bool(better_mae),
        "val_df": val,
        "test_df": test,
        "pred_val": pred_val,
        "pred_test": pred_test,
    }


def final_model(df: pd.DataFrame) -> tuple:
    """Retrain on ALL observed years for the actual one-year-ahead forecasts."""
    X = df[ALL_FEATURES]
    y = df[TARGET_COL]
    model = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=300, max_depth=3, learning_rate=0.06,
        categorical_features=[ALL_FEATURES.index(c) for c in CATEGORICAL_COLS],
        random_state=42,
    )
    model.fit(X, y)
    return model, df[TARGET_COL]


def build_forecast_frame(df: pd.DataFrame, fit: dict, model, residual_std_log: float,
                        chosen: str = "gradient_boosting") -> pd.DataFrame:
    """One-year-ahead (2023-24) forecast per eligible cell with a multiplicate band."""
    cells_anchor = df[df["year"] == LAG1_ANCHOR_YEAR].copy()
    if cells_anchor.empty:
        raise ValueError(f"no observed rows for anchor year {LAG1_ANCHOR_YEAR}")

    history = cell_history_counts(df)
    grid = history[CELL_KEYS].merge(cells_anchor, on=CELL_KEYS, how="inner")

    if chosen == "naive_persistence":
        pred = np.maximum(grid[TARGET_COL], 0.0)
    else:
        Xf = grid[ALL_FEATURES]
        pred = np.maximum(model.predict(Xf), 0.0)
    # Multiplicative residual band: [pred/(1+k), pred*(1+k)].
    k = 1.96 * residual_std_log
    lo = np.maximum(pred / (1.0 + k), 0.0)
    hi = pred * (1.0 + k)

    out = grid[CELL_KEYS + ["state_code", "district_code", "crop_type", "year"]].copy()
    out = out.merge(history, on=CELL_KEYS, how="left")
    out["forecast_year"] = FORECAST_YEAR
    out["latest_observed_year"] = LAG1_ANCHOR_YEAR
    out["observed_last_tonnes"] = grid[TARGET_COL]
    out["forecast_production_tonnes"] = pred
    out["forecast_lower_tonnes"] = lo
    out["forecast_upper_tonnes"] = hi
    out["model_used"] = f"{chosen}-prototype"
    return out.drop(columns=["year"])


def write_no_data_report(reason: str) -> dict:
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_status": "not_generated",
        "target_variable": "production (tonnes)",
        "reason": reason,
        "required_upstream": ["data/india/gold/india_crop_summary/india_crop_year.parquet"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    REPORT_OUT.write_text(json.dumps(payload, indent=2))
    logger.warning(f"India forecast NOT generated ({reason}) -> {REPORT_OUT}")
    return payload


def validation_samples(fit: dict, cap: int = 1000) -> list[dict]:
    tdf = fit["test_df"]
    pred = fit["pred_test"]
    sample = pd.DataFrame({
        **_cell_dict(tdf), "year": tdf["year"].values,
        "observed_tonnes": tdf[TARGET_COL].to_numpy(float),
        "predicted_tonnes": np.asarray(pred, dtype=float),
    })
    rng = np.random.default_rng(0)
    if len(sample) > cap:
        sample = sample.iloc[rng.choice(len(sample), size=cap, replace=False)]
    return sample.to_dict(orient="records")


def _cell_dict(df: pd.DataFrame) -> dict:
    return {c: df[c].values for c in CELL_KEYS}


def run_forecast() -> dict:
    """End-to-end: audit -> fit -> report -> forecasts (or honest no_data)."""
    if not CROP_YEAR_PATH.exists():
        report = write_no_data_report("india_crop_year absent -- agriculture history unavailable")
        return {"status": "no_data", "report": report}

    df = load_crop_year()
    audit_report = audit()
    built = prepare_dataset(df)

    fit = run_fit(built)
    y_test = fit["test_df"][TARGET_COL].to_numpy(float)
    if fit["chosen_on_validation"] == "naive_persistence":
        chosen_pred = fit["test_df"]["lag1_production"].fillna(0).to_numpy(float)
    else:
        chosen_pred = fit["pred_test"]
    # Scale-aware band: log-space residual std of the CHOSEN model, so small
    # cells do not get absurd tonne-scale intervals.
    with np.errstate(divide="ignore"):
        res_log = np.log1p(y_test) - np.log1p(np.maximum(chosen_pred, 0.0))
    residual_std_log = float(np.std(res_log))

    final, _y = final_model(built)
    forecast = build_forecast_frame(built, fit, final, residual_std_log,
                                    chosen=fit["chosen_on_validation"])
    FORECAST_OUT.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_parquet(FORECAST_OUT, index=False)

    report = build_report(fit, audit_report, forecast, residual_std_log)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, indent=2))
    logger.info(f"Wrote india_forecasts.parquet ({len(forecast)} cells) -> {FORECAST_OUT}")
    logger.info(f"Wrote india_model_report.json -> {REPORT_OUT}")
    return {"status": "generated", "forecast": forecast, "report": report}


def build_report(fit: dict, audit_report: dict, forecast: pd.DataFrame, residual_std_log: float) -> dict:
    splits = fit["splits"]
    chosen = fit["chosen_on_validation"]
    return {
        "model_name": (
            "naive-persistence-crop-production-prototype" if chosen == "naive_persistence"
            else "hist-gradient-boosting-crop-production-prototype"
        ),
        "model_family": (
            "last-year persistence baseline (selected on validation)" if chosen == "naive_persistence"
            else "sklearn HistGradientBoostingRegressor (gradient boosting, depth-limited)"
        ),
        "model_status": "prototype",
        "production_ready": False,
        "target": {
            "variable": "production",
            "unit": "tonnes (source: DE&S / MoAFW Area-Production-Yield)",
            "next_unobserved_year": FORECAST_YEAR,
            "why_this_target": (
                "Production is the ONLY India dimension with a real multi-year "
                "time series (26 crop years 1997-98..2022-23 across 737 districts "
                "and 115 crops). It permits a genuinely temporal, leak-free "
                "train/validate/test split. Mandi prices/arrivals have only 61 "
                "daily records (2026-04-01..2026-05-31) and the weather window is "
                "a single recent month disjoint from the agriculture years, so "
                "neither supports defensible time-series forecasting here."
            ),
        },
        "alternatives_not_forecast": [
            {
                "variable": "commodity modal price (INR/quintal)",
                "reason": "only 61 daily records (one season) -- no multi-year daily history for a temporal split",
            },
            {
                "variable": "arrival quantity (Metric Tonnes)",
                "reason": "same 61-day window as prices; one season is not a forecasting history",
            },
            {
                "variable": "weather as a feature",
                "reason": "weather window (Aug-Sep 2026) starts after the agriculture data ends (2022-23); using it would invent history",
            },
        ],
        "data_period": {
            "start": audit_report["agriculture"]["years"]["min"] if audit_report["agriculture"]["years"] else None,
            "end": audit_report["agriculture"]["years"]["max"] if audit_report["agriculture"]["years"] else None,
            "n_years": int(audit_report["agriculture"]["n_years"]),
            "forecast_year": FORECAST_YEAR,
        },
        "dimensions": audit_report["agriculture"],
        "training_rows": int(fit["train_rows"]),
        "forecast_cells": int(len(forecast)),
        "validation_method": (
            "strict temporal split -- no random shuffle. All training years precede "
            "all validation years, which precede all test years. Model selection is "
            "done ONLY on the validation years; the test years are evaluated once. "
            f"train {splits['train'][0]}..{splits['train'][-1]} | "
            f"validate {splits['validate'][0]}..{splits['validate'][-1]} | "
            f"test {splits['test'][0]}..{splits['test'][-1]}"
        ),
        "splits": splits,
        "features": " + ".join(
            ["year_offset (within-cell trend)", "lag1 production (raw tonnes, same cell)",
             "3-year rolling mean production (lags 1-3)", "lag1 area (log ha)", "lag1 yield (t/ha)",
             "categorical: crop, season; ordinal: state, district"]
        ) + " | selected model on validation: " + fit["chosen_on_validation"],
        "metrics": {
            "test_mae_t": fit["test_metrics"]["mae_t"],
            "test_rmse_t": fit["test_metrics"]["rmse_t"],
            "test_r2_vs_observed_mean": fit["test_metrics"]["r2_vs_observed_mean"],
            "test_mape_median_pct": fit["test_metrics"]["mape_median_pct"],
            "test_rows": fit["test_metrics"]["n_rows"],
            "naive_baseline_test_mae_t": fit["naive_metrics"]["test"]["mae_t"],
            "naive_baseline_test_rmse_t": fit["naive_metrics"]["test"]["rmse_t"],
            "model_selected_on_validation": fit["chosen_on_validation"],
            "beat_naive_on_validation_mae": fit["beat_naive_on_val"],
        },
        "validation_metrics": fit["val_metrics"],
        "residual_std_log_scale": round(float(residual_std_log), 4),
        "limitations": [
            "Prototype model -- NOT production-ready, NOT guaranteed, NOT a profit signal.",
            (
                "Model selection on the held-out validation years selected: "
                + ("gradient boosting (beat last-year persistence by MAE)." if chosen == "gradient_boosting"
                   else "last-year persistence (gradient boosting did NOT beat persistence on validation "
                        "MAE; the ML prototype is disclosed as underperforming, not hidden). Forecasts in "
                        "india_forecasts.parquet are persistence-anchored, with gradient-boosting held-out "
                        "metrics still reported for transparency.")
            ),
            "Historical agriculture years are not weather-conditioned: the weather window "
            "postdates the agriculture data, so weather cannot be a feature without fabrication.",
            "Crop/area policy changes, irrigation, variety and credit shifts cannot be inferred "
            "from this panel; forecasts assume the historical regime continues.",
            "The 95% band is residual-based (held-out errors, log scale, multiplicative) and "
            "approximate, not a calibrated interval.",
            "Cells with fewer than 6 observed years are excluded; sparse districts are under-represented.",
            "MAPE is reported as a robust median only over rows with >= 1 tonne observed production.",
            "Price and arrival forecasting was NOT attempted (61 days of daily history is not a "
            "validation-capable time series) -- no price/arrival values are ever fabricated here.",
        ],
        "validation_samples": validation_samples(fit),
        "audit": audit_report,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main():
    result = run_forecast()
    if result["status"] == "no_data":
        print("India forecast GATED: no agriculture history -- see catalog/india_model_report.json")
        return
    fc = result["forecast"]
    print(f"forecast cells: {len(fc)}; sample: {fc.head(5).to_string(index=False)}")


if __name__ == "__main__":
    main()