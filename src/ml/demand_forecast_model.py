"""
demand_forecast_model.py

Downstream ML module for AgriPulse India. Consumes the Gold-layer region
feature table produced by silver_to_gold.py and trains a simple regression
model between a region's weather profile and its MARKET PRICE SIGNAL
(INR per quintal, e-NAM / AGMARKNET standard).

GATING (data-science honesty first):
  * A market forecast REQUIRES a real market target. Until the e-NAM /
    AGMARKNET loader is configured (see india_market_source.py), this module
    writes data/india/catalog/ml_status.json with model_status =
    "not_generated" and does NOT emit forecast_results.parquet or a
    model_report.json. NO forecast is fabricated.
  * When configured, the target is the observed region-level average market
    price (INR per quintal) aggregated from the Gold market columns.

Methodology honesty (when trained):
  * Small samples report metrics as IN-SAMPLE with an explicit warning.
  * No metric is computed when it cannot be computed correctly.
  * The model is labelled a prototype, never a validated production forecaster.

Run:
    python src/ml/demand_forecast_model.py
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("demand_forecast")

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

GOLD = config.GOLD
CATALOG = config.CATALOG

# Minimum sample size before a held-out train/test split is attempted.
MIN_ROWS_FOR_VALID_SPLIT = 15

FEATURE_COLS = ["avg_temp_c", "total_precip_mm", "avg_humidity_pct"]
# Region-level market target aggregated from Gold market columns (INR/quintal).
TARGET_COL = "market_price_signal_inr_per_quintal"


def market_enabled() -> bool:
    """Market-based forecasting only runs with a real e-NAM/AGMARKNET feed."""
    from config import market_enabled as _cfg_enabled
    return _cfg_enabled()


def load_gold_features() -> pd.DataFrame:
    """
    Reads the Gold partitioned dataset the same way a Databricks ML notebook
    would -- via pandas.read_parquet against the partitioned directory.
    """
    path = GOLD / "region_daily_features"
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows, {df['region_name'].nunique()} regions from Gold layer")
    return df


def build_region_level_dataset(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily weather (+ market signal when present) to region-level features."""
    agg = daily_df.groupby("region_name", observed=True).agg(
        avg_temp_c=("temp_avg_c", "mean"),
        total_precip_mm=("precipitation_mm", "sum"),
        avg_humidity_pct=("humidity_pct", "mean"),
    ).reset_index()
    if "avg_market_price_inr_per_quintal" in daily_df.columns:
        agg = agg.merge(
            daily_df.groupby("region_name", observed=True).agg(
                market_price_signal_inr_per_quintal=("avg_market_price_inr_per_quintal", "mean"),
                total_arrivals_quintal=("total_arrivals_quintal", "sum"),
            ).reset_index(),
            on="region_name", how="left",
        )
    return agg


def write_forecast_gate_status(reason: str) -> dict:
    """
    Market forecast cannot be produced without a real market target.
    Persist ml_status.json (machine-readable) so the dashboard can report,
    honestly, that no forecast model has been generated -- instead of showing
    a made-up number.
    """
    CATALOG.mkdir(parents=True, exist_ok=True)
    report = {
        "model_status": "not_generated",
        "target_variable": TARGET_COL,
        "reason": reason,
        "required_upstream": [
            "market source configured (catalog/pipeline_config.json -> market_source: configured)",
            "e-NAM / AGMARKNET loader registered (src/ingestion/india_market_source.py)",
            "data/india/silver/market.parquet present (market table through Bronze->Silver gate)",
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_path = CATALOG / "ml_status.json"
    out_path.write_text(json.dumps(report, indent=2))
    logger.warning(f"Forecast NOT generated (market target absent) -> {out_path}")
    return report


def train_model(region_df: pd.DataFrame) -> dict:
    """
    Train the LinearRegression model and return model artefacts + evaluation
    metadata. Preserves the original in-sample methodology for small datasets
    and never reports a split-based metric that was not actually computed on a
    held-out set.
    """
    feature_cols = FEATURE_COLS
    X = region_df[feature_cols]
    y = region_df[TARGET_COL]

    sample_size = len(region_df)
    methodology = "in-sample fit"
    warning = None

    if sample_size < MIN_ROWS_FOR_VALID_SPLIT:
        warning = (
            f"Only {sample_size} regions available (minimum {MIN_ROWS_FOR_VALID_SPLIT} for a "
            f"held-out split). Metrics are IN-SAMPLE (model trained and evaluated on the same "
            f"rows) and must not be read as out-of-sample accuracy. A production forecaster "
            f"needs many more regions and/or historical periods."
        )
        logger.warning(warning)
        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(X)
        eval_preds, eval_y = preds, y
        methodology = "in-sample fit (no held-out split -- sample too small)"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        eval_preds, eval_y = preds, y_test
        methodology = "80/20 train/test split (random_state=42), metrics on held-out test set"

    mae = mean_absolute_error(eval_y, eval_preds)
    r2 = r2_score(eval_y, eval_preds)
    mape = mean_absolute_percentage_error(eval_y, eval_preds)

    coefs = {name: round(float(c), 6) for name, c in zip(feature_cols, model.coef_)}
    metrics = {
        "mae_inr_per_quintal": round(float(mae), 2),
        "r2": round(float(r2), 4),
        "mape_pct": round(float(mape) * 100.0, 4),
        "metrics_are_in_sample": sample_size < MIN_ROWS_FOR_VALID_SPLIT,
    }
    logger.info(f"[{methodology}] MAE: {mae:.1f} INR/quintal | R^2: {r2:.3f} | MAPE: {mape * 100:.1f}%")
    logger.info(f"Feature coefficients: {coefs}")

    return {
        "model": model,
        "feature_cols": feature_cols,
        "eval_actual": np.asarray(eval_y),
        "eval_predicted": np.asarray(eval_preds),
        "metrics": metrics,
        "coefficients": coefs,
        "methodology": methodology,
        "small_sample_warning": warning,
        "row_count": int(sample_size),
    }


def build_forecast_frame(region_df: pd.DataFrame, model_info: dict) -> pd.DataFrame:
    """Per-region forecast table: actual vs predicted market price signal (INR/quintal)."""
    model = model_info["model"]
    feature_cols = model_info["feature_cols"]
    preds = model.predict(region_df[feature_cols])

    predicted = np.asarray(preds, dtype=float)
    actual = region_df[TARGET_COL].to_numpy(dtype=float)

    frame = pd.DataFrame(
        {
            "region_name": region_df["region_name"].values,
            "actual_signal_inr_per_quintal": actual,
            "predicted_signal_inr_per_quintal": predicted,
        }
    )
    frame["difference_inr_per_quintal"] = (
        frame["predicted_signal_inr_per_quintal"] - frame["actual_signal_inr_per_quintal"]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (frame["difference_inr_per_quintal"] / frame["actual_signal_inr_per_quintal"]) * 100.0
    frame["prediction_error_pct"] = np.where(np.isfinite(pct), pct, np.nan)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def write_forecast_outputs(forecast_df: pd.DataFrame, model_info: dict) -> None:
    """Persist forecast results + model report to data/india/* (market-configured path)."""
    GOLD.mkdir(parents=True, exist_ok=True)
    CATALOG.mkdir(parents=True, exist_ok=True)

    forecast_path = GOLD / "forecast_results.parquet"
    forecast_df.to_parquet(forecast_path, index=False)
    logger.info(f"Wrote forecast results -> {forecast_path}")

    metrics = model_info["metrics"]
    report = {
        "model_name": "linear-regression-weather-to-market-price-prototype",
        "model_family": "scikit-learn LinearRegression",
        "model_status": "prototype",
        "target_variable": TARGET_COL,
        "feature_names": model_info["feature_cols"],
        "row_count": model_info["row_count"],
        "region_count": int(forecast_df.shape[0]),
        "evaluation_methodology": model_info["methodology"],
        "metrics": metrics,
        "coefficients": model_info["coefficients"],
        "small_sample_warning": model_info["small_sample_warning"],
        "limitations": [
            "Prototype model: linear relationship between a short weather window and a "
            "regional market-price signal; not a validated production forecaster.",
            "Price signal is a single cross-section; there is no multi-year history, so the "
            "model cannot demonstrate temporal forecast skill.",
            "In-sample metrics (where reported) do not estimate out-of-sample performance.",
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = CATALOG / "model_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Wrote model report -> {report_path}")


def run_forecast() -> dict:
    """End-to-end forecast. Gated: market target must exist (real source only)."""
    daily_df = load_gold_features()
    region_df = build_region_level_dataset(daily_df)

    if not market_enabled() or TARGET_COL not in region_df.columns:
        reason = (
            "Market prices/arrivals are NOT yet integrated: the e-NAM / AGMARKNET source is "
            "an extension point that has not been registered as configured. Without a real "
            "market target (INR per quintal) no forecast can be trained -- training on "
            "fabricated numbers would be misleading."
        )
        status = write_forecast_gate_status(reason)
        return {
            "forecast": pd.DataFrame(columns=["region_name", "actual_signal_inr_per_quintal"]),
            "model_info": {"model_status": "not_generated", "gate_code": status["model_status"]},
            "region_df": region_df,
            "gated": True,
        }

    model_info = train_model(region_df)
    forecast_df = build_forecast_frame(region_df, model_info)
    write_forecast_outputs(forecast_df, model_info)
    logger.info(
        f"Forecast complete: {len(forecast_df)} regions -> "
        f"{GOLD / 'forecast_results.parquet'}, {CATALOG / 'model_report.json'}"
    )
    return {"forecast": forecast_df, "model_info": model_info, "region_df": region_df, "gated": False}


def main():
    result = run_forecast()
    if result.get("gated"):
        print("Forecast step is GATED (no real market target) -- see data/india/catalog/ml_status.json")
        return
    region_df = result["region_df"]
    forecast_df = result["forecast"]
    print(region_df.to_string(index=False))
    print(forecast_df.to_string(index=False))


if __name__ == "__main__":
    main()