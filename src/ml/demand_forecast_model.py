"""
demand_forecast_model.py

Downstream ML module: consumes the Gold-layer region feature table produced by
silver_to_gold.py and trains a simple regression model predicting a region's
total agricultural export value from its weather profile. This is a prototype
stand-in for a real sales-forecasting model, built on the exact feature
contract a Databricks ML notebook would read from.

The algorithm is deliberately simple (scikit-learn LinearRegression on a few
weather features). This module now also persists machine-readable outputs:

    data/gold/forecast_results.parquet   (per-region actual vs predicted)
    catalog/model_report.json            (features, metrics, limitations)

Data-science honesty rules enforced here:
  * If the dataset is too small for a held-out test split, we explicitly
    label the reported metrics as IN-SAMPLE and warn in the report.
  * No metric is computed when it cannot be computed correctly.
  * The model is labelled a prototype, not a validated production forecaster.

Run directly:
    python src/ml/demand_forecast_model.py
"""

import json
import logging
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
GOLD = BASE / "data" / "gold"
CATALOG = BASE / "catalog"

# Minimum sample size before a held-out train/test split is attempted.
# With fewer rows than this the test set would be too small to be meaningful,
# so we fall back to reporting in-sample fit with an explicit warning.
MIN_ROWS_FOR_VALID_SPLIT = 15

FEATURE_COLS = ["avg_temp_c", "total_precip_mm", "avg_humidity_pct", "avg_windspeed"]
TARGET_COL = "total_exports_musd"


def load_gold_features() -> pd.DataFrame:
    """
    Reads the Gold partitioned dataset the same way a Databricks ML notebook
    would -- via pandas.read_parquet against the partitioned directory,
    which transparently picks up the region_name partition column.
    """
    path = GOLD / "region_daily_features"
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows, {df['region_name'].nunique()} regions from Gold layer")
    return df


def build_region_level_dataset(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily weather to region-level features aligned with the target (export value)."""
    agg = daily_df.groupby("region_name", observed=True).agg(
        avg_temp_c=("temp_avg_c", "mean"),
        total_precip_mm=("precipitation_mm", "sum"),
        avg_humidity_pct=("humidity_pct", "mean"),
        avg_windspeed=("windspeed_max_kmh", "mean"),
        total_exports_musd=("total_exports_musd", "max"),
        corn_musd=("corn_musd", "max"),
        wheat_musd=("wheat_musd", "max"),
    ).reset_index()
    return agg


def train_model(region_df: pd.DataFrame) -> dict:
    """
    Train the LinearRegression model and return a dict of model artefacts +
    evaluation metadata. Preserves the original in-sample methodology for
    small datasets and never reports a split-based metric that was not actually
    computed on a held-out set.
    """
    feature_cols = FEATURE_COLS
    X = region_df[feature_cols]
    y = region_df[TARGET_COL]

    sample_size = len(region_df)
    methodology = "in-sample fit"
    warning = None

    if sample_size < MIN_ROWS_FOR_VALID_SPLIT:
        # Too small for a held-out test split to be meaningful, so we train on
        # all data and report in-sample fit. Flagged explicitly rather than
        # dressing up a toy result as a validated model.
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
        "mae_musd": round(float(mae), 4),
        "r2": round(float(r2), 4),
        "mape_pct": round(float(mape) * 100.0, 4),
        "metrics_are_in_sample": sample_size < MIN_ROWS_FOR_VALID_SPLIT,
    }
    logger.info(
        f"[{methodology}] MAE: {mae:.1f} $M | R^2: {r2:.3f} | MAPE: {mape * 100:.1f}%"
    )
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
    """
    Build the per-region forecast result table: actual vs predicted export
    value plus the raw difference and percentage error. All fields are the
    actual model outputs -- nothing is fabricated.
    """
    model = model_info["model"]
    feature_cols = model_info["feature_cols"]
    preds = model.predict(region_df[feature_cols])

    predicted = np.asarray(preds, dtype=float)
    actual = region_df[TARGET_COL].to_numpy(dtype=float)

    frame = pd.DataFrame(
        {
            "region_name": region_df["region_name"].values,
            "actual_exports_musd": actual,
            "predicted_exports_musd": predicted,
        }
    )
    frame["difference_musd"] = frame["predicted_exports_musd"] - frame["actual_exports_musd"]
    # Percentage error is undefined when actual == 0; represent as NaN rather
    # than inventing a number.
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (frame["difference_musd"] / frame["actual_exports_musd"]) * 100.0
    frame["prediction_error_pct"] = np.where(np.isfinite(pct), pct, np.nan)

    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def write_forecast_outputs(forecast_df: pd.DataFrame, model_info: dict) -> None:
    """
    Persist the machine-readable outputs:
      * data/gold/forecast_results.parquet
      * catalog/model_report.json
    Uses deterministic paths relative to the repo root.
    """
    GOLD.mkdir(parents=True, exist_ok=True)
    CATALOG.mkdir(parents=True, exist_ok=True)

    forecast_path = GOLD / "forecast_results.parquet"
    forecast_df.to_parquet(forecast_path, index=False)
    logger.info(f"Wrote forecast results -> {forecast_path}")

    metrics = model_info["metrics"]
    report = {
        "model_name": "linear-regression-weather-to-export-prototype",
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
            "Prototype model: linear relationship between a short weather window and annual "
            "export value; not a validated production forecaster.",
            "Export figures are a single annual cross-section (2011 USDA dataset); there is no "
            "multi-year history, so the model cannot demonstrate temporal forecast skill.",
            "In-sample metrics (where reported) do not estimate out-of-sample performance.",
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = CATALOG / "model_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Wrote model report -> {report_path}")


def run_forecast() -> dict:
    """End-to-end forecast: load gold -> build features -> train -> persist outputs."""
    daily_df = load_gold_features()
    region_df = build_region_level_dataset(daily_df)
    model_info = train_model(region_df)
    forecast_df = build_forecast_frame(region_df, model_info)
    write_forecast_outputs(forecast_df, model_info)
    logger.info(
        f"Forecast complete: {len(forecast_df)} regions -> "
        f"{GOLD/'forecast_results.parquet'}, {CATALOG/'model_report.json'}"
    )
    return {"forecast": forecast_df, "model_info": model_info, "region_df": region_df}


def main():
    daily_df = load_gold_features()
    region_df = build_region_level_dataset(daily_df)
    print(region_df.to_string(index=False))
    model_info = train_model(region_df)
    forecast_df = build_forecast_frame(region_df, model_info)
    write_forecast_outputs(forecast_df, model_info)
    print(forecast_df.to_string(index=False))


if __name__ == "__main__":
    main()