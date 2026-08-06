"""
demand_forecast_model.py
Represents the "Data Scientist" side of the handoff: consumes the Gold-layer
feature table produced by silver_to_gold.py and trains a simple regression
model predicting a region's total agricultural export value from its
weather profile. This is a stand-in for Syngenta's real sales-forecasting
model, built on the exact same feature contract a Databricks ML notebook
would read from.

Deliberately kept simple (linear regression, few features) -- the point
is the pipeline->ML handoff and feature contract, not model sophistication.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("demand_forecast")

BASE = Path(__file__).resolve().parents[2]
GOLD = BASE / "data" / "gold"


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
    agg = daily_df.groupby("region_name").agg(
        avg_temp_c=("temp_avg_c", "mean"),
        total_precip_mm=("precipitation_mm", "sum"),
        avg_humidity_pct=("humidity_pct", "mean"),
        avg_windspeed=("windspeed_max_kmh", "mean"),
        total_exports_musd=("total_exports_musd", "max"),
        corn_musd=("corn_musd", "max"),
        wheat_musd=("wheat_musd", "max"),
    ).reset_index()
    return agg


def train_model(region_df: pd.DataFrame):
    feature_cols = ["avg_temp_c", "total_precip_mm", "avg_humidity_pct", "avg_windspeed"]
    X = region_df[feature_cols]
    y = region_df["total_exports_musd"]

    # Only 10 regions -- too small for a held-out test split to be meaningful,
    # so this trains on all data and reports in-sample fit. Flagged explicitly
    # rather than dressing up a toy result as a validated model.
    if len(region_df) < 15:
        logger.warning(f"Only {len(region_df)} regions available -- too small for a real "
                        f"train/test split. Reporting in-sample fit only. A production "
                        f"version needs many more regions/history before this number means anything.")
        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(X)
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        logger.info(f"In-sample MAE: {mae:.1f} $M | In-sample R^2: {r2:.3f}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        logger.info(f"Test MAE: {mae:.1f} $M | Test R^2: {r2:.3f}")

    coefs = dict(zip(feature_cols, model.coef_))
    logger.info(f"Feature coefficients: {coefs}")
    return model


def main():
    daily_df = load_gold_features()
    region_df = build_region_level_dataset(daily_df)
    print(region_df.to_string(index=False))
    train_model(region_df)


if __name__ == "__main__":
    main()
