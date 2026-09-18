"""
marketing_segmentation.py

Analytical regional segmentation for AgriPulse India, built ONLY from existing
Gold-layer data (no fabricated inputs).

With the market source still an extension point (e-NAM / AGMARKNET not yet
configured), the segmentation is a WEATHER-EXPOSURE analysis: it groups the
monitored districts by what the observed weather data shows about growing
conditions -- rainfall availability, heat stress and humidity -- using
deterministic, transparent RULE-BASED tercile thresholds computed from the
actual data. Every segment carries a `segment_reason` string citing the real
numbers behind the assignment.

Once a real market feed is integrated, the same module can extend the segments
with market-signal size/tiers (its U.S.-version counterpart grouped regions by
export size and crop mix).

IMPORTANT: the labels ('High-Rainfall', 'Heat-Stress', ...) are ANALYSIS
groupings for exploration only. They are NOT validated agronomic or commercial
classifications and must never be presented as such. Segments describe observed
weather exposure, not a guarantee of crop outcomes.

Output:
    data/india/gold/region_segments.parquet
    columns: region_name, segment, segment_reason, exposure_tier, dominant_factor

Run directly:
    python src/ml/marketing_segmentation.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("marketing_segmentation")

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

GOLD = config.GOLD


def load_region_features() -> pd.DataFrame:
    """One weather-exposure feature row per monitored Indian district."""
    daily_df = pd.read_parquet(GOLD / "region_daily_features")
    features = daily_df.groupby("region_name", observed=True).agg(
        avg_temp_c=("temp_avg_c", "mean"),
        total_precip_mm=("precipitation_mm", "sum"),
        avg_humidity_pct=("humidity_pct", "mean"),
    ).reset_index()
    logger.info(f"Loaded weather-exposure features for {len(features)} districts from Gold layer")
    return features


def _tercile_floor(values: pd.Series, pct: float) -> float:
    """Deterministic quantile threshold on the actual observed values."""
    return float(np.quantile(values, pct))


def _exposure_tier(precip_tier: str, heat_tier: str) -> str:
    """Combine precipitation and heat terciles into an overall exposure tier."""
    strength = {"High-Exposure": 3, "Moderate-Exposure": 2, "Low-Exposure": 1}
    p = 3 if precip_tier == "High-Rainfall" else (2 if precip_tier == "Moderate-Rainfall" else 1)
    h = 3 if heat_tier == "Heat-Stress" else (2 if heat_tier == "Moderate-Heat" else 1)
    score = max(p, h)
    for label, value in strength.items():
        if score == value:
            return label
    return "Moderate-Exposure"


def assign_segments(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based weather-exposure segmentation (terciles of the observed data).
    Every processed region always receives a non-null segment.
    """
    precip_sorted = feature_df["total_precip_mm"].sort_values(ascending=True).to_numpy(dtype=float)
    precip_low = _tercile_floor(precip_sorted, 0.333)
    precip_high = _tercile_floor(precip_sorted, 0.667)
    precip_median = float(np.median(precip_sorted))

    temp_sorted = feature_df["avg_temp_c"].sort_values(ascending=True).to_numpy(dtype=float)
    temp_low = _tercile_floor(temp_sorted, 0.333)
    temp_high = _tercile_floor(temp_sorted, 0.667)

    humid_median = float(np.median(feature_df["avg_humidity_pct"].to_numpy(dtype=float)))

    rows = []
    for _, row in feature_df.iterrows():
        precip = float(row["total_precip_mm"])
        if precip >= precip_high:
            precip_tier = "High-Rainfall"
        elif precip <= precip_low:
            precip_tier = "Water-Limited"
        else:
            precip_tier = "Moderate-Rainfall"

        temp = float(row["avg_temp_c"])
        if temp >= temp_high:
            heat_tier = "Heat-Stress"
        elif temp <= temp_low:
            heat_tier = "Cool-Regime"
        else:
            heat_tier = "Moderate-Heat"

        humidity = float(row["avg_humidity_pct"])
        humidity_label = "Humid" if humidity >= humid_median else "Dry-Air"

        # Dominant driver: whichever signal strays farther from its median.
        precip_delta = abs(precip - precip_median) / (precip_median if precip_median else 1.0)
        temp_delta = abs(temp - np.median(temp_sorted)) / max(abs(temp_low), abs(temp_high), 1.0)
        dominant_factor = "precipitation" if precip_delta >= temp_delta else "temperature"

        segment = f"{precip_tier} > {heat_tier} > {humidity_label}"
        exposure_tier = _exposure_tier(precip_tier, heat_tier)

        reason = (
            f"{row['region_name']}: {precip:.0f}mm total precipitation ({precip_tier.lower()}; "
            f"terciles <= {precip_low:.0f}mm / >= {precip_high:.0f}mm), {temp:.1f}C avg temp "
            f"({heat_tier.lower()}; terciles <= {temp_low:.1f}C / >= {temp_high:.1f}C), "
            f"{humidity:.0f}% humidity ({humidity_label}). Overall weather-exposure tier: "
            f"{exposure_tier} (dominant driver: {dominant_factor}). "
            f"This describes observed weather exposure over the window, not a forecast."
        )

        rows.append({
            "region_name": row["region_name"],
            "segment": segment,
            "segment_reason": reason,
            "exposure_tier": exposure_tier,
            "dominant_factor": dominant_factor,
        })

    return pd.DataFrame(rows)


def write_segments(segment_df: pd.DataFrame) -> Path:
    GOLD.mkdir(parents=True, exist_ok=True)
    out_path = GOLD / "region_segments.parquet"
    segment_df.to_parquet(out_path, index=False)
    logger.info(f"Wrote segmentation results -> {out_path}")
    return out_path


def run_segmentation() -> dict:
    """End-to-end segmentation: load gold features -> assign -> persist."""
    feature_df = load_region_features()
    segment_df = assign_segments(feature_df)
    write_segments(segment_df)
    logger.info(f"Segmentation complete: {len(segment_df)} districts segmented")
    return {"features": feature_df, "segments": segment_df}


def main():
    feature_df = load_region_features()
    segment_df = assign_segments(feature_df)
    write_segments(segment_df)
    print(segment_df[["region_name", "segment", "exposure_tier", "dominant_factor"]].to_string(index=False))


if __name__ == "__main__":
    main()