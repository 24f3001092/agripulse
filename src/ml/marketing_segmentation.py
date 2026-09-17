"""
marketing_segmentation.py

Analytical regional segmentation built ONLY from existing Gold data. It groups
monitored regions by what the data actually shows:

  * export size   (total_exports_musd, from Gold region_daily_features)
  * crop mix      (corn / wheat / cotton / dairy shares)
  * weather       (precipitation, temperature, humidity)

Method choice: with ~10 monitored regions a KMeans fit would be unstable and
hard to explain, so this module uses transparent, deterministic RULE-BASED
segments (tercile thresholds computed from the observed data). Every segment
carries a `segment_reason` string citing the actual numbers behind the
assignment, so nothing is a black box.

IMPORTANT: The labels ("Export Leader", "Corn-Centric", ...) are ANALYSIS
groupings for exploration only. They are NOT validated commercial/customer
classifications and must never be presented as such.

Output:
    data/gold/region_segments.parquet
    columns: region_name, segment, segment_reason, export_tier, dominant_crop

Run directly:
    python src/ml/marketing_segmentation.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("marketing_segmentation")

BASE = Path(__file__).resolve().parents[2]
GOLD = BASE / "data" / "gold"

CROP_COLS = ["corn_musd", "wheat_musd", "cotton_musd", "dairy_musd"]
EXPORT_COL = "total_exports_musd"
# A crop is "dominant" only if it makes up at least this share of the regional
# crop basket; otherwise the region is treated as diversified.
DOMINANT_SHARE = 0.40


def load_region_features() -> pd.DataFrame:
    """
    Build one feature row per region from the Gold daily feature table.
    Uses real columns only (no invented data): export and crop values are the
    observed maxima; weather figures are observed means over the window.
    """
    daily_df = pd.read_parquet(GOLD / "region_daily_features")
    features = daily_df.groupby("region_name", observed=True).agg(
        total_exports_musd=(EXPORT_COL, "max"),
        corn_musd=("corn_musd", "max"),
        wheat_musd=("wheat_musd", "max"),
        cotton_musd=("cotton_musd", "max"),
        dairy_musd=("dairy_musd", "max"),
        avg_temp_c=("temp_avg_c", "mean"),
        total_precip_mm=("precipitation_mm", "sum"),
        avg_humidity_pct=("humidity_pct", "mean"),
    ).reset_index()
    logger.info(f"Loaded features for {len(features)} regions from Gold layer")
    return features


def _tercile_floor(values: pd.Series, pct: float) -> float:
    """Deterministic quantile threshold on the actual observed values."""
    return float(np.quantile(values, pct))


def _export_tier(export: float, low_q: float, high_q: float) -> str:
    if export >= high_q:
        return "High-Export"
    if export <= low_q:
        return "Emerging-Export"
    return "Mid-Export"


def _dominant_crop(row: pd.Series) -> str:
    crop_values = [float(row[c]) for c in CROP_COLS]
    total = sum(crop_values)
    if total <= 0:
        return "diversified"
    best_idx = int(np.argmax(crop_values))
    best_name = CROP_COLS[best_idx].replace("_musd", "")
    best_share = crop_values[best_idx] / total
    dominant = best_name if best_share >= DOMINANT_SHARE else "diversified"
    return dominant, best_name, best_share, total


def assign_segments(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the transparent rule-based segmentation and produce one row per
    region with `segment`, `segment_reason`, `export_tier`, `dominant_crop`.
    Every processed region always receives a non-null segment.
    """
    sorted_exports = feature_df[EXPORT_COL].sort_values(ascending=True).to_numpy(dtype=float)
    low_q = _tercile_floor(sorted_exports, 0.333)
    high_q = _tercile_floor(sorted_exports, 0.667)

    precip_median = float(np.median(feature_df["total_precip_mm"].to_numpy(dtype=float)))

    rows = []
    for _, row in feature_df.iterrows():
        export = float(row[EXPORT_COL])
        tier = _export_tier(export, low_q, high_q)

        dominant, best_crop, best_share, crop_total = _dominant_crop(row)
        crop_label = f"{best_crop}-centric" if dominant != "diversified" else "diversified-crops"

        precip = float(row["total_precip_mm"])
        precip_label = "water-sensitive (low precip)" if precip < precip_median else "precipitation-adequate"

        segment = f"{tier} > {crop_label}"

        reason = (
            f"{row['region_name']}: exports ${export:,.0f}M ({tier.lower()}; tier thresholds: "
            f"<=${low_q:,.0f}M / >=${high_q:,.0f}M). Crop basket ${crop_total:,.0f}M dominated by "
            f"{best_crop} ({best_share * 100:.0f}% share) -> {crop_label}. Weather: {precip:.0f}mm "
            f"total precip, {row['avg_temp_c']:.1f}C avg, {row['avg_humidity_pct']:.0f}% humidity "
            f"({precip_label}) over the window."
        )

        rows.append({
            "region_name": row["region_name"],
            "segment": segment,
            "segment_reason": reason,
            "export_tier": tier,
            "dominant_crop": crop_label.replace("-centric", "") if crop_label != "diversified-crops" else "none",
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
    logger.info(f"Segmentation complete: {len(segment_df)} regions segmented")
    return {"features": feature_df, "segments": segment_df}


def main():
    feature_df = load_region_features()
    segment_df = assign_segments(feature_df)
    write_segments(segment_df)
    print(segment_df[["region_name", "segment", "export_tier", "dominant_crop"]].to_string(index=False))


if __name__ == "__main__":
    main()