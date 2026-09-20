"""
prediction_pipeline.py

Unified ML pipeline (AgriPulse India): coordinates every modelling step that
consumes the Gold layer, reusing the individual modules (no duplicated logic):

    1. load Gold region features
    2. market-price forecasting           -> demand_forecast_model.run_forecast()
         * GATED: while the e-NAM / AGMARKNET market source is an extension
           point (not configured), writes data/india/catalog/ml_status.json
           {model_status: not_generated} and skips forecast outputs -- nothing
           is fabricated.
         * When configured:
             data/india/gold/forecast_results.parquet
             data/india/catalog/model_report.json
    3. regional weather-exposure segmentation
         -> marketing_segmentation.run_segmentation()
             data/india/gold/region_segments.parquet
    4. write processing metadata           -> data/india/catalog/prediction_run.json

Run directly:
    python src/ml/prediction_pipeline.py
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("prediction_pipeline")

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

CATALOG = config.CATALOG
GOLD = config.GOLD

# Make sibling modules importable regardless of how this file is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import demand_forecast_model  # noqa: E402
import marketing_segmentation  # noqa: E402


def run_prediction_pipeline() -> dict:
    """Run forecasting, segmentation and persist all ML outputs + metadata."""
    steps = []
    model = {"name": "N/A", "status": "N/A"}

    forecast_result = demand_forecast_model.run_forecast()
    forecast_df = forecast_result["forecast"]
    if forecast_result.get("gated"):
        steps.append({
            "step": "demand_forecast",
            "status": "skipped_market_gate",
            "output": "data/india/catalog/ml_status.json",
            "rows": 0,
            "detail": ("Forecast requires a real market target (e-NAM / AGMARKNET). Source not "
                       "configured; ml_status.json records not_generated. No forecast fabricated."),
        })
        model = {"name": "not_generated", "status": "not_generated"}
    else:
        steps.append({
            "step": "demand_forecast",
            "status": "success",
            "output": "data/india/gold/forecast_results.parquet",
            "rows": int(len(forecast_df)),
        })

    segmentation_result = marketing_segmentation.run_segmentation()
    segment_df = segmentation_result["segments"]
    steps.append({
        "step": "weather_exposure_segmentation",
        "status": "success",
        "output": "data/india/gold/region_segments.parquet",
        "rows": int(len(segment_df)),
    })

    metadata = {
        "pipeline": "agripulse_prediction_pipeline",
        "country": config.COUNTRY,
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "steps": steps,
        "model": model,
        "market_source": config.market_source_status(),
        "outputs": [
            "data/india/gold/region_segments.parquet",
            "data/india/catalog/ml_status.json",
        ],
    }
    # Keep model metadata in sync with ml_status.json when it exists.
    status_path = CATALOG / "ml_status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text())
            metadata["model"] = {
                "name": status.get("target_variable"),
                "status": status.get("model_status"),
            }
        except json.JSONDecodeError:
            logger.warning("ml_status.json unreadable; keeping generic model metadata")

    metadata_path = CATALOG / "prediction_run.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info(f"Wrote prediction run metadata -> {metadata_path}")

    return {
        "forecast": forecast_df,
        "forecast_metadata": forecast_result["model_info"],
        "segments": segment_df,
        "metadata": metadata,
    }


def main():
    result = run_prediction_pipeline()
    logger.info(
        f"Prediction pipeline complete: {len(result['forecast'])} forecast rows, "
        f"{len(result['segments'])} segment rows written."
    )


if __name__ == "__main__":
    main()