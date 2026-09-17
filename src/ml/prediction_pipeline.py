"""
prediction_pipeline.py

Unified ML pipeline: coordinates every modelling step that consumes the Gold
layer, reusing the individual modules (no duplicated logic):

    1. load Gold region features
    2. demand / export forecasting            -> demand_forecast_model.run_forecast()
         data/gold/forecast_results.parquet
         catalog/model_report.json
    3. regional analytical segmentation        -> marketing_segmentation.run_segmentation()
         data/gold/region_segments.parquet
    4. write processing metadata               -> catalog/prediction_run.json

Run directly:
    python src/ml/prediction_pipeline.py

Also callable as a function (`run_prediction_pipeline()`) and invoked by
run_pipeline.py / dags/agripulse_dag.py.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("prediction_pipeline")

BASE = Path(__file__).resolve().parents[2]
CATALOG = BASE / "catalog"

# Make sibling modules importable regardless of how this file is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import demand_forecast_model  # noqa: E402
import marketing_segmentation  # noqa: E402


def run_prediction_pipeline() -> dict:
    """Run forecasting, segmentation and persist all ML outputs + metadata."""
    steps = []

    forecast_result = demand_forecast_model.run_forecast()
    forecast_df = forecast_result["forecast"]
    steps.append({
        "step": "demand_forecast",
        "status": "success",
        "output": "data/gold/forecast_results.parquet",
        "rows": int(len(forecast_df)),
    })

    segmentation_result = marketing_segmentation.run_segmentation()
    segment_df = segmentation_result["segments"]
    steps.append({
        "step": "marketing_segmentation",
        "status": "success",
        "output": "data/gold/region_segments.parquet",
        "rows": int(len(segment_df)),
    })

    metadata = {
        "pipeline": "agripulse_prediction_pipeline",
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "steps": steps,
        "model": {"name": "N/A", "status": "N/A"},
        "outputs": [
            "data/gold/forecast_results.parquet",
            "data/gold/region_segments.parquet",
            "catalog/model_report.json",
            "catalog/prediction_run.json",
        ],
    }
    # Pull the authoritative model name/status from the report that was just
    # written, so all metadata files stay in sync.
    report_path = CATALOG / "model_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
            metadata["model"] = {"name": report.get("model_name"), "status": report.get("model_status")}
        except json.JSONDecodeError:
            logger.warning("model_report.json unreadable; keeping generic model metadata")

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