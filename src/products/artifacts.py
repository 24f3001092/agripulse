"""Tracked artifacts required by the deployed AgriPulse India dashboard.

The dashboard reads locally generated pipeline outputs (data/india/...) first;
when they are absent (e.g. a fresh Streamlit Community Cloud checkout, where the
generated files are gitignored), it falls back to the real, committed pipeline
outputs under data/india/demo/.

This module is the single source of truth for that mapping so the app and the
deployment-readiness tests can both rely on it. Pure pathlib — no Streamlit,
no live APIs.
"""

from __future__ import annotations

from pathlib import Path

# key -> (locally generated relative path, tracked demo relative path)
REQUIRED_ARTIFACTS: dict[str, tuple[str, str]] = {
    "gold_daily": ("data/india/gold/region_daily_features", "data/india/demo/gold/region_daily_features.parquet"),
    "gold_summary": ("data/india/gold/region_summary", "data/india/demo/gold/region_summary.parquet"),
    "segments": ("data/india/gold/region_segments.parquet", "data/india/demo/gold/region_segments.parquet"),
    "health": ("data/india/catalog/health_report.json", "data/india/demo/catalog/health_report.json"),
    "ml_status": ("data/india/catalog/ml_status.json", "data/india/demo/catalog/ml_status.json"),
    "silver_weather": ("data/india/silver/weather.parquet", "data/india/demo/silver/weather.parquet"),
    "silver_geo": ("data/india/silver/geo.parquet", "data/india/demo/silver/geo.parquet"),
    "silver_mandi": ("data/india/silver/india_mandi.parquet", "data/india/demo/silver/india_mandi.parquet"),
    "silver_agriculture": ("data/india/silver/india_agriculture.parquet", "data/india/demo/silver/india_agriculture.parquet"),
    "gold_mandi_summary": ("data/india/gold/india_market_summary/manifest.json", "data/india/demo/gold/india_market_summary/manifest.json"),
    "gold_mandi_data": ("data/india/gold/india_market_summary/india_market_summary.parquet", "data/india/demo/gold/india_market_summary/india_market_summary.parquet"),
    "gold_region_features": ("data/india/gold/india_region_features/india_region_features.parquet", "data/india/demo/gold/india_region_features.parquet"),
    "gold_crop_summary": ("data/india/gold/india_crop_summary/india_crop_summary.parquet", "data/india/demo/gold/india_crop_summary.parquet"),
    "gold_crop_year": ("data/india/gold/india_crop_summary/india_crop_year.parquet", "data/india/demo/gold/india_crop_year.parquet"),
    "gold_lineage": ("data/india/catalog/gold_lineage.json", "data/india/demo/catalog/gold_lineage.json"),
    "india_forecasts": ("data/india/gold/india_forecasts.parquet", "data/india/demo/gold/india_forecasts.parquet"),
    "india_model_report": ("data/india/catalog/india_model_report.json", "data/india/demo/catalog/india_model_report.json"),
}

# Market-paced artifacts: present only when a real e-NAM / AGMARKNET source is
# configured. Their ABSENCE is the honest state of the product today and is not
# treated as a deployment failure.
OPTIONAL_ARTIFACTS: dict[str, tuple[str, str]] = {
    "forecast": ("data/india/gold/forecast_results.parquet", "data/india/demo/gold/forecast_results.parquet"),
    "model": ("data/india/catalog/model_report.json", "data/india/demo/catalog/model_report.json"),
    "silver_market": ("data/india/silver/market.parquet", "data/india/demo/silver/market.parquet"),
}

DEMO_META_REL = "data/india/demo/bronze/latest_weather.meta.json"


def demo_artifact_map() -> dict[str, str]:
    """key -> demo-relative path for every required artifact."""
    return {k: demo for k, (_live, demo) in REQUIRED_ARTIFACTS.items()}


def missing_demo_artifacts(root: Path, include_optional: bool = False) -> list[str]:
    """Return the demo-relative paths of missing deployment artifacts."""
    mapping = dict(REQUIRED_ARTIFACTS)
    if include_optional:
        mapping.update(OPTIONAL_ARTIFACTS)
    return [
        demo_rel
        for _key, (_live, demo_rel) in mapping.items()
        if not (Path(root) / demo_rel).exists()
    ]


def demo_artifacts_present(root: Path) -> bool:
    """True when every REQUIRED deployment artifact exists under the given root."""
    return len(missing_demo_artifacts(Path(root))) == 0


def resolve_artifact(root: Path, key: str) -> Path:
    """Local-generated artifact when present, else the tracked demo artifact."""
    if key not in REQUIRED_ARTIFACTS and key not in OPTIONAL_ARTIFACTS:
        raise KeyError(f"Unknown artifact key: {key}")
    mapping = dict(REQUIRED_ARTIFACTS)
    mapping.update(OPTIONAL_ARTIFACTS)
    live_rel, demo_rel = mapping[key]
    live = Path(root) / live_rel
    return live if live.exists() else (Path(root) / demo_rel)