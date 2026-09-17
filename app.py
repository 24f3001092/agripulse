"""
app.py

AgriPulse - Regional Agricultural Intelligence & Forecasting MVP (dashboard).

A clean, data-focused Streamlit app that reads ONLY repository-generated
artifacts:

    data/gold/region_daily_features    (gold region feature table)
    data/gold/region_summary           (gold region-level summary)
    data/gold/forecast_results.parquet (forecast outputs)
    data/gold/region_segments.parquet  (analytical segments)
    catalog/health_report.json         (pipeline health)
    catalog/model_report.json          (model/report metadata)

No hard-coded numbers: every metric on this page traces back to a generated
file, and missing files are reported instead of being silently replaced.
Charts read the same frames the metrics are computed from.

Run:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent
GOLD = BASE / "data" / "gold"
SILVER = BASE / "data" / "silver"
BRONZE = BASE / "data" / "bronze"
CATALOG = BASE / "catalog"

ACCENT = "#2E7D32"   # agri green
AMBER = "#F9A825"    # secondary accent
SLATE = "#455A64"    # neutral text

st.set_page_config(
    page_title="AgriPulse - Agricultural Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Presentation helpers (lightweight, no extra dependencies)
# --------------------------------------------------------------------------- #
_CSS = """
<style>
h1, h2, h3 { color: #1f3b2c; }
[data-testid="stMetric"] {
    background: #f6f8f4;
    border: 1px solid #e2e6dc;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #5b6b5f; }
[data-testid="stMetricValue"] { overflow-wrap: anywhere; word-break: normal; }
.app-banner { overflow-wrap: anywhere; }
section[data-testid="stSidebar"] { border-right: 1px solid #e2e6dc; }
.app-banner {
    background: #f6f8f4;
    border: 1px solid #e2e6dc;
    border-left: 4px solid #2E7D32;
    border-radius: 8px;
    padding: 8px 14px;
    color: #2d3a2f;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


def style_fig(fig, height: int | None = None, unified: bool = False) -> None:
    """Apply the app's consistent, clean chart styling."""
    fig.update_layout(
        template="plotly_white",
        colorway=[ACCENT, AMBER, SLATE, "#8D6E63", "#26A69A", "#1565C0"],
        font={"family": "Segoe UI, Arial, sans-serif", "color": SLATE, "size": 12},
        margin={"l": 60, "r": 20, "t": 55, "b": 40},
        hovermode="x unified" if unified else "closest",
        showlegend=True,
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(showgrid=False, linecolor="#DDE2D8")
    fig.update_yaxes(gridcolor="#EDF0E9")


# --------------------------------------------------------------------------- #
# Data loaders (all defensive: return None if the artifact is missing)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_gold_daily() -> pd.DataFrame | None:
    path = GOLD / "region_daily_features"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - surface any read error to the user
        st.error(f"Could not read {path}: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_summary() -> pd.DataFrame | None:
    path = GOLD / "region_summary"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read {path}: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_forecast() -> pd.DataFrame | None:
    path = GOLD / "forecast_results.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read {path}: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_segments() -> pd.DataFrame | None:
    path = GOLD / "region_segments.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read {path}: {exc}")
        return None


def load_json(rel_path: Path) -> dict | None:
    if not rel_path.exists():
        return None
    import json

    try:
        return json.loads(rel_path.read_text())
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read {rel_path}: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_health_report() -> dict | None:
    return load_json(CATALOG / "health_report.json")


@st.cache_data(show_spinner=False)
def load_model_report() -> dict | None:
    return load_json(CATALOG / "model_report.json")


@st.cache_data(show_spinner=False)
def load_latest_ingestion_meta() -> dict | None:
    metas = sorted(BRONZE.glob("*.meta.json")) if BRONZE.exists() else []
    if not metas:
        return None
    return load_json(metas[-1])


def missing_artifact(name: str) -> None:
    st.markdown(f'<div class="app-banner"><b>Data not found</b><br/>{name}</div>', unsafe_allow_html=True)
    st.info("Generate it first with the pipeline, then refresh this page:")
    st.code("python run_pipeline.py --skip-ingestion", language="bash")


def require_columns(df: pd.DataFrame, columns: list[str], artifact: str) -> bool:
    """True if df is non-empty and carries all required columns, else surfaces an error."""
    if df.empty:
        st.error(f"{artifact} is empty. Re-run the pipeline to regenerate it.")
        return False
    missing = [c for c in columns if c not in df.columns]
    if missing:
        st.error(f"{artifact} is missing required columns: {missing}. Re-run the pipeline to regenerate it.")
        return False
    return True


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    st.header("Overview")
    st.caption("Headline figures and pipeline status, all computed from the Gold layer.")

    summary = load_summary()
    daily = load_gold_daily()
    health = load_health_report()
    model = load_model_report()
    meta = load_latest_ingestion_meta()

    if summary is None or not require_columns(
        summary, ["region_name", "total_exports_musd"], "data/gold/region_summary"
    ):
        missing_artifact("data/gold/region_summary")
        return

    total_regions = int(len(summary))
    total_exports = float(summary["total_exports_musd"].sum())
    avg_temp = (
        float(summary["avg_temp_c"].mean())
        if "avg_temp_c" in summary.columns and summary["avg_temp_c"].notna().any()
        else None
    )
    health_status = (health or {}).get("status", "UNKNOWN")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monitored regions", f"{total_regions}")
    c2.metric("Total exports (monitored)", f"${total_exports:,.0f}M")
    c3.metric("Avg temperature", f"{avg_temp:.1f}C" if avg_temp is not None else "n/a")
    c4.metric("Data quality status", health_status)

    c5, c6 = st.columns(2)
    if model:
        c5.metric("Model status", (model.get("model_status") or "UNKNOWN").capitalize())
        c6.metric("Model", model.get("model_name") or "n/a")
    else:
        c5.metric("Model status", "Not generated")
        c6.metric("Model", "n/a")

    if health_status != "PASS":
        st.warning("The last health report is not PASS. Open the Data Health tab for details.")

    st.divider()
    st.subheader("Pipeline activity")
    stamp_a = (health or {}).get("checked_at_utc")
    stamp_b = (model or {}).get("generated_at_utc")
    if stamp_a:
        st.write(f"- Health report checked: `{stamp_a}`")
    if stamp_b:
        st.write(f"- Model report generated: `{stamp_b}`")
    if meta:
        st.write(
            f"- Latest weather ingestion: `{meta.get('ingested_at_utc')}` "
            f"(status: `{meta.get('status')}`)"
        )
    if not (stamp_a or stamp_b or meta):
        st.info("No pipeline metadata found yet.")

    st.divider()
    st.subheader("Exports by region")
    chart_df = summary.sort_values("total_exports_musd", ascending=True)
    fig = px.bar(
        chart_df, x="total_exports_musd", y="region_name", orientation="h",
        title="Total agricultural exports by region",
        labels={"total_exports_musd": "Exports (USD millions)", "region_name": "Region"},
    )
    style_fig(fig, height=440)
    st.plotly_chart(fig, width="stretch")

    st.caption("Charts read the Gold layer exactly as the metrics above. No values are hard-coded.")


def page_regional_intelligence() -> None:
    st.header("Regional Intelligence")
    st.caption(
        "Pick one region to compare its export and weather profile side by side. "
        "All values come from the Gold layer."
    )

    daily = load_gold_daily()
    summary = load_summary()
    if daily is None or summary is None:
        missing_artifact("data/gold/region_daily_features")
        return
    if not require_columns(
        daily,
        ["region_name", "date", "temp_avg_c", "temp_max_c", "temp_min_c",
         "precipitation_mm", "humidity_pct", "windspeed_max_kmh",
         "corn_musd", "wheat_musd", "cotton_musd", "dairy_musd"],
        "data/gold/region_daily_features",
    ):
        return
    if not require_columns(summary, ["region_name", "total_exports_musd"], "data/gold/region_summary"):
        return

    region = st.selectbox("Select region", sorted(daily["region_name"].unique()))
    region_daily = daily[daily["region_name"] == region].copy()
    region_sum = summary[summary["region_name"] == region]
    if region_daily.empty:
        st.error(f"No daily records for region `{region}`. Re-run the pipeline.")
        return

    c1, c2, c3 = st.columns(3)
    exports = float(region_sum["total_exports_musd"].iloc[0]) if len(region_sum) else float("nan")
    c1.metric("Total exports", f"${exports:,.0f}M" if pd.notna(exports) else "n/a")
    c2.metric("Avg temperature", f"{region_daily['temp_avg_c'].mean():.1f}C")
    c3.metric("Total precipitation", f"{region_daily['precipitation_mm'].sum():.0f} mm")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Crop profile")
        crop_map = {
            "corn": float(region_daily["corn_musd"].max()),
            "wheat": float(region_daily["wheat_musd"].max()),
            "cotton": float(region_daily["cotton_musd"].max()),
            "dairy": float(region_daily["dairy_musd"].max()),
        }
        crop_df = pd.DataFrame(
            {"crop": list(crop_map.keys()), "exports_musd": list(crop_map.values())}
        )
        fig = px.bar(
            crop_df, x="crop", y="exports_musd",
            title=f"Export value by crop - {region}",
            labels={"exports_musd": "USD millions", "crop": "Crop"},
        )
        fig.update_traces(marker_color=ACCENT)
        style_fig(fig, height=360)
        st.plotly_chart(fig, width="stretch")
        st.caption("Largest export value observed per crop over the observation window ($M).")

    with col_b:
        st.subheader("Weather - daily temperature")
        fig = px.line(
            region_daily, x="date", y="temp_avg_c",
            title=f"Average temperature - {region}",
            labels={"date": "Date", "temp_avg_c": "deg C"},
        )
        fig.update_traces(line_color=ACCENT)
        style_fig(fig, height=360, unified=True)
        st.plotly_chart(fig, width="stretch")
        st.caption("Daily mean temperature over the observation window.")

    st.divider()
    st.subheader("Daily detail")
    st.caption("Raw daily weather measurements for the selected region.")
    st.dataframe(
        region_daily[
            ["date", "temp_max_c", "temp_min_c", "temp_avg_c", "precipitation_mm",
             "humidity_pct", "windspeed_max_kmh"]
        ].sort_values("date"),
        width="stretch",
    )


def page_forecast() -> None:
    st.header("Forecast")
    st.caption(
        "Prototype export forecast built from Gold-layer weather features. "
        "Keep the limitations below in mind when reading any number here."
    )

    forecast = load_forecast()
    model = load_model_report()
    if forecast is None or not require_columns(
        forecast,
        ["region_name", "actual_exports_musd", "predicted_exports_musd"],
        "data/gold/forecast_results.parquet",
    ):
        missing_artifact("data/gold/forecast_results.parquet")
        return

    c1, c2, c3, c4 = st.columns(4)
    if model:
        c1.metric("Model", model.get("model_name") or "n/a")
        c2.metric("Status", (model.get("model_status") or "UNKNOWN").capitalize())
        metrics = model.get("metrics") or {}
        c3.metric("R2 (in-sample)", f"{metrics.get('r2', float('nan')):.3f}" if isinstance(metrics.get("r2"), (int, float)) else "n/a")
        mae = metrics.get("mae_musd")
        c4.metric("MAE ($M)", f"{mae:,.1f}" if isinstance(mae, (int, float)) else "unavailable")
    else:
        c1.metric("Model", "n/a")
        c2.metric("Status", "Not generated")
        c3.metric("R2 (in-sample)", "n/a")
        c4.metric("MAE ($M)", "unavailable")

    if model:
        st.markdown(f'<div class="app-banner"><b>Evaluation method</b><br/>{model.get("evaluation_methodology") or "n/a"}</div>', unsafe_allow_html=True)
        if model.get("small_sample_warning"):
            st.warning(model["small_sample_warning"])
    else:
        st.info("No model report found yet. Re-run the pipeline to generate one.")

    st.divider()
    st.subheader("Actual vs predicted exports")
    molten = forecast.melt(
        id_vars="region_name",
        value_vars=["actual_exports_musd", "predicted_exports_musd"],
        var_name="series", value_name="exports_musd",
    ).copy()
    molten["series"] = molten["series"].map(
        {"actual_exports_musd": "Actual", "predicted_exports_musd": "Predicted"}
    )
    fig = px.bar(
        molten, x="region_name", y="exports_musd", color="series", barmode="group",
        title="Actual vs predicted exports, per region",
        labels={"exports_musd": "USD millions", "region_name": "Region", "series": ""},
        color_discrete_map={"Actual": SLATE, "Predicted": ACCENT},
    )
    style_fig(fig, height=420)
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Forecast detail")
    st.dataframe(forecast.sort_values("region_name"), width="stretch")

    st.divider()
    st.subheader("Known limitations")
    limitations = (model or {}).get("limitations", []) or ["Not generated yet."]
    for limit in limitations:
        st.markdown(f"- {limit}")
    st.caption("Forecast values are generated by the pipeline (data/gold/forecast_results.parquet).")


def page_segments() -> None:
    st.header("Marketing Segments")
    st.caption(
        "Analytical region groups derived from Gold-layer data for exploration. "
        "A segment combines the observed export tier with the region's dominant crop."
    )

    segments = load_segments()
    if segments is None or not require_columns(
        segments,
        ["region_name", "segment", "segment_reason", "export_tier", "dominant_crop"],
        "data/gold/region_segments.parquet",
    ):
        missing_artifact("data/gold/region_segments.parquet")
        return

    st.warning(
        "These groups are analytical labels computed from observed data only. "
        "They are not validated commercial or customer classifications."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Segment distribution")
        dist = segments["segment"].value_counts().reset_index()
        dist.columns = ["segment", "count"]
        fig = px.bar(
            dist.sort_values("count", ascending=True),
            x="count", y="segment", orientation="h",
            title="Regions per segment",
            labels={"count": "Regions", "segment": "Segment"},
        )
        fig.update_traces(marker_color=ACCENT)
        style_fig(fig, height=380)
        st.plotly_chart(fig, width="stretch")
    with col_b:
        st.subheader("Regions in each segment")
        for seg in sorted(segments["segment"].unique()):
            names = ", ".join(segments[segments["segment"] == seg]["region_name"].tolist())
            st.markdown(f"**{seg}**")
            st.write(names)

    st.divider()
    st.subheader("Segment detail")
    chosen = st.selectbox("Select segment", sorted(segments["segment"].unique()))
    detail = segments[segments["segment"] == chosen]
    st.dataframe(detail[["region_name", "segment", "export_tier", "dominant_crop"]], width="stretch")

    st.subheader("Why each region was assigned")
    for _, row in detail.iterrows():
        st.markdown(f"**{row['region_name']}:** {row['segment_reason']}")


def page_health() -> None:
    st.header("Data Health")
    st.caption("Automated pipeline checks over the artifact layers, with provenance.")

    health = load_health_report()
    if health is None:
        missing_artifact("catalog/health_report.json")
        return

    status = health.get("status", "UNKNOWN")
    st.metric("Overall status", status)

    if status == "PASS":
        st.success("All monitored checks passed on the latest run.")
    else:
        st.error("One or more checks failed. Review the issues below.")

    issues = health.get("issues") or []
    if issues:
        st.subheader("Issues")
        for issue in issues:
            st.error(f"`{issue.get('check')}` - {issue.get('detail', '')}")
    if health.get("checked_at_utc"):
        st.write(f"Checked at: `{health['checked_at_utc']}`")

    st.subheader("Checks performed")
    st.write("- **Freshness** - the latest weather ingestion is within the 24-hour freshness window.")
    st.write("- **Row counts** - weather, ag_exports and geo meet their minimum row thresholds in the Silver layer.")

    meta = load_latest_ingestion_meta()
    if meta:
        st.subheader("Latest weather ingestion")
        st.write(
            f"- Source: `{meta.get('source_name')}`  "
            f"status: `{meta.get('status')}`  "
            f"ingested: `{meta.get('ingested_at_utc')}`  "
            f"rows: `{meta.get('row_count')}`"
        )

    st.subheader("Artifact presence")
    rows = []
    for name, path in [
        ("Silver weather", SILVER / "weather.parquet"),
        ("Silver exports", SILVER / "ag_exports.parquet"),
        ("Silver geo", SILVER / "geo.parquet"),
        ("Gold region_daily_features", GOLD / "region_daily_features"),
        ("Gold region_summary", GOLD / "region_summary"),
        ("Forecast results", GOLD / "forecast_results.parquet"),
        ("Segments", GOLD / "region_segments.parquet"),
        ("Model report", CATALOG / "model_report.json"),
    ]:
        present = "Present" if path.exists() else "Missing"
        rows.append({"artifact": name, "present": present})

    presence = pd.DataFrame(rows)

    def _color(val: str) -> str:
        return "color:#2E7D32;" if val == "Present" else "color:#B71C1C; font-weight:600;"

    st.dataframe(presence.style.map(_color, subset=["present"]), width="stretch")


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("AgriPulse")
    st.caption("Regional Agricultural Intelligence & Forecasting MVP - powered by real pipeline outputs")

    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Regional Intelligence", "Forecast", "Marketing Segments", "Data Health"],
    )

    st.sidebar.caption("Reads Gold-layer and catalog artifacts generated by `run_pipeline.py`.")

    if page == "Overview":
        page_overview()
    elif page == "Regional Intelligence":
        page_regional_intelligence()
    elif page == "Forecast":
        page_forecast()
    elif page == "Marketing Segments":
        page_segments()
    else:
        page_health()


if __name__ == "__main__":
    main()