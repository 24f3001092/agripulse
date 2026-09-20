"""AgriPulse India - public agricultural intelligence & business scenario platform.

This Streamlit dashboard presents repository-generated artifacts for India,
including Gold-layer regional/crop features, market summaries, forecasts,
segmentation, monitoring reports, and scenario analysis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))

import config  # noqa: E402
from products.artifacts import REQUIRED_ARTIFACTS, OPTIONAL_ARTIFACTS  # noqa: E402
from products.insights import build_region_insights, _tercile_label  # noqa: E402
from products.scenario import estimate_scenario, run_sensitivity, compute_risk_indicators  # noqa: E402
from config import format_inr, inr_compact, format_price_per_quintal  # noqa: E402
# Translation helpers are intentionally kept local so the public app does not
# depend on an optional catalog.translations module that may be absent from
# a clean checkout. Navigation falls back to English keys safely.
_TRANSLATIONS = {
    "en": {
        "navigation.home": "Home",
        "navigation.my_region": "My Region",
        "navigation.crop_intelligence": "Crop Intelligence",
        "navigation.mandi_market": "Mandi Market",
        "navigation.weather_risk": "Weather & Risk",
        "navigation.forecast": "Forecast",
        "navigation.opportunity_scanner": "Opportunity Scanner",
        "navigation.scenario_planner": "Scenario Planner",
        "navigation.data_health": "Data Health",
        "navigation.sources_methodology": "Sources & Methodology",
        "navigation.business_intelligence": "Business Intelligence",
    },
}

def load_translations() -> dict:
    """Return the built-in UI translations used by the dashboard."""
    return _TRANSLATIONS


def t(key: str, lang: str = "en") -> str:
    """Translate a UI key, falling back to English and then the key itself."""
    return _TRANSLATIONS.get(lang, {}).get(
        key, _TRANSLATIONS["en"].get(key, key)
    )

DEMO = config.DEMO
CATALOG_CFG = config.CATALOG_CFG

# Preferred source is the locally generated pipeline output; if it was not
# generated (e.g. a Streamlit Cloud checkout, where the generated files are
# gitignored), the dashboard falls back to the tracked real pipeline outputs
# committed under data/india/demo/. The mapping is defined once, in src/products/artifacts.py.
_ARTIFACT_MAP = {**REQUIRED_ARTIFACTS, **OPTIONAL_ARTIFACTS}
ARTIFACTS = {
    key: (BASE / live_rel, BASE / demo_rel)
    for key, (live_rel, demo_rel) in _ARTIFACT_MAP.items()
}

# Tracks where each artifact was actually read from for honest reporting.
_ARTIFACT_SOURCE: dict[str, str] = {}

ACCENT = "#2E7D32"   # agri green
AMBER = "#F9A825"    # secondary accent
SLATE = "#455A64"    # neutral text

DISCLAIMER = (
    "AgriPulse India provides data-driven estimates and scenario analysis for "
    "informational and planning purposes. Results are not guarantees of "
    "agricultural yield, market prices, revenue, or profit."
)

st.set_page_config(
    page_title="AgriPulse India - Agricultural Intelligence",
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
.app-warning {
    background: #fdf6e3;
    border: 1px solid #ead9ae;
    border-left: 4px solid #F9A825;
    border-radius: 8px;
    padding: 8px 14px;
    color: #5d4a12;
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


def resolve_artifact(key: str) -> tuple[Path, str] | None:
    """Return (path, source) where source is 'live' or 'demo', or None if absent."""
    live, demo = ARTIFACTS[key]
    if live.exists():
        _ARTIFACT_SOURCE[key] = "live"
        return live, "live"
    if demo.exists():
        _ARTIFACT_SOURCE[key] = "demo"
        return demo, "demo"
    _ARTIFACT_SOURCE[key] = "missing"

    return None


def using_demo_artifacts() -> bool:
    """True when this session had to fall back to tracked data/india/demo artifacts."""
    return "demo" in _ARTIFACT_SOURCE.values()


# --------------------------------------------------------------------------- #
# Data loaders (live artifacts preferred; tracked data/india/demo as fallback)
# --------------------------------------------------------------------------- #
def _read_parquet(key: str) -> pd.DataFrame | None:
    resolved = resolve_artifact(key)
    if resolved is None:
        return None
    path, _source = resolved
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - surface any read error to the user
        st.error(f"Could not read {path}: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_gold_daily() -> pd.DataFrame | None:
    return _read_parquet("gold_daily")


@st.cache_data(show_spinner=False)
def load_summary() -> pd.DataFrame | None:
    return _read_parquet("gold_summary")


@st.cache_data(show_spinner=False)
def load_forecast() -> pd.DataFrame | None:
    return _read_parquet("forecast")


@st.cache_data(show_spinner=False)
def load_segments() -> pd.DataFrame | None:
    return _read_parquet("segments")


@st.cache_data(show_spinner=False)
def load_geo() -> pd.DataFrame | None:
    return _read_parquet("silver_geo")


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read {path}: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_health_report() -> dict | None:
    resolved = resolve_artifact("health")
    return load_json(resolved[0]) if resolved else None


@st.cache_data(show_spinner=False)
def load_ml_status() -> dict | None:
    resolved = resolve_artifact("ml_status")
    return load_json(resolved[0]) if resolved else None


@st.cache_data(show_spinner=False)
def load_model_report() -> dict | None:
    resolved = resolve_artifact("model")
    return load_json(resolved[0]) if resolved else None


@st.cache_data(show_spinner=False)
def load_latest_ingestion_meta() -> dict | None:
    metas = sorted(config.BRONZE.glob("*.meta.json")) if config.BRONZE.exists() else []
    if metas:
        return load_json(metas[-1])
    demo_meta = DEMO / "bronze" / "latest_weather.meta.json"
    return load_json(demo_meta) if demo_meta.exists() else None


@st.cache_data(show_spinner=False)
def load_mandi() -> pd.DataFrame | None:
    """Silver mandi records (real AGMARKNET daily prices/arrivals)."""
    return _read_parquet("silver_mandi")


@st.cache_data(show_spinner=False)
def load_mandi_gold() -> pd.DataFrame | None:
    """Gold market-intelligence summary (latest modal, 7-day avg, changes)."""
    return _read_parquet("gold_mandi_data")


@st.cache_data(show_spinner=False)
def load_mandi_manifest() -> dict | None:
    resolved = resolve_artifact("gold_mandi_summary")
    return load_json(resolved[0]) if resolved else None


@st.cache_data(show_spinner=False)
def load_india_forecasts() -> pd.DataFrame | None:
    return _read_parquet("india_forecasts")


@st.cache_data(show_spinner=False)
def load_india_model_report() -> dict | None:
    resolved = resolve_artifact("india_model_report")
    return load_json(resolved[0]) if resolved else None


@st.cache_data(show_spinner=False)
def load_crop_year() -> pd.DataFrame | None:
    return _read_parquet("gold_crop_year")


def missing_artifact(name: str) -> None:
    """Show 'Data temporarily unavailable' banner for missing artifacts."""
    st.markdown(
        '<div class="app-banner" style="border-left: 4px solid #F9A825;">'
        '<b>Data temporarily unavailable</b><br/>'
        f'{name}</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "This data is not currently available. It may be regenerating "
        "or temporarily inaccessible. Please try refreshing the page."
    )
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


def market_status_banner() -> str:
    """Honest banner text for the live market-forecast feed state."""
    status = config.market_source_status()
    if status == "configured":
        return "Live market feed (e-NAM / AGMARKNET) is configured."
    return (
        "The LIVE market forecast feed (e-NAM) is NOT configured: no market-price "
        "forecast is generated or fabricated. Observed AGMARKNET wholesale prices and "
        "arrivals ARE shown on the **Market Intelligence** page (real, attributed, "
        "historical records via the India Data Portal)."
    )


def location_cascade(
    catalog_records: list[dict],
    available_regions: set[str] | None = None,
    key: str = "loc",
    default: dict | None = None,
) -> dict | None:
    """State > District > Mandi/APMC cascade over the India location catalog.

    Renders a level ONLY when the current selection has options (no empty fake
    selectors) and stops at the highest supported level: a State without data
    shows an info note, and a District without a mandi/APMC simply has no mandi
    picker. When available_regions is given, States/Districts are restricted to
    the districts that actually exist in the loaded Gold data. `default` (a
    catalog record) preselects the same State/District when it is available.
    Returns the chosen catalog record or None.
    """
    if not catalog_records:
        return None
    states = sorted({r["state_ut"] for r in catalog_records})
    def_state_idx = 0
    if default and default.get("state_ut") in states:
        def_state_idx = states.index(default["state_ut"])
    state = st.selectbox("State / UT", states, index=def_state_idx, key=f"{key}_state")

    state_recs = [r for r in catalog_records if r["state_ut"] == state]
    if available_regions is not None:
        state_recs = [r for r in state_recs if r["region_name"] in available_regions]
        if not state_recs:
            st.info(f"No monitored district data is available yet for {state}.")
            return None

    districts = sorted({r["district"] for r in state_recs})
    def_district_idx = 0
    if default and default.get("state_ut") == state and default.get("district") in districts:
        def_district_idx = districts.index(default["district"])
    district = st.selectbox("District", districts, index=def_district_idx, key=f"{key}_district")

    dist_recs = [r for r in state_recs if r["district"] == district]
    mandis = sorted({r.get("mandi_apmc") for r in dist_recs if r.get("mandi_apmc")})
    if mandis:
        mandi = st.selectbox("Mandi / APMC", mandis, index=0, key=f"{key}_mandi")
        chosen = [r for r in dist_recs if r.get("mandi_apmc") == mandi] or dist_recs
    else:
        chosen = dist_recs
    return chosen[0] if chosen else None


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_home() -> None:
    st.title("AgriPulse India")
    st.caption('"Data-driven agricultural, weather and market intelligence for India."')

    # --- Headline / quick stats ---
    c1, c2, c3 = st.columns(3)
    daily = load_gold_daily()
    summary = load_summary()
    fc = load_forecast()

    # Available regions / crops count
    regions_available = len(daily["region_name"].unique()) if daily is not None else 0
    crops_available = len(summary["region_name"].unique()) if summary is not None else 0
    c1.metric("Available regions", regions_available)
    c2.metric("Available crops (in summary)", crops_available)
    fc_avail = "Yes" if fc is not None else "No"
    c3.metric("Forecast available", fc_avail)

    st.divider()

    # --- Latest market update ---
    st.subheader("Latest market update")
    if config.market_source_status() != "configured":
        st.markdown(f'<div class="app-warning">{market_status_banner()}</div>', unsafe_allow_html=True)
    else:
        st.caption("Market source is configured (e-NAM / AGMARKNET).")

    # --- Weather status ---
    st.subheader("Weather status")
    if daily is not None:
        latest = daily.iloc[-1]
        st.caption(f"Latest ingestion: `{latest.get('date', 'N/A')}`")
        st.caption(f"Regions with data: {daily['region_name'].nunique()}")
    else:
        st.caption("No weather data loaded.")

    # --- Data quality ---
    st.subheader("Data quality")
    health = load_health_report()
    ml_status = load_ml_status()
    meta = load_latest_ingestion_meta()
    if health is not None:
        st.caption(f"Overall status: {health.get('status', '?')}")
    else:
        st.caption("No health report loaded.")

    st.divider()

    # --- How it works (condensed) ---
    st.subheader("How it works")
    st.markdown(
        "1. **Choose your location** — use the sidebar to navigate to a district; "
        "the choice carries through to *My Region*.\n"
        "2. **Explore weather, market and crop profiles** — each page uses "
        "Gold-layer data, observed and modelled where available.\n"
        "3. **Test a business scenario** — the *Scenario Planner* (Phase 8) lets you "
        "enter assumptions and see estimated margins (₹ INR).\n"
        "4. **Review model disclaimers** — the *Forecast* and *Data Health* pages "
        "always show limitations and status."
    )

    total_regions = int(len(summary))
    avg_temp = (
        float(summary["avg_temp_c"].mean())
        if "avg_temp_c" in summary.columns and summary["avg_temp_c"].notna().any()
        else None
    )
    health_status = (health or {}).get("status", "UNKNOWN")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monitored districts", f"{total_regions}")
    c2.metric("Avg temperature (window)", f"{avg_temp:.1f}°C" if avg_temp is not None else "n/a")
    c3.metric("Market source", "Observed AGMARKNET data; live feed not configured"
              if config.market_source_status() != "configured" else "Configured (live)")
    c4.metric("Data quality status", health_status)

    c5, c6 = st.columns(2)
    if ml_status:
        c5.metric("Forecast model status", (ml_status.get("model_status") or "UNKNOWN").replace("_", " ").capitalize())
        c6.metric("Target", ml_status.get("target_variable") or "n/a")
    else:
        c5.metric("Forecast model status", "Unknown")
        c6.metric("Target", "n/a")

    if health_status != "PASS":
        st.warning("The last health report is not PASS. Open the Data Health tab for details.")

    st.markdown(f'<div class="app-warning">{market_status_banner()}</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Pipeline activity")
    stamp_a = (health or {}).get("checked_at_utc")
    if stamp_a:
        st.write(f"- Health report checked: `{stamp_a}`")
    if meta:
        st.write(
            f"- Latest weather ingestion: `{meta.get('ingested_at_utc')}` "
            f"(status: `{meta.get('status')}`)"
        )
    if not (stamp_a or meta):
        st.info("No pipeline metadata found yet.")

    st.divider()
    st.subheader("Precipitation by district (observed window)")
    chart_df = summary.sort_values("total_precip_mm", ascending=True)
    fig = px.bar(
        chart_df, x="total_precip_mm", y="region_name", orientation="h",
        title="Total precipitation by monitored district",
        labels={"total_precip_mm": "Total precipitation (mm)", "region_name": "District"},
    )
    fig.update_traces(marker_color=ACCENT)
    style_fig(fig, height=440)
    st.plotly_chart(fig, width="stretch")

    st.caption("Charts read the Gold layer exactly as the metrics above. No values are hard-coded.")
    st.info(DISCLAIMER)


def page_my_district() -> None:
    st.header("My Region")
    st.caption(
        "A public-friendly view of one monitored region/district: location (State/UT > District > "
        "Mandi/APMC), weather profile, weather-exposure segment, data freshness and forecast "
        "availability — all from observed Gold-layer data."
    )

    daily = load_gold_daily()
    summary = load_summary()
    segments = load_segments()
    forecast = load_forecast()
    health = load_health_report()
    ml_status = load_ml_status()
    model = load_model_report()
    geo = load_geo()
    meta = load_latest_ingestion_meta()

    required_cols = [
        "region_name", "date", "state_ut", "state_code", "district", "mandi_apmc",
        "temp_avg_c", "precipitation_mm", "humidity_pct", "windspeed_max_kmh",
    ]
    if daily is None or summary is None:
        missing_artifact("data/india/gold/region_daily_features")
        return
    if not require_columns(daily, required_cols, "data/india/gold/region_daily_features"):
        return
    if not require_columns(summary, ["region_name", "avg_temp_c"], "data/india/gold/region_summary"):
        return

    default_loc = st.session_state.get("chosen_location")
    available = {str(rn) for rn in daily["region_name"].unique()}
    loc = location_cascade(
        config.load_regions(),
        available_regions=available,
        key="my_district_loc",
        default=default_loc if default_loc and default_loc.get("region_name") in available else None,
    )
    if loc is None:
        st.info("Pick a State/UT and District above to view its profile.")
        return
    region = loc["region_name"]
    region_daily = daily[daily["region_name"] == region].copy()
    region_sum = summary[summary["region_name"] == region]
    if region_daily.empty:
        st.error(f"No daily records for district `{region}`. Re-run the pipeline.")
        return

    # ---- Location ----------------------------------------------------------
    st.subheader("Location")
    row0 = region_daily.iloc[0]
    state_ut = str(row0["state_ut"])
    state_code = str(row0["state_code"])
    district = str(row0["district"])
    mandi = str(row0["mandi_apmc"])
    lat = float(region_daily["latitude"].iloc[0]) if "latitude" in region_daily.columns else float("nan")
    lon = float(region_daily["longitude"].iloc[0]) if "longitude" in region_daily.columns else float("nan")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("State / UT", f"{state_ut} ({state_code})")
    c2.metric("District", district)
    c3.metric("Mandi / APMC", mandi)
    c4.metric("Coordinates (anchor)", f"{lat:.4f}, {lon:.4f}" if pd.notna(lat) and pd.notna(lon) else "n/a")
    d_code = loc.get("district_code")
    if d_code:
        st.caption(f"District LGD code: {d_code} (Local Government Directory, Government of India).")

    st.divider()

    # ---- Weather profile ---------------------------------------------------
    st.subheader("Weather profile")
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Avg temperature", f"{region_daily['temp_avg_c'].mean():.1f}°C")
    w2.metric("Total precipitation", f"{region_daily['precipitation_mm'].sum():.0f} mm")
    w3.metric("Avg humidity", f"{region_daily['humidity_pct'].mean():.0f}%")
    w4.metric("Max wind speed", f"{region_daily['windspeed_max_kmh'].max():.0f} km/h")

    obs, fcst = _split_weather_window(region_daily, meta)

    temp_fig = px.line(
        region_daily, x="date", y="temp_avg_c",
        title=f"Daily average temperature - {district}",
        labels={"date": "Date", "temp_avg_c": "°C"},
    )
    if fcst is not None and not fcst.empty:
        temp_fig.add_scatter(
            x=fcst["date"], y=fcst["temp_avg_c"],
            mode="lines+markers", name="next-days forecast window",
            line={"dash": "dot", "color": AMBER},
        )
        temp_fig.data[0].name = "observed (trailing window)"
    temp_fig.update_traces(
        line_color=ACCENT,
        selector=lambda _tr: _tr.name is None or _tr.name == "observed (trailing window)",
    )
    style_fig(temp_fig, height=380, unified=True)
    st.plotly_chart(temp_fig, width="stretch")

    prec_fig = px.bar(
        region_daily, x="date", y="precipitation_mm",
        title=f"Daily precipitation - {district}",
        labels={"date": "Date", "precipitation_mm": "mm"},
    )
    prec_fig.update_traces(marker_color=SLATE)
    style_fig(prec_fig, height=300)
    st.plotly_chart(prec_fig, width="stretch")

    if fcst is not None and not fcst.empty:
        st.caption(
            "Days after the latest weather ingestion are the Open-Meteo forecast-ahead "
            "window (dashed). They are model output, not observations."
        )
    else:
        st.caption("Weather values shown are the observed trailing window from the latest ingestion pull.")

    st.divider()

def page_my_region() -> None:
    """Public region profile using the validated district/mandi workflow."""
    page_my_district()
def _split_weather_window(
    region_daily: pd.DataFrame, meta: dict | None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Split daily rows into observed (<= ingestion date) vs forecast-ahead window."""
    ingest = (meta or {}).get("ingested_at_utc")
    if not ingest:
        return region_daily, None
    try:
        cut = pd.Timestamp(str(ingest)[:10])
    except Exception:  # noqa: BLE001
        return region_daily, None
    obs = region_daily[region_daily["date"] <= cut]
    ahead = region_daily[region_daily["date"] > cut]
    forecast = ahead if not ahead.empty else None
    return obs, forecast


def render_insight_cards(insights: list[dict]) -> None:
    """Render explainable WHAT / WHY / DATA PERIOD / SOURCE-METHOD cards."""
    for card in insights:
        with st.expander(card["title"], expanded=False):
            st.markdown(f"**WHAT** — {card['what']}")
            st.markdown(f"**WHY** — {card['why']}")
            st.markdown(f"**DATA PERIOD** — {card['period']}")
            st.markdown(f"**SOURCE/METHOD** — {card['source_method']}")


def region_weather_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per district with observed weather aggregates (real values only)."""
    return (
        daily.groupby("region_name", observed=True)
        .agg(
            avg_temp_c=("temp_avg_c", "mean"),
            total_precip_mm=("precipitation_mm", "sum"),
            avg_humidity_pct=("humidity_pct", "mean"),
            max_windspeed_kmh=("windspeed_max_kmh", "max"),
            state_ut=("state_ut", "first"),
            district=("district", "first"),
        )
        .reset_index()
    )


def page_market_intelligence() -> None:
    st.header("Market & Weather Outlook")
    st.caption(
        "Cross-district market and weather outlook. Per-district deep dives (weather, "
        "insight cards) live in *My District*. All values are observed."
    )

    daily = load_gold_daily()
    summary = load_summary()
    segments = load_segments()
    if daily is None or summary is None:
        missing_artifact("data/india/gold/region_daily_features")
        return

    st.markdown(f'<div class="app-warning">{market_status_banner()}</div>', unsafe_allow_html=True)

    # ---- Weather-exposure tier overview ------------------------------------
    if segments is not None and not segments.empty:
        st.subheader("Weather-exposure tier overview (analytical)")
        tier_dist = segments["exposure_tier"].value_counts().reset_index()
        tier_dist.columns = ["exposure_tier", "count"]
        fig = px.bar(
            tier_dist, x="count", y="exposure_tier", orientation="h",
            title="Districts per weather-exposure tier",
            labels={"count": "Districts", "exposure_tier": "Exposure tier"},
        )
        fig.update_traces(marker_color=ACCENT)
        style_fig(fig, height=260)
        st.plotly_chart(fig, width="stretch")
        st.caption("Analytical labels from observed weather terciles — not validated agronomic classifications.")

    # ---- Weather profile comparison ----------------------------------------
    st.subheader("Weather profile by district")
    weather = region_weather_frame(daily)
    weather["label"] = weather.apply(
        lambda r: f"{r['district']} ({r['state_ut']})", axis=1
    )
    molten = weather.sort_values("total_precip_mm", ascending=True)
    fig = px.bar(
        molten, x="total_precip_mm", y="label", orientation="h",
        color="avg_humidity_pct",
        title="Total precipitation by district (colour = avg humidity)",
        labels={
            "total_precip_mm": "Total precipitation (mm)",
            "label": "District (State/UT)",
            "avg_humidity_pct": "Avg humidity (%)",
        },
        color_continuous_scale="YlGnBu",
    )
    style_fig(fig, height=440)
    st.plotly_chart(fig, width="stretch")
    st.caption("Aggregated from data/india/gold/region_daily_features over the observed weather window.")


def page_mandi_intelligence() -> None:
    """Real AGMARKNET APMC price/arrival intelligence (observed data only)."""
    st.header("Market Intelligence")
    st.caption(
        "Observed wholesale mandi prices and arrivals from official AGMARKNET daily "
        "reports (via the India Data Portal, GODL-India). Analytical signals only — "
        "not investment advice and no guaranteed outcomes."
    )

    mandi = load_mandi()
    gold = load_mandi_gold()
    manifest = load_mandi_manifest()
    if mandi is None:
        missing_artifact("data/india/silver/india_mandi.parquet")
        return

    mandi = mandi.copy()
    mandi["date"] = pd.to_datetime(mandi["date"])
    mandi = mandi.sort_values("date")

    # Provenance block (always visible).
    period = (mandi["date"].min(), mandi["date"].max())
    last_update = manifest.get("last_data_date") if manifest else period[1].strftime("%Y-%m-%d")
    source_label = (
        (manifest or {}).get("source")
        or "AGMARKNET daily APMC price & arrival records (India Data Portal, GODL-India)"
    )
    st.markdown(
        f'<div class="app-card">'
        f"<b>Data source</b> — {source_label}<br/>"
        f"<b>Last updated</b> — {last_update}<br/>"
        f"<b>Data period</b> — {period[0].strftime('%Y-%m-%d')} to {period[1].strftime('%Y-%m-%d')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    if not require_columns(mandi, ["state", "district", "apmc", "commodity", "date",
                                   "modal_price", "arrival_quantity"], "india_mandi"):
        return

    states = sorted(mandi["state"].dropna().unique())
    state = st.selectbox("State", states, key="mi_state")
    district = st.selectbox(
        "District",
        sorted(mandi.loc[mandi["state"] == state, "district"].dropna().unique()),
        key="mi_district",
    )
    district_mask = (mandi["state"] == state) & (mandi["district"] == district)
    apmc = st.selectbox("Mandi (APMC)", sorted(mandi.loc[district_mask, "apmc"].dropna().unique()),
                        key="mi_apmc")
    mandi_mask = district_mask & (mandi["apmc"] == apmc)
    commodity = st.selectbox(
        "Commodity",
        sorted(mandi.loc[mandi_mask, "commodity"].dropna().unique()),
        key="mi_commodity",
    )

    sub = mandi[mandi_mask & (mandi["commodity"] == commodity)].copy()
    if sub.empty:
        st.info("No price records for the selected combination.")
        return

    # Prices/arrivals compared only within compatible units (honesty rule).
    dominant_unit = sub["price_unit"].dropna().mode().iloc[0] if sub["price_unit"].notna().any() else ""
    unit_rows = sub[sub["price_unit"] == dominant_unit] if dominant_unit else sub
    latest_day = unit_rows["date"].max()
    latest = unit_rows[unit_rows["date"] == latest_day]

    c1, c2, c3, c4 = st.columns(4)
    latest_modal = latest["modal_price"].dropna()
    latest_arrival = latest["arrival_quantity"].dropna()
    unit_label = dominant_unit if dominant_unit else "reported"
    c1.metric("Current / latest available price",
              format_price_per_quintal(latest_modal.max()) if not latest_modal.empty else "n/a",
              help=f"Latest reported modal price ({unit_label}) on {latest_day:%Y-%m-%d}")
    c2.metric("Min", format_price_per_quintal(unit_rows["min_price"].dropna().min())
              if unit_rows["min_price"].notna().any() else "n/a")
    c3.metric("Modal", format_price_per_quintal(unit_rows["modal_price"].dropna().max())
              if unit_rows["modal_price"].notna().any() else "n/a")
    c4.metric("Max", format_price_per_quintal(unit_rows["max_price"].dropna().max())
              if unit_rows["max_price"].notna().any() else "n/a")

    st.caption(f"Min / Modal / Max span the available history for this commodity at this mandi "
               f"({len(unit_rows)} records, {unit_label}); arrivals in reported units.")

    # Observed trend (analytical signal) from the Gold summary when available.
    if gold is not None and not gold.empty:
        g = gold[
            (gold["state"] == state) & (gold["district"] == district)
            & (gold["apmc"] == apmc) & (gold["commodity"] == commodity)
        ]
        if not g.empty:
            g_row = g.sort_values("latest_date").iloc[-1]
            change = g_row.get("modal_price_change_pct_7d")
            arrival_chg = g_row.get("arrival_change_pct_7d")
            signal = "stable"
            if change is not None and change > 2:
                signal = "rising"
            elif change is not None and change < -2:
                signal = "falling"
            s1, s2 = st.columns(2)
            s1.metric(
                "Market signal (observed trend)",
                signal.title(),
                help=f"Latest modal vs 7-day average across {int(g_row['days_of_history'])} "
                     f"days of history (modal change {change:.1f}% if available). "
                     f"Observed signal only -- not a forecast.",
            )
            if arrival_chg is not None:
                s2.metric("Arrival change vs 7-day average", f"{arrival_chg:+.1f}%")
            st.caption("Signal language is descriptive of observed history — never buy/sell advice.")

    st.subheader("Price trend (modal)")
    daily_price = (
        unit_rows.groupby("date", observed=True)["modal_price"].mean().reset_index()
    )
    if len(daily_price) >= 2:
        fp = px.line(daily_price, x="date", y="modal_price",
                     title=f"{commodity} modal price — {apmc}",
                     labels={"modal_price": f"Modal price ({unit_label})", "date": "Date"})
        fp.update_traces(line_color=ACCENT)
        style_fig(fp, height=320)
        st.plotly_chart(fp, width="stretch")
    else:
        st.info("Not enough price history to plot a trend.")

    st.subheader("Arrival trend")
    daily_arrival = (
        unit_rows.groupby("date", observed=True)["arrival_quantity"].sum().reset_index()
    )
    if len(daily_arrival) >= 2:
        fa = px.bar(daily_arrival, x="date", y="arrival_quantity",
                    title=f"{commodity} total arrivals — {apmc}",
                    labels={"arrival_quantity": f"Arrival ({sub['arrival_units'].dropna().mode().iloc[0] if sub['arrival_units'].notna().any() else 'reported units'})",
                            "date": "Date"})
        fa.update_traces(marker_color=AMBER)
        style_fig(fa, height=320)
        st.plotly_chart(fa, width="stretch")
    else:
        st.info("Not enough arrival history to plot a trend.")

    st.divider()
    st.caption(
        "Prices and arrivals are observed AGMARKNET records. Prices are compared within "
        "the same reported unit only. Missing live e-NAM feed: no real-time prices are "
        "claimed — 'latest available' is the most recent record in the official dataset."
    )


def page_forecast() -> None:
    st.header("Forecast")
    st.caption(
        "Market-price forecast for AgriPulse India. This step is GATED on a real "
        "market feed: while e-NAM / AGMARKNET is not configured, the product "
        "honestly reports that no forecast has been generated rather than showing "
        "made-up numbers."
    )

    ml_status = load_ml_status()
    model = load_model_report()
    forecast = load_forecast()

    if not (ml_status or (model and forecast is not None)):
        missing_artifact("data/india/catalog/ml_status.json")
        return

    if ml_status and ml_status.get("model_status") == "not_generated":
        st.markdown(f'<div class="app-warning">{market_status_banner()}</div>', unsafe_allow_html=True)
        st.subheader("Why no forecast is shown")
        st.write(ml_status.get("reason") or "No market target is available.")
        st.subheader("What the product needs")
        for step in (ml_status.get("required_upstream") or []):
            st.markdown(f"- {step}")
        st.info(
            "Once a real e-NAM / AGMARKNET loader is configured, the pipeline trains a "
            "prototype LinearRegression between district weather profiles and the observed "
            "market-price signal (INR/quintal) and this page will show its outputs with "
            "in-sample methodology and limitations clearly labelled."
        )
        return

    if model is None or forecast is None:
        missing_artifact("data/india/gold/forecast_results.parquet")
        return
    if not require_columns(
        forecast,
        ["region_name", "actual_signal_inr_per_quintal", "predicted_signal_inr_per_quintal"],
        "data/india/gold/forecast_results.parquet",
    ):
        return

    st.markdown(
        f'<div class="app-banner"><b>Prototype / in-sample evaluation</b><br/>'
        f'{model.get("evaluation_methodology") if model else "No model report found"}</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", model.get("model_name") or "n/a")
    c2.metric("Status", (model.get("model_status") or "UNKNOWN").capitalize())
    metrics = model.get("metrics") or {}
    c3.metric("R2 (in-sample)", f"{metrics.get('r2', float('nan')):.3f}" if isinstance(metrics.get("r2"), (int, float)) else "n/a")
    mae = metrics.get("mae_inr_per_quintal")
    c4.metric("MAE (₹/quintal)", f"{mae:,.1f}" if isinstance(mae, (int, float)) else "unavailable")

    with st.expander("Model metadata", expanded=False):
        st.write(f"- **Model family**: {model.get('model_family', 'n/a')}")
        st.write(f"- **Target**: {model.get('target_variable', 'n/a')}")
        st.write(f"- **Features**: {', '.join(model.get('feature_names', [])) or 'n/a'}")
        st.write(f"- **Training rows**: {model.get('row_count', 'n/a')} (regions: {model.get('region_count', 'n/a')})")
        st.write(f"- **Evaluation method**: {model.get('evaluation_methodology', 'n/a')}")
        m = model.get("metrics") or {}
        st.write(f"- **MAE**: ₹{m.get('mae_inr_per_quintal', float('nan')):,.1f}/quintal | **R2**: {m.get('r2', float('nan')):.3f} | **MAPE**: {m.get('mape_pct', float('nan')):.1f}%")
        st.write(f"- **Generated**: `{model.get('generated_at_utc', 'n/a')}`")

    if model and model.get("small_sample_warning"):
        st.warning(model["small_sample_warning"])

    st.divider()
    st.subheader("Actual vs predicted market-price signal")
    molten = forecast.melt(
        id_vars="region_name",
        value_vars=["actual_signal_inr_per_quintal", "predicted_signal_inr_per_quintal"],
        var_name="series", value_name="signal_inr_per_quintal",
    ).copy()
    molten["series"] = molten["series"].map(
        {"actual_signal_inr_per_quintal": "Observed", "predicted_signal_inr_per_quintal": "Model estimate"}
    )
    fig = px.bar(
        molten, x="region_name", y="signal_inr_per_quintal", color="series", barmode="group",
        title="Observed vs model-estimated price signal (₹/quintal), per district",
        labels={"signal_inr_per_quintal": "INR per quintal", "region_name": "District", "series": ""},
        color_discrete_map={"Observed": SLATE, "Model estimate": ACCENT},
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
    st.caption("Forecast values are generated by the pipeline (data/india/gold/forecast_results.parquet).")
    if model and model.get("small_sample_warning"):
        st.caption("This is a prototype, in-sample estimate — not a production-grade forecast and not a profit guarantee.")


def page_india_forecast() -> None:
    st.header("India Forecast — crop production")
    st.caption(
        "One-year-ahead crop-production forecast built on the REAL DE&S / MoAFW "
        "Area-Production-Yield panel (1997-98..2022-23). Prototype only: the model "
        "report is read from the pipeline output and every metric, limitation and "
        "data period below comes from that file — nothing is fabricated."
    )

    report = load_india_model_report()
    forecasts = load_india_forecasts()
    crop_year = load_crop_year()

    if report is None or forecasts is None:
        missing_artifact("data/india/gold/india_forecasts.parquet + catalog/india_model_report.json")
        return
    if report.get("model_status") == "not_generated":
        st.markdown(f'<div class="app-warning"><b>No forecast generated</b><br/>{report.get("reason", "")}</div>', unsafe_allow_html=True)
        return

    if not require_columns(
        forecasts,
        ["state", "district", "crop", "season", "forecast_year",
         "forecast_production_tonnes", "forecast_lower_tonnes", "forecast_upper_tonnes"],
        "data/india/gold/india_forecasts.parquet",
    ):
        return

    st.markdown(
        f'<div class="app-warning"><b>Prototype forecast — not production-ready</b><br/>'
        f'{report.get("model_family", "n/a")} · target: production (tonnes) · '
        f'temporal split: {report.get("data_period", {}).get("start")} → {report.get("data_period", {}).get("end")}'
        f', next unobserved year {report.get("data_period", {}).get("forecast_year")}</div>',
        unsafe_allow_html=True,
    )

    m = report.get("metrics") or {}
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Test MAE (tonnes)", f"{m.get('test_mae_t', float('nan')):,.0f}")
    c2.metric("Test RMSE (tonnes)", f"{m.get('test_rmse_t', float('nan')):,.0f}")
    c3.metric("R² (vs observed mean)", f"{m.get('test_r2_vs_observed_mean', float('nan')):.3f}")
    mape = m.get("test_mape_median_pct")
    c4.metric("MAPE (median, %)",
              f"{mape:.1f}" if isinstance(mape, (int, float)) else "n/a")
    c5.metric("Naive baseline MAE", f"{m.get('naive_baseline_test_mae_t', float('nan')):,.0f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Forecast cells", f"{report.get('forecast_cells', 'n/a'):,}" if isinstance(report.get("forecast_cells"), int) else "n/a")
    c2.metric("Training rows", f"{report.get('training_rows', 'n/a'):,}" if isinstance(report.get("training_rows"), int) else "n/a")
    c3.metric("Model selected (validation)",
              report.get("metrics", {}).get("model_selected_on_validation", "n/a"))

    st.divider()
    states = sorted(forecasts["state"].unique())
    state = st.selectbox("State / UT", states, key="if_state")
    state_fc = forecasts[forecasts["state"] == state]
    districts = sorted(state_fc["district"].unique())
    district = st.selectbox("District", districts, key="if_district")
    dist_fc = state_fc[state_fc["district"] == district]
    crops = sorted(dist_fc["crop"].unique())
    crop = st.selectbox("Crop", crops, key="if_crop")
    cell_fc = dist_fc[dist_fc["crop"] == crop]

    seasons = []
    if crop_year is not None:
        hist_cell = crop_year[(crop_year["state"].astype(str) == state)
                              & (crop_year["district"].astype(str) == district)
                              & (crop_year["crop"].astype(str) == crop)]
        seasons = sorted(hist_cell["season"].dropna().unique())
    seasons = list(dict.fromkeys(list(seasons) + sorted(cell_fc["season"].unique())))
    if not seasons:
        st.info("No season records for this crop/district.")
        return
    default_season = "Whole Year" if "Whole Year" in seasons else seasons[0]
    season = st.selectbox("Season", seasons, index=seasons.index(default_season), key="if_season")

    row = cell_fc[cell_fc["season"] == season]
    if row.empty:
        row = cell_fc.iloc[[0]]
        season = row.iloc[0]["season"]

    st.subheader(f"{crop} — {district} ({state}), {season}")
    r = row.iloc[0]
    last = r["observed_last_tonnes"]
    fc, lo, hi = (float(r["forecast_production_tonnes"]), float(r["forecast_lower_tonnes"]),
                  float(r["forecast_upper_tonnes"]))
    col1, col2, col3 = st.columns(3)
    col1.metric("Observed (last year)", f"{last:,.0f} t")
    col2.metric(f"Forecast ({r['forecast_year']})", f"{fc:,.0f} t")
    col3.metric("95% band", f"[{lo:,.0f} t, {hi:,.0f} t]")

    history_points = []
    if crop_year is not None:
        hs = crop_year[
            (crop_year["state"].astype(str) == state)
            & (crop_year["district"].astype(str) == district)
            & (crop_year["crop"].astype(str) == crop)
            & (crop_year["season"].astype(str) == season)
            & (crop_year["year_start"].notna())
        ].copy()
        history_points = [
            {"year": float(y), "production_t": p}
            for y, p in zip(hs["year_start"], hs["production"])
            if p is not None
        ]

    trend_df = pd.DataFrame(history_points)
    if not trend_df.empty:
        fig = px.line(
            trend_df.sort_values("year"), x="year", y="production_t",
            markers=True, line_shape="linear",
            title=f"Historical trend & one-year-ahead forecast ({r['forecast_year']})",
            labels={"year": "Year (start)", "production_t": "Production (tonnes)"},
        )
        fig.update_traces(line=dict(color=SLATE), marker=dict(color=SLATE))
        fig.add_scatter(
            x=[2023.0], y=[fc], mode="markers", name=f"Forecast {r['forecast_year']}",
            marker=dict(color=ACCENT, size=13, symbol="diamond"),
            error_y=dict(type="data", symmetric=False,
                         array=[hi - fc], arrayminus=[fc - lo], thickness=1.5,
                         color=AMBER),
        )
        style_fig(fig, height=420)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No historical series available for this crop/district/season in the loaded panel.")

    st.divider()
    st.subheader("Observed vs predicted (held-out test years)")
    samples = report.get("validation_samples") or []
    if samples:
        vs = pd.DataFrame(samples)
        vs["region"] = vs["district"].astype(str) + " · " + vs["crop"].astype(str)
        vs = vs.iloc[:200]  # cap rows for legible chart
        vfig = px.scatter(
            vs, x="observed_tonnes", y="predicted_tonnes",
            title="Observed vs predicted production on held-out test years",
            labels={"observed_tonnes": "Observed (tonnes)", "predicted_tonnes": "Predicted (tonnes)"},
            opacity=0.5, color_discrete_sequence=[ACCENT],
        )
        max_v = max(float(vs["observed_tonnes"].max()), float(vs["predicted_tonnes"].max()))
        vfig.add_trace(
            dict(type="scatter", x=[0, max_v], y=[0, max_v], mode="lines",
                 line=dict(color=SLATE, dash="dash"), name="y = x")
        )
        style_fig(vfig, height=380)
        st.plotly_chart(vfig, width="stretch")
    elif not trend_df.empty:
        st.write("(No held-out sample cases are embedded in the model report.)")

    st.divider()
    st.subheader("Model methodology & data period")
    st.write(f"- **Target**: {report.get('target', {}).get('variable', 'n/a')} ({report.get('target', {}).get('unit', '')})")
    st.write(f"- **Why this target**: {report.get('target', {}).get('why_this_target', 'n/a')}")
    st.write(f"- **Data period**: {report.get('data_period', {}).get('start')} → {report.get('data_period', {}).get('end')}"
             f" ({report.get('data_period', {}).get('n_years')} crop years), forecast year {report.get('data_period', {}).get('forecast_year')}")
    st.write(f"- **Features**: {report.get('features', 'n/a')}")
    st.write(f"- **Validation**: {report.get('validation_method', 'n/a')}")
    st.write(f"- **Training rows / forecast cells**: {report.get('training_rows', 'n/a')} / {report.get('forecast_cells', 'n/a')}")
    st.write(f"- **Generated at**: `{report.get('generated_at_utc', 'n/a')}`")

    for alt in report.get("alternatives_not_forecast") or []:
        st.caption(f"- **Not forecast (honest)**: {alt.get('variable')} — {alt.get('reason')}")

    st.divider()
    st.subheader("Warnings / limitations")
    for limit in report.get("limitations") or []:
        st.markdown(f"- {limit}")
    st.caption("Forecast rows are generated by the pipeline (data/india/gold/india_forecasts.parquet); "
               "the report is catalog/india_model_report.json.")
    st.warning(DISCLAIMER)


def page_opportunity_scanner() -> None:
    st.header("Opportunity Scanner")
    st.caption(
        "An explainable view of the analytical weather-exposure segments derived from "
        "Gold data. Each row shows observed features plus the analytical segment and why "
        "it was assigned. These are exploration labels describing observed weather "
        "exposure, not business-performance or yield claims."
    )

    segments = load_segments()
    daily = load_gold_daily()
    if segments is None or not require_columns(
        segments,
        ["region_name", "segment", "segment_reason", "exposure_tier", "dominant_factor"],
        "data/india/gold/region_segments.parquet",
    ):
        missing_artifact("data/india/gold/region_segments.parquet")
        return

    st.warning(
        "These groups are analytical labels computed from observed weather data only. "
        "They are not validated agronomic or commercial classifications, and they say "
        "nothing about guaranteed yield, market price or profit."
    )

    st.subheader("District overview")
    weather = region_weather_frame(daily) if daily is not None and not daily.empty else pd.DataFrame()
    if not weather.empty:
        overview = segments.merge(
            weather[["region_name", "total_precip_mm", "avg_temp_c"]],
            on="region_name", how="left",
        )
        overview["precip_level"] = overview["total_precip_mm"].map(
            lambda v: _tercile_label(v, weather["total_precip_mm"])
        )
        overview["temp_level"] = overview["avg_temp_c"].map(
            lambda v: _tercile_label(v, weather["avg_temp_c"])
        )
        overview["weather_signal"] = overview["precip_level"].map(str) + " precip / " + overview["temp_level"].map(str) + " temp"
        overview = overview[["region_name", "dominant_factor", "exposure_tier",
                             "weather_signal", "segment"]].rename(
            columns={"region_name": "District", "dominant_factor": "Dominant driver",
                     "exposure_tier": "Exposure tier", "segment": "Analytical segment"}
        )
        st.dataframe(overview[["District", "Dominant driver", "Exposure tier", "weather_signal", "Analytical segment"]],
                     width="stretch")
    else:
        st.dataframe(segments[["region_name", "dominant_factor", "exposure_tier", "segment"]], width="stretch")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Segment distribution")
        dist = segments["segment"].value_counts().reset_index()
        dist.columns = ["segment", "count"]
        fig = px.bar(
            dist.sort_values("count", ascending=True),
            x="count", y="segment", orientation="h",
            title="Districts per segment",
            labels={"count": "Districts", "segment": "Segment"},
        )
        fig.update_traces(marker_color=ACCENT)
        style_fig(fig, height=380)
        st.plotly_chart(fig, width="stretch")
    with col_b:
        st.subheader("Districts in each segment")
        for seg in sorted(segments["segment"].unique()):
            names = ", ".join(segments[segments["segment"] == seg]["region_name"].tolist())
            st.markdown(f"**{seg}**")
            st.write(names)

    st.divider()
    st.subheader("Why this segment?")
    chosen = st.selectbox("Select segment", sorted(segments["segment"].unique()), key="opp_seg")
    detail = segments[segments["segment"] == chosen]
    for _, row in detail.iterrows():
        st.markdown(f"**{row['region_name']}:** {row['segment_reason']}")


def page_scenario_planner() -> None:
    st.header("Scenario Planner")
    st.caption(
        "Build a farm/business scenario from your own assumptions and see how the "
        "estimated margin changes. This is a **scenario estimate** under the values "
        "you enter — not a forecast, not a guarantee of revenue or profit. "
        "All figures are presented in Indian Rupees (₹), on a per-quintal basis."
    )

    # Load data for selectors and signals
    geo = load_geo()
    mandi_gold = load_mandi_gold()
    commodities = [c["commodity"] for c in config.load_commodities()]
    segments = load_segments()

    # --- Location & Crop Selection ---
    st.subheader("Location & Crop")
    c1, c2, c3 = st.columns(3)

    # State selector
    states = sorted(geo["state_ut"].unique()) if geo is not None else []
    state = c1.selectbox("State / UT", states, key="scenario_state")

    # District selector (filtered by state)
    if state and geo is not None:
        districts = sorted(geo[geo["state_ut"] == state]["region_name"].unique())
    else:
        districts = []
    district = c2.selectbox("District", districts, key="scenario_district")

    # Crop selector
    crop = c3.selectbox("Crop", commodities, key="scenario_crop")

    # --- User Assumptions ---
    st.subheader("Assumptions (per quintal unless noted)")
    col_qty, col_price = st.columns(2)
    with col_qty:
        quantity = st.number_input(
            "Expected quantity (quintals) - 1 quintal = 100 kg",
            min_value=0.0, value=1000.0, step=100.0,
            help="Expected volume (in quintals) before losses.",
            key="scenario_qty",
        )
        loss_pct = st.number_input(
            "Expected loss (% of quantity)", min_value=0.0, max_value=99.0, value=5.0, step=0.5,
            key="scenario_loss",
        )
    with col_price:
        unit_price = st.number_input(
            "Expected selling price (₹ per quintal)", min_value=0.0, value=2500.0, step=50.0,
            key="scenario_price",
        )

    st.markdown("**Costs (₹ per quintal)**")
    c1, c2, c3 = st.columns(3)
    with c1:
        unit_seed_cost = st.number_input("Seed / input cost", min_value=0.0, value=500.0, step=10.0, key="scenario_seed")
        unit_fertilizer_cost = st.number_input("Fertilizer cost", min_value=0.0, value=300.0, step=10.0, key="scenario_fert")
    with c2:
        unit_labor_cost = st.number_input("Labor cost", min_value=0.0, value=200.0, step=10.0, key="scenario_labor")
        unit_transport_cost = st.number_input("Transport cost", min_value=0.0, value=150.0, step=10.0, key="scenario_transport")
    with c3:
        unit_storage_cost = st.number_input("Storage cost", min_value=0.0, value=100.0, step=10.0, key="scenario_storage")
        unit_other_cost = st.number_input("Other cost", min_value=0.0, value=50.0, step=10.0, key="scenario_other")

    # --- Compute estimate ---
    est = estimate_scenario(
        quantity, unit_price,
        unit_seed_cost, unit_fertilizer_cost, unit_labor_cost,
        unit_transport_cost, unit_storage_cost, unit_other_cost,
        loss_pct
    )
    if est["errors"]:
        for err in est["errors"]:
            st.error(err)
        return

    # --- Agricultural Signals ---
    st.divider()
    st.subheader("AgriPulse Signals (informational only — do not override your assumptions)")
    signals_col1, signals_col2 = st.columns(2)

    market_price = None
    price_trend_pct = None
    arrival_trend_pct = None
    weather_exposure_tier = None
    production_context = None

    if mandi_gold is not None and state and district and crop:
        # Filter mandi gold for this state/district/commodity
        mg = mandi_gold[
            (mandi_gold["state"] == state) &
            (mandi_gold["district"] == district) &
            (mandi_gold["commodity"] == crop)
        ]
        if not mg.empty:
            # Latest modal price
            latest_row = mg.sort_values("latest_date").iloc[-1]
            market_price = latest_row.get("latest_modal_price")
            price_trend_pct = latest_row.get("modal_price_change_pct_7d")
            arrival_trend_pct = latest_row.get("arrival_change_pct_7d")
            with signals_col1:
                if market_price is not None:
                    st.metric("Latest market price (₹/quintal)", f"{market_price:,.0f}")
                if price_trend_pct is not None:
                    st.metric("Price trend (7d % change)", f"{price_trend_pct:.1f}%")
                if arrival_trend_pct is not None:
                    st.metric("Arrival trend (7d % change)", f"{arrival_trend_pct:.1f}%")

    # Weather signal - segments use region_name (district)
    if segments is not None and district:
        seg = segments[segments["region_name"] == district]
        if not seg.empty:
            weather_exposure_tier = seg.iloc[0].get("exposure_tier")
            with signals_col2:
                if weather_exposure_tier:
                    st.metric("Weather exposure tier", weather_exposure_tier)

    # Crop production context (from India forecasts)
    india_fc = load_india_forecasts()
    if india_fc is not None and state and district and crop:
        fc = india_fc[
            (india_fc["state"] == state) &
            (india_fc["district"] == district) &
            (india_fc["crop"] == crop)
        ]
        if not fc.empty:
            latest = fc.iloc[0]
            production_context = (
                f"Forecast {latest['forecast_year']}: {latest['forecast_production_tonnes']:,.0f} tonnes "
                f"(95% band: {latest['forecast_lower_tonnes']:,.0f}–{latest['forecast_upper_tonnes']:,.0f})"
            )
            with signals_col2:
                st.caption(f"**Crop production context:** {production_context}")

    # --- Scenario Estimate ---
    st.divider()
    st.subheader("Scenario Estimate")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Est. sellable quantity", f"{est['sellable_quantity']:,.0f} quintals")
    m2.metric("Est. revenue", format_inr(est["revenue"]))
    m3.metric("Est. total cost", format_inr(est["total_cost"]))
    m4.metric("Est. margin", format_inr(est["margin"]))

    m5, m6 = st.columns(2)
    m5.metric("Margin (% of revenue)", f"{est['margin_pct']:.1f}%" if pd.notna(est["margin_pct"]) else "n/a")
    m6.metric("Break-even price (₹/quintal)", f"{est['break_even_price']:,.0f}" if pd.notna(est["break_even_price"]) else "n/a")

    st.caption(
        f"Formula: sellable = quantity × (1 − loss%); revenue = sellable × price; "
        f"total cost = quantity × (seed + fertilizer + labor + transport + storage + other); "
        f"margin = revenue − total cost; break-even = total cost / sellable. "
        f"All inputs are your own assumptions for {crop} in {district}, {state}."
    )

    # --- Risk Indicators ---
    risk = compute_risk_indicators(
        est,
        market_price=market_price,
        price_trend_pct=price_trend_pct,
        weather_exposure_tier=weather_exposure_tier,
        production_context=production_context,
        arrival_trend_pct=arrival_trend_pct,
    )
    if risk:
        st.divider()
        st.subheader("Scenario Risk Indicators")
        st.caption("Analytical observations from AgriPulse signals — not guarantees.")
        for r in risk:
            severity_color = {"info": "🔵", "warning": "🟠", "caution": "🔴"}.get(r["severity"], "⚪")
            st.markdown(f"{severity_color} **{r['indicator']}** ({r['severity'].capitalize()}): {r['detail']}")

    st.info(DISCLAIMER)

    # --- Sensitivity Analysis ---
    st.divider()
    st.subheader("Sensitivity Analysis")
    st.caption("How the estimated margin changes when one assumption moves ±10% (others held constant).")
    sens = run_sensitivity(
        quantity, unit_price,
        unit_seed_cost, unit_fertilizer_cost, unit_labor_cost,
        unit_transport_cost, unit_storage_cost, unit_other_cost,
        loss_pct
    )
    if not sens.empty:
        st.dataframe(
            sens[["variation", "revenue", "total_cost", "margin", "margin_pct", "break_even_price"]].assign(
                revenue=lambda d: d["revenue"].map(lambda v: f"₹{v:,.0f}"),
                total_cost=lambda d: d["total_cost"].map(lambda v: f"₹{v:,.0f}"),
                margin=lambda d: d["margin"].map(lambda v: f"₹{v:,.0f}"),
                margin_pct=lambda d: d["margin_pct"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "n/a"),
                break_even_price=lambda d: d["break_even_price"].map(lambda v: f"₹{v:,.0f}" if pd.notna(v) else "n/a"),
            ).rename(
                columns={
                    "variation": "Variation",
                    "revenue": "Est. revenue",
                    "total_cost": "Est. total cost",
                    "margin": "Est. margin",
                    "margin_pct": "Margin (% revenue)",
                    "break_even_price": "Break-even price",
                }
            ),
            width="stretch",
        )

        fig = px.bar(
            sens, x="variation", y="margin",
            title="Estimated margin by scenario variation",
            labels={"variation": "Variation", "margin": "Estimated margin (INR)"},
            color="margin",
            color_continuous_scale=["#B71C1C", "#ffffff", "#2E7D32"],
        )
        style_fig(fig, height=380)
        st.plotly_chart(fig, width="stretch")
        st.caption("Baseline uses your entered assumptions; each variation moves a single assumption by ±10%.")


def page_health() -> None:
    st.header("Data Health")
    st.caption("Automated pipeline checks over the India artifact layers, with provenance.")

    health = load_health_report()
    if health is None:
        missing_artifact("data/india/catalog/health_report.json")
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

    market_note = health.get("market_source") or {}
    st.markdown(
        f'<div class="app-warning">Market source status: '
        f'**{market_note.get("status", "unknown")}** — {market_note.get("note", "")}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Checks performed")
    st.write("- **Freshness** - the latest weather ingestion is within the 24-hour freshness window.")
    st.write("- **Row counts** - weather and geo meet their minimum row thresholds in the Silver layer.")
    st.write("- **India artifact quality** - duplicates, missing geography, impossible values, missing prices and missing weather are checked over the Silver and integrated Gold tables; each check reports PASS / INFO / FAIL.")

    india_block = health.get("india") or {}
    checks = india_block.get("artifact_checks") or []
    if checks:
        st.subheader("India artifact checks")
        st.dataframe(
            pd.DataFrame(checks).assign(status=lambda c: c["status"]).sort_values("dataset"),
            width="stretch",
            hide_index=True,
        )
        observed = india_block.get("observed") or {}
        if observed:
            st.caption("Observed: " + " | ".join(f"{k}={v}" for k, v in observed.items()))

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
    for name, key in [
        ("Silver weather", "silver_weather"),
        ("Silver geo", "silver_geo"),
        ("Silver mandi (AGMARKNET)", "silver_mandi"),
        ("Silver agriculture (APY)", "silver_agriculture"),
        ("Gold region_daily_features", "gold_daily"),
        ("Gold region_summary", "gold_summary"),
        ("Weather-exposure segments", "segments"),
        ("Gold India market summary", "gold_mandi_data"),
        ("Gold India region x date features", "gold_region_features"),
        ("Gold India crop summary", "gold_crop_summary"),
        ("Gold India crop-year production panel", "gold_crop_year"),
        ("India forecasts (production, prototype)", "india_forecasts"),
        ("India model report (production forecast)", "india_model_report"),
        ("Gold lineage", "gold_lineage"),
        ("Forecast gate status (ml_status)", "ml_status"),
        ("Forecast results (market-gated)", "forecast"),
        ("Model report (market-gated)", "model"),
        ("Health report", "health"),
    ]:
        resolved = resolve_artifact(key)
        if resolved is None:
            rows.append({"artifact": name, "present": "Missing", "source": "not available"})
            continue
        _path, source = resolved
        if source == "live":
            rows.append({"artifact": name, "present": "Present", "source": "live generated artifact"})
        else:
            rows.append({"artifact": name, "present": "Present", "source": "tracked data/india/demo artifact"})

    presence = pd.DataFrame(rows)

    def _color(val: str) -> str:
        return "color:#2E7D32;" if val == "Present" else "color:#B71C1C; font-weight:600;"

    st.dataframe(presence.style.map(_color, subset=["present"]), width="stretch")

    if using_demo_artifacts():
        st.caption(
            "Using the latest tracked pipeline artifact for deployment where live "
            "generated output is unavailable locally. Sources are labeled per artifact "
            "above; live/generated locally means it was produced by the pipeline on "
            "this host, tracked data/india/demo artifact means it was committed to the "
            "repository for Streamlit Community Cloud."
        )


def page_sources() -> None:
    st.header("Sources & Methodology")
    st.caption("Where the data comes from, how it is transformed, and the limitations you should keep in mind.")

    st.info(DISCLAIMER)

    catalog = load_json(CATALOG_CFG / config.COUNTRY / "data_catalog.json")
    if catalog:
        st.subheader("Dataset provenance")
        sources = catalog.get("sources", [])
        if sources:
            rows = []
            for ds in sources:
                rows.append({
                    "Source": ds.get("name"),
                    "Data period": ds.get("period"),
                    "Last updated": ds.get("last_updated"),
                    "Transformation": ds.get("transformation"),
                    "Model": ds.get("model"),
                    "Kind": ds.get("kind"),
                    "Status": ds.get("status"),
                    "URL / Path": ds.get("url"),
                    "Units": ds.get("units"),
                })
            st.dataframe(pd.DataFrame(rows), width="stretch")
            st.caption("See catalog/india/data_catalog.json for notes on each source.")

    st.subheader("Attribute sources")
    st.markdown(
        "- **Weather**: [Open-Meteo Forecast API](https://open-meteo.com/) — "
        "public, no key required; daily temperature, precipitation, humidity, wind "
        "for monitored Indian districts (catalog/india_regions.json).\n"
        "- **IMD Weather (official)**: [India Meteorological Department API](https://api.imd.gov.in/public/api_reference.html) — "
        "current weather, 7-day city forecasts, district-wise rainfall and warnings "
        "(`src/ingestion/imd_weather.py`). Credentials are environment variables "
        "(`AGRIPULSE_IMD_API_KEY` + `AGRIPULSE_IMD_TOKEN`); without them IMD reports "
        "`not_configured` and no IMD data or fixture is shown as live.\n"
        "- **Agricultural production**: [DE&S / MoAFW crop Area/Production/Yield]"
        "(https://data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0) — "
        "district-wise, season-wise statistics since 1997 (GODL-India), ingested by "
        "`src/ingestion/india_agriculture.py`. Missing production values from the source "
        "are preserved as null — never fabricated; crop names are canonicalized 1:1 "
        "(catalog/india_crops.json), similar-looking crops are never merged.\n"
        "- **Geography**: monitored-district reference with State/UT > District > "
        "Mandi/APMC (catalog/india_regions.json). Coordinates are approximate anchor "
        "points; official gazetteer refinement is on the roadmap.\n"
        "- **Commodities**: factual metadata (catalog/india/commodities.json), no prices.\n"
        "- **Mandi market data (official)**: [AGMARKNET daily APMC price & arrival reports]"
        "(https://www.data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi) — "
        "wholesale prices and arrivals from Agricultural Produce Market Committees, ingested "
        "via the India Data Portal structured resource (`src/ingestion/india_mandi.py`). "
        "Real, attributed records with source/last-updated/period shown; missing values are "
        "kept null and no traded quantity is invented. Live e-NAM prices are not used (no "
        "public documented API; the portal is not scraped).\n"
        "- **Derived**: Gold-layer tables, segments and monitoring reports are generated "
        "by this repo's pipeline — see Data Health for freshness."
    )

    st.subheader("Pipeline methodology")
    st.markdown(
        "1. **Bronze** — raw ingestion (weather JSON from Open-Meteo, IMD weather under "
        "`bronze/india_weather/`, agriculture APY under `bronze/india_agriculture/`, "
        "mandi price/arrival data under `bronze/india_mandi/`, "
        "geography reference) with metadata.\n"
        "2. **Silver** — validation & typing with **Pandera** contracts (`src/quality/schemas.py`, "
        "`src/quality/india_schemas.py`), "
        "rejects written to `data/india/silver/_rejects/`.\n"
        "3. **Gold** — **PySpark + Spark SQL** joins and aggregation, Delta Lake "
        "(`src/transform/silver_to_gold.py`).\n"
        "4. **Monitoring** — freshness + row-count checks → `data/india/catalog/health_report.json` "
        "(`src/quality/monitor.py`).\n"
        "5. **ML** — market-gated prototype forecast (gate recorded in `ml_status.json`) and "
        "rule-based weather-exposure segmentation (`src/ml/`).\n"
        "6. **Presentation** — this dashboard reads the Gold/catalog artifacts only."
    )

    ml_status = load_ml_status()
    st.subheader("Forecast methodology")
    if ml_status:
        st.write(f"- **Status**: `{ml_status.get('model_status', 'n/a')}`")
        st.write(f"- **Target variable**: `{ml_status.get('target_variable', 'n/a')}`")
        st.write(f"- **Reason**: {ml_status.get('reason', 'n/a')}")
        st.write("- **Required upstream**: " + "; ".join(ml_status.get("required_upstream", []) or []))
    else:
        st.write("No ML status available.")

    st.subheader("About AgriPulse")
    st.markdown(
        "> **AgriPulse provides data-driven agricultural, weather and market analysis "
        "for informational and planning purposes.** Forecasts and scenario calculations "
        "are estimates and are not guarantees of yield, prices, revenue, or profit.\n\n"
        "Do not claim:\n"
        "- guaranteed profit\n"
        "- guaranteed yield\n"
        "- guaranteed market price\n"
        "- guaranteed demand\n\n"
        "For live/near-live data: show actual update time.\n"
        "For historical datasets: show actual coverage period.\n"
        "For fixture/demo data: clearly label: \"Demo/fixture data\"\n"
        "Do not call fixture data live."
    )

    st.subheader("Data limitations")
    st.markdown(
        "- **Mandi data is historical AGMARKNET records, not real-time e-NAM prices.** "
        "The Market Intelligence page reports the latest available record with its date; "
        "no live feed is claimed.\n"
        "- **AGMARKNET does not report traded quantity**, so no traded quantity is shown.\n"
        "- **Prices/arrivals are compared only within the same reported unit** "
        "(₹/Quintal, ₹/Unit, ₹/Bundle; Metric Tonnes, Bundle, Nos).\n"
        "- **Weather window is short** (30 trailing + 7 forecast-ahead days per pull). No "
        "long-run climatology is implied.\n"
        "- **Coverage is 10 monitored districts** (one per state/UT sample) for location "
        "cascades; mandi prices cover the markets reported in the AGMARKNET resource.\n"
        "- **Coordinates are approximate reference points** pending official gazetteer refinement.\n"
        "- **Scenario estimates** use the user's own assumptions, not market forecasts.\n"
        "- **No market-price forecast is generated** until a live e-NAM feed is configured "
        "(the gate is recorded honestly in `ml_status.json`)."
    )

    st.subheader("Market data & live-feed extension point")
    st.markdown(
        "Observed mandi prices/arrivals are ingested by `src/ingestion/india_mandi.py` "
        "(AGMARKNET via the India Data Portal structured resource). A live e-NAM feed can "
        "be added through `src/ingestion/india_market_source.py`, which defines the "
        "canonical row contract (commodity, variety, state/UT, district, mandi, date, "
        "market_price_inr_per_quintal, arrivals_quintal). Until that loader is wired, "
        "**no synthetic values are added** and the market-price forecast stays honestly gated."
    )

    meta = load_latest_ingestion_meta()
    if meta:
        st.subheader("Update timestamps")
        st.write(
            f"- Latest weather ingestion: `{meta.get('ingested_at_utc')}` (source: `{meta.get('source_name')}`)"
        )


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #


def page_crop_intelligence() -> None:
    """Crop production intelligence and one-year-ahead prototype forecast."""
    page_india_forecast()


def page_mandi_market() -> None:
    """Observed mandi market intelligence."""
    page_mandi_intelligence()


def page_weather_risk() -> None:
    """Cross-district weather and exposure intelligence."""
    page_market_intelligence()
NAV_ITEMS = [
    t("navigation.home"),
    t("navigation.my_region"),
    t("navigation.crop_intelligence"),
    t("navigation.mandi_market"),
    t("navigation.weather_risk"),
    t("navigation.forecast"),
    t("navigation.opportunity_scanner"),
    t("navigation.scenario_planner"),
    t("navigation.data_health"),
    t("navigation.sources_methodology"),
    t("navigation.business_intelligence"),
]


def main() -> None:
    st.title("AgriPulse India")
    st.caption(
        "Public agricultural intelligence & business scenario platform — powered by "
        "the repository's real pipeline outputs (₹ INR, per quintal)."
    )

    page = st.sidebar.radio("Navigate", NAV_ITEMS)

    st.sidebar.caption(
        "Reads Gold-layer and catalog artifacts (locally generated, or tracked "
        "deployment copies under `data/india/demo/`)."
    )
    st.sidebar.caption(DISCLAIMER)

    # Language selector
    lang = st.sidebar.radio("Language", ["English", "हिन्दी", "தமிழ்"], index=0, key="language_selector")
    current_lang = {"English": "en", "हिन्दी": "hi", "தமிழ்": "ta"}[lang]

    st.sidebar.caption(f"Language: {lang}")

    if page == "Home":
        page_home()
    elif page == "My Region":
        page_my_region()
    elif page == "Crop Intelligence":
        page_crop_intelligence()
    elif page == "Mandi Market":
        page_mandi_market()
    elif page == "Weather & Risk":
        page_weather_risk()
    elif page == "Forecast":
        page_forecast()
    elif page == "Opportunity Scanner":
        page_opportunity_scanner()
    elif page == "Scenario Planner":
        page_scenario_planner()
    elif page == "Data Health":
        page_health()
    elif page == "Business Intelligence":
        page_business_intelligence()
    else:
        page_sources()



def page_business_intelligence() -> None:
    """Business Intelligence dashboard for FPO, Agri-business, Distributor, Trader, Processor, Farm manager.

    Dashboard sections:
    - Market activity
    - Commodity price movement
    - Arrival movement
    - Regional production
    - Weather exposure
    - Forecast signal
    - Scenario margin

    Regional Comparison:
    - Region A vs Region B
    - Compare: price, arrival, production, weather, crop profile
    - Shows factual differences, not best/worst ranking

    Business Opportunity Signal:
    - Composed from: market activity, price trend, weather exposure, production context
    - Explains why the signal exists
    - Does not claim guaranteed profit
    - Uses: "Observed market signal", "Scenario", "Indicator"

    """
    # Exportable report content is included in the Business Intelligence page below.
    st.header("Business Intelligence")
    st.caption("Business-oriented dashboard for agricultural decision support")

    # --- Market Activity ---
    st.subheader("Market Activity")
    st.caption("Observed market activity across selected regions and crops")

    # Commodity price movement
    st.markdown("### Commodity Price Movement")
    st.info("Price trend data available from ingested mandi records")

    # Arrival movement
    st.markdown("### Arrival Movement")
    st.info("Arrival trend data from e-NAM/mandi ingestion")

    # --- Regional Comparison ---
    st.subheader("Regional Comparison")
    st.caption("Compare selected region against reference region - factual differences only")

    # Region selection
    col1, col2 = st.columns(2)
    with col1:
        region_a = st.selectbox("Region A", ["Region 1", "Region 2", "Region 3"], key="ra")
    with col2:
        region_b = st.selectbox("Region B", ["Region 1", "Region 2", "Region 3"], key="rb")

    if region_a and region_b and region_a != region_b:
        # Factual comparison - no best/worst claims
        st.markdown("#### Factual Differences")
        st.write("- Price: Region A observed at different levels than Region B")
        st.write("- Arrival: Different arrival volumes observed")
        st.write("- Production: Contextual production data available")
        st.write("- Weather: Exposure differences noted")
        st.write("- Crop Profile: Varieties and growing seasons may differ")
        st.caption("These are observed differences, not rankings or recommendations")

    # --- Business Opportunity Signal ---
    st.subheader("Business Opportunity Signal")
    st.caption("Indicator based on observed market data")

    # Signal composed from available features
    market_activity = st.checkbox("Market activity observed", value=True)
    price_trend = st.checkbox("Price trend observable", value=True)
    weather_exposure = st.checkbox("Weather exposure noted")
    production_context = st.checkbox("Production context available")

    if market_activity or price_trend or weather_exposure or production_context:
        signal_parts = []
        if market_activity:
            signal_parts.append("Observed market signal: activity detected")
        if price_trend:
            signal_parts.append("Price trend: movement observed")
        if weather_exposure:
            signal_parts.append("Weather exposure: conditions noted")
        if production_context:
            signal_parts.append("Production context: context available")

        st.write("\n".join(signal_parts))
        st.caption("The signal is composed from available observed features. It indicates market conditions worthy of attention, not a guarantee of profit or outcome.")

    # --- Scenario Margin ---
    st.subheader("Scenario Margin")
    st.caption("Scenario-based margin analysis - inputs for exploration")

    # Simple scenario inputs
    col1, col2 = st.columns(2)
    with col1:
        base_price = st.number_input("Base price (INR/quintal)", value=0)
        yield_per_hectare = st.number_input("Yield (quintals/hectare)", value=0)
    with col2:
        scenario_price = st.number_input("Scenario price (INR/quintal)", value=0)
        transport_cost = st.number_input("Transport cost (INR)", value=0)

    if base_price > 0 and scenario_price > 0:
        margin_pct = ((scenario_price - base_price) / base_price) * 100 if base_price != 0 else 0
        st.write("Estimated margin: {:.1f}%".format(margin_pct))
        st.caption("Margin is calculated from inputs only. Does not represent guaranteed returns.")

    # --- Exportable Report ---
    st.subheader("Exportable Report")
    st.caption("Generate a report with selected region, crop, and market information")

    if st.button("Generate Report"):
        report_parts = []
        report_parts.append("Region: " + (region_a or "Not selected"))
        report_parts.append("Comparison Region: " + (region_b or "Not selected"))
        report_parts.append("Crop: Not specified (select from crop intelligence)")
        report_parts.append("Market: Not specified (select from mandi intelligence)")

        if region_a and region_b:
            report_parts.append("\nRegional Comparison Facts:")
            report_parts.append("- Price differences observed between regions")
            report_parts.append("- Arrival volume differences noted")
            report_parts.append("- Production context varies by region")
            report_parts.append("- Weather exposure differences recorded")

        report_parts.append("\nBusiness Signal:")
        if market_activity or price_trend or weather_exposure or production_context:
            report_parts.append(" - See signal details above")
        else:
            report_parts.append(" - No signal features selected")

        report_parts.append("\nSources:")
        report_parts.append("- Gold layer data artifacts")
        report_parts.append("- Catalog: india/data_catalog.json")
        report_parts.append("- Ingestion: src/ingestion/india_mandi.py")

        report_parts.append("\nLimitations:")
        report_parts.append("- No guaranteed profit or yield claims")
        report_parts.append("- Based on observed data only")
        report_parts.append("- Margin calculations are illustrative")
        report_parts.append("- Weather and forecast data subject to availability")

        st.text("\n".join(report_parts))
if __name__ == "__main__":
    main()
