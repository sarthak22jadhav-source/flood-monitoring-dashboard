"""
CWC Flood Monitoring & Predictive Dashboard
============================================
Streamlit app: station search, KPI cards, station profile, Is_Flood
predictive model, water-level trends, GIS map, rainfall-vs-surge logs.

Run:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import charts as C
import data as D
import model as M

st.set_page_config(
    page_title="CWC Flood Monitoring & Predictive Dashboard",
    page_icon="\U0001F30A",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Custom CSS
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f0f9ff 0%, #ffffff 220px); }
    header[data-testid="stHeader"] { background: transparent; }
    .hero {
        background: linear-gradient(120deg, #0c4a6e 0%, #0369a1 55%, #0ea5e9 100%);
        color: white; padding: 18px 26px; border-radius: 14px;
        margin-bottom: 6px;
    }
    .hero h1 { margin: 0; font-size: 1.7rem; }
    .hero p  { margin: 4px 0 0 0; opacity: .85; font-size: .92rem; }
    .kpi {
        background: white; border: 1px solid #e2e8f0; border-left: 5px solid var(--accent, #0ea5e9);
        border-radius: 12px; padding: 14px 16px; min-height: 108px;
        box-shadow: 0 1px 3px rgba(15,23,42,.08);
    }
    .kpi .label { font-size: .78rem; color: #64748b; text-transform: uppercase;
                  letter-spacing: .04em; }
    .kpi .value { font-size: 1.85rem; font-weight: 700; color: #0f172a; line-height: 1.25; }
    .kpi .sub   { font-size: .75rem; color: #94a3b8; }
    .badge {
        display: inline-block; padding: 10px 22px; border-radius: 999px;
        font-weight: 800; font-size: 1.05rem; color: white; letter-spacing: .03em;
    }
    div[data-testid="stMetric"] {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 10px 14px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached data + model
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading telemetry workbook\u2026")
def load_data() -> tuple[pd.DataFrame, int]:
    raw_rows, df = D.load_dataset()
    return df, raw_rows


@st.cache_resource(show_spinner="Training Is_Flood classifier (Random Forest)\u2026")
def train(df: pd.DataFrame) -> M.ModelBundle:
    return M.train_model(df)


df, RAW_ROWS = load_data()
bundle = train(df)

TMIN = df[D.CWC_TIME].min().date()
TMAX = df[D.CWC_TIME].max().date()


def kpi_card(label: str, value: str, sub: str, accent: str) -> str:
    return (
        f'<div class="kpi" style="--accent:{accent}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


# --------------------------------------------------------------------------
# Sidebar: context & data quality
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### \U0001F4CA Data Source")
    st.caption(f"`{D.resolve_data_path().name}`")
    for note in D.data_quality_notes(df, raw_rows=RAW_ROWS):
        st.caption(f"\u2022 {note}")
    with st.expander("Model notes"):
        st.caption(f"Train {bundle.n_train:,} / Test {bundle.n_test:,} rows "
                   f"\u2014 flood prevalence {bundle.positive_rate:.0%}.")
        for n in bundle.notes:
            st.caption(n)
    st.caption("Random Forest \u2192 `Is_Flood` (0/1). "
               "Features: water level, rainfall, 1h change, warning & "
               "danger levels, and margin gaps to both thresholds.")

# --------------------------------------------------------------------------
# Hero + GLOBAL TOP BAR: station search, filters, KPI cards
# --------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>\U0001F30A CWC Flood Monitoring &amp; '
    'Predictive Dashboard</h1>'
    "<p>Cauvery basin telemetry \u2014 real-time status, flood-risk ML "
    "prediction, trends &amp; GIS overview</p></div>",
    unsafe_allow_html=True,
)

# Global filters -----------------------------------------------------------
fcol1, fcol2, fcol3, fcol4 = st.columns([3, 2, 2, 3])
with fcol1:
    state_opts = ["All"] + sorted(df["State"].dropna().unique().tolist())
    sel_state = st.selectbox("State", state_opts, key="f_state")
with fcol2:
    basin_pool = df if sel_state == "All" else df[df["State"] == sel_state]
    basin_opts = ["All"] + sorted(basin_pool["Basin"].dropna().unique().tolist())
    sel_basin = st.selectbox("Basin", basin_opts, key="f_basin")
with fcol3:
    river_pool = basin_pool if sel_basin == "All" else basin_pool[basin_pool["Basin"] == sel_basin]
    river_opts = ["All"] + sorted(river_pool["River"].dropna().unique().tolist())
    sel_river = st.selectbox("River", river_opts, key="f_river")
with fcol4:
    date_range = st.date_input(
        "Date Range", value=(TMIN, TMAX), min_value=TMIN, max_value=TMAX,
        key="f_dates",
    )

# Normalise date_input (returns 1 or 2 dates while user is picking).
if isinstance(date_range, tuple) and len(date_range) == 2:
    d_start, d_end = date_range
elif isinstance(date_range, tuple) and len(date_range) == 1:
    d_start, d_end = date_range[0], TMAX
else:
    d_start, d_end = date_range, TMAX


def geo_filter(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply State / Basin / River dropdowns."""
    out = frame
    if sel_state != "All":
        out = out[out["State"] == sel_state]
    if sel_basin != "All":
        out = out[out["Basin"] == sel_basin]
    if sel_river != "All":
        out = out[out["River"] == sel_river]
    return out


filtered = geo_filter(df)
filtered = filtered[
    (filtered[D.CWC_TIME].dt.date >= d_start) & (filtered[D.CWC_TIME].dt.date <= d_end)
]

# Station search bar -------------------------------------------------------
avail_stations = sorted(filtered["Station"].unique().tolist()) if len(filtered) else []
if not avail_stations:
    st.warning("No stations match the current filters \u2014 widen the selection.")
    st.stop()

sc1, sc2 = st.columns([2.2, 1])
with sc1:
    if st.session_state.get("pending_station") in avail_stations:
        st.session_state["station"] = st.session_state.pop("pending_station")
    if st.session_state.get("station") not in avail_stations:
        st.session_state["station"] = avail_stations[0]
    station = st.selectbox(
        "\U0001F50D Station Search \u2014 type a station name",
        avail_stations, key="station",
        help="Type to filter. Selecting a station re-filters every box below.",
    )
with sc2:
    st.markdown(
        f'<div class="kpi" style="--accent:#0ea5e9;min-height:0;padding:10px 14px">'
        f'<div class="value" style="font-size:1.3rem">{len(avail_stations)}</div>'
        f'<div class="sub">stations in view / {df["Station"].nunique()} total</div></div>',
        unsafe_allow_html=True,
    )

# KPI cards ----------------------------------------------------------------
scope = filtered[filtered["Station"].isin(avail_stations)]
latest = scope.sort_values(D.CWC_TIME).groupby("Station", as_index=False).tail(1)
warn_danger = int((latest[D.CWC_STATUS] != "Normal").sum())
flood_events = int(scope[D.TARGET].sum())
avg_rain = float(scope[D.CWC_RAIN].mean()) if len(scope) else 0.0
total_stations = len(avail_stations)

k1, k2, k3, k4 = st.columns(4)
k1.markdown(kpi_card("Total Telemetry Stations", f"{total_stations}",
                     "unique stations matching filters", "#0ea5e9"),
            unsafe_allow_html=True)
k2.markdown(kpi_card("Average Rainfall (mm)", f"{avg_rain:.1f}",
                     "mean hourly rainfall in period", "#6366f1"),
            unsafe_allow_html=True)
k3.markdown(kpi_card("Warning / Danger Stations", f"{warn_danger}",
                     "latest reading at or above warning",
                     "#f59e0b" if warn_danger else "#22c55e"),
            unsafe_allow_html=True)
k4.markdown(kpi_card("Total Flood Events", f"{flood_events:,}",
                     "`Is_Flood = 1` readings in period", "#ef4444"),
            unsafe_allow_html=True)

st.divider()

# ==========================================================================
# BOX 1 â€” Station Search & Detailed Profile
# ==========================================================================
st.subheader("\U0001F4CD Station Profile: " + station)

hist_all = D.station_history(df, station)
hist = hist_all[
    (hist_all[D.CWC_TIME].dt.date >= d_start) & (hist_all[D.CWC_TIME].dt.date <= d_end)
]
if hist.empty:  # station exists but outside date range -> fall back to all data
    hist = hist_all

meta_row = df[df["Station"] == station].iloc[-1]
cur = hist.iloc[-1]
wl_now = float(cur[D.CWC_WL])
warn_m = float(cur[D.CWC_WARN])
danger_m = float(cur[D.CWC_DANGER])
chg_now = float(cur[D.CWC_CHANGE])
status_now = str(cur[D.CWC_STATUS])

bc1, bc2 = st.columns([1.15, 1])

with bc1:
    st.markdown("**Station Metadata**")
    mdf = pd.DataFrame(
        [
            ("River", meta_row.get("River")),
            ("Basin", meta_row.get("Basin")),
            ("Tributary", meta_row.get("Tributary")),
            ("District", meta_row.get("District")),
            ("State", meta_row.get("State")),
            ("Latitude", f"{meta_row.get('Latitude'):.6f}"),
            ("Longitude", f"{meta_row.get('Longitude'):.6f}"),
            ("Zero Gauge RL (m)", f"{meta_row.get('RL_of_zeroGauge'):.3f}"),
            ("Mean Sea Level (m)", f"{meta_row.get('MeanSeaLevel'):.3f}"),
            ("Discharge Data", meta_row.get("Is_DischargeDataAvailable")),
            ("Last Reading", f"{cur[D.CWC_TIME]:%d %b %Y %H:%M}"),
        ],
        columns=["Field", "Value"],
    ).set_index("Field")
    st.dataframe(mdf, height=372, width="stretch")

with bc2:
    st.markdown("**Current Status Gauge**")
    st.plotly_chart(C.status_gauge(status_now, wl_now, warn_m, danger_m),
                    width="stretch")

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Latest Water Level", f"{wl_now:,.3f} m",
           f"{cur[D.CWC_TIME]:%d %b %H:%M}")
mc2.metric("Warning Level", f"{warn_m:,.3f} m",
           f"gap {wl_now - warn_m:+.3f} m")
mc3.metric("Danger Level", f"{danger_m:,.3f} m",
           f"gap {wl_now - danger_m:+.3f} m")
mc4.metric("1-Hour WL Change", f"{chg_now:+.3f} m",
           "rising" if chg_now > 0 else "falling" if chg_now < 0 else "steady")

st.divider()

# ==========================================================================
# BOX 2 â€” Predictive Model & Flood Possibility Calculator
# ==========================================================================
st.subheader("\U0001F916 Flood Risk & Possibility Model")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{bundle.metrics['Accuracy']:.1%}")
m2.metric("Precision", f"{bundle.metrics['Precision']:.1%}")
m3.metric("Recall", f"{bundle.metrics['Recall']:.1%}")
m4.metric("ROC-AUC", f"{bundle.metrics['ROC-AUC']:.3f}")

with st.expander("Confusion matrix & ROC curve"):
    e1, e2 = st.columns(2)
    cm = bundle.confusion
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"],
                         columns=["Pred 0", "Pred 1"])
    e1.dataframe(cm_df, width=320)
    e2.plotly_chart(C.roc_curve_chart(bundle.fpr, bundle.tpr,
                                      bundle.metrics["ROC-AUC"]),
                    width="stretch")

# --- interactive predictor: auto-loaded from the searched station ---
st.markdown(
    f"**Latest telemetry for `{station}` is preloaded below \u2014 "
    "adjust any slider to run what-if scenarios.**"
)

lo_wl = float(hist[D.CWC_WL].min())
hi_wl = float(hist[D.CWC_WL].max())
span_wl = max(hi_wl - lo_wl, 0.5)
lo_ch = float(hist[D.CWC_CHANGE].min())
hi_ch = float(hist[D.CWC_CHANGE].max())
span_ch = max(hi_ch - lo_ch, 0.2)

p1, p2, p3 = st.columns(3)
with p1:
    in_wl = st.slider(
        "River Water Level (m)",
        min_value=round(lo_wl - span_wl * 0.1, 2),
        max_value=round(hi_wl + span_wl * 0.1, 2),
        value=min(max(round(wl_now, 2), round(lo_wl - span_wl * 0.1, 2)),
                  round(hi_wl + span_wl * 0.1, 2)),
        step=0.01, key=f"p_wl_{station}",
        help=f"Auto-loaded from {station} latest reading: {wl_now:.3f} m",
    )
with p2:
    in_ch = st.slider(
        "1-Hour Water Level Change (m)",
        min_value=round(lo_ch - span_ch * 0.1, 2),
        max_value=round(hi_ch + span_ch * 0.1, 2),
        value=min(max(round(chg_now, 2), round(lo_ch - span_ch * 0.1, 2)),
                  round(hi_ch + span_ch * 0.1, 2)),
        step=0.01, key=f"p_ch_{station}",
        help=f"Auto-loaded: {chg_now:+.3f} m",
    )
with p3:
    max_rain = int(max(df[D.CWC_RAIN].max(), 50))
    in_rain = st.slider(
        "Rainfall (mm/h)", min_value=0, max_value=max_rain,
        value=min(int(cur[D.CWC_RAIN]), max_rain),
        step=1, key=f"p_rain_{station}",
        help="Source dataset reports 0 mm everywhere \u2014 use this slider to "
             "explore hypothetical rainfall.",
    )

pred_row = {
    D.CWC_WL: float(in_wl),
    D.CWC_RAIN: float(in_rain),
    D.CWC_CHANGE: float(in_ch),
    D.CWC_WARN: warn_m,
    D.CWC_DANGER: danger_m,
    D.GAP_WARNING: float(in_wl) - warn_m,
    D.GAP_DANGER: float(in_wl) - danger_m,
}
prob = bundle.predict_proba_one(pred_row)
label, emoji, colour = M.risk_category(prob)

g1, g2 = st.columns([1, 1.1])
with g1:
    st.plotly_chart(C.probability_gauge(prob), width="stretch")
    st.markdown(
        f'<div style="text-align:center;margin-top:-14px">'
        f'<span class="badge" style="background:{colour}">{emoji} {label}</span>'
        f'<div style="color:#475569;font-size:.85rem;margin-top:8px">'
        f'Flood probability = <b>{prob:.1%}</b></div></div>',
        unsafe_allow_html=True,
    )
with g2:
    contrib = bundle.contributions(pred_row)
    st.plotly_chart(C.contributions_chart(contrib), width="stretch")

i1, i2 = st.columns(2)
i1.plotly_chart(C.importance_chart(bundle.importances), width="stretch")
with i2:
    st.markdown("**Top drivers of this prediction**")
    for feat, val in contrib.head(3).items():
        direction = "pushes risk UP" if val >= 0 else "pulls risk DOWN"
        st.write(
            f"- `{feat}` \u2192 **{direction}** by "
            f"**{abs(val) * 100:.1f} percentage points**"
        )
    if M.HAS_SHAP:
        st.caption(
            "Explanation method: **SHAP TreeExplainer** \u2014 exact per-feature "
            "contributions (base rate + all contributions = the predicted "
            "probability shown above)."
        )
    else:
        st.caption(
            "Explanation method: leave-one-feature-at-training-median ablation "
            "\u2014 each bar is the probability change when that single feature "
            "is reset to its training median."
        )

st.divider()

# ==========================================================================
# BOX 3 â€” Real-Time Water Level Trends
# ==========================================================================
st.subheader("\U0001F4CA Real-Time Water Level Trends")
st.plotly_chart(C.water_level_trend(hist), width="stretch")
n_flood = int(hist[D.TARGET].sum())
st.caption(
    f"{len(hist):,} readings \u00b7 {d_start} \u2192 {d_end} \u00b7 "
    f"**{n_flood:,} flood hours** (`Is_Flood = 1`) shaded in red \u00b7 "
    "yellow dashed = warning level, red dashed = danger level."
)
st.divider()

# ==========================================================================
# BOX 4 â€” GIS Interactive Map
# ==========================================================================
st.subheader("\U0001F5FA\uFE0F Telemetry Station Map")

map_df = scope.sort_values(D.CWC_TIME).groupby("Station", as_index=False).tail(1).copy()
map_df = map_df.reset_index(drop=True)
map_df["Flood_Probability"] = bundle.model.predict_proba(
    map_df[D.FEATURE_COLUMNS]
)[:, 1]

fig_map = C.station_map(map_df, station)
evt = st.plotly_chart(
    fig_map, width="stretch", key="map_chart",
    on_select="rerun", selection_mode="points",
)

# Click / point-selection -> make that station active.
pts = ((evt or {}).get("selection") or {}).get("points") or []
if pts:
    p0 = pts[0]
    clicked = None
    cd = p0.get("customdata")
    if cd is not None and len(cd):
        cand = str(cd[0])
        if cand in avail_stations:
            clicked = cand
    if clicked is None:
        idx = p0.get("point_number", p0.get("point_index"))
        if idx is not None and 0 <= int(idx) < len(map_df):
            cand = map_df.iloc[int(idx)]["Station"]
            if cand in avail_stations:
                clicked = cand
    if clicked and clicked != station:
        st.session_state["pending_station"] = clicked
        st.rerun()

mc1, mc2 = st.columns([2, 1])
with mc2:
    def _map_pick_changed():
        st.session_state["pending_station"] = st.session_state["map_station_pick"]

    # Keep the widget in sync with the active station (programmatic set does
    # not trigger on_change, so no feedback loop).
    st.session_state["map_station_pick"] = station
    st.selectbox(
        "\u2691 Or set active station manually", avail_stations,
        key="map_station_pick", on_change=_map_pick_changed,
        help="Marker colours: green = Normal, yellow = Warning, red = Danger/Flood.",
    )
st.caption(
    "Hover any marker for name, current river level, rainfall and live flood "
    "probability. Click a marker to load that station across the dashboard "
    "(large \u2b50 star = active station)."
)
st.divider()

# ==========================================================================
# BOX 5 â€” Rainfall vs. Water Surge Correlation & Logs
# ==========================================================================
st.subheader("\U0001F327\uFE0F Rainfall vs. Water Surge Analysis")

st.plotly_chart(C.rainfall_vs_surge(hist), width="stretch")

if hist[D.CWC_RAIN].nunique() <= 1:
    st.caption(
        "Correlation (rainfall vs. 1h surge): **undefined** \u2014 `Rainfall_mm` "
        "is constant (0 mm) in the source workbook, so it has no variance to "
        "correlate. The chart above will populate once real rain-gauge data is "
        "loaded."
    )
else:
    cor = float(hist[D.CWC_RAIN].corr(hist[D.CWC_CHANGE]))
    st.caption(f"Correlation (rainfall vs. 1h surge): **r = {cor:+.3f}**")

# --- searchable telemetry log ---------------------------------------------
st.markdown("**Telemetry log \u2014 search, and flood rows are highlighted red**")
q = st.text_input(
    "\U0001F50D Search logs", key="log_search",
    placeholder="e.g. Danger, 2026-05, Mandya, warning\u2026",
)

LOG_COLS = [
    D.CWC_TIME, "Station", "District", "Tehsil", "River",
    D.CWC_WL, D.CWC_WARN, D.CWC_DANGER, D.CWC_CHANGE,
    D.CWC_STATUS, D.CWC_RAIN, D.TARGET,
]
logs = hist[LOG_COLS].sort_values(D.CWC_TIME, ascending=False).copy()
logs["Time"] = logs[D.CWC_TIME].dt.strftime("%d-%m-%Y %H:%M")
logs = logs.drop(columns=[D.CWC_TIME])

if q.strip():
    mask = logs["Time"].str.contains(q, case=False, na=False)
    for col in ["Station", "District", "Tehsil", "River", D.CWC_STATUS]:
        mask |= logs[col].astype(str).str.contains(q, case=False, na=False)
    for col in [D.CWC_WL, D.CWC_RAIN, D.TARGET]:
        mask |= logs[col].astype(str).str.contains(q, case=False, na=False)
    logs = logs[mask]

ORDER = ["Time", "Station", "District", "Tehsil", "River",
         D.CWC_WL, D.CWC_WARN, D.CWC_DANGER, D.CWC_CHANGE,
         D.CWC_STATUS, D.CWC_RAIN, D.TARGET]
logs = logs[ORDER]

st.caption(f"{len(logs):,} rows \u00b7 sorted newest first")
st.dataframe(
    logs.style
    .format({
        D.CWC_WL: "{:.3f}", D.CWC_WARN: "{:.2f}", D.CWC_DANGER: "{:.2f}",
        D.CWC_CHANGE: "{:+.3f}",
    }, na_rep="\u2014")
    .apply(
        lambda row: ["background-color:#fee2e2"] * len(row)
        if row[D.TARGET] == 1 else [""] * len(row),
        axis=1,
    ),
    height=420, width="stretch", hide_index=True,
)

st.caption(
    "\U0001F534 soft red rows = historical flood events (`Is_Flood = 1`). "
    "Use the search box to filter by time, status, district, level\u2026"
)

