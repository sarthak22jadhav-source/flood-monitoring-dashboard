"""
Plotly chart builders for every dashboard box.

All functions are pure: DataFrame/list in, plotly Figure out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import (
    CWC_CHANGE,
    CWC_DANGER,
    CWC_RAIN,
    CWC_STATUS,
    CWC_TIME,
    CWC_WARN,
    CWC_WL,
    TARGET,
)

COLOR_NORMAL = "#22c55e"
COLOR_WARN = "#eab308"
COLOR_DANGER = "#ef4444"
COLOR_WATER = "#2563eb"
COLOR_RAIN = "#0ea5e9"

STATUS_COLORS = {
    "Normal": COLOR_NORMAL,
    "Warning": COLOR_WARN,
    "Danger (Flood)": COLOR_DANGER,
}

PLOT_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    font=dict(size=12),
)


def _flood_intervals(hist: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous time ranges where Is_Flood == 1 (for shaded bands)."""
    if TARGET not in hist.columns or not hist[TARGET].any():
        return []
    flood = hist[TARGET].eq(1)
    groups = (flood != flood.shift()).cumsum()
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for _, chunk in hist[flood].groupby(groups[flood]):
        intervals.append((chunk[CWC_TIME].iloc[0], chunk[CWC_TIME].iloc[-1]))
    return intervals


def water_level_trend(hist: pd.DataFrame) -> go.Figure:
    """Box 3: hourly WL line + warning/danger thresholds + flood zone bands."""
    fig = go.Figure()

    # Shaded flood zones first (so the line draws on top).
    for start, end in _flood_intervals(hist):
        if start == end:
            end = end + pd.Timedelta(hours=1)
        fig.add_vrect(
            x0=start, x1=end, fillcolor="rgba(239,68,68,0.13)",
            line_width=0, layer="below",
        )

    fig.add_trace(
        go.Scatter(
            x=hist[CWC_TIME], y=hist[CWC_WL],
            mode="lines", name="Water Level (m)",
            line=dict(color=COLOR_WATER, width=2),
            hovertemplate="%{x|%d %b %Y %H:%M}<br>%{y:.3f} m<extra></extra>",
        )
    )

    warn = float(hist[CWC_WARN].iloc[-1])
    danger = float(hist[CWC_DANGER].iloc[-1])
    fig.add_hline(y=warn, line=dict(color=COLOR_WARN, width=2, dash="dash"),
                  annotation_text=f"Warning {warn:g} m",
                  annotation_position="bottom right",
                  annotation_font_color="#a16207")
    fig.add_hline(y=danger, line=dict(color=COLOR_DANGER, width=2, dash="dash"),
                  annotation_text=f"Danger {danger:g} m",
                  annotation_position="top right",
                  annotation_font_color="#b91c1c")

    yvals = hist[CWC_WL]
    y0 = min(yvals.min(), warn) - abs(yvals.min()) * 0.01 - 0.5
    y1 = max(yvals.max(), danger) + abs(yvals.max()) * 0.01 + 0.5
    fig.update_layout(
        **PLOT_LAYOUT,
        title="Hourly River Water Level vs. Warning / Danger Thresholds",
        xaxis_title="Time",
        yaxis_title="Water Level (m)",
        yaxis=dict(range=[y0, y1]),
        height=430,
        hovermode="x unified",
    )
    return fig


def rainfall_vs_surge(hist: pd.DataFrame) -> go.Figure:
    """Box 5: dual-axis Rainfall_mm bars vs Water_Level_Change_1h line."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=hist[CWC_TIME], y=hist[CWC_RAIN],
            name="Rainfall (mm)", marker_color=COLOR_RAIN,
            opacity=0.75,
            hovertemplate="%{x|%d %b %Y %H:%M}<br>%{y} mm<extra>Rainfall</extra>",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=hist[CWC_TIME], y=hist[CWC_CHANGE],
            mode="lines", name="dWater Level 1h (m)",
            line=dict(color="#f97316", width=1.6),
            hovertemplate="%{x|%d %b %Y %H:%M}<br>%{y:.3f} m<extra>dLevel</extra>",
        ),
        secondary_y=False,
    )

    fig.update_layout(**PLOT_LAYOUT, height=400, hovermode="x unified",
                      title="Rainfall vs. Water Surge (hourly)")
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Water Level Change (m)", secondary_y=False)
    fig.update_yaxes(title_text="Rainfall (mm)", showgrid=False, secondary_y=True)
    return fig


def probability_gauge(probability: float) -> go.Figure:
    """Box 2: 0-100% flood probability gauge with risk colour bands."""
    pct = probability * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            number={"suffix": "%", "font": {"size": 42}},
            delta={"reference": 50, "suffix": " vs 50% baseline",
                   "font": {"size": 14}},
            title={"text": "Flood Possibility",
                   "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#1f2937", "thickness": 0.25},
                "steps": [
                    {"range": [0, 30], "color": "rgba(34,197,94,0.45)"},
                    {"range": [30, 70], "color": "rgba(234,179,8,0.45)"},
                    {"range": [70, 100], "color": "rgba(239,68,68,0.55)"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 4},
                    "value": pct,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=50, r=50, t=60, b=10))
    return fig


def contributions_chart(contrib: pd.Series) -> go.Figure:
    """Box 2: horizontal bar of signed local feature contributions (%-points)."""
    labels = contrib.index[::-1]
    values = (contrib.values[::-1]) * 100  # probability points
    colors = ["#16a34a" if v >= 0 else "#2563eb" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:+.2f} pp<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        title="Why this risk score? (feature push on flood probability)",
        xaxis_title="Contribution to probability (percentage points)",
        yaxis_title="",
        height=360,
    )
    fig.add_vline(x=0, line_color="#9ca3af", line_width=1)
    return fig


def importance_chart(importances: pd.Series) -> go.Figure:
    """Box 2: global model feature importance."""
    fig = go.Figure(
        go.Bar(
            x=importances.values[::-1] * 100,
            y=importances.index[::-1],
            orientation="h",
            marker_color="#6366f1",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        title="Global Feature Importance (model)",
        xaxis_title="Importance (%)",
        height=360,
    )
    return fig


def roc_curve_chart(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"ROC (AUC = {auc:.3f})",
                             line=dict(color=COLOR_WATER, width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Chance", line=dict(dash="dash", color="#9ca3af")))
    fig.update_layout(**PLOT_LAYOUT, height=300,
                      xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig


def status_gauge(status: str, level: float, warn: float, danger: float) -> go.Figure:
    """Box 1: semicircular gauge showing current level against thresholds."""
    top = max(danger, level) * 1.05 if max(danger, level) > 0 else 1
    color = STATUS_COLORS.get(status, "#9ca3af")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=level,
            number={"suffix": " m", "font": {"size": 30}},
            title={"text": f"Status: <b>{status}</b>", "font": {"size": 17}},
            gauge={
                "axis": {"range": [0, top], "ticksuffix": " m"},
                "bar": {"color": color, "thickness": 0.3},
                "steps": [
                    {"range": [0, warn], "color": "rgba(34,197,94,0.35)"},
                    {"range": [warn, danger], "color": "rgba(234,179,8,0.4)"},
                    {"range": [danger, top], "color": "rgba(239,68,68,0.45)"},
                ],
                "threshold": {"line": {"color": "#111827", "width": 3},
                              "value": level},
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=40, r=40, t=60, b=10))
    return fig


def station_map(meta: pd.DataFrame, selected: str) -> go.Figure:
    """
    Box 4: token-free Plotly map (OpenStreetMap tiles).

    Marker colour = latest status; hover carries name, level, rainfall and
    flood probability (click handling is done in app.py).
    """
    m = meta.copy().reset_index(drop=True)
    m["color"] = m[CWC_STATUS].map(STATUS_COLORS).fillna("#9ca3af")
    # Active station: bigger + star symbol (map markers have no outline prop).
    m["size"] = np.where(m["Station"] == selected, 26.0, 15.0)
    m["symbol"] = np.where(m["Station"] == selected, "star", "circle")

    has_prob = "Flood_Probability" in m.columns
    if has_prob:
        hover = (
            "<b>%{customdata[0]}</b><br>"
            "Level: %{customdata[1]:.2f} m<br>"
            "Status: %{customdata[2]}<br>"
            "Rainfall: %{customdata[3]} mm<br>"
            "Flood probability: %{customdata[4]:.0%}"
            "<extra>Click to select</extra>"
        )
        custom = m[["Station", CWC_WL, CWC_STATUS, CWC_RAIN,
                    "Flood_Probability"]].values
    else:
        hover = (
            "<b>%{customdata[0]}</b><br>"
            "Level: %{customdata[1]:.2f} m<br>"
            "Status: %{customdata[2]}<br>"
            "Rainfall: %{customdata[3]} mm"
            "<extra>Click to select</extra>"
        )
        custom = m[["Station", CWC_WL, CWC_STATUS, CWC_RAIN]].values

    fig = go.Figure(
        go.Scattermap(
            lat=m["Latitude"], lon=m["Longitude"],
            mode="markers",
            marker=dict(size=m["size"], color=m["color"],
                        opacity=0.9, sizemode="diameter",
                        symbol=m["symbol"]),
            text=m["Station"],
            customdata=custom,
            hovertemplate=hover,
            name="Stations",
        )
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        title="Telemetry Station Network \u2014 click a marker to load it",
        clickmode="event+select",
        map=dict(
            style="open-street-map",
            center={"lat": float(m["Latitude"].mean()),
                    "lon": float(m["Longitude"].mean())},
            zoom=6,
        ),
        showlegend=False,
        height=520,
    )
    return fig
