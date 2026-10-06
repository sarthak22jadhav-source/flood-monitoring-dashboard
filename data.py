"""
Data loading & preparation for the Flood Monitoring Dashboard.

- Reads the CWC telemetry Excel workbook.
- Cleans / parses the `Data Acquisition Time` column.
- Removes duplicated (Station, timestamp) blocks that exist in the source file
  (50,000 raw rows -> 20,507 unique telemetry readings, SlNo 1..20507).
- Engineers the margin-gap features used by the ML model.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# Column names (kept as constants so renames are one-line changes).
CWC_TIME = "Data Acquisition Time"
CWC_WL = "River Water Level Telemetry Hourly (meter)"
CWC_WARN = "Warning_Level_m"
CWC_DANGER = "Danger_Level_m"
CWC_CHANGE = "Water_Level_Change_1h"
CWC_STATUS = "Water_Level_Status"
CWC_FLOOD = "Is_Flood"
CWC_RAIN = "Rainfall_mm"

GAP_WARNING = "Gap_to_Warning_m"   # WL - Warning (negative = below warning)
GAP_DANGER = "Gap_to_Danger_m"     # WL - Danger  (negative = below danger)

TARGET = CWC_FLOOD

# Model feature set required by the specification.
FEATURE_COLUMNS = [
    CWC_WL,
    CWC_RAIN,
    CWC_CHANGE,
    CWC_WARN,
    CWC_DANGER,
    GAP_WARNING,
    GAP_DANGER,
]

METADATA_COLUMNS = [
    "Station",
    "State",
    "District",
    "Tehsil",
    "River",
    "Basin",
    "Tributary",
    "Local River",
    "Latitude",
    "Longitude",
    "RL_of_zeroGauge",
    "MeanSeaLevel",
    "Is_DischargeDataAvailable",
]

# Candidate locations for the workbook, checked in order:
# 1) env var override, 2) alongside this file, 3) ./data/, 4) repo root, 5) OneDrive desktop.
_CANDIDATE_PATHS = [
    os.environ.get("CWC_DATA_PATH"),
    Path(__file__).parent / "cwc_telemetry_flood_ready_50000.xlsx",
    Path(__file__).parent / "data" / "cwc_telemetry_flood_ready_50000.xlsx",
    Path(__file__).parent.parent / "cwc_telemetry_flood_ready_50000.xlsx",
    Path.home() / "OneDrive" / "Desktop" / "cwc_telemetry_flood_ready_50000.xlsx",
]


def resolve_data_path() -> Path:
    """Return the first existing path for the telemetry workbook."""
    for p in _CANDIDATE_PATHS:
        if p:
            path = Path(p)
            if path.exists():
                return path
    searched = "\n".join(f"  - {p}" for p in _CANDIDATE_PATHS if p)
    raise FileNotFoundError(
        "Could not find 'cwc_telemetry_flood_ready_50000.xlsx'. Searched:\n"
        f"{searched}\n"
        "Set the CWC_DATA_PATH environment variable or place the file next to app.py."
    )


def load_dataset() -> tuple[int, pd.DataFrame]:
    """Read + clean the workbook; return (raw_row_count, cleaned_df)."""
    path = resolve_data_path()
    raw = pd.read_excel(path)
    return len(raw), clean(raw)


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the workbook with timestamp parsing applied."""
    path = path or resolve_data_path()
    df = pd.read_excel(path)
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps, de-duplicate and engineer model features."""
    df = df.copy()

    # 1. Parse 'Data Acquisition Time' (format: dd-mm-yyyy HH:MM).
    df[CWC_TIME] = pd.to_datetime(df[CWC_TIME], format="%d-%m-%Y %H:%M", errors="coerce")
    bad_ts = df[CWC_TIME].isna().sum()
    if bad_ts:
        df = df.dropna(subset=[CWC_TIME])

    # 2. The workbook stores the same telemetry block 2-3 times (identical SlNo).
    #    Keep the first occurrence of each (Station, timestamp) reading.
    df = df.sort_values(["Station", CWC_TIME, "SlNo"], kind="mergesort")
    df = df.drop_duplicates(subset=["Station", CWC_TIME], keep="first")

    # 3. Margin gaps to warning / danger thresholds (model features).
    df[GAP_WARNING] = df[CWC_WL] - df[CWC_WARN]
    df[GAP_DANGER] = df[CWC_WL] - df[CWC_DANGER]

    # 4. Ensure target & numeric feature columns are numeric.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Convenience derived columns used by charts / filters.
    df["Date"] = df[CWC_TIME].dt.normalize()
    df["Hour"] = df[CWC_TIME].dt.hour
    df = df.reset_index(drop=True)
    return df


def station_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """One row per station with static metadata + latest reading."""
    cols = [c for c in METADATA_COLUMNS if c in df.columns]
    latest = latest_readings(df)
    meta = df.sort_values(CWC_TIME).groupby("Station", as_index=False).first()[cols]
    meta = meta.merge(
        latest[["Station", CWC_TIME, CWC_WL, CWC_STATUS, CWC_RAIN, CWC_CHANGE, TARGET]],
        on="Station",
        how="left",
        suffixes=("", "_latest"),
    )
    return meta


def latest_readings(df: pd.DataFrame) -> pd.DataFrame:
    """Most recent telemetry row per station."""
    return df.sort_values(CWC_TIME).groupby("Station", as_index=False).tail(1)


def station_history(df: pd.DataFrame, station: str) -> pd.DataFrame:
    """All readings for one station, time-ascending."""
    return df[df["Station"] == station].sort_values(CWC_TIME).reset_index(drop=True)


def data_quality_notes(df: pd.DataFrame, raw_rows: int | None = None) -> list[str]:
    """Return human-readable data quality notes for the sidebar disclosure."""
    notes: list[str] = []
    if raw_rows and raw_rows != len(df):
        notes.append(
            f"De-duplicated {raw_rows:,} raw rows -> {len(df):,} unique "
            "(Station, timestamp) readings."
        )
    if df[CWC_RAIN].nunique() <= 1:
        notes.append(
            f"`{CWC_RAIN}` is constant (= {df[CWC_RAIN].iloc[0]}) in this dataset, "
            "so rainfall contributes no predictive signal until real rain-gauge "
            "data is supplied."
        )
    notes.append(
        f"Time range: {df[CWC_TIME].min():%d %b %Y %H:%M} \u2192 "
        f"{df[CWC_TIME].max():%d %b %Y %H:%M} ({df['Station'].nunique()} stations)."
    )
    return notes
