"""Load, validate, and normalize all three CSV types."""

import ast
import pandas as pd
import numpy as np
import re


def _parse_timezone_offset(tz_str: str) -> str | None:
    """Convert 'UTC-05:00' or 'UTC' into a pandas-compatible fixed offset string."""
    if not tz_str or pd.isna(tz_str):
        return None
    tz_str = str(tz_str).strip()
    if tz_str == "UTC":
        return "UTC"
    m = re.match(r"UTC([+-]\d{2}:\d{2})", tz_str)
    if m:
        return f"Etc/GMT{'+' if m.group(1)[0] == '-' else '-'}{int(m.group(1)[1:3])}"
    return None


def _resolve_tz(tz_name: str) -> str:
    """Best-effort timezone string to pytz-compatible name."""
    if not tz_name or pd.isna(tz_name):
        return "UTC"
    tz_name = str(tz_name).strip()
    offset = _parse_timezone_offset(tz_name)
    if offset:
        return offset
    try:
        import zoneinfo
        zoneinfo.ZoneInfo(tz_name)
        return tz_name
    except Exception:
        return "UTC"


def load_test_results(path: str) -> pd.DataFrame:
    """Load a test-results CSV (reaction results format).

    Returns a cleaned DataFrame with:
    - datetime index in UTC
    - local_time column converted from device_timezone
    - filtered: is_deleted=false, is_failed=false
    - score cast to float
    """
    df = pd.read_csv(path)

    if "type" not in df.columns or "score" not in df.columns:
        raise ValueError(f"{path} does not look like a test-results CSV (missing 'type' or 'score' columns)")

    df["created_at"] = pd.to_datetime(df["created_at"], format="mixed", utc=True)

    for col in ("is_deleted", "is_failed"):
        if col in df.columns:
            df = df[df[col].astype(str).str.lower() != "true"].copy()

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["is_baseline"] = df["is_baseline"].astype(str).str.lower() == "true"

    tz_col = "device_timezone"
    local_tz = None
    if tz_col in df.columns:
        tz_counts = df[tz_col].dropna().value_counts()
        if len(tz_counts):
            local_tz = _resolve_tz(tz_counts.index[0])

    if local_tz:
        df["local_time"] = df["created_at"].dt.tz_convert(local_tz)
    else:
        df["local_time"] = df["created_at"]

    df["date"] = df["local_time"].dt.date
    df["hour"] = df["local_time"].dt.hour + df["local_time"].dt.minute / 60.0

    df = df.sort_values("created_at").reset_index(drop=True)
    return df


def load_sleep_sessions(path: str) -> pd.DataFrame:
    """Load a sleep-sessions CSV (fatigue results format).

    Returns a cleaned DataFrame with per-night sleep architecture,
    recovery scores, sleep debt, and HRV data.
    """
    df = pd.read_csv(path)

    required = {"total_sleep_time_min", "sleep_start_u_t_c"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} does not look like a sleep-sessions CSV (missing {required - set(df.columns)})")

    df["sleep_start"] = pd.to_datetime(df["sleep_start_u_t_c"], unit="s", utc=True)
    df["sleep_end"] = pd.to_datetime(df["sleep_end_u_t_c"], unit="s", utc=True)

    tz_col = "time_zone"
    local_tz = "US/Eastern"
    if tz_col in df.columns:
        tz_counts = df[tz_col].dropna().value_counts()
        if len(tz_counts):
            local_tz = _resolve_tz(tz_counts.index[0])

    df["sleep_start_local"] = df["sleep_start"].dt.tz_convert(local_tz)
    df["sleep_end_local"] = df["sleep_end"].dt.tz_convert(local_tz)
    df["night_date"] = df["sleep_start_local"].dt.date

    numeric_cols = [
        "total_session_time_min", "total_sleep_time_min", "total_wake_time_min",
        "total_wake", "total_light", "total_deep", "total_rem",
        "stress_score", "recovery_score", "sleep_needed_min",
        "sleep_debt_min", "avg_hr_bpm_when_wake", "max_hr_bpm",
        "avg_hrv_rmssd_ms", "circadian_compliance", "day_split_hour",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("sleep_start").reset_index(drop=True)
    return df


def load_decoded_metrics(path: str) -> pd.DataFrame:
    """Load a decoded-metrics CSV (30-sec epoch sensor data).

    Returns a DataFrame with:
    - datetime column derived from unix timestamp
    - numeric columns cast properly
    """
    df = pd.read_csv(path)

    if "acc_x_count" not in df.columns:
        raise ValueError(f"{path} does not look like a decoded-metrics CSV (missing 'acc_x_count')")

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["datetime_et"] = df["datetime"].dt.tz_convert("US/Eastern")

    numeric_cols = df.columns.difference(["wear_mode", "lifecycle_state", "datetime", "datetime_et"])
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
